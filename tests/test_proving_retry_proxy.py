import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, List

import pytest

from proving.retry_proxy import RetryProxy


class Upstream(BaseHTTPRequestHandler):
    """Refuses the first `refusals` requests with 429, then answers (streamed in two chunks)."""

    refusals: int = 0
    seen: List[Any] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Upstream.seen.append((self.path, self.headers.get("Authorization"), json.loads(body)))
        if Upstream.refusals > 0:
            Upstream.refusals -= 1
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error": "slow down"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: one\n\n")
        self.wfile.flush()
        self.wfile.write(b"data: two\n\n")

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def upstream() -> Iterator[str]:
    Upstream.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def post(url: str) -> Any:
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps({"model": "m"}).encode(),
                                     headers={"Authorization": "Bearer k", "Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=10)


def test_rate_limits_are_waited_out_and_the_answer_passed_through(upstream: str) -> None:
    Upstream.refusals = 2
    waited: List[float] = []
    with RetryProxy(upstream, waits=[1, 2, 3], sleep=waited.append) as proxy:
        response = post(proxy.base_url)
        assert response.status == 200 and response.read() == b"data: one\n\ndata: two\n\n"
        assert waited == [1, 2] and proxy.retries == 2
    assert [s[0] for s in Upstream.seen] == ["/v1/chat/completions"] * 3
    assert all(s[1] == "Bearer k" and s[2] == {"model": "m"} for s in Upstream.seen)


def test_after_the_last_wait_the_refusal_is_passed_on(upstream: str) -> None:
    Upstream.refusals = 5
    with RetryProxy(upstream, waits=[0, 0], sleep=lambda s: None) as proxy:
        with pytest.raises(urllib.error.HTTPError) as error:
            post(proxy.base_url)
        assert error.value.code == 429 and proxy.retries == 2


def test_closing_the_relay_does_not_wait_for_a_request_still_waiting(upstream: str) -> None:
    import time
    Upstream.refusals = 100
    released = threading.Event()
    proxy = RetryProxy(upstream, waits=[30] * 3, sleep=lambda s: released.wait(s)).__enter__()
    threading.Thread(target=lambda: post(proxy.base_url), daemon=True).start()
    time.sleep(0.5)  # the request is now waiting out its first refusal
    start = time.time()
    proxy.__exit__()
    assert time.time() - start < 5
    released.set()
