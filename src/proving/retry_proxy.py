"""
A local relay between an agent and an OpenAI-style API that waits out rate limits.

Mistral Vibe retries a refused request only 3 times within about 2 seconds, and Requesty's free
access refuses requests (HTTP 429) for longer than that: on 15 Nov the Leanstral agent stopped
after 14 minutes of work. The relay forwards each request to the API and, on 429 / 500-504,
waits and tries again (WAITS), before anything is sent back; the answer, streamed or not, is
then passed through unchanged. It listens on 127.0.0.1 only.
"""
import logging
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional, Tuple

logger: logging.Logger = logging.getLogger(__name__)

RETRYABLE: Tuple[int, ...] = (429, 500, 502, 503, 504)
WAITS: List[float] = [5, 10, 20, 30, 60, 60, 60, 60, 60, 60]  # seconds before each new try (about 7 min in all)
HOP_HEADERS: Tuple[str, ...] = ("host", "content-length", "connection", "accept-encoding", "transfer-encoding")


class RetryProxy:
    """`with RetryProxy(upstream) as proxy:` gives `proxy.base_url` (http://127.0.0.1:<port>/v1)."""

    def __init__(self, upstream: str, waits: Optional[List[float]] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.upstream: str = upstream.rstrip("/")  # e.g. https://router.requesty.ai/v1
        self.waits: List[float] = list(WAITS if waits is None else waits)
        self.sleep: Callable[[float], None] = sleep
        self.retries: int = 0  # requests tried again, in all
        self.server: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        # closing the relay must not wait for a request still waiting out a rate limit (on 15 Nov
        # that kept a stopped agent's run open for an hour)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.thread: threading.Thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> "RetryProxy":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _handler(self) -> type:
        proxy: "RetryProxy" = self

        class Handler(BaseHTTPRequestHandler):
            def _forward(self, method: str) -> None:
                length: int = int(self.headers.get("Content-Length") or 0)
                body: Optional[bytes] = self.rfile.read(length) if length else None
                path: str = self.path[len("/v1"):] if self.path.startswith("/v1") else self.path
                headers: Dict[str, str] = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
                for attempt in range(len(proxy.waits) + 1):
                    request = urllib.request.Request(proxy.upstream + path, data=body, headers=headers, method=method)
                    try:
                        response = urllib.request.urlopen(request, timeout=900)
                        status: int = response.status
                    except urllib.error.HTTPError as error:
                        if error.code in RETRYABLE and attempt < len(proxy.waits):
                            proxy.retries += 1
                            logger.warning(f"upstream HTTP {error.code}, try {attempt + 2} in {proxy.waits[attempt]:.0f} s")
                            proxy.sleep(proxy.waits[attempt])
                            continue
                        response, status = error, error.code
                    self.send_response(status)
                    for key, value in response.headers.items():
                        if key.lower() not in HOP_HEADERS:
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while True:  # pass the answer through as it comes (streamed answers too)
                        chunk: bytes = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    return

            def do_POST(self) -> None:
                self._forward("POST")

            def do_GET(self) -> None:
                self._forward("GET")

            def log_message(self, *args: object) -> None:
                pass

        return Handler
