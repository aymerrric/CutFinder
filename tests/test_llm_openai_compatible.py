import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from utils import parameters

from evolution.prompts import parse_reply
from llm.base import LLMError, Message
from llm.openai_compatible import MissingAPIKey, OpenAICompatibleClient, api_key, retry_after

MESSAGES: List[Message] = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
KEY: str = "sk-test"


class FakeAPI(BaseHTTPRequestHandler):
    """Answers like OpenAI's chat completions; 401 without the key, 404 for unknown models, and
    `limited` times a 429 (rate limit) before answering."""

    received: List[Dict[str, Any]] = []
    limited: int = 0
    busy: int = 0  # times to answer 503 (server overloaded) before answering
    quota: bool = False  # answer 429 "daily quota used up" (as Gemini's free tier does)
    reply: str = "IDEA: i\nCODE: m.addConstr(A)"

    def _json(self, code: int, body: Dict[str, Any], headers: Dict[str, str] = {}) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self) -> None:
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            return self._json(401, {"error": "bad key"})
        self._json(200, {"data": [{"id": "known"}, {"id": "models/gemini-x"}]})

    def do_POST(self) -> None:
        if self.headers.get("User-Agent", "").startswith("Python-urllib"):  # like Groq's Cloudflare
            return self._json(403, {"error": "error code: 1010"})
        body: Dict[str, Any] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeAPI.received.append(body)
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            return self._json(401, {"error": "bad key"})
        if self.path != "/v1/chat/completions" or body["model"] != "known":
            return self._json(404, {"error": "unknown model"})
        if FakeAPI.quota:
            return self._json(429, {"error": {"message": "Quota exceeded for metric: generate_content_free_tier_requests",
                                              "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}})
        if FakeAPI.busy > 0:
            FakeAPI.busy -= 1
            return self._json(503, {"error": "overloaded"}, {"Retry-After": "0.2"})
        if FakeAPI.limited > 0:
            FakeAPI.limited -= 1
            return self._json(429, {"error": "slow down"}, {"Retry-After": "0.2"})
        self._json(200, {"choices": [{"message": {"role": "assistant", "content": FakeAPI.reply},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 3, "completion_tokens": 5}})

    def log_message(self, *args: Any) -> None:  # keep the test output clean
        pass


@pytest.fixture
def server() -> Iterator[str]:
    FakeAPI.received, FakeAPI.limited, FakeAPI.reply = [], 0, "IDEA: i\nCODE: m.addConstr(A)"
    httpd: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)  # free port
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def client(url: str, model: str = "known", key: str = KEY, **kwargs: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(model, url, key, retry_delay=0.01, **kwargs)


def test_sends_the_conversation_and_returns_the_answer(server: str) -> None:
    answer: str = client(server, options={"temperature": 0.3}).send(MESSAGES)
    assert parse_reply(answer) == ("i", "m.addConstr(A)")
    assert FakeAPI.received == [{"model": "known", "messages": MESSAGES, "temperature": 0.3}]


def test_lists_the_models_gemini_names_included(server: str) -> None:
    assert client(server).available_models() == ["known", "models/gemini-x", "gemini-x"]


def test_a_wrong_key_or_model_fails_after_the_retries(server: str) -> None:
    with pytest.raises(LLMError, match="HTTP 401"):
        client(server, key="wrong").send(MESSAGES)
    with pytest.raises(LLMError, match="HTTP 404"):
        client(server, model="nope").send(MESSAGES)
    assert len(FakeAPI.received) == 2 * parameters.API_MAX_RETRIES  # every try of each (LLMClient's retries)


def test_a_rate_limit_is_waited_out(server: str) -> None:
    FakeAPI.limited = 2  # two 429s, then an answer: within the 3 tries
    start: float = time.time()
    assert parse_reply(client(server).send(MESSAGES))[1] == "m.addConstr(A)"
    assert time.time() - start >= 0.4  # waited Retry-After (0.2 s) twice


def test_the_wait_is_capped(server: str) -> None:
    FakeAPI.limited = 1
    start: float = time.time()
    client(server, max_wait=0.05).send(MESSAGES)
    assert time.time() - start < 0.2


def test_retry_after_header() -> None:
    assert retry_after("3", 1.0) == 3.0 and retry_after(None, 1.0) == 1.0
    assert retry_after("Wed, 21 Oct 2026 07:28:00 GMT", 1.0) == 1.0  # a date: the default


def test_a_think_block_is_removed(server: str) -> None:
    FakeAPI.reply = "<think>let me see\nthe cut</think>\nIDEA: i\nCODE: m.addConstr(A)"
    assert client(server).send(MESSAGES).startswith("IDEA: i")


def test_key_from_the_environment_or_the_keys_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keys: Path = tmp_path / "llm_keys.env"
    keys.write_text("# my keys\nGROQ_API_KEY = gsk_file\nGEMINI_API_KEY=\"AIza\"\n")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert api_key("GROQ_API_KEY", str(keys)) == "gsk_file"
    assert api_key("GEMINI_API_KEY", str(keys)) == "AIza"  # quotes removed
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    assert api_key("GROQ_API_KEY", str(keys)) == "gsk_env"  # the environment wins
    with pytest.raises(MissingAPIKey, match="OPENROUTER_API_KEY=<key>"):
        api_key("OPENROUTER_API_KEY", str(keys))


def test_an_overloaded_server_is_waited_out_too(server: str) -> None:
    FakeAPI.busy = 2  # two 503s, then an answer: waited like a rate limit, within the 3 tries
    start: float = time.time()
    assert parse_reply(client(server).send(MESSAGES))[1] == "m.addConstr(A)"
    assert time.time() - start >= 0.4
    FakeAPI.busy = 0


def test_a_daily_quota_stops_at_once(server: str) -> None:
    from llm.base import QuotaExhausted
    FakeAPI.received.clear()
    FakeAPI.quota = True
    try:
        with pytest.raises(QuotaExhausted, match="daily quota"):
            client(server).send(MESSAGES)
        assert len(FakeAPI.received) == 1  # not retried: no try can succeed before tomorrow
    finally:
        FakeAPI.quota = False


def test_daily_quota_messages() -> None:
    from llm.openai_compatible import daily_quota
    assert daily_quota('{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}')
    assert daily_quota("You have exceeded your requests per day") and not daily_quota("slow down")
