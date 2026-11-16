import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List

import pytest

from evolution.prompts import parse_reply
from llm.base import LLMError, Message
from llm.ollama import OllamaClient

MESSAGES: List[Message] = [{"role": "user", "content": "hi"}]


class FakeOllama(BaseHTTPRequestHandler):
    """Answers POST /api/chat like Ollama; records the request bodies. 404 for unknown models."""

    received: List[Dict[str, Any]] = []

    def do_GET(self) -> None:
        if self.path != "/api/tags":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"models": [{"name": "known"}, {"name": "other:7b"}]}).encode())

    def do_POST(self) -> None:
        body: Dict[str, Any] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.received.append(body)
        if self.path != "/api/chat" or body["model"] != "known":
            self.send_response(404)
            self.end_headers()
            return
        content: str = "" if body["messages"][-1]["content"] == "think only" else "IDEA: i\nCODE: m.addConstr(A)"
        answer: bytes = json.dumps(
            {"message": {"role": "assistant", "content": content, "thinking": "the analysis, then ```lean\nproof\n```"},
             "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 5, "total_duration": 10**9}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(answer)

    def log_message(self, *args: Any) -> None:  # keep the test output clean
        pass


@pytest.fixture
def server() -> Iterator[str]:
    FakeOllama.received = []
    httpd: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)  # free port
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_sends_the_conversation_and_returns_the_answer(server: str) -> None:
    llm: OllamaClient = OllamaClient("known", host=server)
    assert llm.send(MESSAGES) == "IDEA: i\nCODE: m.addConstr(A)"
    assert FakeOllama.received == [{"model": "known", "messages": MESSAGES, "stream": False}]


def test_available_models(server: str) -> None:
    assert OllamaClient("known", host=server).available_models() == ["known", "other:7b"]


def test_options_are_forwarded(server: str) -> None:
    OllamaClient("known", host=server, options={"temperature": 0.2, "seed": 1}).send(MESSAGES)
    assert FakeOllama.received[0]["options"] == {"temperature": 0.2, "seed": 1}


def test_host_with_trailing_slash(server: str) -> None:
    assert OllamaClient("known", host=server + "/").send(MESSAGES)


def test_unknown_model_is_retried_then_llm_error(server: str) -> None:
    llm: OllamaClient = OllamaClient("missing", host=server, max_retries=2, retry_delay=0)
    with pytest.raises(LLMError, match="404"):
        llm.send(MESSAGES)
    assert len(FakeOllama.received) == 2


def test_unreachable_server_is_an_llm_error() -> None:
    llm: OllamaClient = OllamaClient("known", host="http://127.0.0.1:1", max_retries=1, timeout=2)
    with pytest.raises(LLMError):
        llm.send(MESSAGES)


@pytest.mark.skipif(os.environ.get("OLLAMA_LIVE") != "1", reason="set OLLAMA_LIVE=1 (needs the tunnel)")
def test_live_server_follows_the_format() -> None:
    """Real Ollama behind the tunnel: OLLAMA_LIVE=1 pytest tests/test_llm_ollama.py"""
    llm: OllamaClient = OllamaClient("qwen3-coder:30b", options={"temperature": 0})
    reply: str = llm.send(
        [{"role": "user", "content": "Answer exactly in this format:\nIDEA: <one sentence>\n"
          "CODE:\n```python\n<one line of Python>\n```\nTask: print hello."}]
    )
    idea, code = parse_reply(reply)
    assert idea and "print" in code


def test_an_answer_left_in_the_thinking_field_is_returned(server: str) -> None:
    client = OllamaClient("known", host=server, max_retries=1)
    assert client.send([{"role": "user", "content": "think only"}]) == "the analysis, then ```lean\nproof\n```"
    assert client.send(MESSAGES) == "IDEA: i\nCODE: m.addConstr(A)"  # a normal answer: the thinking is ignored
