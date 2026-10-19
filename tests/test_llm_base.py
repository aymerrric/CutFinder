from typing import List

import pytest

from llm.base import LLMClient, LLMError, Message

MESSAGES: List[Message] = [{"role": "user", "content": "hi"}]


class Flaky(LLMClient):
    """Fails `n_failures` times, then answers."""

    def __init__(self, n_failures: int, max_retries: int) -> None:
        super().__init__(max_retries=max_retries, retry_delay=0.0)
        self.n_failures: int = n_failures
        self.calls: int = 0

    def _send(self, messages: List[Message]) -> str:
        self.calls += 1
        if self.calls <= self.n_failures:
            raise ConnectionError("server down")
        return "  answer  "


class Stripping(Flaky):
    def clean(self, text: str) -> str:
        return text.strip()


def test_base_class_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]


def test_failed_calls_are_retried() -> None:
    llm: Flaky = Flaky(n_failures=2, max_retries=3)
    assert llm.send(MESSAGES) == "  answer  "
    assert llm.calls == 3


def test_llm_error_after_every_try_failed() -> None:
    llm: Flaky = Flaky(n_failures=5, max_retries=3)
    with pytest.raises(LLMError, match="server down"):
        llm.send(MESSAGES)
    assert llm.calls == 3


def test_clean_hook_is_applied() -> None:
    assert Stripping(n_failures=0, max_retries=1).send(MESSAGES) == "answer"


def test_max_retries_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Flaky(n_failures=0, max_retries=0)
