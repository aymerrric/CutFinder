from typing import List

from llm.base import LLMClient, Message


class FakeLLM(LLMClient):
    """
    Test double: returns the given replies in order, and records every conversation it
    receives, so tests can run the genetic algorithm without a real LLM.
    """

    def __init__(self, replies: List[str], max_retries: int = 1, retry_delay: float = 0.0) -> None:
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self.replies: List[str] = list(replies)  # copy: the caller's list is left untouched
        self.received: List[List[Message]] = []  # one entry per call, as sent at that time

    def _send(self, messages: List[Message]) -> str:
        self.received.append([dict(m) for m in messages])  # copy: the caller keeps appending
        if not self.replies:
            raise IndexError("FakeLLM has no reply left")
        return self.replies.pop(0)
