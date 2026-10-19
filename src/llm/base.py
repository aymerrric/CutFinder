import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from utils import parameters

# One chat message: {"role": "system" | "user" | "assistant", "content": "..."}
Message = Dict[str, str]

logger: logging.Logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Every try of a request failed: the LLM could not be reached or did not answer."""


class LLMClient(ABC):
    """
    Sends a conversation to an LLM and returns its answer as text. Nothing else: it knows
    nothing about cuts, prompts or the reply format (see evolution/prompts.py).

    A subclass only implements `_send` (one call to its backend). The retries live here, once
    for every backend. They are for failed *calls* (server down, timeout); a bad *answer* is
    the genetic algorithm's business (it asks again with feedback).
    """

    def __init__(
        self,
        max_retries: int = parameters.LLM_MAX_RETRIES,
        retry_delay: float = parameters.LLM_RETRY_DELAY,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay

    def send(self, messages: List[Message]) -> str:
        """The LLM answer to `messages`, passed through `clean`. Raises LLMError if every try fails."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.clean(self._send(messages))
            except Exception as e:  # noqa: BLE001 (any backend error is retried)
                last_error = e
                logger.warning(
                    f"LLM call {attempt}/{self.max_retries} failed: {type(e).__name__}: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * 2 ** (attempt - 1))  # 1 s, 2 s, 4 s, ...
        raise LLMError(f"{self.max_retries} tries failed, last error: {last_error}") from last_error

    @abstractmethod
    def _send(self, messages: List[Message]) -> str:
        """One call to the backend: return the raw text of the answer, or raise."""

    def clean(self, text: str) -> str:
        """Hook for model-specific quirks in the raw answer (e.g. a <think> block)."""
        return text
