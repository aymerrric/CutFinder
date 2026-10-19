import json
import logging
import urllib.request
from typing import Any, Dict, List, Optional

from llm.base import LLMClient, Message
from utils import parameters

logger: logging.Logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """
    Client for an Ollama server (https://github.com/ollama/ollama/blob/main/docs/api.md).
    Any model the server has can be used: changing model = changing `model`.
    """

    def __init__(
        self,
        model: str,
        host: str = parameters.OLLAMA_HOST,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = parameters.LLM_TIMEOUT,
        max_retries: int = parameters.LLM_MAX_RETRIES,
        retry_delay: float = parameters.LLM_RETRY_DELAY,
    ) -> None:
        """
        model: an Ollama model name, e.g. "qwen3-coder:30b".
        options: Ollama generation options, e.g. {"temperature": 0.7, "seed": 0}
                 (None = the model's defaults).
        timeout: seconds to wait for one answer.
        """
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self.model: str = model
        self.host: str = host.rstrip("/")
        self.url: str = self.host + "/api/chat"
        self.options: Dict[str, Any] = dict(options or {})
        self.timeout: float = timeout

    def available_models(self) -> List[str]:
        """Names of the models the server has (GET /api/tags). Raises if it cannot be reached."""
        with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as response:
            return [model["name"] for model in json.load(response)["models"]]

    def _send(self, messages: List[Message]) -> str:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if self.options:
            body["options"] = self.options
        request: urllib.request.Request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        # an HTTP error (e.g. 404 unknown model) or a network error raises: LLMClient retries
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            answer: Dict[str, Any] = json.load(response)
        if answer.get("done_reason") == "length":
            logger.warning("answer cut off: the context or the token limit was reached")
        logger.info(
            f"{self.model}: {answer.get('prompt_eval_count')} tokens in, "
            f"{answer.get('eval_count')} out, {answer.get('total_duration', 0) / 1e9:.1f}s"
        )
        return answer["message"]["content"]
