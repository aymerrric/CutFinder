import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from llm.base import LLMClient, Message, QuotaExhausted
from utils import parameters

logger: logging.Logger = logging.getLogger(__name__)

# Groq sits behind Cloudflare, which refuses urllib's default "Python-urllib/3.x" (HTTP 403, error 1010)
USER_AGENT: str = "my-evo-cut/1.0"
# a reasoning model may think aloud before its answer (DeepSeek-R1, Qwen3, ...): removed by `clean`
SERVER_BUSY: tuple = (500, 502, 503, 504)  # overloaded or failing for a moment: worth waiting
_THINK_RE: "re.Pattern[str]" = re.compile(r"<think>.*?</think>", re.DOTALL)


def daily_quota(text: str) -> bool:
    """A 429 whose body names a per-day quota (Gemini: "...RequestsPerDay..."; others: "per day")."""
    lowered: str = text.lower()
    return "perday" in lowered or "per day" in lowered or "daily" in lowered


def spend_limit(code: int, text: str) -> bool:
    """The account's money is used up: HTTP 402 (Payment Required), or a body naming a spend limit
    or missing credit. 22 Nov: Muse answered 402 "monthly_spend_limit_reached" to every call, the
    runs took it for a passing error and filled 20 generations with copies of the seed cut."""
    lowered: str = text.lower()
    return code == 402 or any(w in lowered for w in ("spend_limit", "spend limit", "insufficient_quota",
                                                     "insufficient credit", "credit balance"))


def retry_after(value: Optional[str], default: float) -> float:
    """Seconds of a Retry-After header (a number; an HTTP date or nothing gives `default`)."""
    try:
        return max(0.0, float(value)) if value else default
    except ValueError:
        return default


class MissingAPIKey(ValueError):
    """No key for the provider, neither in the environment nor in LLM_KEYS_FILE."""


def api_key(variable: str, keys_file: str = parameters.LLM_KEYS_FILE) -> str:
    """The key in the environment variable `variable`, else on its `variable=...` line of `keys_file`."""
    if os.environ.get(variable):
        return os.environ[variable]
    if os.path.exists(keys_file):
        with open(keys_file) as f:
            for line in f:
                name, sep, value = line.strip().partition("=")
                if sep and name.strip() == variable and value.strip():
                    return value.strip().strip("\"'")
    raise MissingAPIKey(
        f"no API key: set {variable} in the environment or add a line {variable}=<key> to {keys_file}"
    )


class OpenAICompatibleClient(LLMClient):
    """
    Client for any API that follows OpenAI's chat completions format (POST <base_url>/chat/completions
    with a Bearer key): Groq, Gemini, OpenRouter, ... (parameters.API_PROVIDERS). Changing provider
    = changing `base_url` and the key; changing model = changing `model`.

    A rate limit (HTTP 429, frequent on free tiers) is waited out: the call sleeps for the time the
    server asks (Retry-After, at most `max_wait` seconds), then fails, so LLMClient retries it.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        key: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = parameters.LLM_TIMEOUT,
        max_retries: int = parameters.API_MAX_RETRIES,
        retry_delay: float = parameters.LLM_RETRY_DELAY,
        max_wait: float = parameters.API_MAX_WAIT,
        busy_wait: float = parameters.API_BUSY_WAIT,
    ) -> None:
        """
        model: the provider's model name, e.g. "llama-3.3-70b-versatile" (Groq), "gemini-2.5-flash".
        options: extra fields of the request, e.g. {"temperature": 0.7} (None = the model's defaults).
        """
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self.model: str = model
        self.base_url: str = base_url.rstrip("/")
        self.key: str = key
        self.options: Dict[str, Any] = dict(options or {})
        self.timeout: float = timeout
        self.max_wait: float = max_wait
        self.busy_wait: float = busy_wait  # wait after a 429 / 5xx without Retry-After

    def _request(self, path: str, body: Optional[Dict[str, Any]] = None, timeout: float = 10) -> Dict[str, Any]:
        request: urllib.request.Request = urllib.request.Request(
            self.base_url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
                     "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def available_models(self) -> List[str]:
        """Names of the models the provider offers (GET /models). Raises if it cannot be reached."""
        names: List[str] = [model["id"] for model in self._request("/models")["data"]]
        return names + [name.split("/", 1)[1] for name in names if name.startswith("models/")]  # Gemini

    def _send(self, messages: List[Message]) -> str:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, **self.options}
        start: float = time.time()
        try:
            answer: Dict[str, Any] = self._request("/chat/completions", body, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            text: str = e.read().decode(errors="replace")
            if spend_limit(e.code, text):
                raise QuotaExhausted(f"{self.model}: spend limit or credit used up (HTTP {e.code}): {text[:300]}") from e
            if e.code == 429 and daily_quota(text):
                raise QuotaExhausted(f"{self.model}: daily quota used up (HTTP 429): {text[:300]}") from e
            if e.code == 429 or e.code in SERVER_BUSY:  # rate limit or overload: wait, then LLMClient retries
                wait: float = min(retry_after(e.headers.get("Retry-After"), self.busy_wait), self.max_wait)
                reason: str = "rate limited" if e.code == 429 else "server busy"
                logger.warning(f"{self.model}: {reason} (HTTP {e.code}), waiting {wait:.0f}s")
                time.sleep(wait)
            raise RuntimeError(f"HTTP {e.code}: {text[:300]}") from e
        choice: Dict[str, Any] = answer["choices"][0]
        if choice.get("finish_reason") == "length":
            logger.warning("answer cut off: the context or the token limit was reached")
        usage: Dict[str, Any] = answer.get("usage") or {}
        logger.info(f"{self.model}: {usage.get('prompt_tokens')} tokens in, "
                    f"{usage.get('completion_tokens')} out, {time.time() - start:.1f}s")
        return choice["message"]["content"] or ""

    def clean(self, text: str) -> str:
        return _THINK_RE.sub("", text).strip()
