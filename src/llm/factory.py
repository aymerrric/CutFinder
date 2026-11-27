"""The LLM client of a run, from its settings (main.py and prompt_bench.py: --llm_backend, --model_name)."""
from typing import Any, Dict, List, Optional

from llm.base import LLMClient
from llm.claude_code import ClaudeCodeClient, claude
from llm.ollama import OllamaClient
from llm.openai_compatible import OpenAICompatibleClient, api_key
from utils import parameters

CLAUDE_CODE: str = "claude_code"  # Claude through `claude -p`: the user's subscription
BACKENDS: List[str] = ["ollama", *parameters.API_PROVIDERS, CLAUDE_CODE]  # the choices of --llm_backend


def make_llm(
    backend: str, model: str, temperature: Optional[float] = None,
    ollama_host: str = parameters.OLLAMA_HOST, check: bool = True,
) -> LLMClient:
    """
    backend: "ollama", a provider of parameters.API_PROVIDERS ("groq", "gemini", "openrouter", ...)
             or "claude_code" (model: an alias such as "sonnet" or "opus"; no temperature).
    check: refuse a model the server or provider does not offer (one cheap request), so that a
           typo fails now and not at the first child.
    """
    options: Dict[str, Any] = {} if temperature is None else {"temperature": temperature}
    if backend == "ollama":
        llm: Any = OllamaClient(model, host=ollama_host, options=options)
        where: str = f"the Ollama server {ollama_host}"
    elif backend in parameters.API_PROVIDERS:
        base_url, variable = parameters.API_PROVIDERS[backend]
        llm = OpenAICompatibleClient(model, base_url, api_key(variable), options=options)
        where = backend
    elif backend == CLAUDE_CODE:
        if temperature is not None:
            raise ValueError("claude -p has no temperature option: leave --temperature out")
        if check and claude() is None:
            raise ValueError("the claude command is not installed (https://claude.ai/install.sh)")
        return ClaudeCodeClient(model)
    else:
        raise ValueError(f"unknown LLM backend {backend!r} ({', '.join(BACKENDS)})")
    if check and model not in llm.available_models():
        raise ValueError(f"{model} is not offered by {where}")
    return llm
