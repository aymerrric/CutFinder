import pytest

from llm.factory import BACKENDS, make_llm
from llm.ollama import OllamaClient
from llm.openai_compatible import MissingAPIKey, OpenAICompatibleClient
from utils import parameters


def test_backends_are_ollama_and_the_providers() -> None:
    assert BACKENDS == ["ollama", "groq", "gemini", "openrouter", "muse", "mistral", "requesty", "claude_code"]


def test_ollama_backend() -> None:
    llm = make_llm("ollama", "qwen3-coder:30b", temperature=0.5, check=False)
    assert isinstance(llm, OllamaClient) and llm.options == {"temperature": 0.5}


def test_api_backend_uses_the_providers_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    llm = make_llm("groq", "llama-3.3-70b-versatile", check=False)
    assert isinstance(llm, OpenAICompatibleClient) and llm.key == "gsk_x"
    assert llm.base_url == parameters.API_PROVIDERS["groq"][0] == "https://api.groq.com/openai/v1"


def test_missing_key_and_unknown_backend_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(parameters, "LLM_KEYS_FILE", "/nonexistent/llm_keys.env")
    import llm.openai_compatible as module
    monkeypatch.setattr(module.api_key, "__defaults__", ("/nonexistent/llm_keys.env",))
    with pytest.raises(MissingAPIKey):
        make_llm("gemini", "gemini-2.5-flash", check=False)
    with pytest.raises(ValueError, match="unknown LLM backend 'claude'"):
        make_llm("claude", "x", check=False)
