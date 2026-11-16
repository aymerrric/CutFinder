"""
The models an LLM backend offers (also a quick check that its API key works).
Run from src/:  python list_models.py groq        (or gemini, openrouter, ollama)
Then run with one of them:  python main.py --llm_backend groq --model_name <name> ...
"""
import sys
from typing import List, Optional

from llm.factory import BACKENDS, make_llm


def list_models(backend: str) -> List[str]:
    return sorted(make_llm(backend, model="", check=False).available_models())


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in BACKENDS:
        sys.exit(f"usage: python list_models.py {{{','.join(BACKENDS)}}}")
    print("\n".join(list_models(argv[0])))


if __name__ == "__main__":
    main()
