import pytest

import list_models


def test_lists_the_backends_models(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    class Fake:
        def available_models(self) -> list:
            return ["b", "a"]
    monkeypatch.setattr(list_models, "make_llm", lambda backend, model, check: Fake())
    list_models.main(["groq"])
    assert capsys.readouterr().out == "a\nb\n"


def test_unknown_backend_shows_the_usage() -> None:
    with pytest.raises(SystemExit, match="usage: python list_models.py"):
        list_models.main(["claude"])
