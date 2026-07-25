import json
from urllib.error import URLError

import pytest

from llm_engineering.application.llm.providers import LLMProvider, OllamaLLMProvider, extract_json


class BadThenGoodProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> str:
        self.calls += 1
        return "not json" if self.calls == 1 else '{"ok": true}'


def test_extract_json_from_markdown_response() -> None:
    assert extract_json('```json\n{"answer": "local"}\n```') == {"answer": "local"}


def test_generate_json_retries_until_valid_json() -> None:
    provider = BadThenGoodProvider()

    assert provider.generate_json("Return JSON", retries=1) == {"ok": True}
    assert provider.calls == 2


def test_ollama_provider_sends_local_generate_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"response": "local answer"}).encode("utf-8")

    def fake_urlopen(request: object, timeout: int) -> Response:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("llm_engineering.application.llm.providers.urlopen", fake_urlopen)
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="qwen2.5:7b-instruct", timeout_seconds=7)

    answer = provider.generate("hello", temperature=0.2, top_p=0.8, max_new_tokens=64)

    assert answer == "local answer"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 7
    assert captured["payload"] == {
        "model": "qwen2.5:7b-instruct",
        "prompt": "hello",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.8,
            "num_predict": 64,
        },
    }


def test_ollama_provider_wraps_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr("llm_engineering.application.llm.providers.urlopen", fake_urlopen)
    provider = OllamaLLMProvider(base_url="http://localhost:11434", model="qwen2.5:7b-instruct")

    with pytest.raises(RuntimeError, match="Ollama is not reachable"):
        provider.generate("hello")
