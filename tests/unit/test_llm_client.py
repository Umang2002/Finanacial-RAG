"""Unit tests for OllamaClient (src/utils/llm_client.py).

WHY monkeypatch ollama.Client instead of hitting a real local server: same
reasoning as faking SentenceTransformer in test_embeddings.py — no running
Ollama server dependency in unit tests, and we need to assert exactly what
was sent to `.chat()`.
"""

from __future__ import annotations

from src.utils import llm_client as llm_client_module
from src.utils.llm_client import OllamaClient


class _FakeOllamaClient:
    def __init__(self, host: str) -> None:
        self.host = host
        self.chat_calls: list[dict] = []

    def chat(self, model, messages, options):
        self.chat_calls.append({"model": model, "messages": messages, "options": options})
        return {"message": {"content": "fake response"}}


def test_complete_sends_user_message_and_returns_content(monkeypatch) -> None:
    monkeypatch.setattr(llm_client_module.ollama, "Client", _FakeOllamaClient)
    client = OllamaClient(model="llama3.2", host="http://localhost:11434")

    result = client.complete("hello")

    assert result == "fake response"
    call = client._client.chat_calls[0]
    assert call["model"] == "llama3.2"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["options"] == {"temperature": 0.0}


def test_complete_prepends_system_message_when_given(monkeypatch) -> None:
    monkeypatch.setattr(llm_client_module.ollama, "Client", _FakeOllamaClient)
    client = OllamaClient(model="llama3.2", host="http://localhost:11434")

    client.complete("hello", system="be terse", temperature=0.5)

    call = client._client.chat_calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]
    assert call["options"] == {"temperature": 0.5}
