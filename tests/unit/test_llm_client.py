"""Unit tests for GroqClient (src/utils/llm_client.py).

WHY monkeypatch groq.Groq instead of hitting the real API: same reasoning
as faking SentenceTransformer in test_embeddings.py — no real API key or
network call needed in unit tests, and we need to assert exactly what was
sent to `.chat.completions.create()`.
"""

from __future__ import annotations

from src.utils import llm_client as llm_client_module
from src.utils.llm_client import GroqClient


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, parent: _FakeChat) -> None:
        self._parent = parent

    def create(self, model, messages, temperature):
        self._parent.calls.append(
            {"model": model, "messages": messages, "temperature": temperature}
        )
        return _FakeCompletion("fake response")


class _FakeChat:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.completions = _FakeCompletions(self)


class _FakeGroq:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.chat = _FakeChat()


def test_complete_sends_user_message_and_returns_content(monkeypatch) -> None:
    monkeypatch.setattr(llm_client_module, "Groq", _FakeGroq)
    client = GroqClient(model="llama-3.3-70b-versatile", api_key="fake-key")

    result = client.complete("hello")

    assert result == "fake response"
    call = client._client.chat.calls[0]
    assert call["model"] == "llama-3.3-70b-versatile"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["temperature"] == 0.0


def test_complete_prepends_system_message_when_given(monkeypatch) -> None:
    monkeypatch.setattr(llm_client_module, "Groq", _FakeGroq)
    client = GroqClient(model="llama-3.3-70b-versatile", api_key="fake-key")

    client.complete("hello", system="be terse", temperature=0.5)

    call = client._client.chat.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]
    assert call["temperature"] == 0.5
