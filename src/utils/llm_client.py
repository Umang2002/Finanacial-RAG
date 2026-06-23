"""Thin wrapper over local Ollama chat — shared by Phase 4 (query) and Phase 6 (generation).

WHAT: OllamaClient.complete() sends a single-turn chat request to a local
Ollama server and returns the response text.
WHY shared here instead of duplicated in query_transformer.py and
generator.py: both phases need "send a prompt to the local LLM, get text
back" with no other behavior — pulling it into one module means Phase 6's
generator.py reuses this instead of re-wiring ollama.Client.
"""

from __future__ import annotations

import ollama

from src.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Minimal sync wrapper over `ollama.Client.chat` — one prompt in, one string out."""

    def __init__(self, model: str, host: str) -> None:
        """Bind to a model name and Ollama server URL (e.g. http://localhost:11434)."""
        self.model = model
        self._client = ollama.Client(host=host)

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        """Run one chat completion, returning the assistant message content.

        Args:
            prompt: user-turn content.
            system: optional system-turn content (e.g. task instructions).
            temperature: passed through to Ollama's `options.temperature`.

        WHY no streaming/multi-turn: every caller in this project (HyDE,
        multi-query, decomposition, citation-aware answers) is a single
        request/response — streaming and chat history add complexity none
        of them need.
        """
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]
