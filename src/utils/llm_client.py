"""Thin wrapper over the Groq chat API — shared by Phase 4 (query) and Phase 6 (generation).

WHAT: GroqClient.complete() sends a single-turn chat request to Groq's
hosted inference API and returns the response text.
WHY shared here instead of duplicated in query_transformer.py and
generator.py: both phases need "send a prompt to the LLM, get text back"
with no other behavior — pulling it into one module means Phase 6's
generator.py reuses this instead of re-wiring groq.Groq.
WHY Groq over local Ollama: Groq's hosted inference (free tier, OpenAI-
compatible chat API) is far faster than running a 3B model locally on CPU/
MPS — Phase 7's eval runs were minutes-per-example on local Ollama. This
moves every LLM call (query transform, generation, RAGAS judge) off-device;
embeddings/reranking stay local (BGE-M3/BGE-reranker, no API for those).
"""

from __future__ import annotations

from groq import Groq

from src.utils.logging import get_logger

logger = get_logger(__name__)


class GroqClient:
    """Minimal sync wrapper over `groq.Groq().chat.completions.create` — prompt in, string out."""

    def __init__(self, model: str, api_key: str) -> None:
        """Bind to a Groq model id (e.g. llama-3.3-70b-versatile) and API key."""
        self.model = model
        self._client = Groq(api_key=api_key)

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        """Run one chat completion, returning the assistant message content.

        Args:
            prompt: user-turn content.
            system: optional system-turn content (e.g. task instructions).
            temperature: passed through to Groq's chat completion request.

        WHY no streaming/multi-turn: every caller in this project (HyDE,
        multi-query, decomposition, citation-aware answers) is a single
        request/response — streaming and chat history add complexity none
        of them need.
        """
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
