"""Text embedding using OpenAI's text-embedding-3-small model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmbedderConfig:
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100


class Embedder:
    """Wraps OpenAI embeddings API to produce vectors from text.

    Handles batching and truncation. Used to embed document chunks
    at ingestion time and queries at retrieval time.
    """

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self._client = None

    def _get_client(self):
        """Lazily initialise the OpenAI client.
        
        Reads OPENAI_API_KEY from environment via the openai library.
        Lazy init because the client import is heavy and not always needed.
        """
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI()
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per text.

        Batches requests to stay within API limits. Truncates each text
        to 8000 chars before sending to avoid token limit errors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors in the same order as input.
        """
        client = self._get_client()
        all_embeddings: list[list[float]] = []

        # Process in batches to avoid hitting API rate limits
        for i in range(0, len(texts), self.config.batch_size):
            batch = [self._truncate(t) for t in texts[i:i + self.config.batch_size]]
            response = await client.embeddings.create(
                model=self.config.model,
                input=batch,
            )
            # Response data is ordered to match input order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string.

        Convenience wrapper around embed() for query-time use.

        Args:
            text: Input string.

        Returns:
            Single embedding vector of length config.dimension.
        """
        results = await self.embed([text])
        return results[0]

    def _truncate(self, text: str, max_chars: int = 8000) -> str:
        """Truncate text to stay within token limits.

        text-embedding-3-small supports up to 8191 tokens.
        8000 chars is a safe heuristic — roughly 2000 tokens for
        dense technical text.

        Args:
            text: Input string.
            max_chars: Maximum character length before truncation.

        Returns:
            Original string if under limit, truncated otherwise.
        """
        return text[:max_chars] if len(text) > max_chars else text