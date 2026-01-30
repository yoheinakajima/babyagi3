"""
Embedding generation utilities for the memory system.

Supports both OpenAI embeddings and local models.
"""

import os
from functools import lru_cache
from typing import Literal

# Default embedding model
DEFAULT_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class EmbeddingProvider:
    """Base class for embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except ImportError:
                raise ImportError("openai package required for OpenAI embeddings")
        return self._client

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # Truncate very long texts
        if len(text) > 8000:
            text = text[:8000]

        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Truncate texts
        texts = [t[:8000] if len(t) > 8000 else t for t in texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([d.embedding for d in response.data])

        return all_embeddings


class AnthropicVoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embeddings (Anthropic's recommended provider)."""

    def __init__(self, model: str = "voyage-2"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import voyageai

                self._client = voyageai.Client()
            except ImportError:
                raise ImportError("voyageai package required for Voyage embeddings")
        return self._client

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        result = self.client.embed([text], model=self.model)
        return result.embeddings[0]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.embed(batch, model=self.model)
            all_embeddings.extend(result.embeddings)
        return all_embeddings


class LocalEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required for local embeddings"
                )
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


class MockEmbeddings(EmbeddingProvider):
    """Mock embeddings for testing (returns zero vectors)."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic mock embedding based on text hash."""
        import hashlib

        # Create a deterministic but varied embedding based on text content
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Use hash bytes to seed values
        embedding = []
        for i in range(self.dim):
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1]
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(value)
        return embedding


# Global provider instance
_provider: EmbeddingProvider | None = None


def get_provider() -> EmbeddingProvider:
    """Get the current embedding provider."""
    global _provider
    if _provider is None:
        # Auto-detect based on available packages and API keys
        if os.environ.get("OPENAI_API_KEY"):
            _provider = OpenAIEmbeddings()
        elif os.environ.get("VOYAGE_API_KEY"):
            _provider = AnthropicVoyageEmbeddings()
        else:
            # Fall back to mock embeddings if no API keys
            # In production, you'd want to use local embeddings
            _provider = MockEmbeddings()
    return _provider


def set_provider(provider: EmbeddingProvider):
    """Set the embedding provider."""
    global _provider
    _provider = provider


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text.

    Uses the configured provider (OpenAI by default if OPENAI_API_KEY is set).
    """
    return get_provider().embed(text)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    More efficient than calling get_embedding in a loop.
    """
    return get_provider().embed_batch(texts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def find_most_similar(
    query_embedding: list[float],
    embeddings: list[list[float]],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """
    Find the most similar embeddings to a query.

    Returns list of (index, similarity_score) tuples, sorted by similarity.
    """
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
