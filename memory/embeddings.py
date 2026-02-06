"""
Embedding generation utilities for the memory system.

Supports multiple embedding providers with smart selection:
- OpenAI embeddings (text-embedding-3-small, etc.) - requires OPENAI_API_KEY
- Voyage AI embeddings - Anthropic's recommended provider, requires VOYAGE_API_KEY
- Local models (sentence-transformers) - no API key required, used as fallback

Provider selection logic:
1. If embedding_model is "local" → use LocalEmbeddings
2. If embedding_model is OpenAI model + OPENAI_API_KEY exists → use LiteLLM/OpenAI
3. If VOYAGE_API_KEY exists → use Voyage
4. Fallback to LocalEmbeddings (sentence-transformers)
5. Final fallback to MockEmbeddings

Note: Anthropic (Claude) does not provide embedding models.
When using ANTHROPIC_API_KEY only, the system will use local embeddings.

Includes caching and graceful fallback.
"""

import hashlib
import os
import sqlite3
import struct
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import logging

logger = logging.getLogger(__name__)

from metrics import InstrumentedOpenAI, InstrumentedLiteLLMEmbeddings

# Default embedding model
DEFAULT_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


# ═══════════════════════════════════════════════════════════
# EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════


@dataclass
class CacheConfig:
    """Configuration for embedding cache."""

    enabled: bool = True
    max_entries: int = 10000
    ttl_seconds: int = 86400 * 30  # 30 days
    cache_path: str = "~/.babyagi/memory/embedding_cache.db"


class EmbeddingCache:
    """
    SQLite-based cache for embeddings.

    Caches embeddings by text hash to avoid redundant API calls.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._conn is None:
            cache_path = Path(self.config.cache_path).expanduser()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            if not self._initialized:
                self._initialize_schema()
                self._initialized = True
        return self._conn

    def _initialize_schema(self):
        """Create cache table."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL
            )
        """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_created ON embedding_cache(created_at)"
        )
        self.conn.commit()

    def _hash_text(self, text: str, model: str) -> str:
        """Generate hash for text + model combination."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding."""
        if not self.config.enabled:
            return None

        text_hash = self._hash_text(text, model)
        cur = self.conn.cursor()
        cur.execute(
            "SELECT embedding, created_at FROM embedding_cache WHERE text_hash = ?",
            (text_hash,),
        )
        row = cur.fetchone()

        if row is None:
            return None

        # Check TTL
        if time.time() - row["created_at"] > self.config.ttl_seconds:
            cur.execute("DELETE FROM embedding_cache WHERE text_hash = ?", (text_hash,))
            self.conn.commit()
            return None

        # Deserialize embedding
        data = row["embedding"]
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))

    def put(self, text: str, model: str, embedding: list[float]):
        """Store embedding in cache."""
        if not self.config.enabled:
            return

        text_hash = self._hash_text(text, model)
        data = struct.pack(f"{len(embedding)}f", *embedding)

        self.conn.execute(
            """
            INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (text_hash, model, data, time.time()),
        )
        self.conn.commit()

        # Prune if needed
        self._prune_if_needed()

    def _prune_if_needed(self):
        """Remove old entries if cache exceeds max size."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM embedding_cache")
        count = cur.fetchone()[0]

        if count > self.config.max_entries:
            # Remove oldest 10%
            remove_count = count // 10
            cur.execute(
                """
                DELETE FROM embedding_cache
                WHERE text_hash IN (
                    SELECT text_hash FROM embedding_cache
                    ORDER BY created_at ASC
                    LIMIT ?
                )
            """,
                (remove_count,),
            )
            self.conn.commit()

    def clear(self):
        """Clear all cached embeddings."""
        self.conn.execute("DELETE FROM embedding_cache")
        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Global cache instance
_cache: EmbeddingCache | None = None


def get_cache() -> EmbeddingCache:
    """Get the global embedding cache."""
    global _cache
    if _cache is None:
        _cache = EmbeddingCache()
    return _cache


def set_cache(cache: EmbeddingCache):
    """Set the global embedding cache."""
    global _cache
    _cache = cache


# ═══════════════════════════════════════════════════════════
# EMBEDDING PROVIDERS
# ═══════════════════════════════════════════════════════════


class EmbeddingProvider:
    """Base class for embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider with caching."""

    def __init__(self, model: str = DEFAULT_MODEL, use_cache: bool = True):
        self.model = model
        self.use_cache = use_cache
        self._client = None

    @property
    def client(self):
        """Get instrumented OpenAI client for metrics tracking."""
        if self._client is None:
            try:
                self._client = InstrumentedOpenAI()
            except ImportError:
                raise ImportError("openai package required for OpenAI embeddings")
        return self._client

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # Truncate very long texts
        if len(text) > 8000:
            text = text[:8000]

        # Check cache first
        if self.use_cache:
            cache = get_cache()
            cached = cache.get(text, self.model)
            if cached is not None:
                return cached

        # Generate embedding
        response = self.client.embeddings.create(input=text, model=self.model)
        embedding = response.data[0].embedding

        # Store in cache
        if self.use_cache:
            cache.put(text, self.model, embedding)

        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Truncate texts
        texts = [t[:8000] if len(t) > 8000 else t for t in texts]

        # Check cache for each text
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self.use_cache:
            cache = get_cache()
            for i, text in enumerate(texts):
                cached = cache.get(text, self.model)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Generate embeddings for uncached texts
        if uncached_texts:
            all_embeddings = []
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i : i + batch_size]
                response = self.client.embeddings.create(input=batch, model=self.model)
                all_embeddings.extend([d.embedding for d in response.data])

            # Store in cache and results
            cache = get_cache() if self.use_cache else None
            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                if cache:
                    cache.put(texts[idx], self.model, embedding)

        return results


class LiteLLMEmbeddings(EmbeddingProvider):
    """LiteLLM-based embeddings supporting multiple providers."""

    def __init__(self, model: str = DEFAULT_MODEL, use_cache: bool = True):
        self.model = model
        self.use_cache = use_cache
        self._client = None

    @property
    def client(self):
        """Get instrumented LiteLLM embedding client for metrics tracking."""
        if self._client is None:
            try:
                self._client = InstrumentedLiteLLMEmbeddings(default_model=self.model)
            except ImportError:
                raise ImportError("litellm package required for LiteLLM embeddings")
        return self._client

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        # Truncate very long texts
        if len(text) > 8000:
            text = text[:8000]

        # Check cache first
        if self.use_cache:
            cache = get_cache()
            cached = cache.get(text, self.model)
            if cached is not None:
                return cached

        # Generate embedding
        response = self.client.create(input=text, model=self.model)
        embedding = response.data[0]["embedding"]

        # Store in cache
        if self.use_cache:
            cache.put(text, self.model, embedding)

        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Truncate texts
        texts = [t[:8000] if len(t) > 8000 else t for t in texts]

        # Check cache for each text
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self.use_cache:
            cache = get_cache()
            for i, text in enumerate(texts):
                cached = cache.get(text, self.model)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Generate embeddings for uncached texts
        if uncached_texts:
            all_embeddings = []
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i : i + batch_size]
                response = self.client.create(input=batch, model=self.model)
                all_embeddings.extend([d["embedding"] for d in response.data])

            # Store in cache and results
            cache = get_cache() if self.use_cache else None
            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                if cache:
                    cache.put(texts[idx], self.model, embedding)

        return results


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
_fallback_provider: EmbeddingProvider | None = None


def get_provider() -> EmbeddingProvider:
    """Get the current embedding provider."""
    global _provider
    if _provider is None:
        _provider = _create_provider()
    return _provider


def _create_provider() -> EmbeddingProvider:
    """Create an embedding provider based on available packages and API keys."""
    # Get embedding configuration from llm_config if available
    embedding_model = DEFAULT_MODEL
    try:
        from llm_config import get_llm_config
        config = get_llm_config()
        embedding_model = config.embedding_model
    except ImportError:
        pass

    # If configured for local embeddings, skip API providers
    if embedding_model == "local":
        try:
            provider = LocalEmbeddings()
            provider.model  # Test
            return provider
        except Exception as e:
            logger.debug("Local embeddings unavailable: %s", e)

    # Try LiteLLM only if we have the right API key for the model
    # OpenAI embedding models require OPENAI_API_KEY
    if embedding_model.startswith("text-embedding") and os.environ.get("OPENAI_API_KEY"):
        try:
            provider = LiteLLMEmbeddings(model=embedding_model)
            provider.client  # Test that it works
            return provider
        except Exception as e:
            logger.debug("LiteLLM embeddings unavailable for model '%s': %s", embedding_model, e)

    # Fall back to direct OpenAI if key is available
    if os.environ.get("OPENAI_API_KEY"):
        try:
            provider = OpenAIEmbeddings(model="text-embedding-3-small")
            provider.client  # This will raise if there's an issue
            return provider
        except Exception as e:
            logger.debug("OpenAI embeddings unavailable: %s", e)

    # Try Voyage (Anthropic's recommended embedding provider)
    if os.environ.get("VOYAGE_API_KEY"):
        try:
            provider = AnthropicVoyageEmbeddings()
            provider.client  # Test
            return provider
        except Exception as e:
            logger.debug("Voyage embeddings unavailable: %s", e)

    # Try local embeddings as fallback (works without API key)
    try:
        provider = LocalEmbeddings()
        provider.model  # Test
        return provider
    except Exception as e:
        logger.debug("Local embeddings fallback unavailable: %s", e)

    # Fall back to mock embeddings (always works)
    return MockEmbeddings()


def get_fallback_provider() -> EmbeddingProvider:
    """Get the fallback provider (mock embeddings)."""
    global _fallback_provider
    if _fallback_provider is None:
        _fallback_provider = MockEmbeddings()
    return _fallback_provider


def set_provider(provider: EmbeddingProvider):
    """Set the embedding provider."""
    global _provider
    _provider = provider


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text.

    Uses the configured provider with graceful fallback to mock embeddings
    if the primary provider fails.
    """
    try:
        return get_provider().embed(text)
    except Exception as e:
        logger.warning("Embedding generation failed, using fallback: %s", e)
        return get_fallback_provider().embed(text)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    More efficient than calling get_embedding in a loop.
    Uses graceful fallback if primary provider fails.
    """
    try:
        return get_provider().embed_batch(texts)
    except Exception as e:
        logger.warning("Batch embedding generation failed, using fallback: %s", e)
        return get_fallback_provider().embed_batch(texts)


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
