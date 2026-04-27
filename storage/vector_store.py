"""Vector store abstraction supporting pgvector and Qdrant backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ingestion.chunker import Chunk


@dataclass
class VectorDocument:
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    backend: Literal["pgvector", "qdrant"] = "qdrant"
    collection_name: str = "paper_chunks"
    embedding_dim: int = 1536
    # pgvector settings
    pg_dsn: str = "postgresql://localhost:5432/papers"
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    distance_metric: Literal["cosine", "dot", "euclidean"] = "cosine"


class VectorStoreBase(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    async def upsert(self, documents: list[VectorDocument]) -> int:
        """Insert or update a batch of vector documents.

        Args:
            documents: List of VectorDocument objects with pre-computed embeddings.

        Returns:
            Number of documents successfully upserted.
        """

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the top-k nearest neighbours for a query embedding.

        Args:
            query_embedding: Dense float vector of the same dimension as stored documents.
            top_k: Maximum number of results to return.
            filters: Metadata key-value filters to apply before ranking.

        Returns:
            Ordered list of SearchResult objects, highest score first.
        """

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete documents by their chunk IDs.

        Args:
            chunk_ids: List of chunk_id strings to remove.

        Returns:
            Number of documents deleted.
        """

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of vectors currently stored."""


class QdrantVectorStore(VectorStoreBase):
    """Qdrant-backed vector store using the qdrant-client Python SDK.

    Creates the collection on first use if it does not already exist.
    Supports metadata filtering via Qdrant's Filter payload.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self._client = None  # initialised lazily in _get_client()

    async def _get_client(self):
        """Lazily initialise and return the async Qdrant client.

        Creates the target collection if it does not exist, using the
        configured distance metric and embedding dimension.

        Returns:
            Initialised qdrant_client.AsyncQdrantClient instance.
        """
        raise NotImplementedError

    async def upsert(self, documents: list[VectorDocument]) -> int:
        """Upsert documents into the Qdrant collection in batches of 100.

        Converts VectorDocument metadata to Qdrant payload dicts and uses
        qdrant_client.models.PointStruct for serialisation.

        Args:
            documents: Documents with pre-computed embeddings.

        Returns:
            Count of upserted points.
        """
        raise NotImplementedError

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run a nearest-neighbour search using Qdrant's query_points API.

        Converts the filters dict to a qdrant_client.models.Filter object
        if provided.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            filters: Metadata equality filters.

        Returns:
            Ranked SearchResult list.
        """
        raise NotImplementedError

    async def delete(self, chunk_ids: list[str]) -> int:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError


class PgVectorStore(VectorStoreBase):
    """PostgreSQL + pgvector backed store using asyncpg.

    Stores embeddings in a table with columns: chunk_id, text, embedding,
    metadata (jsonb). Creates the table and the pgvector extension if absent.
    """

    CREATE_TABLE_SQL = """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS {table} (
        chunk_id  TEXT PRIMARY KEY,
        text      TEXT NOT NULL,
        embedding vector({dim}),
        metadata  JSONB DEFAULT '{{}}'
    );
    CREATE INDEX IF NOT EXISTS {table}_embedding_idx
        ON {table} USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self._pool = None

    async def _get_pool(self):
        """Lazily create and return an asyncpg connection pool.

        Also runs CREATE_TABLE_SQL on first connection to ensure the schema
        exists.

        Returns:
            asyncpg.Pool connected to the configured DSN.
        """
        raise NotImplementedError

    async def upsert(self, documents: list[VectorDocument]) -> int:
        """Bulk upsert using asyncpg executemany with ON CONFLICT DO UPDATE.

        Serialises embeddings with pgvector's vector type string format.

        Args:
            documents: List of VectorDocument objects.

        Returns:
            Number of rows affected.
        """
        raise NotImplementedError

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run a cosine similarity search using the <=> operator.

        Builds a WHERE clause from filters dict using JSONB containment (@>)
        for simple key-value filtering.

        Args:
            query_embedding: Query vector.
            top_k: Number of neighbours to return.
            filters: Metadata equality constraints.

        Returns:
            Ranked SearchResult list.
        """
        raise NotImplementedError

    async def delete(self, chunk_ids: list[str]) -> int:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError


def create_vector_store(config: VectorStoreConfig) -> VectorStoreBase:
    """Factory that returns a VectorStoreBase implementation for the given config.

    Args:
        config: VectorStoreConfig with backend set to 'qdrant' or 'pgvector'.

    Returns:
        Concrete VectorStoreBase instance.

    Raises:
        ValueError: If config.backend is not recognised.
    """
    if config.backend == "qdrant":
        return QdrantVectorStore(config)
    if config.backend == "pgvector":
        return PgVectorStore(config)
    raise ValueError(f"Unknown vector store backend: {config.backend}")
