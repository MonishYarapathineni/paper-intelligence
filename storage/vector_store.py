"""Vector store abstraction supporting pgvector and Qdrant backends."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

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
    pg_dsn: str = "postgresql://localhost:5432/papers"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    distance_metric: Literal["cosine", "dot", "euclidean"] = "cosine"


class VectorStoreBase(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    async def upsert(self, documents: list[VectorDocument]) -> int:
        """Insert or update a batch of vector documents."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the top-k nearest neighbours for a query embedding."""

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete documents by their chunk IDs."""

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of vectors currently stored."""


class QdrantVectorStore(VectorStoreBase):
    """Qdrant-backed vector store using the qdrant-client Python SDK."""

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self._client = None

    def _to_uuid(self, chunk_id: str) -> str:
        """Convert hex chunk_id to UUID format for Qdrant compatibility."""
        padded = chunk_id.ljust(32, '0')
        return str(uuid.UUID(padded))

    async def _get_client(self) -> AsyncQdrantClient:
        """Lazily initialise and return the async Qdrant client."""
        if self._client is not None:
            return self._client

        self._client = AsyncQdrantClient(url=self.config.qdrant_url)

        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.config.collection_name not in existing:
            distance_map = {
                "cosine": Distance.COSINE,
                "dot": Distance.DOT,
                "euclidean": Distance.EUCLID,
            }
            await self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim,
                    distance=distance_map[self.config.distance_metric],
                ),
            )

        return self._client

    async def upsert(self, documents: list[VectorDocument]) -> int:
        """Upsert documents in batches of 100."""
        client = await self._get_client()
        batch_size = 100
        total = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            points = [
                PointStruct(
                    id=self._to_uuid(doc.chunk_id),
                    vector=doc.embedding,
                    payload={
                        "chunk_id": doc.chunk_id,
                        "text": doc.text,
                        **doc.metadata,
                    }
                )
                for doc in batch
            ]
            await client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )
            total += len(batch)

        return total

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Nearest-neighbour search with optional payload filtering."""
        client = await self._get_client()

        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )

        results = await client.query_points(
            collection_name=self.config.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            SearchResult(
                chunk_id=point.payload.get("chunk_id", str(point.id)),
                text=point.payload.get("text", ""),
                score=point.score,
                metadata={
                    k: v for k, v in point.payload.items()
                    if k not in ("chunk_id", "text")
                },
            )
            for point in results.points
        ]

    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete points by chunk_id."""
        client = await self._get_client()
        uuids = [self._to_uuid(cid) for cid in chunk_ids]

        await client.delete(
            collection_name=self.config.collection_name,
            points_selector=PointIdsList(points=uuids),
        )
        return len(chunk_ids)

    async def count(self) -> int:
        """Return total number of vectors in the collection."""
        client = await self._get_client()
        info = await client.get_collection(
            collection_name=self.config.collection_name
        )
        return info.points_count


class PgVectorStore(VectorStoreBase):
    """PostgreSQL + pgvector backed store — stub, not primary backend."""

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
        raise NotImplementedError

    async def upsert(self, documents: list[VectorDocument]) -> int:
        raise NotImplementedError

    async def search(self, query_embedding, top_k=10, filters=None):
        raise NotImplementedError

    async def delete(self, chunk_ids: list[str]) -> int:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError


def create_vector_store(config: VectorStoreConfig) -> VectorStoreBase:
    """Factory returning the configured VectorStoreBase implementation."""
    if config.backend == "qdrant":
        return QdrantVectorStore(config)
    if config.backend == "pgvector":
        return PgVectorStore(config)
    raise ValueError(f"Unknown vector store backend: {config.backend}")