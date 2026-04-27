"""FalkorDB graph store with schema registry for paper knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Node / Edge type definitions (schema registry)
# ---------------------------------------------------------------------------

NODE_TYPES: dict[str, dict[str, str]] = {
    "Paper": {
        "arxiv_id": "STRING",
        "title": "STRING",
        "published": "STRING",
        "abstract": "STRING",
    },
    "Author": {
        "name": "STRING",
        "email": "STRING",
    },
    "Institution": {
        "name": "STRING",
        "country": "STRING",
    },
    "Dataset": {
        "name": "STRING",
        "description": "STRING",
    },
    "Method": {
        "name": "STRING",
        "description": "STRING",
    },
    "Concept": {
        "name": "STRING",           # general research concept/term
        "category": "STRING",
    },
}

EDGE_TYPES: dict[str, dict[str, str]] = {
    "AUTHORED_BY": {"order": "INTEGER"},
    "AFFILIATED_WITH": {},
    "USES_DATASET": {"role": "STRING"},         # training / evaluation
    "PROPOSES_METHOD": {},
    "BUILDS_ON": {"relation": "STRING"},        # paper → paper
    "ADDRESSES_CONCEPT": {},
    "COMPARES_TO": {"notes": "STRING"},         # paper → paper (baseline)
}


@dataclass
class FalkorDBConfig:
    host: str = "localhost"
    port: int = 6379
    graph_name: str = "paper_intelligence"
    password: str | None = None
    decode_responses: bool = True


@dataclass
class GraphNode:
    label: str
    properties: dict[str, Any]
    node_id: str | None = None


@dataclass
class GraphEdge:
    rel_type: str
    src_label: str
    src_id: str
    dst_label: str
    dst_id: str
    properties: dict[str, Any] = field(default_factory=dict)


class GraphStore:
    """FalkorDB-backed knowledge graph store for paper relationships.

    Nodes represent Papers, Authors, Institutions, Datasets, Methods, and
    Concepts. Edges encode relationships extracted from PaperExtraction
    objects. Uses the falkordb Python client which communicates over the
    Redis protocol.
    """

    def __init__(self, config: FalkorDBConfig | None = None) -> None:
        self.config = config or FalkorDBConfig()
        self._conn = None    # falkordb.FalkorDB connection, set in connect()
        self._graph = None   # falkordb.Graph handle for self.config.graph_name

    async def connect(self) -> None:
        """Open a connection to FalkorDB and obtain a graph handle.

        Creates the graph if it does not already exist. Also calls
        _ensure_indexes() to set up label indexes for common lookup patterns.

        Raises:
            ConnectionError: If FalkorDB is unreachable at configured host/port.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Close the FalkorDB connection gracefully."""
        raise NotImplementedError

    async def upsert_paper(self, extraction) -> str:
        """Upsert a complete PaperExtraction as a subgraph.

        Creates or merges the Paper node, then calls _upsert_authors,
        _upsert_institutions, _upsert_datasets, and _upsert_methods to
        populate related nodes and edges in a single query batch.

        Args:
            extraction: PaperExtraction object from extraction/schemas.py.

        Returns:
            The graph node ID of the Paper node.
        """
        raise NotImplementedError

    async def upsert_node(self, node: GraphNode) -> str:
        """Merge a single node by label + a unique property key.

        Uses Cypher MERGE to avoid duplicates. The unique key is 'name'
        for most node types and 'arxiv_id' for Paper nodes.

        Args:
            node: GraphNode with label and properties.

        Returns:
            Internal FalkorDB node ID string.
        """
        raise NotImplementedError

    async def upsert_edge(self, edge: GraphEdge) -> None:
        """Create or update a directed relationship between two nodes.

        Uses MERGE on (src)-[rel]->(dst) to avoid duplicate edges.
        Sets or updates edge properties on match.

        Args:
            edge: GraphEdge describing the relationship.
        """
        raise NotImplementedError

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        """Execute an arbitrary read Cypher query and return results as dicts.

        Args:
            cypher: Cypher query string (read-only recommended here).
            params: Named parameters to substitute into the query.

        Returns:
            List of result row dicts keyed by column name.
        """
        raise NotImplementedError

    async def find_related_papers(
        self,
        arxiv_id: str,
        relation_types: list[str] | None = None,
        max_hops: int = 2,
    ) -> list[dict]:
        """Return papers related to the given paper within max_hops graph hops.

        Traverses BUILDS_ON and COMPARES_TO edges. When relation_types is
        provided, only traverses edges of those types.

        Args:
            arxiv_id: Source paper identifier.
            relation_types: Edge type filter (None = all edge types).
            max_hops: Maximum traversal depth.

        Returns:
            List of dicts with keys: arxiv_id, title, relation_path, distance.
        """
        raise NotImplementedError

    async def find_papers_by_method(self, method_name: str) -> list[dict]:
        """Return papers that propose or use a given method.

        Matches Method nodes by name (case-insensitive prefix search) and
        traverses PROPOSES_METHOD edges back to Paper nodes.

        Args:
            method_name: Method name to search for.

        Returns:
            List of paper property dicts.
        """
        raise NotImplementedError

    async def find_papers_by_dataset(self, dataset_name: str) -> list[dict]:
        """Return papers that use a given dataset for training or evaluation.

        Args:
            dataset_name: Dataset name to search for (case-insensitive).

        Returns:
            List of paper property dicts with a 'role' field (train/eval).
        """
        raise NotImplementedError

    async def get_author_coauthors(self, author_name: str) -> list[dict]:
        """Return all co-authors of a given author across indexed papers.

        Traverses Paper nodes shared by the author via AUTHORED_BY edges.

        Args:
            author_name: Exact or partial author name.

        Returns:
            List of dicts: {name, shared_paper_count, paper_titles}.
        """
        raise NotImplementedError

    def _ensure_indexes(self) -> None:
        """Create label property indexes for efficient node lookups.

        Creates indexes on Paper.arxiv_id, Author.name, Method.name, and
        Dataset.name. Uses CREATE INDEX IF NOT EXISTS syntax supported by
        FalkorDB >= 3.0.
        """
        raise NotImplementedError

    def _upsert_authors(self, paper_node_id: str, authors: list) -> None:
        """Upsert Author nodes and AUTHORED_BY edges for a paper.

        Also creates AFFILIATED_WITH edges to Institution nodes if the author
        has affiliations in the extraction.

        Args:
            paper_node_id: The Paper node ID to connect authors to.
            authors: List of Author Pydantic objects from PaperExtraction.
        """
        raise NotImplementedError

    def _upsert_datasets(self, paper_node_id: str, datasets: list) -> None:
        """Upsert Dataset nodes and USES_DATASET edges for a paper.

        Args:
            paper_node_id: The Paper node ID.
            datasets: List of Dataset Pydantic objects from PaperExtraction.
        """
        raise NotImplementedError

    def _upsert_methods(self, paper_node_id: str, methods: list) -> None:
        """Upsert Method nodes and PROPOSES_METHOD edges for a paper.

        Args:
            paper_node_id: The Paper node ID.
            methods: List of Method Pydantic objects from PaperExtraction.
        """
        raise NotImplementedError
