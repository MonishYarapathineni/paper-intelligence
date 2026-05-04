"""FalkorDB graph store with schema registry and HITL escalation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema registry — canonical node and edge types
# ---------------------------------------------------------------------------

NODE_TYPES: dict[str, dict[str, str]] = {
    "Paper": {
        "arxiv_id": "STRING",
        "title": "STRING",
        "abstract": "STRING",
        "problem_statement": "STRING",
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
    "Result": {
        "metric_name": "STRING",
        "metric_value": "STRING",
        "conditions": "STRING",
    },
    "Metric": {
        "name": "STRING",
    },
}

EDGE_TYPES: dict[str, dict[str, str]] = {
    "AUTHORED_BY": {"order": "INTEGER"},
    "AFFILIATED_WITH": {},
    "USES_DATASET": {"role": "STRING"},
    "PROPOSES_METHOD": {},
    "BUILDS_ON": {"relation": "STRING"},
    "COMPARES_TO": {"notes": "STRING"},
    "HAS_RESULT": {},
    "ON_DATASET": {},
    "USES_METRIC": {},
}


# ---------------------------------------------------------------------------
# Schema registry — persisted to JSON, governs HITL escalation
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA = {
    "node_types": {
        k: list(v.keys()) for k, v in NODE_TYPES.items()
    },
    "edge_types": {
        k: list(v.keys()) for k, v in EDGE_TYPES.items()
    },
}


@dataclass
class SchemaRegistry:
    """Persists canonical node/edge types and governs HITL escalation.

    On init, loads from registry_path if it exists, otherwise seeds from
    DEFAULT_SCHEMA. All additions require explicit approval — call
    add_node_type / add_edge_type only after human confirmation.
    """

    registry_path: Path
    schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.registry_path.exists():
            self.schema = json.loads(self.registry_path.read_text())
        else:
            self.schema = DEFAULT_SCHEMA
            self.save()

    def save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.schema, indent=2))

    @property
    def node_types(self) -> dict[str, list[str]]:
        return self.schema["node_types"]

    @property
    def edge_types(self) -> dict[str, list[str]]:
        return self.schema["edge_types"]

    def add_node_type(self, name: str, properties: list[str]) -> None:
        """Add a new node type after HITL approval."""
        self.schema["node_types"][name] = properties
        self.save()

    def add_edge_type(self, name: str, properties: list[str]) -> None:
        """Add a new edge type after HITL approval."""
        self.schema["edge_types"][name] = properties
        self.save()

    def as_prompt_str(self) -> str:
        """Render the registry as a readable string for LLM prompts."""
        lines = ["NODE TYPES:"]
        for name, props in self.node_types.items():
            lines.append(f"  {name}: {', '.join(props)}")
        lines.append("EDGE TYPES:")
        for name, props in self.edge_types.items():
            lines.append(f"  {name}: {', '.join(props) or 'no properties'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HITL escalation
# ---------------------------------------------------------------------------

@dataclass
class HITLEscalation:
    """Represents a field that couldn't be mapped to an existing schema type.

    Queued for human review when LLM confidence is below threshold.
    Human approves by calling registry.add_node_type / add_edge_type.
    """
    field_name: str
    field_value: Any
    candidate_types: list[str]   # top-k closest existing types
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass
class FalkorDBConfig:
    host: str = "localhost"
    port: int = 6379
    graph_name: str = "paper_intelligence"
    password: str | None = None
    decode_responses: bool = True
    registry_path: Path = Path("data/schema_registry.json")


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


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """FalkorDB-backed knowledge graph store for paper relationships.

    Nodes: Paper, Author, Institution, Dataset, Method, Result, Metric.
    Edges: AUTHORED_BY, AFFILIATED_WITH, USES_DATASET, PROPOSES_METHOD,
           BUILDS_ON, COMPARES_TO, HAS_RESULT, ON_DATASET, USES_METRIC.

    Schema evolution goes through SchemaRegistry — new types require HITL
    approval before being added to the graph.
    """

    def __init__(self, config: FalkorDBConfig | None = None) -> None:
        self.config = config or FalkorDBConfig()
        self.registry = SchemaRegistry(self.config.registry_path)
        self._conn = None
        self._graph = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open a connection to FalkorDB and obtain a graph handle.

        Creates the graph if it does not already exist and sets up
        label indexes for common lookup patterns.
        """
        import falkordb

        self._conn = falkordb.FalkorDB(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
            decode_responses=self.config.decode_responses,
        )
        self._graph = self._conn.select_graph(self.config.graph_name)
        self._ensure_indexes()

    async def close(self) -> None:
        """Close the FalkorDB connection gracefully."""
        if self._conn:
            try:
                self._conn.connection.close()
            except Exception:
                pass
            self._conn = None
            self._graph = None

    # ------------------------------------------------------------------
    # Core upsert primitives
    # ------------------------------------------------------------------

    async def upsert_node(self, node: GraphNode) -> str:
        """Merge a single node by label + unique key, set all other properties.

        Uses MERGE to avoid duplicates. Unique key is arxiv_id for Paper,
        name for everything else.

        Returns:
            Internal FalkorDB node id as a string.
        """
        unique_key = "arxiv_id" if node.label == "Paper" else "name"
        unique_val = node.properties.get(unique_key, "")

        # Build SET clause for non-unique properties
        other_props = {k: v for k, v in node.properties.items() if k != unique_key}
        set_stmt = ""
        if other_props:
            set_clauses = ", ".join(f"n.{k} = ${k}" for k in other_props)
            set_stmt = f"SET {set_clauses}"

        cypher = f"""
            MERGE (n:{node.label} {{{unique_key}: $unique_val}})
            {set_stmt}
            RETURN id(n) AS node_id
        """
        params = {"unique_val": unique_val, **other_props}
        result = self._graph.query(cypher, params)
        return str(result.result_set[0][0])

    async def upsert_edge(self, edge: GraphEdge) -> None:
        """Create or update a directed relationship between two nodes.

        Uses MERGE on (src)-[rel]->(dst) to avoid duplicate edges.
        """
        unique_src = "arxiv_id" if edge.src_label == "Paper" else "name"
        unique_dst = "arxiv_id" if edge.dst_label == "Paper" else "name"

        prop_str = ""
        if edge.properties:
            props = ", ".join(f"{k}: ${k}" for k in edge.properties)
            prop_str = f" {{{props}}}"

        cypher = f"""
            MATCH (src:{edge.src_label} {{{unique_src}: $src_id}})
            MATCH (dst:{edge.dst_label} {{{unique_dst}: $dst_id}})
            MERGE (src)-[r:{edge.rel_type}{prop_str}]->(dst)
        """
        params = {
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            **edge.properties,
        }
        self._graph.query(cypher, params)

    # ------------------------------------------------------------------
    # Paper ingestion
    # ------------------------------------------------------------------

    async def upsert_paper(self, extraction) -> str:
        """Upsert a complete PaperExtraction as a subgraph.

        Creates or merges Paper, Author, Institution, Dataset, Method
        nodes and all connecting edges in order.

        Returns:
            The paper node id (arxiv_id or title hash).
        """
        paper_id = extraction.arxiv_id or self._paper_id(extraction.title)

        await self.upsert_node(GraphNode(
            label="Paper",
            properties={
                "arxiv_id": paper_id,
                "title": extraction.title,
                "abstract": extraction.abstract or "",
                "problem_statement": extraction.problem_statement or "",
            }
        ))

        await self._upsert_authors(paper_id, extraction.authors)
        await self._upsert_datasets(paper_id, extraction.datasets)
        await self._upsert_methods(paper_id, extraction.methods)

        return paper_id

    async def _upsert_authors(self, paper_id: str, authors: list) -> None:
        """Upsert Author nodes and AUTHORED_BY + AFFILIATED_WITH edges."""
        for i, author in enumerate(authors):
            await self.upsert_node(GraphNode(
                label="Author",
                properties={
                    "name": author.name,
                    "email": author.email or "",
                }
            ))
            await self.upsert_edge(GraphEdge(
                rel_type="AUTHORED_BY",
                src_label="Paper", src_id=paper_id,
                dst_label="Author", dst_id=author.name,
                properties={"order": i},
            ))
            for affiliation in author.affiliations:
                await self.upsert_node(GraphNode(
                    label="Institution",
                    properties={"name": affiliation, "country": ""},
                ))
                await self.upsert_edge(GraphEdge(
                    rel_type="AFFILIATED_WITH",
                    src_label="Author", src_id=author.name,
                    dst_label="Institution", dst_id=affiliation,
                ))

    async def _upsert_datasets(self, paper_id: str, datasets: list) -> None:
        """Upsert Dataset nodes and USES_DATASET edges."""
        for dataset in datasets:
            await self.upsert_node(GraphNode(
                label="Dataset",
                properties={
                    "name": dataset.name,
                    "description": dataset.description or "",
                }
            ))
            await self.upsert_edge(GraphEdge(
                rel_type="USES_DATASET",
                src_label="Paper", src_id=paper_id,
                dst_label="Dataset", dst_id=dataset.name,
            ))

    async def _upsert_methods(self, paper_id: str, methods: list) -> None:
        """Upsert Method nodes and PROPOSES_METHOD edges."""
        for method in methods:
            await self.upsert_node(GraphNode(
                label="Method",
                properties={
                    "name": method.name,
                    "description": method.description or "",
                }
            ))
            await self.upsert_edge(GraphEdge(
                rel_type="PROPOSES_METHOD",
                src_label="Paper", src_id=paper_id,
                dst_label="Method", dst_id=method.name,
            ))

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict]:
        """Execute a Cypher query and return results as a list of dicts."""
        result = self._graph.query(cypher, params or {})
        if not result.result_set:
            return []
        headers = result.header
        return [
            {headers[i]: row[i] for i in range(len(headers))}
            for row in result.result_set
        ]

    async def find_related_papers(
        self,
        arxiv_id: str,
        relation_types: list[str] | None = None,
        max_hops: int = 2,
    ) -> list[dict]:
        """Find papers related via BUILDS_ON or COMPARES_TO within max_hops.

        This is the core GraphRAG query — vector search can't express
        multi-hop relationships like "papers that build on papers that
        use ImageNet."
        """
        rel_filter = ""
        if relation_types:
            rel_filter = ":" + "|".join(relation_types)

        cypher = f"""
            MATCH (src:Paper {{arxiv_id: $arxiv_id}})
            MATCH (src)-[r{rel_filter}*1..{max_hops}]-(related:Paper)
            WHERE related.arxiv_id <> $arxiv_id
            RETURN DISTINCT related.arxiv_id AS arxiv_id,
                            related.title AS title,
                            length(r) AS distance
            ORDER BY distance
        """
        return await self.query(cypher, {"arxiv_id": arxiv_id})

    async def find_papers_by_dataset(self, dataset_name: str) -> list[dict]:
        """Find all papers that used a specific dataset."""
        cypher = """
            MATCH (p:Paper)-[:USES_DATASET]->(d:Dataset)
            WHERE toLower(d.name) CONTAINS toLower($name)
            RETURN p.arxiv_id AS arxiv_id, p.title AS title, d.name AS dataset
        """
        return await self.query(cypher, {"name": dataset_name})

    async def find_papers_by_method(self, method_name: str) -> list[dict]:
        """Find all papers that proposed or used a specific method."""
        cypher = """
            MATCH (p:Paper)-[:PROPOSES_METHOD]->(m:Method)
            WHERE toLower(m.name) CONTAINS toLower($name)
            RETURN p.arxiv_id AS arxiv_id, p.title AS title, m.name AS method
        """
        return await self.query(cypher, {"name": method_name})

    async def get_author_coauthors(self, author_name: str) -> list[dict]:
        """Return all co-authors of a given author across indexed papers."""
        cypher = """
            MATCH (a:Author {name: $name})<-[:AUTHORED_BY]-(p:Paper)
            MATCH (p)-[:AUTHORED_BY]->(coauthor:Author)
            WHERE coauthor.name <> $name
            RETURN coauthor.name AS name, count(p) AS shared_papers
            ORDER BY shared_papers DESC
        """
        return await self.query(cypher, {"name": author_name})

    # ------------------------------------------------------------------
    # Schema evolution — HITL gated
    # ------------------------------------------------------------------

    async def map_to_schema(
        self,
        field_name: str,
        field_value: Any,
        confidence_threshold: float = 0.7,
    ) -> tuple[str | None, float, HITLEscalation | None]:
        """Use Claude to map a new field to existing schema types.

        Sends field_name + field_value + registry to Claude Haiku.
        Returns matched type if confidence >= threshold, otherwise
        returns a HITLEscalation for human review.

        Returns:
            (matched_type, confidence, escalation_or_None)
        """
        import httpx

        registry_str = self.registry.as_prompt_str()
        prompt = f"""You are a schema mapping assistant.

Given this schema registry:
{registry_str}

Map this field to the most appropriate existing node or edge type:
Field name: {field_name}
Field value: {field_value}

Respond with JSON only:
{{"matched_type": "TypeName", "confidence": 0.0-1.0, "reason": "..."}}
"""
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        data = response.json()
        import json as json_mod
        result = json_mod.loads(data["content"][0]["text"])

        matched_type = result.get("matched_type")
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "")

        if confidence >= confidence_threshold:
            return matched_type, confidence, None

        # Below threshold — escalate to HITL
        escalation = HITLEscalation(
            field_name=field_name,
            field_value=field_value,
            candidate_types=[matched_type] if matched_type else [],
            confidence=confidence,
            reason=reason,
        )
        return None, confidence, escalation

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create label indexes for efficient node lookups."""
        indexes = [
            "CREATE INDEX FOR (p:Paper) ON (p.arxiv_id)",
            "CREATE INDEX FOR (a:Author) ON (a.name)",
            "CREATE INDEX FOR (d:Dataset) ON (d.name)",
            "CREATE INDEX FOR (m:Method) ON (m.name)",
            "CREATE INDEX FOR (i:Institution) ON (i.name)",
        ]
        for idx in indexes:
            try:
                self._graph.query(idx)
            except Exception:
                pass  # index already exists

    def _paper_id(self, title: str) -> str:
        """Generate a stable paper id from title when arxiv_id is absent."""
        return sha1(title.encode()).hexdigest()[:12]