"""Graph-augmented retrieval using FalkorDB relationship traversal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from storage.graph_store import GraphStore


@dataclass
class GraphRAGConfig:
    max_hops: int = 2
    max_nodes_per_hop: int = 10
    include_paper_abstracts: bool = True
    include_method_descriptions: bool = True
    context_max_tokens: int = 2000


@dataclass
class GraphContext:
    query: str
    subgraph_nodes: list[dict[str, Any]]
    subgraph_edges: list[dict[str, Any]]
    formatted_context: str
    retrieval_latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphRAG:
    """Retrieval-augmented generation using the FalkorDB knowledge graph.

    Complements VectorRAG by providing structured relational context:
    which methods a paper proposes, which papers use the same dataset,
    and what prior work a paper builds on. This structured context reduces
    hallucination for questions about explicit paper relationships.
    """

    def __init__(self, store: GraphStore, config: GraphRAGConfig | None = None) -> None:
        self.store = store
        self.config = config or GraphRAGConfig()

    async def retrieve_for_paper(
        self,
        arxiv_id: str,
        query: str | None = None,
    ) -> GraphContext:
        """Build a graph context centred on a specific paper.

        Fetches the paper's direct neighbours (authors, institutions, datasets,
        methods) and optionally the papers it builds on / compares to up to
        config.max_hops.

        Args:
            arxiv_id: arXiv ID of the focal paper.
            query: Optional natural language query to filter which graph
                   neighbours are most relevant.

        Returns:
            GraphContext with subgraph nodes, edges, and a formatted string.
        """
        raise NotImplementedError

    async def retrieve_by_concept(
        self,
        concept: str,
        top_papers: int = 5,
    ) -> GraphContext:
        """Retrieve papers, methods, and datasets related to a concept string.

        Performs a fuzzy match on Method and Concept nodes, then traverses
        back to Paper nodes via PROPOSES_METHOD and ADDRESSES_CONCEPT edges.

        Args:
            concept: Research concept or technique name (e.g. 'contrastive learning').
            top_papers: Maximum number of papers to include in the context.

        Returns:
            GraphContext with a subgraph of relevant papers and relationships.
        """
        raise NotImplementedError

    async def retrieve_comparison_context(
        self,
        arxiv_ids: list[str],
    ) -> GraphContext:
        """Build a comparison context for multiple papers.

        Extracts the union of methods and datasets across all specified papers
        and identifies shared and unique elements. Useful for answering
        comparative questions like "how do these papers differ in methodology?".

        Args:
            arxiv_ids: List of arXiv IDs to compare.

        Returns:
            GraphContext with overlap/diff analysis in the formatted_context.
        """
        raise NotImplementedError

    def format_graph_context(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        query: str | None = None,
    ) -> str:
        """Render a subgraph as a concise, LLM-readable structured text block.

        Organises nodes by type and represents relationships as prose sentences
        (e.g. "Paper X proposes Method Y and uses Dataset Z").
        Truncates to config.context_max_tokens by dropping low-priority nodes.

        Args:
            nodes: List of node property dicts with 'label' and 'id' keys.
            edges: List of edge property dicts with 'type', 'src', 'dst' keys.
            query: Optional query to guide which facts to emphasise.

        Returns:
            Structured text string for injection into an LLM prompt.
        """
        raise NotImplementedError

    async def hybrid_retrieve(
        self,
        vector_context,
        arxiv_ids: list[str],
        query: str,
    ) -> str:
        """Fuse vector and graph context into a single prompt-ready string.

        Takes a VectorRAG RAGContext and supplements it with graph-derived
        relational facts about the papers whose chunks were retrieved.
        Interleaves the two context types with clear section headers.

        Args:
            vector_context: RAGContext from VectorRAG.retrieve().
            arxiv_ids: arXiv IDs of the papers referenced in the vector context.
            query: Original query for relevance weighting.

        Returns:
            Combined context string with both dense and structured facts.
        """
        raise NotImplementedError

    def _nodes_to_dict(self, raw_results: list) -> list[dict[str, Any]]:
        """Convert raw FalkorDB query result rows into normalised node dicts.

        Args:
            raw_results: List of result rows from GraphStore.query().

        Returns:
            List of dicts with standardised 'id', 'label', and property keys.
        """
        raise NotImplementedError
