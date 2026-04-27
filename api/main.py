"""FastAPI application for the paper-intelligence pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extraction.schemas import PaperExtraction
from ingestion.arxiv_fetcher import ArxivFetcher, FetchConfig
from storage.graph_store import FalkorDBConfig, GraphStore
from storage.vector_store import VectorStoreConfig, create_vector_store


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    arxiv_id: str = Field(..., examples=["2304.01373"])
    force_reprocess: bool = Field(
        False, description="Re-ingest even if the paper is already in the store."
    )


class IngestResponse(BaseModel):
    arxiv_id: str
    status: str          # "queued" | "completed" | "already_exists"
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(10, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)
    use_graph_context: bool = False


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    graph_context: str | None = None
    retrieval_latency_ms: float


class ExtractRequest(BaseModel):
    arxiv_id: str
    fields: list[str] | None = Field(
        None, description="Subset of PaperExtraction fields to return. None returns all."
    )


class EvalRequest(BaseModel):
    arxiv_id: str
    run_field_eval: bool = True
    run_rag_eval: bool = False
    run_hallucination_eval: bool = True
    source_text: str | None = None


class EvalResponse(BaseModel):
    arxiv_id: str
    field_scores: dict[str, float] | None = None
    overall_score: float | None = None
    hallucination_rate: float | None = None
    consistency_score: float | None = None


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    """Holds shared async clients initialised at startup."""
    fetcher: ArxivFetcher
    vector_store: Any
    graph_store: GraphStore


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and teardown shared resources on application start/stop.

    Creates the ArxivFetcher, vector store, and graph store on startup.
    Closes connections gracefully on shutdown.
    """
    app_state.fetcher = ArxivFetcher(FetchConfig())
    app_state.vector_store = create_vector_store(VectorStoreConfig())
    app_state.graph_store = GraphStore(FalkorDBConfig())
    await app_state.graph_store.connect()
    yield
    await app_state.graph_store.close()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Paper Intelligence API",
    description="Multimodal document intelligence pipeline for scientific papers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_paper(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Ingest a paper by arXiv ID into the pipeline.

    Downloads the PDF, parses it with PDFParser, routes pages through the
    complexity router, chunks the result, runs VLM extraction, and stores
    embeddings in the vector store and the knowledge graph in FalkorDB.

    The heavy processing is offloaded to a background task so the endpoint
    returns immediately with status='queued'.

    Args:
        request: IngestRequest with the arXiv ID to process.
        background_tasks: FastAPI background task manager.

    Returns:
        IngestResponse with status and a message.
    """
    raise NotImplementedError


@app.get("/papers/{arxiv_id}", response_model=PaperExtraction, tags=["Papers"])
async def get_paper(arxiv_id: str) -> PaperExtraction:
    """Retrieve the full structured extraction for a previously ingested paper.

    Looks up the paper's PaperExtraction stored in the graph store and
    returns it as a Pydantic model.

    Args:
        arxiv_id: arXiv identifier (e.g. '2304.01373').

    Returns:
        PaperExtraction or 404 if not found.
    """
    raise NotImplementedError


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(request: SearchRequest) -> SearchResponse:
    """Search for relevant paper chunks using dense vector retrieval.

    Embeds the query, retrieves the top-k chunks from the vector store,
    and optionally augments results with FalkorDB graph context.

    Args:
        request: SearchRequest with query, top_k, filters, and graph flag.

    Returns:
        SearchResponse with ranked chunks and optional graph context.
    """
    raise NotImplementedError


@app.get("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_get(
    q: str = Query(..., min_length=3, max_length=1000),
    top_k: int = Query(10, ge=1, le=50),
) -> SearchResponse:
    """GET convenience wrapper around the POST /search endpoint.

    Args:
        q: Query string.
        top_k: Number of results.

    Returns:
        SearchResponse.
    """
    raise NotImplementedError


@app.post("/extract", response_model=PaperExtraction, tags=["Extraction"])
async def extract_paper(request: ExtractRequest) -> PaperExtraction:
    """Re-run VLM extraction for an already-ingested paper.

    Useful for re-extracting with a different model or updated prompt.
    The PDF must already be cached locally (i.e. the paper was previously
    ingested successfully).

    Args:
        request: ExtractRequest specifying the arXiv ID and optional field subset.

    Returns:
        Updated PaperExtraction.
    """
    raise NotImplementedError


@app.post("/eval", response_model=EvalResponse, tags=["Evaluation"])
async def evaluate_extraction(request: EvalRequest) -> EvalResponse:
    """Run the evaluation suite on a paper's extraction.

    Supports field-level G-Eval, hallucination detection, and consistency
    checking. Pass source_text to enable reference-free evaluation without
    a ground-truth annotation.

    Args:
        request: EvalRequest specifying which evals to run.

    Returns:
        EvalResponse with scores for each requested evaluation type.
    """
    raise NotImplementedError


@app.get("/papers/{arxiv_id}/related", tags=["Knowledge Graph"])
async def get_related_papers(
    arxiv_id: str,
    max_hops: int = Query(2, ge=1, le=3),
) -> list[dict[str, Any]]:
    """Return papers related to the given paper via graph traversal.

    Traverses BUILDS_ON and COMPARES_TO edges up to max_hops hops in the
    FalkorDB knowledge graph.

    Args:
        arxiv_id: Source paper identifier.
        max_hops: Traversal depth limit.

    Returns:
        List of related paper dicts with arxiv_id, title, and relation_path.
    """
    raise NotImplementedError


@app.get("/health", tags=["Meta"])
async def health() -> dict[str, str]:
    """Return the service health status.

    Returns:
        Dict with status='ok' when all backend connections are live.
    """
    return {"status": "ok"}
