"""Pydantic output schemas for structured paper extraction."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class Author(BaseModel):
    name: str = Field(..., description="Full name of the author.")
    affiliations: list[str] = Field(
        default_factory=list,
        description="List of institution names this author is affiliated with.",
    )
    email: str | None = Field(None, description="Contact email if present in the paper.")
    is_corresponding: bool = Field(
        False, description="True if marked as the corresponding author."
    )


class Institution(BaseModel):
    name: str = Field(..., description="Full institution name.")
    department: str | None = Field(None, description="Department or lab name if given.")
    country: str | None = Field(None, description="Country of the institution.")


class Dataset(BaseModel):
    name: str = Field(..., description="Dataset name as used in the paper.")
    description: str | None = Field(None, description="Brief description of the dataset.")
    size: str | None = Field(
        None, description="Dataset size (e.g. '1.2M samples', '50K images')."
    )
    source_url: str | None = Field(None, description="URL or citation if provided.")
    split_info: str | None = Field(
        None, description="Train/val/test split sizes if reported."
    )


class Metric(BaseModel):
    name: str = Field(..., description="Metric name (e.g. 'F1', 'BLEU', 'Accuracy').")
    value: float | str = Field(..., description="Reported value, may be a percentage string.")
    dataset: str | None = Field(None, description="Dataset this metric was measured on.")
    split: str | None = Field(None, description="Data split (train/val/test).")
    conditions: str | None = Field(
        None, description="Additional conditions (e.g. 'zero-shot', 'fine-tuned')."
    )


class Method(BaseModel):
    name: str = Field(..., description="Method or component name.")
    description: str = Field(..., description="What this method does in the paper's context.")
    novelty: str | None = Field(
        None, description="How this method differs from prior work according to the authors."
    )
    components: list[str] = Field(
        default_factory=list,
        description="Sub-components or steps of this method.",
    )


class Baseline(BaseModel):
    name: str = Field(..., description="Name of the baseline system or approach.")
    citation: str | None = Field(None, description="Citation key or reference label.")
    metrics: list[Metric] = Field(
        default_factory=list,
        description="Performance metrics reported for this baseline.",
    )
    notes: str | None = Field(
        None, description="Any caveats about how the baseline was set up.",
    )


class ResultEntry(BaseModel):
    method_name: str = Field(
        ..., description="Method (proposed or ablation) that achieved these results."
    )
    metrics: list[Metric] = Field(default_factory=list)
    dataset: str | None = None
    is_proposed: bool = Field(
        True, description="True if this is the paper's proposed method, False for ablations."
    )


class LimitationType(str, Enum):
    COMPUTATIONAL = "computational"
    DATA = "data"
    SCOPE = "scope"
    REPRODUCIBILITY = "reproducibility"
    ETHICAL = "ethical"
    OTHER = "other"


class Limitation(BaseModel):
    description: str = Field(..., description="Description of the limitation as stated by the authors.")
    type: LimitationType = Field(LimitationType.OTHER, description="Category of limitation.")
    is_author_stated: bool = Field(
        True, description="False if this limitation was inferred, not explicitly stated."
    )


# ---------------------------------------------------------------------------
# Root schema
# ---------------------------------------------------------------------------

class FieldConfidence(BaseModel):
    """Confidence scores for each extracted field, on a 0.0 to 1.0 scale."""

    title: float = 0.0
    authors: float = 0.0
    datasets: float = 0.0
    methods: float = 0.0
    results: float = 0.0
    baselines: float = 0.0
    limitations: float = 0.0
    problem_statement: float = 0.0

class PaperExtraction(BaseModel):
    """Complete structured extraction of a scientific paper."""

    arxiv_id: str | None = Field(None, description="arXiv identifier if available.")
    title: str = Field(..., description="Full title of the paper exactly as written.")
    authors: list[Author] = Field(
        default_factory=list,
        description="Ordered list of authors.",
    )
    institutions: list[Institution] = Field(
        default_factory=list,
        description="Deduplicated list of affiliated institutions.",
    )
    problem_statement: str = Field(
        ...,
        description=(
            "1–3 sentence description of the core problem the paper addresses. "
            "Should cover: what is lacking in prior work and why it matters."
        ),
    )
    datasets: list[Dataset] = Field(
        default_factory=list,
        description="All datasets used for training, evaluation, or analysis.",
    )
    methods: list[Method] = Field(
        default_factory=list,
        description="Key methods, models, or algorithmic contributions proposed.",
    )
    results: list[ResultEntry] = Field(
        default_factory=list,
        description="Quantitative results reported in the paper.",
    )
    baselines: list[Baseline] = Field(
        default_factory=list,
        description="Baseline systems compared against the proposed method.",
    )
    limitations: list[Limitation] = Field(
        default_factory=list,
        description="Limitations acknowledged by the authors or inferable from the work.",
    )
    abstract: str | None = Field(None, description="Full abstract text.")
    conclusion_summary: str | None = Field(
        None,
        description="1–2 sentence summary of the paper's conclusions.",
    )
    extraction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence score assigned by the VLM extractor.",
    )
    extraction_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance info: model used, timestamp, source pages, etc.",
    )

    field_confidence: FieldConfidence = Field(default_factory=FieldConfidence) 

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be empty")
        return v.strip()

    @field_validator("problem_statement")
    @classmethod
    def problem_statement_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("problem_statement must not be empty")
        return v.strip()


class PartialExtraction(PaperExtraction):
    """Relaxed version of PaperExtraction used during incremental/page-level extraction.

    All required fields from PaperExtraction become optional here so that a
    partial result can be stored while remaining pages are processed.
    """

    title: str = Field(default="", description="Title — may be empty until the title page is processed.")
    problem_statement: str = Field(default="", description="Problem statement — filled in after abstract/intro.")

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:  # type: ignore[override]
        return v  # allow empty in partial results

    @field_validator("problem_statement")
    @classmethod
    def problem_statement_not_empty(cls, v: str) -> str:  # type: ignore[override]
        return v
