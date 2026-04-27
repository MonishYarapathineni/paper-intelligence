"""Hallucination detection for PaperExtraction fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from extraction.schemas import PaperExtraction


class HallucinationType(str, Enum):
    FABRICATED_METRIC = "fabricated_metric"
    FABRICATED_AUTHOR = "fabricated_author"
    FABRICATED_DATASET = "fabricated_dataset"
    FABRICATED_CITATION = "fabricated_citation"
    UNSUPPORTED_CLAIM = "unsupported_claim"     # claim in problem_statement not in source
    WRONG_AFFILIATION = "wrong_affiliation"
    OTHER = "other"


@dataclass
class HallucinationFlag:
    hallucination_type: HallucinationType
    field_name: str
    flagged_value: str
    evidence: str      # quote from source text that contradicts or fails to support the value
    confidence: float  # 0–1, how confident the evaluator is this is a hallucination
    is_confirmed: bool = False    # True once a human or external oracle confirms it


@dataclass
class HallucinationReport:
    arxiv_id: str | None
    flags: list[HallucinationFlag]
    hallucination_rate: float     # fraction of checked claims that are flagged
    checked_claims: int
    evaluation_model: str
    source_text_used: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class HallucinationEvaluator:
    """Detects hallucinations in a PaperExtraction against the source document.

    Uses a combination of:
    1. Lexical presence checks — numerical metrics and proper nouns must appear
       verbatim or near-verbatim in the source text.
    2. NLI-based checking — uses an LLM as an NLI model to judge whether each
       extracted claim is entailed by, neutral to, or contradicted by the source.
    3. Cross-reference checks — author names are verified against the arXiv API.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        use_arxiv_verification: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.use_arxiv_verification = use_arxiv_verification

    async def evaluate(
        self,
        extraction: PaperExtraction,
        source_text: str,
    ) -> HallucinationReport:
        """Run the full hallucination detection suite on an extraction.

        Runs metric, author, dataset, and claim checks in parallel and
        aggregates all flags into a HallucinationReport.

        Args:
            extraction: PaperExtraction to check.
            source_text: Full concatenated Markdown text of the source PDF.

        Returns:
            HallucinationReport with all flags and overall hallucination rate.
        """
        raise NotImplementedError

    async def check_metrics(
        self,
        extraction: PaperExtraction,
        source_text: str,
    ) -> list[HallucinationFlag]:
        """Verify that every reported metric value appears in the source text.

        Searches for each metric's value (with tolerance for formatting
        differences like '0.912' vs '91.2%') in the source text. Flags
        values that cannot be found as FABRICATED_METRIC.

        Args:
            extraction: PaperExtraction with results and baselines.
            source_text: Raw source document text.

        Returns:
            List of FABRICATED_METRIC HallucinationFlag objects.
        """
        raise NotImplementedError

    async def check_authors(
        self,
        extraction: PaperExtraction,
        source_text: str,
    ) -> list[HallucinationFlag]:
        """Verify that author names appear in the source document.

        Checks each author name against the source text using fuzzy matching.
        If use_arxiv_verification is True, also queries the arXiv API to
        confirm author names for papers with known arxiv_id.

        Args:
            extraction: PaperExtraction with authors field.
            source_text: Raw source document text.

        Returns:
            List of FABRICATED_AUTHOR flags.
        """
        raise NotImplementedError

    async def check_datasets(
        self,
        extraction: PaperExtraction,
        source_text: str,
    ) -> list[HallucinationFlag]:
        """Verify that dataset names appear in the source document.

        Each dataset name should appear verbatim (case-insensitive) in the
        source text. Flags names not found.

        Args:
            extraction: PaperExtraction with datasets field.
            source_text: Raw source document text.

        Returns:
            List of FABRICATED_DATASET flags.
        """
        raise NotImplementedError

    async def check_unsupported_claims(
        self,
        extraction: PaperExtraction,
        source_text: str,
    ) -> list[HallucinationFlag]:
        """Use NLI prompting to check if the problem_statement is entailed by the source.

        Splits the problem_statement into sentences, then for each sentence
        prompts the LLM with the source abstract + intro to determine whether
        the sentence is (a) entailed, (b) neutral, or (c) contradicted.
        Flags sentences labelled (b) or (c).

        Args:
            extraction: PaperExtraction with problem_statement.
            source_text: Raw source document (typically abstract + introduction).

        Returns:
            List of UNSUPPORTED_CLAIM flags.
        """
        raise NotImplementedError

    def _value_in_source(self, value: str, source_text: str, tolerance: float = 0.05) -> bool:
        """Check whether a string value appears in source_text within tolerance.

        For numeric values, also checks for common representation variants
        (e.g. fractional vs percentage). For strings, uses case-insensitive
        substring matching.

        Args:
            value: Extracted value to search for.
            source_text: Full document text.
            tolerance: Fractional tolerance for numeric comparisons.

        Returns:
            True if the value is found in the source text.
        """
        raise NotImplementedError

    def _compute_hallucination_rate(self, flags: list[HallucinationFlag], total: int) -> float:
        """Compute the fraction of checked claims that are flagged.

        Args:
            flags: Confirmed hallucination flags.
            total: Total number of claims checked.

        Returns:
            Float in [0, 1].
        """
        if total == 0:
            return 0.0
        return len(flags) / total
