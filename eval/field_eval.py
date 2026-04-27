"""G-Eval style per-field evaluation for PaperExtraction outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from extraction.schemas import PaperExtraction


@dataclass
class FieldScore:
    field_name: str
    score: float                   # 0–5 G-Eval score, normalised to 0–1
    reasoning: str
    criteria: str
    raw_scores: list[float] = field(default_factory=list)   # per-sample scores


@dataclass
class ExtractionEvalResult:
    arxiv_id: str | None
    field_scores: list[FieldScore]
    overall_score: float           # weighted mean of field scores
    evaluation_model: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


FIELD_CRITERIA: dict[str, str] = {
    "title": (
        "Does the extracted title exactly match the title in the source paper? "
        "Score 5 if identical, 3 if minor punctuation differs, 1 if significantly wrong."
    ),
    "authors": (
        "Are all authors present with correct name spelling and affiliation? "
        "Deduct 1 point per missing or misspelled author."
    ),
    "institutions": (
        "Are all institutions correctly identified and deduplicated? "
        "Score 5 if all institutions are present and correctly named."
    ),
    "problem_statement": (
        "Does the problem statement accurately and completely describe the research "
        "problem as presented in the paper? Score 5 if complete and accurate, "
        "3 if partially correct, 1 if mostly wrong or missing key aspects."
    ),
    "datasets": (
        "Are all datasets mentioned in the paper captured with correct names and sizes? "
        "Score 5 if all datasets are present and correctly described."
    ),
    "methods": (
        "Are the key methodological contributions correctly described? "
        "Score based on completeness and accuracy of method descriptions."
    ),
    "results": (
        "Are numerical results (metrics, values, conditions) correctly extracted? "
        "Score 5 if all key results match exactly, deduct per error."
    ),
    "baselines": (
        "Are comparison baselines correctly identified with their reported performance? "
        "Score based on completeness and numerical accuracy."
    ),
    "limitations": (
        "Are the paper's limitations correctly and completely captured? "
        "Score 5 if all author-stated limitations are present."
    ),
}


class FieldEvaluator:
    """Evaluates each field of a PaperExtraction against a ground-truth or source.

    Uses the G-Eval methodology: generates multiple LLM judgements for each
    field using a chain-of-thought critique prompt, then averages the scores.
    Supports evaluation against a ground-truth PaperExtraction or directly
    against the source PDF text.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        num_samples: int = 3,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.num_samples = num_samples
        self.api_key = api_key

    async def evaluate(
        self,
        prediction: PaperExtraction,
        reference: PaperExtraction | None = None,
        source_text: str | None = None,
    ) -> ExtractionEvalResult:
        """Run G-Eval across all fields of a PaperExtraction.

        Either reference or source_text must be provided. If reference is
        given, scores are computed by comparing predicted and reference values.
        If only source_text is provided, the LLM is asked to judge accuracy
        against the raw document text.

        Args:
            prediction: The extracted PaperExtraction to evaluate.
            reference: Optional ground-truth PaperExtraction.
            source_text: Optional raw document text for reference-free evaluation.

        Returns:
            ExtractionEvalResult with per-field scores and an overall score.
        """
        raise NotImplementedError

    async def evaluate_field(
        self,
        field_name: str,
        predicted_value: Any,
        reference_value: Any | None = None,
        source_text: str | None = None,
    ) -> FieldScore:
        """Run G-Eval for a single extraction field.

        Constructs a critique prompt that includes the field's criteria string,
        the predicted value, and the reference/source. Samples num_samples
        completions, parses an integer score (1–5) from each, and averages.

        Args:
            field_name: Name of the PaperExtraction field being evaluated.
            predicted_value: The model's extracted value for this field.
            reference_value: Ground-truth value if available.
            source_text: Source document text for reference-free scoring.

        Returns:
            FieldScore with averaged score and one reasoning string.
        """
        raise NotImplementedError

    def _build_eval_prompt(
        self,
        field_name: str,
        predicted_value: Any,
        reference_value: Any | None,
        source_text: str | None,
    ) -> str:
        """Build the G-Eval critique prompt for one field.

        Injects the field's FIELD_CRITERIA string, the predicted value
        (serialised to JSON), and either the reference value or a truncated
        excerpt of the source text as context.

        Args:
            field_name: Field to evaluate.
            predicted_value: Extracted value.
            reference_value: Ground truth.
            source_text: Source document.

        Returns:
            Complete prompt string.
        """
        raise NotImplementedError

    def _parse_score(self, completion: str) -> float:
        """Extract a numeric score (1–5) from a G-Eval completion string.

        Searches for patterns like 'Score: 4' or '4/5' in the text.
        Returns 1.0 if no score is found to avoid inflating averages.

        Args:
            completion: Raw LLM completion text.

        Returns:
            Float score in [1.0, 5.0].
        """
        raise NotImplementedError

    def compute_overall_score(self, field_scores: list[FieldScore]) -> float:
        """Compute a weighted mean score across all evaluated fields.

        Fields with more informational content (results, methods) are weighted
        higher than structural fields (title, authors). Weights are hardcoded
        but configurable via a class-level FIELD_WEIGHTS dict.

        Args:
            field_scores: List of per-field FieldScore objects.

        Returns:
            Weighted mean score normalised to [0, 1].
        """
        raise NotImplementedError
