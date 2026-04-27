"""Cross-field consistency evaluation for PaperExtraction outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from extraction.schemas import PaperExtraction


class InconsistencyType(str, Enum):
    AUTHOR_INSTITUTION_MISMATCH = "author_institution_mismatch"
    METRIC_BASELINE_CONFLICT = "metric_baseline_conflict"
    DATASET_RESULT_MISMATCH = "dataset_result_mismatch"
    METHOD_RESULT_MISMATCH = "method_result_mismatch"
    DUPLICATE_ENTRY = "duplicate_entry"
    IMPOSSIBLE_VALUE = "impossible_value"     # e.g. accuracy > 100%
    OTHER = "other"


@dataclass
class Inconsistency:
    type: InconsistencyType
    fields_involved: list[str]
    description: str
    severity: float                # 0–1, higher = more likely to affect downstream use
    suggested_fix: str | None = None


@dataclass
class ConsistencyReport:
    arxiv_id: str | None
    is_consistent: bool
    inconsistencies: list[Inconsistency]
    consistency_score: float      # 1.0 minus weighted sum of severity scores
    checked_rules: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class ConsistencyEvaluator:
    """Detects cross-field inconsistencies in a PaperExtraction.

    Runs a battery of rule-based and LLM-assisted checks to find
    contradictions between fields — for example, a result claiming 95% F1
    on a dataset that the extraction lists as a training-only split.
    """

    def evaluate(self, extraction: PaperExtraction) -> ConsistencyReport:
        """Run all consistency checks on a PaperExtraction.

        Calls each check_* method, collects any Inconsistency objects, and
        computes an overall consistency score.

        Args:
            extraction: PaperExtraction to evaluate.

        Returns:
            ConsistencyReport with all detected inconsistencies.
        """
        raise NotImplementedError

    def check_author_institution_alignment(
        self,
        extraction: PaperExtraction,
    ) -> list[Inconsistency]:
        """Verify that every author affiliation appears in the institutions list.

        Flags authors whose affiliation strings do not fuzzy-match any
        institution in the institutions field. Uses Levenshtein distance
        for matching to handle minor OCR differences.

        Args:
            extraction: PaperExtraction to check.

        Returns:
            List of AUTHOR_INSTITUTION_MISMATCH inconsistencies.
        """
        raise NotImplementedError

    def check_metric_ranges(
        self,
        extraction: PaperExtraction,
    ) -> list[Inconsistency]:
        """Flag metric values that are outside plausible ranges.

        Checks for:
        - Percentage metrics > 100 or < 0
        - Named metrics with known ranges (perplexity > 0, BLEU in [0, 1])
        - Duplicate metric entries with conflicting values

        Args:
            extraction: PaperExtraction to check.

        Returns:
            List of IMPOSSIBLE_VALUE inconsistencies.
        """
        raise NotImplementedError

    def check_dataset_result_alignment(
        self,
        extraction: PaperExtraction,
    ) -> list[Inconsistency]:
        """Verify that result entries reference datasets present in the datasets list.

        Flags ResultEntry objects whose dataset field does not match any
        Dataset.name in the extraction, suggesting a hallucinated or
        misextracted dataset name.

        Args:
            extraction: PaperExtraction to check.

        Returns:
            List of DATASET_RESULT_MISMATCH inconsistencies.
        """
        raise NotImplementedError

    def check_baseline_completeness(
        self,
        extraction: PaperExtraction,
    ) -> list[Inconsistency]:
        """Check that baselines referenced in results also appear in the baselines list.

        When a ResultEntry.method_name is not_proposed (is_proposed=False),
        it should correspond to a Baseline entry. Flags mismatches.

        Args:
            extraction: PaperExtraction to check.

        Returns:
            List of METRIC_BASELINE_CONFLICT inconsistencies.
        """
        raise NotImplementedError

    def check_duplicate_entries(
        self,
        extraction: PaperExtraction,
    ) -> list[Inconsistency]:
        """Detect duplicate authors, datasets, or methods by name.

        Uses normalised (lowercase, stripped) name comparison. Flags
        exact duplicates with DUPLICATE_ENTRY and near-duplicates with
        a lower severity score.

        Args:
            extraction: PaperExtraction to check.

        Returns:
            List of DUPLICATE_ENTRY inconsistencies.
        """
        raise NotImplementedError

    def _compute_score(self, inconsistencies: list[Inconsistency]) -> float:
        """Compute a [0, 1] consistency score from a list of inconsistencies.

        Score = 1.0 - clamp(sum(i.severity for i in inconsistencies), 0, 1).
        Returns 1.0 for no inconsistencies.

        Args:
            inconsistencies: Detected inconsistency objects.

        Returns:
            Consistency score in [0, 1].
        """
        if not inconsistencies:
            return 1.0
        total = sum(i.severity for i in inconsistencies)
        return max(0.0, 1.0 - min(total, 1.0))
