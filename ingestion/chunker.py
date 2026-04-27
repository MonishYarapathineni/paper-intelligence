"""Section-aware chunker that splits parsed documents into LLM-ready chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Sequence

from ingestion.pdf_parser import ParsedPage


class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    UNKNOWN = "unknown"


@dataclass
class Chunk:
    chunk_id: str               # deterministic hash of (source, section, index)
    source_path: str
    section_type: SectionType
    section_heading: str
    text: str
    page_numbers: list[int]
    token_estimate: int
    chunk_index: int            # position within the section
    total_chunks_in_section: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkerConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64
    min_chunk_tokens: int = 50
    # avg chars per token, used for fast estimation without a full tokeniser
    chars_per_token: float = 4.0
    respect_paragraph_boundaries: bool = True


class SectionAwareChunker:
    """Splits a document's pages into semantically coherent, overlap-aware chunks.

    Uses Markdown heading signals emitted by pymupdf4llm to detect section
    boundaries and classify each section. Within each section, text is split
    into fixed-size windows with configurable overlap so that retrieval can
    find content spanning a chunk boundary.
    """

    # Regex patterns for Markdown headings produced by pymupdf4llm
    HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    SECTION_KEYWORDS: dict[SectionType, list[str]] = {
        SectionType.ABSTRACT: ["abstract"],
        SectionType.INTRODUCTION: ["introduction", "1 introduction"],
        SectionType.RELATED_WORK: ["related work", "background", "prior work"],
        SectionType.METHODOLOGY: ["method", "approach", "model", "architecture", "framework"],
        SectionType.EXPERIMENTS: ["experiment", "experimental", "setup", "implementation"],
        SectionType.RESULTS: ["result", "evaluation", "performance", "comparison"],
        SectionType.DISCUSSION: ["discussion", "analysis", "ablation"],
        SectionType.CONCLUSION: ["conclusion", "future work", "summary"],
        SectionType.REFERENCES: ["references", "bibliography"],
        SectionType.APPENDIX: ["appendix"],
    }

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(
        self,
        pages: Sequence[ParsedPage],
        source_path: str | Path,
    ) -> list[Chunk]:
        """Convert a full document's pages into an ordered list of Chunks.

        1. Concatenates page Markdown in order, tagging each line with its
           source page number so page provenance is preserved in each Chunk.
        2. Splits the concatenated text on Markdown headings to extract
           labelled sections.
        3. Calls chunk_section for each section.
        4. Returns all chunks in document order.

        Args:
            pages: Ordered ParsedPage list (skip-routed pages should be excluded).
            source_path: Original PDF path, stored on each Chunk for retrieval.

        Returns:
            Flat list of Chunk objects covering the entire document.
        """
        raise NotImplementedError

    def chunk_section(
        self,
        section_text: str,
        section_type: SectionType,
        section_heading: str,
        page_numbers: list[int],
        source_path: str,
    ) -> list[Chunk]:
        """Split a single section's text into overlapping token-bounded chunks.

        Uses a sliding-window approach over paragraphs (if
        respect_paragraph_boundaries is set) or over raw tokens. Overlap
        is prepended from the end of the previous chunk so the retriever
        can reconstruct context across boundaries.

        Args:
            section_text: Raw Markdown text of the section.
            section_type: Classified SectionType enum value.
            section_heading: Heading string to store on each Chunk.
            page_numbers: Pages that this section spans.
            source_path: Source PDF path string.

        Returns:
            List of Chunk objects for this section.
        """
        raise NotImplementedError

    def iter_sections(
        self,
        full_text: str,
    ) -> Iterator[tuple[str, str, int, int]]:
        """Yield (heading, section_text, start_char, end_char) tuples.

        Splits on Markdown headings (# through ####) detected by HEADING_RE.
        The text between consecutive headings is treated as one section.

        Args:
            full_text: Full concatenated Markdown string for the document.

        Yields:
            Tuples of (heading_text, body_text, start_char_offset, end_char_offset).
        """
        raise NotImplementedError

    def classify_section(self, heading: str) -> SectionType:
        """Map a section heading string to the closest SectionType enum value.

        Lowercases the heading and checks for each SectionType's keyword list
        in priority order. Falls back to SectionType.UNKNOWN.

        Args:
            heading: Raw heading text (without # prefix).

        Returns:
            Best-matching SectionType.
        """
        lower = heading.lower()
        for section_type, keywords in self.SECTION_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return section_type
        return SectionType.UNKNOWN

    def estimate_tokens(self, text: str) -> int:
        """Estimate the token count for a string without running a tokeniser.

        Uses a configurable chars_per_token ratio. Sufficient for chunking
        heuristics; not a replacement for tiktoken in downstream model calls.

        Args:
            text: Input string.

        Returns:
            Estimated token count as an integer.
        """
        return max(1, int(len(text) / self.config.chars_per_token))

    def _make_chunk_id(self, source_path: str, section_heading: str, index: int) -> str:
        """Generate a deterministic, URL-safe chunk ID.

        Combines source_path stem, normalised section heading, and chunk
        index, then hex-digests with SHA-1 to produce a stable 12-char ID.

        Args:
            source_path: Source PDF path string.
            section_heading: Section heading text.
            index: Zero-based chunk index within the section.

        Returns:
            12-character hex string.
        """
        raise NotImplementedError
