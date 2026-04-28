"""Section-aware chunker for Marker-parsed research paper documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha1
from pathlib import Path
from typing import Any

from ingestion.pdf_parser import ParsedDocument, ParsedPage, PageBlock


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


SECTION_KEYWORDS: dict[SectionType, list[str]] = {
    SectionType.ABSTRACT: ["abstract"],
    SectionType.INTRODUCTION: ["introduction"],
    SectionType.RELATED_WORK: ["related work", "background", "prior work"],
    SectionType.METHODOLOGY: ["method", "approach", "model", "architecture", "framework"],
    SectionType.EXPERIMENTS: ["experiment", "experimental", "setup", "implementation"],
    SectionType.RESULTS: ["result", "evaluation", "performance", "comparison"],
    SectionType.DISCUSSION: ["discussion", "analysis", "ablation"],
    SectionType.CONCLUSION: ["conclusion", "future work", "summary"],
    SectionType.REFERENCES: ["references", "bibliography"],
    SectionType.APPENDIX: ["appendix"],
}

VISUAL_BLOCK_TYPES = {"Figure", "FigureGroup", "PictureGroup", "TableGroup"}
TEXT_BLOCK_TYPES   = {"Text", "ListGroup", "Caption"}
SKIP_BLOCK_TYPES   = {"PageHeader", "PageFooter", "Footnote"}


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    section: str
    section_type: SectionType
    chunk_type: str          # "text" | "figure" | "table"
    content: str
    page_numbers: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        return self.chunk_type in ("table", "figure")

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class ChunkerConfig:
    min_chunk_words: int = 30   # text chunks below this are dropped


class Chunker:
    """Splits a ParsedDocument into section-aware chunks.

    Strategy:
    - SectionHeader blocks are chunk boundaries
    - Text/ListGroup/Caption blocks accumulate within the current section
    - Figure/FigureGroup/PictureGroup blocks → standalone figure chunks
    - TableGroup blocks → standalone table chunks
    - Each chunk gets a deterministic SHA-1 id
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """Chunk an entire ParsedDocument into ordered Chunk objects."""
        chunks: list[Chunk] = []
        current_section = "preamble"
        current_section_type = SectionType.UNKNOWN
        current_blocks: list[PageBlock] = []
        current_pages: set[int] = set()

        for page in doc.pages:
            for block in page.blocks:
                btype = block.block_type

                if btype in SKIP_BLOCK_TYPES:
                    continue

                if btype == "SectionHeader":
                    # Flush accumulator
                    if current_blocks:
                        chunk = self._make_text_chunk(
                            doc.title or doc.source_path.stem,
                            current_section,
                            current_section_type,
                            current_blocks,
                            sorted(current_pages),
                        )
                        if chunk:
                            chunks.append(chunk)

                    current_section = self._strip_html(block.content)
                    current_section_type = self.classify_section(current_section)
                    current_blocks = []
                    current_pages = set()

                elif btype in VISUAL_BLOCK_TYPES:
                    chunk_type = "table" if btype == "TableGroup" else "figure"
                    chunks.append(self._make_visual_chunk(
                        doc.title or doc.source_path.stem,
                        current_section,
                        current_section_type,
                        block,
                        page.page_number,
                        chunk_type,
                    ))

                elif btype in TEXT_BLOCK_TYPES:
                    current_blocks.append(block)
                    current_pages.add(page.page_number)

        # Flush remaining
        if current_blocks:
            chunk = self._make_text_chunk(
                doc.title or doc.source_path.stem,
                current_section,
                current_section_type,
                current_blocks,
                sorted(current_pages),
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def classify_section(self, heading: str) -> SectionType:
        # Strip leading section numbers: "3.1 Method" → "method"
        lower = re.sub(r'^[\d\.\s]+', '', heading.lower()).strip()
        for section_type, keywords in SECTION_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return section_type
        return SectionType.UNKNOWN

    def estimate_tokens(self, text: str, chars_per_token: float = 4.0) -> int:
        """Rough token estimate without running a tokeniser."""
        return max(1, int(len(text) / chars_per_token))

    def _make_text_chunk(self, doc_title, section, section_type, blocks, page_numbers):
        # Strip HTML from each block before joining
        content = "\n\n".join(
            self._strip_html(b.content) for b in blocks if b.content
        )
        if len(content.split()) < self.config.min_chunk_words:
            return None
        return Chunk(
            chunk_id=self._make_id(content),
            doc_title=doc_title,
            section=section,
            section_type=section_type,
            chunk_type="text",
            content=content,
            page_numbers=page_numbers,
            metadata={"word_count": len(content.split())},
        )

    def _make_visual_chunk(
        self,
        doc_title: str,
        section: str,
        section_type: SectionType,
        block: PageBlock,
        page_number: int,
        chunk_type: str,
    ) -> Chunk:
        content = block.content or ""
        return Chunk(
            chunk_id=self._make_id(content + str(page_number)),
            doc_title=doc_title,
            section=section,
            section_type=section_type,
            chunk_type=chunk_type,
            content=content,
            page_numbers=[page_number],
            metadata={"block_type": block.block_type},
        )

    def _make_id(self, content: str) -> str:
        return sha1(content.encode()).hexdigest()[:16]

    def _strip_html(self, html: str | None) -> str:
        if not html:
            return ""
        return re.sub(r"<[^>]+>", "", html).strip()