"""PDF parsing with Marker, producing typed block structures per page."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PageBlock:
    block_type: str        # Caption, Footnote, Formula, Picture,
                           # Section-header, Table, Text, Title
    content: str           # text content or image path
    page_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPage:
    page_number: int
    blocks: list[PageBlock]
    has_figures: bool
    has_tables: bool
    has_equations: bool
    word_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def markdown(self) -> str:
        """Reconstruct plain markdown from text blocks for downstream use."""
        return "\n\n".join(
            b.content for b in self.blocks
            if b.block_type in ("Text", "Title", "Section-header", "Caption")
        )


@dataclass
class ParsedDocument:
    source_path: Path
    pages: list[ParsedPage]
    title: str | None
    total_pages: int
    doc_metadata: dict[str, Any] = field(default_factory=dict)


class PDFParser:
    """Converts PDF files to structured block representations using Marker.

    Marker uses LayoutLMv3 for block classification, giving us typed blocks
    (Table, Picture, Formula, Text, etc.) without heuristic signal detection.
    """

    def __init__(self) -> None:
        pass


    def parse(self, pdf_path: Path) -> ParsedDocument:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        config_parser = ConfigParser({"output_format": "json"})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        rendered = converter(str(pdf_path))

        VLM_BLOCK_TYPES = {
            "Figure", "FigureGroup", "PictureGroup", "TableGroup", "Equation"
        }
        SKIP_BLOCK_TYPES = {"PageHeader", "PageFooter", "Footnote"}
        TEXT_BLOCK_TYPES = {
            "Text", "SectionHeader", "Caption", "ListGroup"
        }

        pages = []
        for page_output in rendered.children:
            # Parse 0-indexed page number from id: "/page/0/Page/240"
            page_num = int(page_output.id.split("/")[2])

            blocks = []
            has_figures = False
            has_tables = False
            has_equations = False
            word_count = 0

            for block in (page_output.children or []):
                btype = block.block_type

                if btype in SKIP_BLOCK_TYPES:
                    continue

                if btype in VLM_BLOCK_TYPES:
                    if btype in ("Figure", "FigureGroup", "PictureGroup"):
                        has_figures = True
                    elif btype == "TableGroup":
                        has_tables = True
                    elif btype == "Equation":
                        has_equations = True

                content = block.html or ""
                word_count += len(content.split())

                blocks.append(PageBlock(
                    block_type=btype,
                    content=content,
                    page_number=page_num,
                    metadata={
                        "bbox": page_output.bbox,
                        "section_hierarchy": block.section_hierarchy,
                    }
                ))

            pages.append(ParsedPage(
                page_number=page_num,
                blocks=blocks,
                has_figures=has_figures,
                has_tables=has_tables,
                has_equations=has_equations,
                word_count=word_count,
                metadata={
                    "page_stats": rendered.metadata.get(
                        "page_stats", {}
                    )
                }
            ))

        doc_metadata = self.extract_metadata(pdf_path)

        return ParsedDocument(
            source_path=pdf_path,
            pages=pages,
            title=doc_metadata.get("title") or pdf_path.stem,
            total_pages=len(pages),
            doc_metadata=doc_metadata,
        )

    def extract_metadata(self, pdf_path: Path) -> dict:
        import fitz

        doc = fitz.open(str(pdf_path))
        meta = doc.metadata
        page_count = doc.page_count
        doc.close()

        return {
            "title": meta.get("title"),
            "author": meta.get("author"),
            "subject": meta.get("subject"),
            "keywords": meta.get("keywords"),
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "creation_date": meta.get("creationDate"),
            "modification_date": meta.get("modDate"),
            "page_count": page_count,
        }

   
