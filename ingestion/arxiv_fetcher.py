"""Fetches papers from the arXiv API and downloads their PDFs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    pdf_url: str
    published: str
    updated: str
    local_pdf_path: Path | None = None


@dataclass
class FetchConfig:
    download_dir: Path = Path("./data/pdfs")
    max_results: int = 10
    timeout_seconds: float = 30.0


class ArxivFetcher:
    """Client for the arXiv API that fetches metadata and downloads PDFs."""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, config: FetchConfig | None = None) -> None:
        self.config = config or FetchConfig()
        self.config.download_dir.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, max_results: int | None = None) -> list[ArxivPaper]:
        """Search arXiv for papers matching a free-text query.

        Uses the arXiv Atom feed API. Parses title, authors, abstract,
        categories, and the PDF download URL from the response.

        Args:
            query: Free-text search string (e.g. "retrieval augmented generation").
            max_results: Override the config-level limit for this call.

        Returns:
            List of ArxivPaper dataclasses ordered by relevance.
        """
        raise NotImplementedError

    def fetch_by_id(self, arxiv_id: str) -> ArxivPaper:
        """Fetch a single paper by its arXiv identifier (e.g. '2304.01373').

        Handles both versioned IDs (2304.01373v2) and bare IDs. Raises
        ValueError if the paper is not found.

        Args:
            arxiv_id: The arXiv paper identifier.

        Returns:
            Populated ArxivPaper dataclass.
        """
        raise NotImplementedError

    def fetch_by_ids(self, arxiv_ids: list[str]) -> list[ArxivPaper]:
        """Batch-fetch metadata for multiple arXiv IDs in a single API call.

        Chunks requests to stay within the arXiv API's recommended batch size
        of 100 IDs per request.

        Args:
            arxiv_ids: List of arXiv identifiers.

        Returns:
            List of ArxivPaper dataclasses in the same order as input IDs.
        """
        raise NotImplementedError

    def download_pdf(self, paper: ArxivPaper) -> Path:
        """Download the PDF for a paper and save it to the configured directory.

        Skips the download if a file with the same name already exists
        (idempotent). Sets paper.local_pdf_path on success.

        Args:
            paper: ArxivPaper with a valid pdf_url.

        Returns:
            Path to the saved PDF file.
        """
        raise NotImplementedError

    def stream_category(
        self,
        category: str,
        start_date: str,
        end_date: str,
    ) -> Iterator[ArxivPaper]:
        """Yield papers from a category submitted within a date range.

        Uses the arXiv OAI-PMH or search API to paginate through results.
        Useful for bulk ingestion pipelines. Dates should be in ISO-8601
        format (YYYY-MM-DD).

        Args:
            category: arXiv category string (e.g. "cs.CL", "cs.LG").
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Yields:
            ArxivPaper instances one by one without buffering all in memory.
        """
        raise NotImplementedError

    def _parse_atom_entry(self, entry: dict) -> ArxivPaper:
        """Parse a single Atom feed entry dict into an ArxivPaper.

        Extracts the clean arXiv ID from the entry id URL, normalises the
        PDF link, and flattens the author list.

        Args:
            entry: Raw dict parsed from the Atom feed.

        Returns:
            ArxivPaper dataclass.
        """
        raise NotImplementedError

    def _clean_arxiv_id(self, raw_id: str) -> str:
        """Strip URL prefix and version suffix from a raw arXiv ID string.

        Examples:
            'http://arxiv.org/abs/2304.01373v2' -> '2304.01373'

        Args:
            raw_id: Raw identifier string from the Atom feed.

        Returns:
            Clean numeric arXiv ID.
        """
        match = re.search(r"(\d{4}\.\d{4,5})", raw_id)
        if not match:
            raise ValueError(f"Cannot parse arXiv ID from: {raw_id}")
        return match.group(1)
