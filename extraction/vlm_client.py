"""SGLang VLM client for structured extraction from paper pages."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from extraction.schemas import PartialExtraction, PaperExtraction


@dataclass
class SGLangConfig:
    base_url: str = "http://localhost:30000"
    model: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_tokens: int = 4096
    temperature: float = 0.1          # low temp for deterministic extraction
    timeout_seconds: float = 120.0
    max_retries: int = 3
    # JSON schema constrained decoding — requires SGLang >= 0.3
    use_constrained_decoding: bool = True


@dataclass
class ExtractionRequest:
    page_image_path: Path | None = None
    page_text: str | None = None        # Markdown from PDFParser
    system_prompt: str = ""
    user_prompt: str = ""
    schema_name: str = "PaperExtraction"


@dataclass
class ExtractionResponse:
    raw_text: str
    parsed: PartialExtraction | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    success: bool
    error: str | None = None


class VLMClient:
    """Client for structured extraction via an SGLang-served vision language model.

    Sends page images and/or Markdown text to SGLang's OpenAI-compatible
    /v1/chat/completions endpoint and decodes the response against a Pydantic
    schema. When use_constrained_decoding is enabled, the JSON schema is
    forwarded to SGLang's guided_json parameter so the model is constrained to
    valid schema output.
    """

    SYSTEM_PROMPT = (
        "You are a precise scientific paper information extractor. "
        "Extract structured information from the provided paper content exactly "
        "as stated by the authors. Do not infer or hallucinate facts. "
        "Return only valid JSON matching the provided schema."
    )

    def __init__(self, config: SGLangConfig | None = None) -> None:
        self.config = config or SGLangConfig()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    async def extract_page(
        self,
        image_path: Path | None,
        page_text: str,
        target_fields: list[str] | None = None,
    ) -> ExtractionResponse:
        """Extract structured fields from a single paper page.

        Sends the page image (if available) and Markdown text together as a
        multimodal message. The target_fields parameter narrows the extraction
        prompt to specific fields, reducing hallucination risk for pages where
        only certain fields are relevant (e.g. the results table).

        Args:
            image_path: Optional path to the page PNG for visual grounding.
            page_text: Markdown text of the page from PDFParser.
            target_fields: Subset of PaperExtraction field names to focus on.
                           None means extract all fields.

        Returns:
            ExtractionResponse with parsed PartialExtraction and token counts.
        """
        raise NotImplementedError

    async def extract_document(
        self,
        pages: list[tuple[Path | None, str]],
        merge_strategy: str = "union",
    ) -> PaperExtraction:
        """Run page-level extraction across a full document and merge results.

        Extracts each page independently (or in parallel with asyncio.gather)
        and merges the per-page PartialExtraction objects into one final
        PaperExtraction. The merge_strategy controls how conflicts are resolved:
        - 'union': combine lists, keep first non-empty scalar
        - 'last_wins': later pages overwrite earlier values

        Args:
            pages: List of (image_path, markdown_text) tuples, one per page.
            merge_strategy: How to resolve field conflicts across pages.

        Returns:
            Fully merged PaperExtraction.
        """
        raise NotImplementedError

    async def extract_with_retry(
        self,
        request: ExtractionRequest,
    ) -> ExtractionResponse:
        """Wrap a single extraction request with exponential-backoff retries.

        Retries on HTTP 5xx errors and JSON parse failures up to
        config.max_retries times. Raises the last exception if all retries
        are exhausted.

        Args:
            request: Fully populated ExtractionRequest.

        Returns:
            ExtractionResponse from the first successful attempt.
        """
        raise NotImplementedError

    def merge_extractions(
        self,
        partials: list[PartialExtraction],
        strategy: str = "union",
    ) -> PaperExtraction:
        """Merge multiple PartialExtraction objects into one PaperExtraction.

        List fields (authors, datasets, methods, etc.) are deduplicated by
        name. Scalar fields (title, problem_statement) use the strategy to
        pick the winning value. Sets extraction_confidence to the mean of
        per-partial confidence scores.

        Args:
            partials: Ordered list of per-page extraction results.
            strategy: 'union' or 'last_wins'.

        Returns:
            A valid PaperExtraction (raises ValidationError if required fields
            could not be populated from any page).
        """
        raise NotImplementedError

    def _build_messages(
        self,
        request: ExtractionRequest,
    ) -> list[dict[str, Any]]:
        """Build the messages array for the chat completions API call.

        Encodes the image as a base64 data URI if image_path is provided.
        Appends the page_text and a JSON schema instruction to the user turn.

        Args:
            request: Extraction request with prompt and optional image.

        Returns:
            List of message dicts ready for the /v1/chat/completions body.
        """
        raise NotImplementedError

    def _build_extraction_prompt(
        self,
        target_fields: list[str] | None,
    ) -> str:
        """Construct the user-facing extraction instruction.

        When target_fields is provided, lists only those fields in the prompt
        to focus the model. Appends the JSON schema description so the model
        knows the expected output shape.

        Args:
            target_fields: Optional list of field names to extract.

        Returns:
            Formatted prompt string.
        """
        raise NotImplementedError

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """Read an image file and return it as a base64-encoded data URI.

        Args:
            image_path: Path to a PNG or JPEG file.

        Returns:
            String of the form 'data:image/png;base64,...'.
        """
        suffix = image_path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg") else "png"
        data = base64.b64encode(image_path.read_bytes()).decode()
        return f"data:image/{mime};base64,{data}"

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "VLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
