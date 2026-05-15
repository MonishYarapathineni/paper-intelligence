"""SGLang VLM client for structured extraction from paper pages."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from extraction.schemas import PartialExtraction, PaperExtraction
import asyncio
import time

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
        
        prompt = self._build_extraction_prompt(target_fields)

        request = ExtractionRequest(
            page_image_path=image_path,
            page_text=page_text,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        return await self.extract_with_retry(request)
        

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
        partials = []

        for i, (image_path, page_text) in enumerate(pages):
            if not page_text.strip() and image_path is None:
                continue  # skip empty pages

            print(f"  Extracting page {i + 1}/{len(pages)}...")
            response = await self.extract_page(image_path, page_text)

            if response.success and response.parsed:
                partials.append(response.parsed)
            else:
                print(f"  Page {i + 1} extraction failed: {response.error}")

        if not partials:
            raise ValueError("No pages successfully extracted")

        return self.merge_extractions(partials, strategy=merge_strategy)

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

        last_exception = None

        for attempt in range(self.config.max_retries):
            start_ms = time.monotonic() * 1000
            try:
                messages = self._build_messages(request)

                body = {
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                }

                # Constrained decoding — forces valid JSON matching schema
                if self.config.use_constrained_decoding:
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "PartialExtraction",
                            "schema": PartialExtraction.model_json_schema(),
                        }
                    }

                response = await self._client.post(
                    "/v1/chat/completions",
                    json=body,
                )

                latency_ms = time.monotonic() * 1000 - start_ms

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                data = response.json()
                raw_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                # Parse into PartialExtraction
                try:
                    parsed = PartialExtraction.model_validate_json(raw_text)
                except Exception as parse_err:
                    # Try stripping markdown fences if constrained decoding failed
                    clean = raw_text.strip().strip("```json").strip("```").strip()
                    try:
                        parsed = PartialExtraction.model_validate_json(clean)
                    except Exception:
                        raise ValueError(f"JSON parse failed: {parse_err}") from parse_err

                return ExtractionResponse(
                    raw_text=raw_text,
                    parsed=parsed,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    latency_ms=latency_ms,
                    model=self.config.model,
                    success=True,
                )

            except Exception as e:
                last_exception = e
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)

        # All retries exhausted
        return ExtractionResponse(
            raw_text="",
            parsed=None,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            model=self.config.model,
            success=False,
            error=str(last_exception),
        )

            
    

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
        if not partials:
            raise ValueError("No partial extractions to merge")

        # Filter out completely empty partials
        non_empty = [
            p for p in partials
            if any([
                p.title, p.problem_statement, p.abstract,
                p.authors, p.datasets, p.methods, p.results,
                p.baselines, p.limitations, p.conclusion_summary
            ])
]
        if not non_empty:
            non_empty = partials  # fall back to all if none have content

        # --- Scalar fields ---
        # 'union': take first non-empty value (title page comes first)
        # 'last_wins': take last non-empty value
        def pick_scalar(values: list[str]) -> str:
            non_empty_vals = [v for v in values if v and v.strip()]
            if not non_empty_vals:
                return ""
            return non_empty_vals[-1] if strategy == "last_wins" else non_empty_vals[0]

        title = pick_scalar([p.title for p in non_empty])
        problem_statement = pick_scalar([p.problem_statement for p in non_empty])
        abstract = pick_scalar([p.abstract or "" for p in non_empty])
        conclusion_summary = pick_scalar([p.conclusion_summary or "" for p in non_empty])
        arxiv_id = pick_scalar([p.arxiv_id or "" for p in non_empty])

        # --- List fields — deduplicate by name ---
        def merge_by_name(lists):
            seen = {}
            for item in (i for sublist in lists for i in sublist):
                key = getattr(item, "name", None) or getattr(item, "description", str(item))
                if key not in seen:
                    seen[key] = item
            return list(seen.values())

        authors = merge_by_name([p.authors for p in non_empty])
        institutions = merge_by_name([p.institutions for p in non_empty])
        datasets = merge_by_name([p.datasets for p in non_empty])
        methods = merge_by_name([p.methods for p in non_empty])
        baselines = merge_by_name([p.baselines for p in non_empty])

        # Results deduplicated by method_name + dataset combination
        seen_results = {}
        for partial in non_empty:
            for r in partial.results:
                key = f"{r.method_name}_{r.dataset}"
                if key not in seen_results:
                    seen_results[key] = r
        results = list(seen_results.values())

        # Limitations deduplicated by description
        seen_limitations = {}
        for partial in non_empty:
            for lim in partial.limitations:
                if lim.description not in seen_limitations:
                    seen_limitations[lim.description] = lim
        limitations = list(seen_limitations.values())

        # --- Confidence scores ---
        # Aggregate extraction_confidence as mean across partials
        confidences = [p.extraction_confidence for p in non_empty if p.extraction_confidence > 0]
        aggregate_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Per-field confidence: max across pages (if any page got it right, we trust it)
        from extraction.schemas import FieldConfidence
        field_confidence = FieldConfidence(
            title=max((p.field_confidence.title for p in non_empty), default=0.0),
            authors=max((p.field_confidence.authors for p in non_empty), default=0.0),
            datasets=max((p.field_confidence.datasets for p in non_empty), default=0.0),
            methods=max((p.field_confidence.methods for p in non_empty), default=0.0),
            results=max((p.field_confidence.results for p in non_empty), default=0.0),
            baselines=max((p.field_confidence.baselines for p in non_empty), default=0.0),
            limitations=max((p.field_confidence.limitations for p in non_empty), default=0.0),
            problem_statement=max((p.field_confidence.problem_statement for p in non_empty), default=0.0),
        )

        return PaperExtraction(
            arxiv_id=arxiv_id or None,
            title=title,
            problem_statement=problem_statement,
            abstract=abstract or None,
            conclusion_summary=conclusion_summary or None,
            authors=authors,
            institutions=institutions,
            datasets=datasets,
            methods=methods,
            results=results,
            baselines=baselines,
            limitations=limitations,
            extraction_confidence=aggregate_confidence,
            field_confidence=field_confidence,
            extraction_metadata={
                "pages_processed": len(partials),
                "pages_with_content": len(non_empty),
                "merge_strategy": strategy,
            }
        )

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
        # System message — sets the extraction persona
        messages = [
            {
                "role": "system",
                "content": request.system_prompt or self.SYSTEM_PROMPT,
            }
        ]

        # User message — text content always present, image optional
        user_content: list[dict[str, Any]] = []

        # Add image first if available — VLMs attend better when image precedes text
        if request.page_image_path and request.page_image_path.exists():
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._encode_image(request.page_image_path),
                },
            })

        # Add the extraction prompt
        user_content.append({
            "type": "text",
            "text": request.user_prompt,
        })

        # Add the page text if present
        if request.page_text:
            user_content.append({
                "type": "text",
                "text": f"Page content:\n\n{request.page_text}",
            })

        messages.append({
            "role": "user",
            "content": user_content,
        })

        return messages

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

        all_fields = {
            "title": "Full paper title exactly as written",
            "authors": "All authors with names, affiliations, emails if present",
            "institutions": "All affiliated institutions with name, department, country",
            "problem_statement": "1-3 sentences: what problem does this paper solve and why it matters",
            "abstract": "Full abstract text verbatim",
            "datasets": "All datasets used — name, description, size, splits",
            "methods": "Key methods/architectures proposed — name, description, novelty",
            "results": "Quantitative results — method name, metrics, dataset, conditions",
            "baselines": "Baseline systems compared against — name, metrics, notes",
            "limitations": "Limitations stated or inferable — description, type",
            "conclusion_summary": "1-2 sentence summary of conclusions",
        }

        fields_to_extract = (
            {k: v for k, v in all_fields.items() if k in target_fields}
            if target_fields
            else all_fields
        )

        field_instructions = "\n".join(
            f"- {field}: {description}"
            for field, description in fields_to_extract.items()
        )

        return (
            f"Extract the following fields from this research paper page:\n\n"
            f"{field_instructions}\n\n"
            f"Rules:\n"
            f"- Only extract information explicitly present in the provided content\n"
            f"- Do not infer, hallucinate, or fill gaps with prior knowledge\n"
            f"- For fields not present on this page, use empty string or empty list\n"
            f"- Extract confidence scores per field: 1.0 if clearly present, "
            f"0.5 if partially present, 0.0 if absent\n"
            f"- Return valid JSON matching the PartialExtraction schema exactly"
            f"- Set extraction_confidence to a float 0.0-1.0 reflecting overall extraction quality\n"
            f"- Set field_confidence.title, field_confidence.authors, etc. individually\n"
        )

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
