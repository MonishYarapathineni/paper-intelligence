"""Simple routing based on Marker block types — no heuristics needed."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ingestion.pdf_parser import ParsedPage


class ProcessingRoute(str, Enum):
    TEXT_ONLY = "text_only"
    VLM_REQUIRED = "vlm"
    SKIP = "skip"

STRUCTURAL_SECTION_HEADERS = {
    "references", "bibliography", "acknowledgments",
    "acknowledgements", "funding", "author contributions",
}

VLM_BLOCK_TYPES = {"Picture", "Table", "Formula"}

SKIP_BLOCK_TYPES = {"Page-header", "Page-footer", "Footnote"}


@dataclass
class RoutingDecision:
    page_number: int
    route: ProcessingRoute
    confidence: float
    reasons: list[str]


class PageRouter:
    """Routes pages based on Marker block type composition."""

    def route_page(self, page: ParsedPage) -> RoutingDecision:
        reasons = []

        # Check for structural-only pages — all blocks are headers/footers
        block_types = {b.block_type for b in page.blocks}

        if not page.blocks or block_types.issubset({"PageHeader", "PageFooter"}):
            return RoutingDecision(
                page_number=page.page_number,
                route=ProcessingRoute.SKIP,
                confidence=0.95,
                reasons=["no content blocks after filtering headers/footers"],
            )

        # Check for structural section pages (references, acknowledgements)
        for block in page.blocks:
            if block.block_type == "SectionHeader":
                header_text = block.content.lower().strip()
                if any(kw in header_text for kw in STRUCTURAL_SECTION_HEADERS):
                    # Only skip if this is the dominant content
                    non_text = [
                        b for b in page.blocks
                        if b.block_type in VLM_BLOCK_TYPES
                    ]
                    if not non_text:
                        reasons.append(f"structural section: {header_text}")
                        return RoutingDecision(
                            page_number=page.page_number,
                            route=ProcessingRoute.SKIP,
                            confidence=0.85,
                            reasons=reasons,
                        )

        # VLM routing based on block types — no heuristics needed
        vlm_signals = 0

        if page.has_figures:
            vlm_signals += 1
            reasons.append("Figure/FigureGroup/PictureGroup block detected")

        if page.has_tables:
            vlm_signals += 1
            reasons.append("TableGroup block detected")

        if page.has_equations:
            vlm_signals += 1
            reasons.append("Equation block detected")

        if vlm_signals > 0:
            confidence = min(0.7 + (vlm_signals * 0.1), 0.95)
            return RoutingDecision(
                page_number=page.page_number,
                route=ProcessingRoute.VLM_REQUIRED,
                confidence=confidence,
                reasons=reasons,
            )

        reasons.append("text-only blocks")
        return RoutingDecision(
            page_number=page.page_number,
            route=ProcessingRoute.TEXT_ONLY,
            confidence=0.9,
            reasons=reasons,
        )

    def route_document(self, pages: list[ParsedPage]) -> list[RoutingDecision]:
        decisions = [self.route_page(p) for p in pages]

        # Context promotion — same logic as before, still valid
        for i, decision in enumerate(decisions):
            if decision.route == ProcessingRoute.TEXT_ONLY:
                neighbors = []
                if i > 0:
                    neighbors.append(decisions[i - 1])
                if i < len(decisions) - 1:
                    neighbors.append(decisions[i + 1])
                if any(n.route == ProcessingRoute.VLM_REQUIRED for n in neighbors):
                    decisions[i] = RoutingDecision(
                        page_number=decision.page_number,
                        route=ProcessingRoute.VLM_REQUIRED,
                        confidence=0.65,
                        reasons=decision.reasons + ["promoted: adjacent to VLM page"],
                    )

        return decisions

    def split_by_route(self, pages, decisions):
        text_pages, vlm_pages = [], []
        decision_map = {d.page_number: d for d in decisions}

        for page in pages:
            route = decision_map[page.page_number].route
            if route == ProcessingRoute.TEXT_ONLY:
                text_pages.append(page)
            elif route == ProcessingRoute.VLM_REQUIRED:
                vlm_pages.append(page)

        return text_pages, vlm_pages