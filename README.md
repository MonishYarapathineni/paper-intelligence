# Paper Intelligence

Multimodal document intelligence pipeline for research papers.

## What this is
<!-- fill in later -->

## Architecture
<!-- fill in later -->

## Tech stack
<!-- fill in later -->

## Design decisions
<!-- running log — add an entry every time a real decision is made -->

### [date] PDF page routing strategy
- Per-page complexity router, not whole-document
- Signals: word count, image area ratio, find_tables(), equation density
- TEXT_ONLY / VLM_REQUIRED / SKIP routes
- Context promotion pass: adjacent pages to VLM pages get promoted

### [date] Chunking strategy
- Section-aware chunking on header boundaries
- Figures and tables get their own chunks tagged by type
- Semantic chunking only as fallback for poorly structured papers

### [date] Retrieval strategy
- Both VectorRAG and GraphRAG implemented and tracked on MLflow
- Comparison is the deliverable, not picking a winner upfront

### [date] Schema evolution
- Schema registry in JSON — canonical node/edge types with definitions
- LLM maps new fields to existing schema with confidence score
- Below threshold → HITL escalation, not auto-create



### [4/27] Figure detection threshold — drawing_count > 50
- pymupdf4llm omits embedded images, leaving "intentionally omitted" markers
- vector drawings count varies wildly: 333 for a real figure (ViT page 2),
  4 for table formatting (ViT page 4), 2-3 for title page borders
- Three independent signals: raster count, omitted markers, drawing count > 50
- Threshold derived empirically from actual test papers, not assumed

### [4/28] Switched from pymupdf4llm + heuristic router to Marker
- pymupdf4llm required regex, drawing counts, equation density, 
  keyword matching to route pages — fragile and hard to tune
- Marker (LayoutLMv3 + Surya) classifies every block by type:
  Figure, TableGroup, Equation, SectionHeader, Text, etc.
- Routing becomes a block type lookup — no heuristics
- Fixed 2 false positive SKIPs that heuristic router got wrong
- Tradeoff: Marker is heavier (downloads ~3GB of models), 
  slower per document, requires GPU for reasonable speed
- For this project that tradeoff is worth it — accuracy over speed


### [4/28] Section classification — keyword matching + parent inheritance
- Numbered headings stripped before keyword matching: "3.1 Method" → "method"  
- Subsections that don't match any keyword inherit parent section type
- Appendix subsections inherit from last known classified section
- Known edge case: h1 title page hits "transformer" keyword → classified as methodology
- Good enough for retrieval — section type is metadata, not primary search signal


## Eval framework
<!-- fill in later -->

## Benchmark results
<!-- fill in later — real numbers only -->

## Setup
<!-- fill in later -->

