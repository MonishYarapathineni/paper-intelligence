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



### [today] Figure detection threshold — drawing_count > 50
- pymupdf4llm omits embedded images, leaving "intentionally omitted" markers
- vector drawings count varies wildly: 333 for a real figure (ViT page 2),
  4 for table formatting (ViT page 4), 2-3 for title page borders
- Three independent signals: raster count, omitted markers, drawing count > 50
- Threshold derived empirically from actual test papers, not assumed

## Eval framework
<!-- fill in later -->

## Benchmark results
<!-- fill in later — real numbers only -->

## Setup
<!-- fill in later -->

