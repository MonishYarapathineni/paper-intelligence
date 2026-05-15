"""Serialize parsed chunks to JSON for cross-session use."""
import json
from pathlib import Path
from ingestion.chunker import Chunk

def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Save chunks to JSON file for loading in extraction session."""
    data = [
        {
            "chunk_id": c.chunk_id,
            "doc_title": c.doc_title,
            "section": c.section,
            "section_type": c.section_type.value,
            "chunk_type": c.chunk_type,
            "content": c.content,
            "page_numbers": c.page_numbers,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(chunks)} chunks to {output_path}")

def load_chunks(input_path: Path) -> list[dict]:
    """Load chunks from JSON file in extraction session."""
    return json.loads(input_path.read_text())