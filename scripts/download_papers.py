import urllib.request
import time
from pathlib import Path

papers = [
    # Clean structured NLP paper
    ("2005.14165", "attention_is_all_you_need.pdf"),
    # Figure-heavy vision paper
    ("2010.11929", "vit_image_worth_16x16.pdf"),
    # Math-heavy theory paper
    ("1706.03762", "transformer_original.pdf"),
    # Systems paper with architecture diagrams
    ("2304.11490", "llama2.pdf"),
    # Table-heavy benchmark paper
    ("2108.07258", "codex.pdf"),
]

output_dir = Path("data/pdfs")
output_dir.mkdir(parents=True, exist_ok=True)

for arxiv_id, filename in papers:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out_path = output_dir / filename
    if out_path.exists():
        print(f"Already exists: {filename}")
        continue
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, out_path)
    time.sleep(2)  # be polite to arXiv
    print(f"Saved: {out_path}")

print("Done.")
