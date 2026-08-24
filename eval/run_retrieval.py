"""Hit Rate@5 against eval/golden.jsonl (requires running API ingest first)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.pipeline import answer_question  # noqa: E402


def load_golden(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def hit_at_k(text_blob: str, needles: list[str]) -> bool:
    blob = text_blob.lower().replace(",", "")
    return all(n.lower().replace(",", "") in blob for n in needles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    args = parser.parse_args()
    rows = load_golden(Path(args.golden))
    hits = 0
    for row in rows:
        result = answer_question(row["query"])
        blob = " ".join((c.parent_text or c.text) for c in result.chunks[:5])
        ok = hit_at_k(blob, row["relevant_contains"])
        hits += int(ok)
        status = "HIT" if ok else "MISS"
        print(f"{status}\t{row['intent']}\t{row['query']}")
    print(f"Hit Rate@5: {hits / max(len(rows), 1):.3f} ({hits}/{len(rows)})")


if __name__ == "__main__":
    main()
