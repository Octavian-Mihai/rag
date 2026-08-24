from __future__ import annotations

import re

from rag.config import load_config
from rag.models import RetrievedChunk

CITE_RE = re.compile(r"\[([^:\]]+):(\d+)\]")


def citation_keys(chunks: list[RetrievedChunk]) -> set[tuple[str, int]]:
    return {(c.doc_id, c.page) for c in chunks}


def extract_citations(answer: str) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in CITE_RE.finditer(answer)]


def citations_grounded(answer: str, chunks: list[RetrievedChunk]) -> bool:
    allowed = citation_keys(chunks)
    found = extract_citations(answer)
    # Missing citations: still answer. Invented [doc_id:page] tags: refuse.
    if not found:
        return True
    return all(cite in allowed for cite in found)


def max_score(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    return max(c.score for c in chunks)


def should_abstain_low_similarity(chunks: list[RetrievedChunk]) -> bool:
    threshold = float(load_config()["guards"]["min_similarity"])
    return max_score(chunks) < threshold
