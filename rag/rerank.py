from __future__ import annotations

import math
from functools import lru_cache

from rag.config import load_config
from rag.models import RetrievedChunk


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder

    cfg = load_config()
    return CrossEncoder(cfg["models"]["reranker"])


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    model = _reranker()
    pairs = [(query, c.parent_text or c.text) for c in chunks]
    scores = model.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk.score = _sigmoid(float(score))
    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
