from __future__ import annotations


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    return float(any(item in relevant_ids for item in retrieved_ids[:k]))


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def context_relevancy_stub(question: str, contexts: list[str]) -> float:
    """Phase 2: replace with RAGAS context relevancy."""
    raise NotImplementedError("RAGAS context relevancy is a phase-2 metric")


def faithfulness_stub(answer: str, contexts: list[str]) -> float:
    """Phase 2: replace with RAGAS faithfulness."""
    raise NotImplementedError("RAGAS faithfulness is a phase-2 metric")
