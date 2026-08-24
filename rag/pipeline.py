from __future__ import annotations

import time

from rag.citations import citations_grounded
from rag.config import load_config
from rag.debug_log import debug_log
from rag.generate import GenerationError, generate_answer
from rag.guards import pre_generation_guard, ungrounded_message
from rag.metrics import METRICS
from rag.models import QueryResult
from rag.rerank import rerank
from rag.retrievers import retrieve_ensemble, retrieve_graph, retrieve_hybrid, retrieve_sql
from rag.router import route_query


def _retrieve(intent: str, question: str):
    if intent == "comparative":
        return retrieve_graph(question)
    if intent == "aggregate":
        return retrieve_sql(question)
    if intent == "ensemble":
        return retrieve_ensemble(question)
    return retrieve_hybrid(question)


def answer_question(question: str) -> QueryResult:
    started = time.perf_counter()
    # #region agent log
    debug_log("pipeline.py:answer_question", "query_start", {"question_len": len(question)}, hypothesis_id="H3")
    # #endregion
    decision = route_query(question)
    # #region agent log
    debug_log(
        "pipeline.py:answer_question",
        "route_decided",
        {"intent": decision.intent, "confidence": decision.confidence, "ensemble": decision.use_ensemble},
        hypothesis_id="H4",
    )
    # #endregion
    chunks = _retrieve(decision.intent, question)
    top_k = int(load_config()["retrieval"]["rerank_top_k"])
    ranked = rerank(question, chunks, top_k=top_k)
    # #region agent log
    debug_log(
        "pipeline.py:answer_question",
        "retrieval_done",
        {"chunk_count": len(ranked), "top_score": ranked[0].score if ranked else 0.0},
        hypothesis_id="H2",
    )
    # #endregion

    abstain = pre_generation_guard(ranked)
    if abstain:
        result = QueryResult(
            answer=abstain,
            route=decision.intent,
            confidence=decision.confidence,
            chunks=ranked,
            latency_ms=(time.perf_counter() - started) * 1000,
            abstained=True,
            reason="low_similarity",
        )
        METRICS.record(result.route, result.latency_ms, True)
        # #region agent log
        debug_log("pipeline.py:answer_question", "abstain_low_similarity", {"top_score": ranked[0].score if ranked else 0.0}, hypothesis_id="H4")
        # #endregion
        return result

    try:
        raw = generate_answer(question, ranked)
    except GenerationError as exc:
        result = QueryResult(
            answer=str(exc),
            route=decision.intent,
            confidence=decision.confidence,
            chunks=ranked,
            latency_ms=(time.perf_counter() - started) * 1000,
            abstained=True,
            reason="generation_error",
        )
        METRICS.record(result.route, result.latency_ms, True)
        # #region agent log
        debug_log("pipeline.py:answer_question", "generation_error", {"error": str(exc)}, hypothesis_id="H1")
        # #endregion
        return result
    if not citations_grounded(raw, ranked):
        result = QueryResult(
            answer=ungrounded_message(),
            route=decision.intent,
            confidence=decision.confidence,
            chunks=ranked,
            latency_ms=(time.perf_counter() - started) * 1000,
            abstained=True,
            reason="ungrounded_citation",
        )
        METRICS.record(result.route, result.latency_ms, True)
        return result

    result = QueryResult(
        answer=raw,
        route=decision.intent,
        confidence=decision.confidence,
        chunks=ranked,
        latency_ms=(time.perf_counter() - started) * 1000,
        abstained=False,
        reason=decision.reason,
    )
    METRICS.record(result.route, result.latency_ms, False)
    return result
