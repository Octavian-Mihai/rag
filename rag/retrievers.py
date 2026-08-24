from __future__ import annotations

import sqlite3

from rag.config import load_config
from rag.embedder import embed_query
from rag.graph import GRAPH
from rag.models import RetrievedChunk
from rag.sparse import BM25, tokenize
from rag.store import connect, get_parent
from rag.vectorstore import hybrid_search


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        key = chunk.parent_id or chunk.chunk_id
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def retrieve_hybrid(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    cfg = load_config()
    limit = top_k or int(cfg["retrieval"]["hybrid_top_k"])
    dense = embed_query(query)
    idx, vals = BM25.encode(query)
    return hybrid_search(dense, idx, vals, limit)


def retrieve_graph(query: str) -> list[RetrievedChunk]:
    hops = int(load_config()["retrieval"]["graph_hops"])
    seeds = retrieve_hybrid(query, top_k=5)
    GRAPH.rebuild()
    extra: list[RetrievedChunk] = []
    for seed in seeds:
        neighbors = GRAPH.neighbors(seed.parent_id, hops=hops)
        extra.extend(GRAPH.as_retrieved(neighbors, score=max(0.55, seed.score * 0.9)))
    return _dedupe(seeds + extra)


def _aggregate_summary() -> RetrievedChunk | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(amount_usd), 0) AS total_amount,
                   AVG(term_months) AS avg_term
            FROM clauses
            """
        ).fetchone()
        sample = conn.execute(
            "SELECT doc_id, page, heading FROM clauses LIMIT 1"
        ).fetchone()
    if not row or not row["n"]:
        return None
    text = (
        f"Structured aggregate over extracted clauses: count={row['n']}, "
        f"total_amount_usd={row['total_amount']}, avg_term_months={row['avg_term']}."
    )
    doc_id = sample["doc_id"] if sample else "corpus"
    page = int(sample["page"] or 1) if sample else 1
    return RetrievedChunk(
        chunk_id="sql-aggregate",
        doc_id=doc_id,
        parent_id="sql-aggregate",
        page=page,
        text=text,
        heading="Aggregate",
        score=0.85,
        parent_text=text,
        source="sql",
    )


def retrieve_sql(query: str) -> list[RetrievedChunk]:
    tokens = [t for t in tokenize(query) if len(t) > 2]
    like_clauses = " OR ".join(["heading LIKE ? OR party LIKE ? OR date_iso LIKE ?"] * max(len(tokens), 1))
    params: list[str] = []
    if tokens:
        for tok in tokens:
            wildcard = f"%{tok}%"
            params.extend([wildcard, wildcard, wildcard])
    else:
        like_clauses = "1=1"

    sql = f"""
        SELECT c.doc_id, c.page, c.heading, c.amount_usd, c.term_months, c.party, c.date_iso,
               ch.chunk_id, ch.parent_id, ch.text
        FROM clauses c
        LEFT JOIN chunks ch ON ch.doc_id = c.doc_id AND ch.is_parent = 1 AND ch.heading = c.heading
        WHERE {like_clauses}
        ORDER BY c.amount_usd DESC NULLS LAST
        LIMIT 20
    """
    # SQLite has no NULLS LAST; emulate
    sql = sql.replace("DESC NULLS LAST", "DESC")
    rows: list[sqlite3.Row]
    with connect() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            rows = []

    chunks: list[RetrievedChunk] = []
    for row in rows:
        parent_id = row["parent_id"] or ""
        parent = get_parent(parent_id) if parent_id else None
        text = parent.text if parent else (
            f"amount={row['amount_usd']} term_months={row['term_months']} "
            f"party={row['party']} date={row['date_iso']}"
        )
        chunks.append(
            RetrievedChunk(
                chunk_id=row["chunk_id"] or f"sql-{row['doc_id']}-{row['page']}",
                doc_id=row["doc_id"],
                parent_id=parent_id or (row["chunk_id"] or ""),
                page=int(row["page"] or 1),
                text=text,
                heading=row["heading"] or "clause",
                score=0.8,
                parent_text=text,
                source="sql",
            )
        )
    summary = _aggregate_summary()
    if summary:
        chunks.insert(0, summary)
    if len(chunks) <= (1 if summary else 0):
        fallback = retrieve_hybrid(query)
        for item in fallback:
            item.source = "sql_fallback_hybrid"
        chunks.extend(fallback)
    return _dedupe(chunks)


def retrieve_ensemble(query: str) -> list[RetrievedChunk]:
    merged = retrieve_hybrid(query) + retrieve_graph(query)
    if any(t in query.lower() for t in ("how many", "total", "count", "average", "sum")):
        merged.extend(retrieve_sql(query))
    return _dedupe(merged)
