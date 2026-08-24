from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.config import ROOT, load_config
from rag.debug_log import debug_log
from rag.graph import GRAPH
from rag.ingest import ingest_directory, ingest_pdf
from rag.metrics import METRICS
from rag.models import RetrievedChunk
from rag.pipeline import answer_question
from rag.store import list_documents
from rag.vectorstore import collection_count, ensure_collection
from rag.warmup import warmup_ollama

app = FastAPI(title="Adaptive Local Legal RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = ROOT / "data" / "uploads"
SAMPLES = ROOT / "data" / "samples"


class QueryBody(BaseModel):
    question: str


def _chunk_payload(chunk: RetrievedChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page": chunk.page,
        "heading": chunk.heading,
        "score": chunk.score,
        "source": chunk.source,
        "text": (chunk.parent_text or chunk.text)[:1200],
    }


def _maybe_ingest_samples() -> None:
    try:
        if collection_count() == 0 and SAMPLES.exists() and any(SAMPLES.glob("*.pdf")):
            ingest_directory(SAMPLES)
        else:
            GRAPH.rebuild()
    except Exception:
        pass


@app.on_event("startup")
def startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ensure_collection()
    except Exception:
        pass
    threading.Thread(target=_maybe_ingest_samples, daemon=True).start()
    threading.Thread(target=warmup_ollama, daemon=True).start()


@app.get("/health")
def health() -> dict:
    qdrant_ok = False
    count = 0
    try:
        count = collection_count()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    return {
        "ok": True,
        "qdrant": qdrant_ok,
        "chunk_count": count,
        "embedding": load_config()["models"]["embedding"],
    }


@app.get("/metrics")
def metrics() -> dict:
    return METRICS.snapshot()


@app.get("/documents")
def documents() -> dict:
    return {"documents": list_documents()}


@app.post("/ingest")
async def ingest(
    file: UploadFile | None = File(default=None),
    use_samples: bool = Query(default=False),
) -> dict:
    if use_samples:
        if not SAMPLES.exists():
            raise HTTPException(400, "Sample directory missing")
        return {"ingested": ingest_directory(SAMPLES)}
    if file is None or not file.filename:
        raise HTTPException(400, "Upload a PDF or set use_samples=true")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    dest = UPLOAD_DIR / Path(file.filename).name
    dest.write_bytes(await file.read())
    try:
        result = ingest_pdf(dest)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ingested": [result]}


@app.post("/query")
def query(body: QueryBody) -> dict:
    if not body.question.strip():
        raise HTTPException(400, "question is required")
    # #region agent log
    debug_log("app/main.py:query", "api_query_received", {"question_len": len(body.question.strip())}, hypothesis_id="H3")
    # #endregion
    try:
        result = answer_question(body.question.strip())
    except Exception as exc:
        # #region agent log
        debug_log("app/main.py:query", "api_query_unhandled_error", {"error": str(exc), "type": type(exc).__name__}, hypothesis_id="H1")
        # #endregion
        raise HTTPException(503, f"Query failed: {exc}") from exc
    # #region agent log
    debug_log(
        "app/main.py:query",
        "api_query_ok",
        {"abstained": result.abstained, "reason": result.reason, "route": result.route},
        hypothesis_id="H3",
    )
    # #endregion
    return {
        "answer": result.answer,
        "route": result.route,
        "confidence": result.confidence,
        "chunks": [_chunk_payload(c) for c in result.chunks],
        "latency_ms": result.latency_ms,
        "abstained": result.abstained,
        "reason": result.reason,
    }
