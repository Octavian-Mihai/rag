from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from rag.config import load_config
from rag.models import Chunk, RetrievedChunk
from rag.store import get_parent

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        cfg = load_config()
        _client = QdrantClient(url=cfg["qdrant"]["url"], check_compatibility=False)
    return _client


def ensure_collection() -> None:
    cfg = load_config()
    name = cfg["qdrant"]["collection"]
    dim = int(cfg["models"]["embedding_dim"])
    dense = cfg["qdrant"]["dense_name"]
    sparse = cfg["qdrant"]["sparse_name"]
    q = client()
    if q.collection_exists(name):
        return
    q.create_collection(
        collection_name=name,
        vectors_config={dense: VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={sparse: SparseVectorParams()},
    )


def upsert_chunks(
    children: list[Chunk],
    dense: list[list[float]],
    sparse_pairs: list[tuple[list[int], list[float]]],
) -> None:
    cfg = load_config()
    name = cfg["qdrant"]["collection"]
    dense_name = cfg["qdrant"]["dense_name"]
    sparse_name = cfg["qdrant"]["sparse_name"]
    ensure_collection()
    points = []
    for chunk, vec, (idx, vals) in zip(children, dense, sparse_pairs):
        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector={
                    dense_name: vec,
                    sparse_name: SparseVector(indices=idx, values=vals),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "parent_id": chunk.parent_id,
                    "page": chunk.page,
                    "heading": chunk.heading,
                    "text": chunk.text,
                },
            )
        )
    if points:
        client().upsert(collection_name=name, points=points)


def hybrid_search(
    dense_query: list[float],
    sparse_idx: list[int],
    sparse_vals: list[float],
    limit: int,
) -> list[RetrievedChunk]:
    cfg = load_config()
    name = cfg["qdrant"]["collection"]
    dense_name = cfg["qdrant"]["dense_name"]
    sparse_name = cfg["qdrant"]["sparse_name"]
    ensure_collection()
    result = client().query_points(
        collection_name=name,
        prefetch=[
            Prefetch(query=dense_query, using=dense_name, limit=limit),
            Prefetch(
                query=SparseVector(indices=sparse_idx, values=sparse_vals),
                using=sparse_name,
                limit=limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    out: list[RetrievedChunk] = []
    for point in result.points:
        payload = point.payload or {}
        parent = get_parent(str(payload.get("parent_id", "")))
        out.append(
            RetrievedChunk(
                chunk_id=str(payload.get("chunk_id", point.id)),
                doc_id=str(payload.get("doc_id", "")),
                parent_id=str(payload.get("parent_id", "")),
                page=int(payload.get("page", 1)),
                text=str(payload.get("text", "")),
                heading=str(payload.get("heading", "")),
                score=float(point.score or 0.0),
                parent_text=parent.text if parent else str(payload.get("text", "")),
                source="hybrid",
            )
        )
    return out


def collection_count() -> int:
    cfg = load_config()
    name = cfg["qdrant"]["collection"]
    if not client().collection_exists(name):
        return 0
    return int(client().count(name).count)
