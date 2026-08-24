from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    parent_id: str
    page: int
    text: str
    heading: str
    is_parent: bool = False


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    parent_id: str
    page: int
    text: str
    heading: str
    score: float
    parent_text: str = ""
    source: str = "hybrid"


@dataclass
class RouteDecision:
    intent: str
    confidence: float
    use_ensemble: bool
    reason: str


@dataclass
class QueryResult:
    answer: str
    route: str
    confidence: float
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    abstained: bool = False
    reason: str = ""
