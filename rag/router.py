from __future__ import annotations

import re

import numpy as np

from rag.config import load_config
from rag.embedder import embed_query, embed_texts
from rag.models import RouteDecision

COMPARATIVE_RE = re.compile(
    r"\b(compar(e|ison)|versus|vs\.?|differ(ence|s)?|contrast|between)\b",
    re.IGNORECASE,
)
AGGREGATE_RE = re.compile(
    r"\b(how many|count|total|sum|average|avg|aggregate|across (all )?contracts)\b",
    re.IGNORECASE,
)

_prototypes: dict[str, np.ndarray] | None = None


def _prototype_matrix() -> dict[str, np.ndarray]:
    global _prototypes
    if _prototypes is None:
        cfg = load_config()["router"]
        _prototypes = {
            "factual": np.mean(embed_texts(cfg["factual_prototypes"]), axis=0),
            "comparative": np.mean(embed_texts(cfg["comparative_prototypes"]), axis=0),
            "aggregate": np.mean(embed_texts(cfg["aggregate_prototypes"]), axis=0),
        }
    return _prototypes


def _rule_intent(query: str) -> tuple[str, float] | None:
    if COMPARATIVE_RE.search(query):
        return "comparative", 0.92
    if AGGREGATE_RE.search(query):
        return "aggregate", 0.9
    return None


def route_query(query: str) -> RouteDecision:
    cfg = load_config()
    threshold = float(cfg["guards"]["router_ensemble_below"])
    ruled = _rule_intent(query)
    qvec = np.asarray(embed_query(query), dtype=np.float32)
    protos = _prototype_matrix()
    sims = {name: float(np.dot(qvec, vec) / (np.linalg.norm(qvec) * np.linalg.norm(vec) + 1e-9)) for name, vec in protos.items()}
    proto_intent = max(sims, key=sims.get)
    proto_conf = sims[proto_intent]

    if ruled:
        intent, rule_conf = ruled
        confidence = max(rule_conf, proto_conf if proto_intent == intent else rule_conf * 0.85)
        reason = f"rule:{intent}; proto:{proto_intent}={proto_conf:.2f}"
    else:
        intent, confidence = proto_intent, proto_conf
        reason = f"prototype:{intent}={confidence:.2f}"

    use_ensemble = confidence < threshold
    if use_ensemble:
        intent = "ensemble"
        reason += "; ensemble"
    return RouteDecision(intent=intent, confidence=confidence, use_ensemble=use_ensemble, reason=reason)
