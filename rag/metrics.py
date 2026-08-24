from __future__ import annotations

import statistics
import threading
from collections import Counter


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.queries = 0
        self.abstains = 0
        self.routes: Counter[str] = Counter()
        self.latencies: list[float] = []

    def record(self, route: str, latency_ms: float, abstained: bool) -> None:
        with self._lock:
            self.queries += 1
            self.routes[route] += 1
            self.latencies.append(latency_ms)
            if abstained:
                self.abstains += 1

    def snapshot(self) -> dict:
        with self._lock:
            lats = list(self.latencies)
            routes = dict(self.routes)
            queries = self.queries
            abstains = self.abstains
        p50 = statistics.median(lats) if lats else 0.0
        p95 = sorted(lats)[max(0, int(len(lats) * 0.95) - 1)] if lats else 0.0
        return {
            "queries": queries,
            "abstains": abstains,
            "routes": routes,
            "latency_ms": {"p50": p50, "p95": p95},
        }


METRICS = Metrics()
