from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from rag.config import ROOT

TOKEN_RE = re.compile(r"[a-z0-9]+")
VOCAB = 30_000
STATE_PATH = Path(ROOT) / "data" / "bm25.json"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def term_index(term: str) -> int:
    digest = hashlib.md5(term.encode()).hexdigest()
    return int(digest, 16) % VOCAB


class BM25Encoder:
    """Hashed BM25 sparse vectors for Qdrant hybrid search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avgdl = 0.0
        self.df: Counter[str] = Counter()
        self._total_len = 0

    def fit_document(self, text: str) -> None:
        tokens = tokenize(text)
        self.doc_count += 1
        self._total_len += len(tokens)
        self.avgdl = self._total_len / max(self.doc_count, 1)
        for term in set(tokens):
            self.df[term] += 1

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        if not tokens:
            return [0], [0.0]
        tf = Counter(tokens)
        dl = len(tokens)
        n = max(self.doc_count, 1)
        avgdl = self.avgdl or dl
        merged: dict[int, float] = {}
        for term, freq in tf.items():
            df = self.df.get(term, 1)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * dl / avgdl)
            score = idf * (freq * (self.k1 + 1)) / denom
            idx = term_index(term)
            merged[idx] = merged.get(idx, 0.0) + float(score)
        indices = sorted(merged)
        values = [merged[i] for i in indices]
        return indices, values

    def save(self, path: Path = STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "doc_count": self.doc_count,
                    "avgdl": self.avgdl,
                    "total_len": self._total_len,
                    "df": dict(self.df),
                }
            )
        )

    def load(self, path: Path = STATE_PATH) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self.doc_count = int(data.get("doc_count", 0))
        self.avgdl = float(data.get("avgdl", 0.0))
        self._total_len = int(data.get("total_len", 0))
        self.df = Counter(data.get("df", {}))


BM25 = BM25Encoder()
BM25.load()
