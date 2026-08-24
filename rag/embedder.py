from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from rag.config import load_config


@lru_cache
def _embedder():
    from sentence_transformers import SentenceTransformer

    cfg = load_config()
    model_name = cfg["models"]["embedding"]
    return SentenceTransformer(model_name, cache_folder=os.environ.get("HF_HOME"))


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _embedder()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0].tolist()
