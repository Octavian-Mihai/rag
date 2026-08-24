from __future__ import annotations

from rag.citations import should_abstain_low_similarity
from rag.config import load_config
from rag.models import RetrievedChunk


def low_similarity_message() -> str:
    return load_config()["guards"]["abstain_message"]


def ungrounded_message() -> str:
    return load_config()["guards"]["ungrounded_message"]


def pre_generation_guard(chunks: list[RetrievedChunk]) -> str | None:
    if should_abstain_low_similarity(chunks):
        return low_similarity_message()
    return None
