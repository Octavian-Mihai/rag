from rag.guards import pre_generation_guard
from rag.models import RetrievedChunk


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c",
        doc_id="d",
        parent_id="p",
        page=1,
        text="t",
        heading="h",
        score=score,
        parent_text="t",
    )


def test_low_similarity_abstains(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag.citations.load_config",
        lambda: {"guards": {"min_similarity": 0.6}},
    )
    msg = pre_generation_guard([_chunk(0.4)])
    assert msg is not None
    assert "couldn't find relevant information" in msg.lower()


def test_high_similarity_passes(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag.citations.load_config",
        lambda: {"guards": {"min_similarity": 0.6}},
    )
    assert pre_generation_guard([_chunk(0.81)]) is None
