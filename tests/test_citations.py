from rag.citations import citations_grounded, extract_citations
from rag.models import RetrievedChunk


def _chunk(doc_id: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id=doc_id,
        parent_id="p1",
        page=page,
        text="example",
        heading="PAYMENT",
        score=0.9,
        parent_text="Payment is $50,000.",
    )


def test_extract_citations() -> None:
    assert extract_citations("The fee is $50,000 [abc:2] and [xyz:1].") == [
        ("abc", 2),
        ("xyz", 1),
    ]


def test_grounded_when_all_cites_exist() -> None:
    answer = "The payment is $50,000 [msa:1]."
    assert citations_grounded(answer, [_chunk("msa", 1)]) is True


def test_ungrounded_unknown_doc() -> None:
    answer = "The payment is $50,000 [other:1]."
    assert citations_grounded(answer, [_chunk("msa", 1)]) is False


def test_ungrounded_missing_citation() -> None:
    assert citations_grounded("There is a payment of $50,000.", [_chunk("msa", 1)]) is True
