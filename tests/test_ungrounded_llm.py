from rag.citations import citations_grounded
from rag.models import RetrievedChunk


def test_mocked_llm_ungrounded_string() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="1",
            doc_id="acme-beta",
            parent_id="p",
            page=1,
            text="Payment is $50,000.",
            heading="PAYMENT",
            score=0.9,
            parent_text="Payment is $50,000.",
        )
    ]
    hallucinated = "Payment is $1,000,000 [invented-doc:99]."
    assert citations_grounded(hallucinated, retrieved) is False
