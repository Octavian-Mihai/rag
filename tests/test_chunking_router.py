from rag.chunking import is_heading, semantic_parent_child
from rag.router import AGGREGATE_RE, COMPARATIVE_RE, _rule_intent


def test_heading_detection() -> None:
    assert is_heading("ARTICLE 2 PAYMENT")
    assert is_heading("Section 3")
    assert not is_heading("This is a normal sentence about payment terms.")


def test_parent_child_splits_articles() -> None:
    pages = [
        (
            1,
            "Preamble text.\n\nARTICLE 1 PARTIES\nAcme Corp and Beta LLC.\n\n"
            "ARTICLE 2 PAYMENT\nThe fee is $50,000.\n",
        )
    ]
    parents, children = semantic_parent_child("doc-1", pages)
    headings = {p.heading for p in parents}
    assert "ARTICLE 1 PARTIES" in headings
    assert "ARTICLE 2 PAYMENT" in headings
    assert children
    assert all(not c.is_parent for c in children)


def test_rule_intent() -> None:
    assert _rule_intent("Compare the liability caps")[0] == "comparative"
    assert _rule_intent("What is the total contract value?")[0] == "aggregate"
    assert _rule_intent("What is the governing law?") is None
    assert COMPARATIVE_RE.search("versus")
    assert AGGREGATE_RE.search("how many clauses")
