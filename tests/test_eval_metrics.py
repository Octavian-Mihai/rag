from eval.metrics import hit_rate_at_k, mrr


def test_hit_rate_and_mrr() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    assert hit_rate_at_k(retrieved, {"c"}, k=5) == 1.0
    assert hit_rate_at_k(retrieved, {"z"}, k=5) == 0.0
    assert mrr(retrieved, {"c"}) == 1.0 / 3
    assert mrr(retrieved, {"z"}) == 0.0
