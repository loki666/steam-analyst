import pytest

from steam_game_analyst.eval import ablation as ab


def _row(qid, rank, ndcg, precision, strict_rank=None):
    return {
        "qid": qid,
        "doc_type": "news",
        "rank": rank,
        "strict_rank": strict_rank,
        "ndcg": ndcg,
        "precision": precision,
    }


def _result(rows, unjudged=0):
    return {"k": 10, "n": len(rows), "unjudged_hits": unjudged, "per_question": rows}


BASE = [_row("a", 1, 1.0, 0.3, 1), _row("b", 7, 0.2, 0.1), _row("c", None, 0.0, 0.0)]
BETTER = [_row("a", 1, 1.0, 0.5, 1), _row("b", 2, 0.6, 0.3, 2), _row("c", 4, 0.3, 0.1)]


def test_question_values_per_metric():
    row = _row("q", 4, 0.5, 0.2, strict_rank=None)
    assert ab.value(row, "hit@5", 10) == 1.0
    assert ab.value(row, "hit@1", 10) == 0.0
    assert ab.value(row, "mrr@10", 10) == 0.25
    assert ab.value(row, "ndcg@10", 10) == 0.5
    assert ab.value(row, "p@10", 10) == 0.2
    assert ab.value(row, "strict_hit@5", 10) == 0.0
    assert ab.value(_row("q", None, 0.0, 0.0), "mrr@10", 10) == 0.0


def test_bootstrap_interval_brackets_the_mean_and_is_seeded():
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    mean, lo, hi = ab.bootstrap_ci(values, seed=3)
    assert lo <= mean == 0.625 <= hi
    assert ab.bootstrap_ci(values, seed=3) == (mean, lo, hi)


def test_identical_systems_differ_by_exactly_zero():
    assert ab.bootstrap_ci([0.0] * 5) == (0.0, 0.0, 0.0)


def test_ablation_compares_each_mode_with_the_baseline_question_by_question():
    table = ab.ablation({"hybrid": _result(BASE), "rerank": _result(BETTER)}, baseline="hybrid")
    rows = {r["mode"]: r for r in table["rows"]}
    assert rows["hybrid"]["vs_baseline"] == {}
    d = rows["rerank"]["vs_baseline"]["ndcg@10"]
    assert d["mean"] == pytest.approx((0.0 + 0.4 + 0.3) / 3)
    assert d["better"] == 2 and d["worse"] == 0
    assert rows["rerank"]["metrics"]["hit@5"]["mean"] == 1.0
    assert table["n"] == 3 and table["baseline"] == "hybrid"


def test_ablation_refuses_runs_over_different_questions():
    other = [_row("a", 1, 1.0, 0.3), _row("z", 1, 1.0, 0.3), _row("c", 1, 1.0, 0.3)]
    with pytest.raises(ValueError, match="same questions"):
        ab.ablation({"hybrid": _result(BASE), "dense": _result(other)}, baseline="hybrid")


def test_ablation_needs_judged_questions_and_the_baseline():
    with pytest.raises(ValueError, match="no question has a judged relevant chunk"):
        ab.ablation({"hybrid": _result([])}, baseline="hybrid")
    with pytest.raises(ValueError, match="baseline"):
        ab.ablation({"dense": _result(BASE)}, baseline="hybrid")


def test_markdown_table_marks_the_baseline_and_flags_unjudged_hits():
    table = ab.ablation(
        {"hybrid": _result(BASE), "rerank": _result(BETTER, unjudged=2)}, baseline="hybrid"
    )
    md = ab.to_markdown(table, split="dev")
    assert "| hybrid (baseline) |" in md
    assert "| rerank |" in md
    assert "Δ nDCG@10 vs hybrid" in md
    assert "2 unjudged" in md
