import math
from dataclasses import dataclass

import pytest

from steam_game_analyst.eval import judged as ev
from steam_game_analyst.retrieval.store import Hit


@dataclass
class FakeRetrieval:
    hits: tuple


class FakeRetriever:
    """Returns pre-set chunk ids per question."""

    def __init__(self, ranked):
        self.ranked = ranked
        self.calls = []

    def retrieve(self, question, k=8, overrides=None):
        self.calls.append((question, k))
        ids = self.ranked.get(question, [])[:k]
        return FakeRetrieval(tuple(Hit(c, 1.0, _payload(c)) for c in ids))


class FakeStore:
    def fetch(self, ids):
        return [Hit(c, 0.0, _payload(c)) for c in ids]


def _payload(chunk_id):
    return {
        "source_id": "src-" + chunk_id,
        "doc_type": "review",
        "game": "Hades",
        "title": "",
        "text": "text of " + chunk_id,
    }


ITEMS = [
    {"qid": "a", "split": "dev", "question": "qa", "chunk_id": "gen-a", "doc_type": "review"},
    {"qid": "b", "split": "dev", "question": "qb", "chunk_id": "gen-b", "doc_type": "wiki"},
    {"qid": "t", "split": "test", "question": "qt", "chunk_id": "gen-t", "doc_type": "wiki"},
]
CANDIDATE_KEYS = {"chunk_id", "source_id", "doc_type", "game", "title", "text"}


# ----------------------------------------------------------------------------- pooling
def test_pool_unions_every_mode_plus_the_generating_chunk_without_saying_who_found_what():
    retrievers = {
        "dense": FakeRetriever({"qa": ["c1", "c2"], "qb": ["c9"]}),
        "bm25": FakeRetriever({"qa": ["c2", "c3"], "qb": []}),
    }
    pool = ev.build_pool(ITEMS, retrievers, FakeStore(), split="dev", depth=10)

    assert [p["qid"] for p in pool] == ["a", "b"]  # dev only
    a = pool[0]
    assert {c["chunk_id"] for c in a["candidates"]} == {"c1", "c2", "c3", "gen-a"}
    assert all(set(c) == CANDIDATE_KEYS for c in a["candidates"])
    # blind order: not the retrieval order, and the same on every run
    ids = [c["chunk_id"] for c in a["candidates"]]
    assert ids == sorted(ids)
    assert all(call[1] == 10 for r in retrievers.values() for call in r.calls)


def test_pool_refuses_the_test_split_without_final():
    with pytest.raises(PermissionError):
        ev.build_pool(ITEMS, {"dense": FakeRetriever({})}, FakeStore(), split="test")


# ----------------------------------------------------------------------------- judged scoring
def _judgments(rows):
    return {(q, c): g for q, c, g in rows}


def test_judged_metrics_count_any_relevant_chunk_and_grade_ndcg():
    retriever = FakeRetriever({"qa": ["x", "r2", "r1"], "qb": ["y", "z"]})
    judgments = _judgments(
        [
            ("a", "x", 0),
            ("a", "r2", 2),
            ("a", "r1", 1),
            ("a", "gen-a", 2),  # relevant but not retrieved: counts in the ideal ranking
            ("b", "y", 0),
            ("b", "z", 0),
            ("b", "gen-b", 1),
        ]
    )
    res = ev.evaluate_judged(ITEMS, retriever, judgments, split="dev", k=10)

    assert res["n"] == 2 and res["no_relevant"] == 0 and res["unjudged_hits"] == 0
    qa = next(q for q in res["per_question"] if q["qid"] == "a")
    assert qa["rank"] == 2  # first chunk graded >= 1
    dcg = (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    idcg = (2**2 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    assert qa["ndcg"] == pytest.approx(dcg / idcg)
    qb = next(q for q in res["per_question"] if q["qid"] == "b")
    assert qb["rank"] is None and qb["ndcg"] == 0.0
    assert res["overall"]["hit@1"] == 0.0
    assert res["overall"]["hit@5"] == pytest.approx(0.5)
    assert res["overall"]["mrr@10"] == pytest.approx(0.25)
    assert res["overall"]["p@10"] == pytest.approx((2 / 10 + 0) / 2)
    assert res["strict"]["hit@5"] == pytest.approx(0.5)  # grade 2 only: r2 at rank 2
    assert res["by_doc_type"]["wiki"]["n"] == 1


def test_questions_with_nothing_relevant_are_reported_not_averaged():
    retriever = FakeRetriever({"qa": ["x"], "qb": ["y"]})
    judgments = _judgments([("a", "x", 1), ("b", "y", 0), ("b", "gen-b", 0)])
    res = ev.evaluate_judged(ITEMS, retriever, judgments, split="dev", k=10)
    assert res["n"] == 1 and res["no_relevant"] == 1
    assert res["overall"]["hit@1"] == 1.0


def test_unjudged_hits_are_counted_so_new_systems_trigger_more_judging():
    retriever = FakeRetriever({"qa": ["x", "new"], "qb": ["y"]})
    judgments = _judgments([("a", "x", 2), ("b", "y", 1)])
    res = ev.evaluate_judged(ITEMS, retriever, judgments, split="dev", k=10)
    assert res["unjudged_hits"] == 1


def test_judged_eval_keeps_the_test_split_frozen():
    with pytest.raises(PermissionError):
        ev.evaluate_judged(ITEMS, FakeRetriever({}), {}, split="test")


def test_load_judgments_keeps_the_last_grade_per_pair(tmp_path):
    path = tmp_path / "j.jsonl"
    path.write_text(
        '{"qid": "a", "chunk_id": "x", "grade": 0}\n'
        "\n"
        '{"qid": "a", "chunk_id": "x", "grade": 2}\n'
        '{"qid": "b", "chunk_id": "y", "grade": 1}\n'
    )
    assert ev.load_judgments(path) == {("a", "x"): 2, ("b", "y"): 1}


def test_load_judgments_rejects_grades_outside_the_scale(tmp_path):
    path = tmp_path / "j.jsonl"
    path.write_text('{"qid": "a", "chunk_id": "x", "grade": 3}\n')
    with pytest.raises(ValueError, match="grade"):
        ev.load_judgments(path)
