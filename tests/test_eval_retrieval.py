from dataclasses import dataclass

import pytest

from steam_game_analyst.eval import retrieval as ev
from steam_game_analyst.retrieval.store import Hit


@dataclass
class FakeRetrieval:
    hits: tuple


class FakeRetriever:
    """Returns pre-set source ids per question; several chunks may share a source."""

    def __init__(self, ranked):
        self.ranked = ranked

    def retrieve(self, question, k=8, overrides=None):
        ids = self.ranked[question][:k]
        return FakeRetrieval(tuple(Hit(f"c{i}", 1.0, {"source_id": s}) for i, s in enumerate(ids)))


ITEMS = [
    {"qid": "a", "split": "dev", "question": "qa", "source_id": "s1", "doc_type": "review"},
    {"qid": "b", "split": "dev", "question": "qb", "source_id": "s2", "doc_type": "wiki"},
    {"qid": "c", "split": "dev", "question": "qc", "source_id": "s3", "doc_type": "wiki"},
    {"qid": "t", "split": "test", "question": "qt", "source_id": "s9", "doc_type": "wiki"},
]


def test_document_level_ranks_dedupe_chunks_of_the_same_source():
    retriever = FakeRetriever(
        {
            "qa": ["s1", "s1", "x"],  # rank 1
            "qb": ["x", "x", "y", "s2"],  # chunks of x count once -> s2 is document rank 3
            "qc": ["x"] * 30,  # miss
        }
    )
    res = ev.evaluate(ITEMS, retriever, split="dev", k=20)
    assert res["n"] == 3
    assert res["overall"]["recall@5"] == pytest.approx(2 / 3)
    assert res["overall"]["mrr@20"] == pytest.approx((1 + 1 / 3) / 3)
    assert res["by_doc_type"]["wiki"]["n"] == 2
    assert res["by_doc_type"]["wiki"]["recall@20"] == pytest.approx(0.5)


def test_test_split_needs_explicit_final_flag():
    retriever = FakeRetriever({"qt": ["s9"]})
    with pytest.raises(PermissionError):
        ev.evaluate(ITEMS, retriever, split="test")
    assert ev.evaluate(ITEMS, retriever, split="test", final=True)["overall"]["recall@5"] == 1.0


def test_results_keep_each_questions_rank_for_paired_comparison():
    retriever = FakeRetriever({"qa": ["s1"], "qb": ["x", "s2"], "qc": ["x"]})
    res = ev.evaluate(ITEMS, retriever, split="dev")
    assert res["per_question"] == [
        {"qid": "a", "doc_type": "review", "rank": 1},
        {"qid": "b", "doc_type": "wiki", "rank": 2},
        {"qid": "c", "doc_type": "wiki", "rank": None},
    ]


def test_paired_comparison_counts_wins_and_losses():
    before = {
        "per_question": [
            {"qid": "a", "rank": 3},
            {"qid": "b", "rank": None},
            {"qid": "c", "rank": 1},
        ]
    }
    after = {
        "per_question": [
            {"qid": "a", "rank": 1},
            {"qid": "b", "rank": None},
            {"qid": "c", "rank": 4},
        ]
    }
    assert ev.compare(before, after) == {"better": 1, "worse": 1, "same": 1, "n": 3}
