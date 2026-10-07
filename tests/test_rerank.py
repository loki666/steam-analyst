import pytest

from steam_game_analyst.retrieval import rag
from test_rag import retriever  # noqa: F401  (fixture)


class KeywordReranker:
    """Scores a passage by whether it contains `word`; records what it was given."""

    def __init__(self, word):
        self.word = word
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [1.0 if self.word in p.lower() else 0.0 for p in passages]


def _rerank(retriever, reranker, depth=50):  # noqa: F811
    return rag.Retriever(
        retriever.store,
        retriever.embedder,
        retriever.games,
        mode="rerank",
        reranker=reranker,
        rerank_depth=depth,
    )


def test_rerank_reorders_hybrid_candidates(retriever):  # noqa: F811
    question = "crafting workbench forge"
    hybrid = retriever.retrieve(question, k=5).hits
    reranker = KeywordReranker("boons")
    hits = _rerank(retriever, reranker).retrieve(question, k=2).hits
    assert len(hits) == 2
    assert hits[0].payload["title"] == "Boons"
    assert {h.id for h in hits} <= {h.id for h in hybrid} | {hits[0].id}
    assert hits[0].score == 1.0  # the reranker's score replaces the fusion score


def test_rerank_scores_depth_candidates_with_a_source_header(retriever):  # noqa: F811
    reranker = KeywordReranker("zzz")
    _rerank(retriever, reranker, depth=50).retrieve("crafting workbench forge", k=1)
    query, passages = reranker.calls[0]
    assert query == "crafting workbench forge"
    assert len(passages) > 1  # rerank_depth candidates, not k
    crafting = next(p for p in passages if "Crafting" in p.splitlines()[0])
    assert crafting.splitlines()[0].startswith("Valheim · wiki")  # describe() header first


def test_ties_keep_the_hybrid_order(retriever):  # noqa: F811
    question = "crafting workbench forge"
    hybrid = [h.id for h in retriever.retrieve(question, k=3).hits]
    hits = _rerank(retriever, KeywordReranker("zzz")).retrieve(question, k=3).hits
    assert [h.id for h in hits] == hybrid


def test_rerank_mode_needs_a_reranker(retriever):  # noqa: F811
    with pytest.raises(ValueError, match="needs a reranker"):
        rag.Retriever(retriever.store, retriever.embedder, retriever.games, mode="rerank")
