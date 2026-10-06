from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest
from qdrant_client import QdrantClient

from steam_game_analyst.data.schemas import NEWS_SCHEMA, REVIEWS_SCHEMA, WIKI_SCHEMA
from steam_game_analyst.retrieval import chunking, index, query
from steam_game_analyst.retrieval.filters import SearchFilters
from steam_game_analyst.retrieval.store import ChunkStore

NAMES = {1245620: "ELDEN RING", 892970: "Valheim", 1145350: "Hades II", 1145360: "Hades"}


class FakeEmbedder:
    """Deterministic bag-of-words hashing: texts sharing words get similar vectors."""

    dim = 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w.strip(".,!?")) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def _reviews():
    rows = [
        (
            "r1",
            1245620,
            "Runs great on the Steam Deck after the patch.",
            True,
            4000,
            datetime(2023, 5, 1),
        ),
        (
            "r2",
            1245620,
            "Steam Deck performance is awful, constant stutter.",
            False,
            600,
            datetime(2022, 3, 1),
        ),
        (
            "r3",
            892970,
            "Building a longhouse with friends is the best part.",
            True,
            3000,
            datetime(2021, 4, 2),
        ),
    ]
    df = pl.DataFrame(
        {
            "review_id": [r[0] for r in rows],
            "appid": [r[1] for r in rows],
            "language": ["english"] * 3,
            "review_text": [r[2] for r in rows],
            "voted_up": [r[3] for r in rows],
            "votes_up": [5, 2, 9],
            "votes_funny": [0, 0, 0],
            "weighted_vote_score": [None] * 3,
            "playtime_at_review_min": [None] * 3,
            "playtime_forever_min": [r[4] for r in rows],
            "steam_purchase": [None] * 3,
            "received_for_free": [None] * 3,
            "written_during_early_access": [False] * 3,
            "created_at": [r[5] for r in rows],
            "n_chars": [len(r[2]) for r in rows],
        }
    )
    return df.cast(REVIEWS_SCHEMA)


def _news():
    return pl.DataFrame(
        {
            "news_id": ["n1"],
            "appid": [892970],
            "title": ["Patch 0.217.14"],
            "contents": ["Fixed crafting station bug.\n\nImproved performance in large bases."],
            "author": ["dev"],
            "feed_name": ["steam_community_announcements"],
            "feed_label": ["Community Announcements"],
            "url": ["https://store.steampowered.com/news/1"],
            "published_at": [datetime(2023, 9, 1)],
            "tags": [["patchnotes"]],
            "is_patch_note": [True],
        }
    ).cast(NEWS_SCHEMA)


def _wiki():
    return pl.DataFrame(
        {
            "section_id": ["w1", "w2"],
            "slug": ["valheim", "hades"],
            "appids": [[892970], [1145350, 1145360]],
            "pageid": [10, 20],
            "title": ["Crafting", "Boons"],
            "heading": ["", "Zeus"],
            "url": [
                "https://valheim.fandom.com/wiki/Crafting",
                "https://hades.fandom.com/wiki/Boons",
            ],
            "text": [
                "Crafting combines materials at a workbench or forge.",
                "Zeus boons deal lightning damage.",
            ],
            "n_chars": [52, 33],
        }
    ).cast(WIKI_SCHEMA)


# ----------------------------------------------------------------------------- chunking
def test_split_text_packs_paragraphs_and_splits_long_ones():
    text = "one two three\n\nfour five\n\n" + " ".join(f"w{i}" for i in range(25))
    parts = chunking.split_text(text, max_words=10, overlap=2)
    assert parts[0] == "one two three\n\nfour five"
    assert all(len(p.split()) <= 10 for p in parts)
    assert parts[1].split()[0] == "w0"
    assert parts[2].split()[:2] == parts[1].split()[-2:]  # overlap carried over


def test_review_chunks_carry_filterable_payload_and_no_author():
    chunks = chunking.review_chunks(_reviews(), NAMES)
    c = chunks[0]
    assert c.payload["doc_type"] == "review"
    assert c.payload["appids"] == [1245620]
    assert c.payload["game"] == "ELDEN RING"
    assert c.payload["playtime_min"] == 4000
    assert c.payload["voted_up"] is True
    assert c.payload["created_at"] == int(datetime(2023, 5, 1, tzinfo=UTC).timestamp())
    assert c.text.startswith("ELDEN RING review (recommended)")
    assert "user" not in c.payload and "author" not in c.payload


def test_chunk_ids_are_stable_and_unique():
    a = chunking.review_chunks(_reviews(), NAMES) + chunking.news_chunks(_news(), NAMES)
    b = chunking.review_chunks(_reviews(), NAMES) + chunking.news_chunks(_news(), NAMES)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_news_and_wiki_chunks():
    news = chunking.news_chunks(_news(), NAMES)
    assert news[0].payload["doc_type"] == "patch_note"
    assert news[0].payload["title"] == "Patch 0.217.14"
    wiki = chunking.wiki_chunks(_wiki(), NAMES)
    hades = next(c for c in wiki if c.payload["title"] == "Boons")
    assert hades.payload["appids"] == [1145350, 1145360]
    assert hades.payload["heading"] == "Zeus"
    assert hades.text.startswith("Hades II wiki: Boons > Zeus")


# ----------------------------------------------------------------------------- filters + store
@pytest.fixture
def store():
    s = ChunkStore(QdrantClient(":memory:"), collection="test")
    s.ensure(dim=FakeEmbedder.dim, recreate=True)
    emb = FakeEmbedder()
    chunks = (
        chunking.review_chunks(_reviews(), NAMES)
        + chunking.news_chunks(_news(), NAMES)
        + chunking.wiki_chunks(_wiki(), NAMES)
    )
    s.upsert(chunks, emb.encode([c.text for c in chunks]))
    return s


def _search(store, text, filters=None, k=5):
    return store.search(FakeEmbedder().encode([text])[0], filters or SearchFilters(), k=k)


def test_store_counts_and_searches(store):
    assert store.count() == 3 + 1 + 2
    hits = _search(store, "crafting workbench forge")
    assert hits[0].payload["title"] == "Crafting"


def test_filters_appid_doc_type_playtime_and_text(store):
    f = SearchFilters(
        appids=(1245620,),
        doc_types=("review",),
        min_playtime_min=50 * 60,
        text_contains="steam deck",
    )
    hits = _search(store, "steam deck", f)
    assert [h.payload["source_id"] for h in hits] == ["r1"]


def test_filters_date_range_and_shared_wiki_appids(store):
    hits = _search(
        store, "review", SearchFilters(date_from=date(2023, 1, 1), doc_types=("review",))
    )
    assert [h.payload["source_id"] for h in hits] == ["r1"]
    hades = _search(store, "zeus lightning", SearchFilters(appids=(1145360,)))
    assert hades and hades[0].payload["title"] == "Boons"


def test_empty_filters_mean_no_qdrant_filter():
    assert SearchFilters().to_qdrant() is None


# ----------------------------------------------------------------------------- query parsing
GAMES = query.GameIndex.from_names(NAMES)


def test_game_index_prefers_longest_match():
    assert GAMES.find("What do players complain about most in Hades II?") == [(1145350, "Hades II")]
    assert GAMES.find("Is Hades better than elden ring?") == [
        (1145360, "Hades"),
        (1245620, "ELDEN RING"),
    ]
    assert GAMES.find("Tell me about shadows") == []


def test_single_word_names_need_a_capitalised_mention():
    games = query.GameIndex.from_names({1: "Satisfactory", 2: "Journey", 3: "ELDEN RING"})
    assert games.find("is the performance satisfactory on a long journey?") == []
    assert games.find("Is Satisfactory worth it?") == [(1, "Satisfactory")]
    assert games.find("elden ring bosses") == [(3, "ELDEN RING")]  # multi-word: any case
    assert games.find("satisfactory", strict=False) == [(1, "Satisfactory")]  # explicit --game


def test_numeric_names_cjk_names_and_ambiguous_aliases():
    games = query.GameIndex.from_names(
        {
            1: "1980",
            2: "东北之夏",
            3: "觅长生",
            4: "One Dreamer: Prologue",
            5: "Hell Architect: Prologue",
            6: "Warhammer 40,000: Darktide",
        }
    )
    assert games.find("What happened in 1980 to gaming?") == []
    assert games.find("东北之夏好玩吗") == [(2, "东北之夏")]
    assert games.find("Is Prologue good?") == []  # alias shared by two games
    assert games.find("Darktide crashes") == []  # one-word alias: too risky
    assert games.find("tell me about 觅长生") == [(3, "觅长生")]


def test_quoted_game_name_is_not_a_text_filter_and_after_excludes_year():
    q = query.parse_question('Reviews about "Hades" after 2023', GAMES)
    assert q.filters.appids == (1145360,)
    assert q.filters.text_contains is None
    assert q.filters.date_from == date(2024, 1, 1)
    assert query.parse_question("Hades reviews since 2023", GAMES).filters.date_from == date(
        2023, 1, 1
    )


def test_cjk_text_is_split_by_characters():
    parts = chunking.split_text("字" * 3000, max_words=300, max_chars=1200)
    assert len(parts) >= 3 and all(len(p) <= 1200 for p in parts)


def test_review_chunks_scrub_profile_links():
    df = _reviews().with_columns(
        pl.lit("Add me https://steamcommunity.com/id/some_player123/ for co-op").alias(
            "review_text"
        )
    )
    text = chunking.review_chunks(df.head(1), NAMES)[0].payload["text"]
    assert "some_player123" not in text and "[profile link]" in text


def test_parse_question_extracts_playtime_and_mention():
    q = query.parse_question(
        "Show me reviews mentioning Steam Deck for Elden Ring from players with 50+ hours.", GAMES
    )
    assert q.filters.appids == (1245620,)
    assert q.filters.min_playtime_min == 50 * 60
    assert q.filters.text_contains == "steam deck"
    assert q.filters.doc_types == ("review",)


def test_parse_question_without_filters():
    q = query.parse_question("How does the crafting system work in Valheim?", GAMES)
    assert q.filters == SearchFilters(appids=(892970,))


# ----------------------------------------------------------------------------- index
def test_build_index_from_processed_tables(tmp_path):
    _reviews().write_parquet(tmp_path / "reviews.parquet")
    _news().write_parquet(tmp_path / "news.parquet")
    _wiki().write_parquet(tmp_path / "wiki.parquet")
    pl.DataFrame({"appid": list(NAMES), "name": list(NAMES.values())}).write_parquet(
        tmp_path / "games.parquet"
    )
    s = ChunkStore(QdrantClient(":memory:"), collection="idx")

    counts = index.build_index(tmp_path, s, FakeEmbedder(), recreate=True)

    assert counts == {"review": 3, "news": 1, "wiki": 2}
    assert s.count() == 6


def test_reindexing_a_source_replaces_its_stale_chunks(tmp_path):
    _news().write_parquet(tmp_path / "news.parquet")
    _wiki().write_parquet(tmp_path / "wiki.parquet")
    pl.DataFrame({"appid": list(NAMES), "name": list(NAMES.values())}).write_parquet(
        tmp_path / "games.parquet"
    )
    s = ChunkStore(QdrantClient(":memory:"), collection="re")
    _reviews().write_parquet(tmp_path / "reviews.parquet")
    index.build_index(tmp_path, s, FakeEmbedder(), recreate=True)

    _reviews().head(1).write_parquet(tmp_path / "reviews.parquet")  # new, smaller sample
    index.build_index(tmp_path, s, FakeEmbedder(), sources=("review",))

    assert s.count() == 1 + 1 + 2  # 1 review left, news and wiki untouched


class FailingEmbedder(FakeEmbedder):
    def encode(self, texts):
        raise RuntimeError("CUDA out of memory")


def test_failed_reindex_keeps_previous_chunks(tmp_path):
    _reviews().write_parquet(tmp_path / "reviews.parquet")
    pl.DataFrame({"appid": list(NAMES), "name": list(NAMES.values())}).write_parquet(
        tmp_path / "games.parquet"
    )
    s = ChunkStore(QdrantClient(":memory:"), collection="fail")
    index.build_index(tmp_path, s, FakeEmbedder(), sources=("review",), recreate=True)

    with pytest.raises(RuntimeError):
        index.build_index(tmp_path, s, FailingEmbedder(), sources=("review",))

    assert s.count() == 3


def test_build_index_streams_in_small_batches(tmp_path):
    _reviews().write_parquet(tmp_path / "reviews.parquet")
    pl.DataFrame({"appid": list(NAMES), "name": list(NAMES.values())}).write_parquet(
        tmp_path / "games.parquet"
    )
    s = ChunkStore(QdrantClient(":memory:"), collection="small")
    counts = index.build_index(
        tmp_path, s, FakeEmbedder(), sources=("review",), recreate=True, encode_batch=2
    )
    assert counts == {"review": 3} and s.count() == 3
