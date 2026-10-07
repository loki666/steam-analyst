import json

import httpx
import pytest

from steam_game_analyst.data import wiki


# ----------------------------------------------------------------------------- naming
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ELDEN RING", ["eldenring", "elden-ring"]),
        ("Hades II", ["hadesii", "hades-ii", "hades"]),
        (
            "Warhammer 40,000: Darktide",
            ["warhammer40000darktide", "warhammer-40000-darktide", "warhammer40000", "darktide"],
        ),
        ("Cyberpunk 2077", ["cyberpunk2077", "cyberpunk-2077", "cyberpunk"]),
        ("Valheim™", ["valheim"]),
    ],
)
def test_slug_candidates(name, expected):
    assert wiki.slug_candidates(name) == expected


@pytest.mark.parametrize(
    ("sitename", "game", "ok"),
    [
        ("Valheim Wiki", "Valheim", True),
        ("Cyberpunk Wiki", "Cyberpunk 2077", True),
        ("Hades Wiki", "Hades II", True),
        ("Elden Ring Wiki", "ELDEN RING", True),
        ("Rust Wiki", "Rusty Lake Paradise", False),
        ("Minecraft Wiki", "Valheim", False),
    ],
)
def test_sitename_matches_game(sitename, game, ok):
    assert wiki.sitename_matches(sitename, game) is ok


# ----------------------------------------------------------------------------- wikitext
WIKITEXT = """{{Infobox item|name=Workbench|type=Crafting station|size=2x2}}
[[File:Workbench.png|thumb|A workbench]]
'''Workbenches''' are used for [[crafting]] and [[Repair|repairing]] items.

== Recipe ==
Needs {{Item link|Wood|10}} to build.

=== Upgrades ===
* [[Chopping Block]]
* [[Tanning Rack]]
"""


def test_wikitext_to_sections_splits_on_headings_and_renders_templates():
    sections = wiki.wikitext_to_sections(WIKITEXT)

    assert [s.heading for s in sections] == ["", "Recipe", "Recipe > Upgrades"]
    intro, recipe, upgrades = (s.text for s in sections)
    assert "type: Crafting station" in intro
    assert "Workbenches are used for crafting and repairing items." in intro
    assert "thumb" not in intro and "File:" not in intro
    assert "Needs Wood 10 to build." in recipe
    assert "Chopping Block" in upgrades and "Tanning Rack" in upgrades


def test_wikitext_to_sections_drops_empty_sections():
    sections = wiki.wikitext_to_sections("== Gallery ==\n[[File:a.png]]\n== Lore ==\nOld text.")
    assert [s.heading for s in sections] == ["Lore"]


# ----------------------------------------------------------------------------- API client
def _api(handler_by_slug):
    def handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.host.split(".")[0]
        if slug not in handler_by_slug:
            return httpx.Response(404)
        return handler_by_slug[slug](dict(request.url.params))

    return httpx.Client(transport=httpx.MockTransport(handler))


def _siteinfo(sitename, articles=500, lang="en"):
    return lambda params: httpx.Response(
        200,
        json={
            "query": {
                "general": {"sitename": sitename, "lang": lang},
                "statistics": {"articles": articles},
            }
        },
    )


def test_resolve_wiki_tries_candidates_and_validates_sitename():
    client = _api({"hades": _siteinfo("Hades Wiki")})
    found = wiki.FandomClient(client, rate_per_sec=0, backoff_s=0).resolve("Hades II")
    assert found is not None and found.slug == "hades" and found.sitename == "Hades Wiki"


def test_resolve_wiki_rejects_unrelated_or_non_english_wiki():
    client = _api(
        {
            "darktide": _siteinfo("Minecraft Wiki"),
            "cyberpunk": _siteinfo("Cyberpunk Wiki", lang="de"),
        }
    )
    fc = wiki.FandomClient(client, rate_per_sec=0, backoff_s=0)
    assert fc.resolve("Warhammer 40,000: Darktide") is None
    assert fc.resolve("Cyberpunk 2077") is None


def test_largest_pages_and_fetch_wikitext():
    def handler(params):
        if params.get("generator") == "allpages":
            pages = [
                {"pageid": 1, "title": "Crafting", "length": 32000},
                {"pageid": 2, "title": "Stub", "length": 120},
                {"pageid": 3, "title": "Workbench", "length": 6000},
            ]
            return httpx.Response(200, json={"query": {"pages": pages}})
        titles = params["titles"].split("|")
        pages = [
            {
                "pageid": i,
                "title": t,
                "revisions": [{"slots": {"main": {"content": f"Text of {t}"}}}],
            }
            for i, t in enumerate(titles)
        ]
        return httpx.Response(200, json={"query": {"pages": pages}})

    fc = wiki.FandomClient(_api({"valheim": handler}), rate_per_sec=0, backoff_s=0)
    top = fc.largest_pages("valheim", n=2, min_length=500)
    assert [p["title"] for p in top] == ["Crafting", "Workbench"]
    texts = fc.fetch_wikitext("valheim", [p["title"] for p in top])
    assert texts == {"Crafting": "Text of Crafting", "Workbench": "Text of Workbench"}


def test_download_wikis_writes_jsonl_and_map(tmp_path):
    def valheim(params):
        if params.get("meta") == "siteinfo":
            return _siteinfo("Valheim Wiki")(params)
        if params.get("generator") == "allpages":
            pages = [{"pageid": 7, "title": "Crafting", "length": 9000}]
            return httpx.Response(200, json={"query": {"pages": pages}})
        content = "== Basics ==\nCombine materials."
        page = {
            "pageid": 7,
            "title": "Crafting",
            "revisions": [{"slots": {"main": {"content": content}}}],
        }
        return httpx.Response(200, json={"query": {"pages": [page]}})

    fc = wiki.FandomClient(_api({"valheim": valheim}), rate_per_sec=0, backoff_s=0)
    games = [(892970, "Valheim"), (999, "Unknown Indie Game")]

    out = wiki.download_wikis(tmp_path, games, fc, max_pages=10)

    mapping = json.loads((out / "wiki_map.json").read_text())
    assert mapping == [{"slug": "valheim", "sitename": "Valheim Wiki", "appids": [892970]}]
    rows = [json.loads(line) for line in (out / "valheim.jsonl").read_text().splitlines()]
    assert rows == [
        {
            "slug": "valheim",
            "pageid": 7,
            "title": "Crafting",
            "url": "https://valheim.fandom.com/wiki/Crafting",
            "sections": [{"heading": "Basics", "text": "Combine materials."}],
        }
    ]


def test_build_wiki_one_row_per_section_limited_to_subset(tmp_path):
    from steam_game_analyst.data import transform

    (tmp_path / "wiki_map.json").write_text(
        json.dumps(
            [
                {"slug": "hades", "sitename": "Hades Wiki", "appids": [1145350, 1145360]},
                {"slug": "oldgame", "sitename": "Old Wiki", "appids": [42]},
            ]
        )
    )
    page = {
        "slug": "hades",
        "pageid": 3,
        "title": "Boons",
        "url": "https://hades.fandom.com/wiki/Boons",
        "sections": [
            {"heading": "", "text": "Boons are gifts."},
            {"heading": "Zeus", "text": "Lightning."},
        ],
    }
    (tmp_path / "hades.jsonl").write_text(json.dumps(page) + "\n")
    (tmp_path / "oldgame.jsonl").write_text(json.dumps({**page, "slug": "oldgame"}) + "\n")

    df = transform.build_wiki(tmp_path, appids=[1145350, 1145360, 892970])

    assert df["slug"].unique().to_list() == ["hades"]
    assert df["heading"].to_list() == ["", "Zeus"]
    assert df["appids"].to_list() == [[1145350, 1145360]] * 2
    assert df["section_id"].n_unique() == 2
    assert df["n_chars"].to_list() == [16, 10]


def test_build_wiki_gives_repeated_headings_distinct_ids_and_drops_exact_copies(tmp_path):
    from steam_game_analyst.data import clean, transform

    (tmp_path / "wiki_map.json").write_text(
        json.dumps([{"slug": "loop", "sitename": "Loop Wiki", "appids": [1282730]}])
    )
    page = {
        "slug": "loop",
        "pageid": 7,
        "title": "Sand Dunes",
        "url": "https://loophero.fandom.com/wiki/Sand_Dunes",
        "sections": [
            {"heading": "Notes", "text": "8 cards per deck."},
            {"heading": "Notes", "text": "Lowers max HP."},
            {"heading": "Notes", "text": "8 cards per deck."},  # exact copy
        ],
    }
    (tmp_path / "loop.jsonl").write_text(json.dumps(page) + "\n")

    df = transform.build_wiki(tmp_path, appids=[1282730])

    assert df["text"].to_list() == ["8 cards per deck.", "Lowers max HP."]
    assert df["section_id"].n_unique() == 2
    # the first occurrence keeps the id it had before repeats were numbered
    assert df["section_id"][0] == clean.stable_id("loop", 7, "Notes")


def test_resolve_treats_closed_wiki_as_missing_and_retries_rate_limit():
    calls = {"n": 0}

    def limited(params):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return _siteinfo("Red Dead Redemption Wiki")(params)

    def handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.host.split(".")[0]
        if slug == "reddeadredemption2":
            return httpx.Response(410)
        if slug == "reddeadredemption":
            return limited(dict(request.url.params))
        return httpx.Response(404)

    fc = wiki.FandomClient(
        httpx.Client(transport=httpx.MockTransport(handler)), rate_per_sec=0, backoff_s=0
    )
    found = fc.resolve("Red Dead Redemption 2")
    assert found is not None and found.slug == "reddeadredemption"
    assert calls["n"] == 2


def test_resolution_cache_keeps_good_games_when_one_fails(tmp_path):
    class Boom(wiki.FandomClient):
        def resolve(self, game_name):
            if game_name == "Bad":
                raise RuntimeError("network down")
            return wiki.WikiInfo("good", "Good Wiki", 100)

    fc = Boom(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
        rate_per_sec=0,
        backoff_s=0,
    )
    wiki.download_wikis(tmp_path, [(1, "Good"), (2, "Bad")], fc)  # "Bad" skipped, not cached

    cache = json.loads((tmp_path / "wiki" / "resolved.json").read_text())
    assert cache == {"1": {"slug": "good", "sitename": "Good Wiki", "articles": 100}}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_forbidden_is_an_error_not_a_missing_wiki():
    fc = wiki.FandomClient(
        _client(lambda r: httpx.Response(403)), rate_per_sec=0, backoff_s=0, retries=2
    )
    with pytest.raises(RuntimeError):
        fc.site_info("valheim")


def test_html_or_redirected_reply_means_no_wiki():
    def handler(request):
        return httpx.Response(200, text="<html>Not a valid wiki</html>")

    assert (
        wiki.FandomClient(_client(handler), rate_per_sec=0, backoff_s=0).site_info("valheim")
        is None
    )


def test_transient_failure_is_not_cached_as_no_wiki(tmp_path):
    class Flaky(wiki.FandomClient):
        def resolve(self, game_name):
            raise RuntimeError("fandom: giving up")

    fc = Flaky(_client(lambda r: httpx.Response(404)), rate_per_sec=0, backoff_s=0)
    wiki.download_wikis(tmp_path, [(1, "Valheim")], fc)
    cache = json.loads((tmp_path / "wiki" / "resolved.json").read_text())
    assert "1" not in cache  # retried next run


def test_bad_page_is_skipped_and_failed_batch_writes_no_file(tmp_path, monkeypatch):
    def handler(request):
        p = dict(request.url.params)
        if p.get("meta") == "siteinfo":
            return _siteinfo("Valheim Wiki")(p)
        if p.get("generator") == "allpages":
            pages = [
                {"pageid": 1, "title": "Good", "length": 9000},
                {"pageid": 2, "title": "Bad", "length": 9000},
            ]
            return httpx.Response(200, json={"query": {"pages": pages}})
        pages = [
            {
                "pageid": i,
                "title": t,
                "revisions": [{"slots": {"main": {"content": f"== S ==\n{t} text"}}}],
            }
            for i, t in enumerate(p["titles"].split("|"))
        ]
        return httpx.Response(200, json={"query": {"pages": pages}})

    real = wiki.wikitext_to_sections

    def flaky_parse(text):
        if "Bad" in text:
            raise RecursionError("deep templates")
        return real(text)

    monkeypatch.setattr(wiki, "wikitext_to_sections", flaky_parse)
    fc = wiki.FandomClient(_client(handler), rate_per_sec=0, backoff_s=0)
    wiki.download_wikis(tmp_path, [(892970, "Valheim")], fc)
    rows = (tmp_path / "wiki" / "valheim.jsonl").read_text().splitlines()
    assert [json.loads(r)["title"] for r in rows] == ["Good"]

    def failing_batch(request):
        p = dict(request.url.params)
        if "titles" in p:
            return httpx.Response(503)
        return handler(request)

    (tmp_path / "wiki" / "valheim.jsonl").unlink()
    fc = wiki.FandomClient(_client(failing_batch), rate_per_sec=0, backoff_s=0, retries=1)
    wiki.download_wikis(tmp_path, [(892970, "Valheim")], fc)
    assert not (tmp_path / "wiki" / "valheim.jsonl").exists()


def test_build_wiki_survives_unicode_line_separators(tmp_path):
    from steam_game_analyst.data import transform

    (tmp_path / "wiki_map.json").write_text(
        json.dumps([{"slug": "w", "sitename": "W Wiki", "appids": [1]}])
    )
    page = {
        "slug": "w",
        "pageid": 1,
        "title": "T",
        "url": "u",
        "sections": [{"heading": "", "text": "a b"}],
    }
    (tmp_path / "w.jsonl").write_text(json.dumps(page, ensure_ascii=False) + "\n", encoding="utf-8")
    assert transform.build_wiki(tmp_path, appids=[1])["text"].to_list() == ["a b"]


# ----------------------------------------------------------------------------- per-game categories
@pytest.mark.parametrize(
    "category, spec, ok",
    [
        ("Battlefield 1", wiki.CategorySpec(("Battlefield 1",)), True),
        ("Battlefield 1: Turning Tides", wiki.CategorySpec(("Battlefield 1",)), True),
        ("Battlefield 1942", wiki.CategorySpec(("Battlefield 1",)), False),
        ("Little Nightmares III", wiki.CategorySpec(("Little Nightmares II",)), False),
        ("Wasteland 3 weapons", wiki.CategorySpec(("Wasteland 3",)), True),
        ("Hades II Boons", wiki.CategorySpec(("Hades",), exclude=("Hades II",)), False),
        ("Hades I Boons", wiki.CategorySpec(("Hades",), exclude=("Hades II",)), True),
        ("Hades Wiki", wiki.CategorySpec(("Hades",), exclude=("Hades II",)), False),
    ],
)
def test_category_matches_whole_words_and_skips_excluded_and_meta(category, spec, ok):
    assert wiki.category_matches(category, spec) is ok


def _hades_wiki(params):
    """Mock franchise wiki: Hades and Hades II pages, one page in both games' categories."""
    if params.get("meta") == "siteinfo":
        return _siteinfo("Hades Wiki")(params)
    if params.get("list") == "allcategories":
        names = ["Hades", "Hades I Boons", "Hades II", "Hades II Boons", "Hades Wiki"]
        cats = [{"category": n} for n in names if n.startswith(params["acprefix"])]
        return httpx.Response(200, json={"query": {"allcategories": cats}})
    if params.get("list") == "categorymembers":  # subcategories
        subs = {"Category:Hades": ["Category:Characters", "Category:Hades II"]}
        found = [{"title": t} for t in subs.get(params["cmtitle"], [])]
        return httpx.Response(200, json={"query": {"categorymembers": found}})
    if params.get("generator") == "allpages":
        pages = [{"pageid": 99, "title": "Wiki rules", "length": 9999}]
        return httpx.Response(200, json={"query": {"pages": pages}})
    if params.get("generator") == "categorymembers":
        members = {
            "Category:Hades": [(1, "Zagreus", 9000), (3, "Nyx", 5000)],
            "Category:Hades I Boons": [(2, "Zeus (Hades)", 7000)],
            "Category:Characters": [
                (4, "Hypnos", 6000),
                (13, "Hecate", 6500),
                (14, "Boons/Hades II", 9500),
            ],
            "Category:Hades II": [(10, "Melinoë", 8000), (3, "Nyx", 5000), (13, "Hecate", 6500)],
            "Category:Hades II Boons": [(11, "Zeus (Hades II)", 6000), (12, "Stub", 100)],
            "Category:Hades Wiki": [(99, "Wiki rules", 9999)],
        }[params["gcmtitle"]]
        pages = [{"pageid": i, "title": t, "length": n} for i, t, n in members]
        return httpx.Response(200, json={"query": {"pages": pages}})
    titles = params["titles"].split("|")
    pages = [
        {"pageid": 0, "title": t, "revisions": [{"slots": {"main": {"content": f"About {t}."}}}]}
        for t in titles
    ]
    return httpx.Response(200, json={"query": {"pages": pages}})


HADES_SPECS = {
    1145360: wiki.CategorySpec(("Hades",), exclude=("Hades II",)),
    1145350: wiki.CategorySpec(("Hades II",)),
}


def test_franchise_wiki_pages_are_tagged_with_the_games_whose_categories_hold_them(tmp_path):
    """Hades takes its prefixed categories plus their direct subcategories ("Characters"),
    minus every page filed under the excluded "Hades II" (Hecate, and Nyx, who is in both) or
    titled after it ("Boons/Hades II")."""
    fc = wiki.FandomClient(_api({"hades": _hades_wiki}), rate_per_sec=0, backoff_s=0)
    games = [(1145360, "Hades"), (1145350, "Hades II")]

    out = wiki.download_wikis(
        tmp_path, games, fc, max_pages=10, min_length=500, categories=HADES_SPECS
    )

    rows = {r["title"]: r for r in map(json.loads, (out / "hades.jsonl").read_text().splitlines())}
    assert set(rows) == {
        "Zagreus",
        "Zeus (Hades)",
        "Nyx",
        "Hypnos",
        "Hecate",
        "Melinoë",
        "Zeus (Hades II)",
    }
    assert rows["Zagreus"]["appids"] == [1145360]
    assert rows["Hypnos"]["appids"] == [1145360]
    assert rows["Melinoë"]["appids"] == [1145350]
    assert rows["Nyx"]["appids"] == [1145350]
    assert rows["Hecate"]["appids"] == [1145350]
    assert "Boons/Hades II" not in rows
    mapping = json.loads((out / "wiki_map.json").read_text())
    assert mapping == [{"slug": "hades", "sitename": "Hades Wiki", "appids": [1145350, 1145360]}]


def test_game_without_a_category_on_a_franchise_wiki_gets_no_pages(tmp_path):
    fc = wiki.FandomClient(_api({"hades": _hades_wiki}), rate_per_sec=0, backoff_s=0)
    games = [(1145360, "Hades"), (1145350, "Hades II")]
    specs = {1145350: HADES_SPECS[1145350]}

    out = wiki.download_wikis(tmp_path, games, fc, max_pages=10, min_length=500, categories=specs)

    rows = [json.loads(line) for line in (out / "hades.jsonl").read_text().splitlines()]
    assert all(r["appids"] == [1145350] for r in rows)
    mapping = json.loads((out / "wiki_map.json").read_text())
    assert mapping[0]["appids"] == [1145350]


def test_changed_page_selection_refetches_an_existing_wiki_file(tmp_path):
    fc = wiki.FandomClient(_api({"hades": _hades_wiki}), rate_per_sec=0, backoff_s=0)
    games = [(1145360, "Hades"), (1145350, "Hades II")]
    wiki.download_wikis(tmp_path, games, fc, max_pages=10, min_length=500)  # old: largest pages
    stale = (tmp_path / "wiki" / "hades.jsonl").read_text()

    wiki.download_wikis(tmp_path, games, fc, max_pages=10, min_length=500, categories=HADES_SPECS)

    fresh = (tmp_path / "wiki" / "hades.jsonl").read_text()
    assert fresh != stale and '"appids"' in fresh


def test_build_wiki_uses_page_level_appids_when_present(tmp_path):
    from steam_game_analyst.data import transform

    (tmp_path / "wiki_map.json").write_text(
        json.dumps([{"slug": "hades", "sitename": "Hades Wiki", "appids": [1145350, 1145360]}])
    )
    pages = [
        {
            "slug": "hades",
            "pageid": 1,
            "title": "Zagreus",
            "url": "u1",
            "appids": [1145360],
            "sections": [{"heading": "", "text": "Prince of the Underworld."}],
        },
        {
            "slug": "hades",
            "pageid": 10,
            "title": "Melinoë",
            "url": "u2",
            "appids": [1145350],
            "sections": [{"heading": "", "text": "Princess of the Underworld."}],
        },
    ]
    (tmp_path / "hades.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pages))

    df = transform.build_wiki(tmp_path, appids=[1145360])

    assert df["title"].to_list() == ["Zagreus"]
    assert df["appids"].to_list() == [[1145360]]


def test_load_category_specs_accepts_lists_and_mappings_and_rejects_empty():
    specs = wiki.load_category_specs(
        {"397540": ["Borderlands 3"], 1145360: {"prefixes": ["Hades"], "exclude": ["Hades II"]}}
    )
    assert specs == {
        397540: wiki.CategorySpec(("Borderlands 3",)),
        1145360: wiki.CategorySpec(("Hades",), ("Hades II",)),
    }
    with pytest.raises(ValueError):
        wiki.load_category_specs({1: []})
    with pytest.raises(ValueError):
        wiki.load_category_specs({1: {"prefixes": "Hades"}})
