# Data card — Steam Game Analyst

As of 2026-10-06 (`make data` + `sga data download-wiki`, configs in `configs/`).
Raw data is downloaded at build time and never committed; this card describes
`data/processed/` and the Qdrant index built from it.

## Sources

| Source | Used for | Licence / terms |
| --- | --- | --- |
| [FronkonGames Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) (`games.csv`) | catalog: 125,855 apps, price, genres, tags, positive/negative counts, owners, playtime | MIT |
| [Steam Games Metadata and Player Reviews 2020–2024](https://data.mendeley.com/datasets/jxy85cr3th/2) (Mendeley, v2) | review text: one CSV per app (`user, playtime, post_date, helpfulness, review, recommend, early_access_review`) | see dataset page |
| [Steam Web API `GetNewsForApp`](https://steamcommunity.com/dev/apiterms) | patch notes and announcements for the subset | free, attribution |
| [Fandom](https://www.fandom.com) wikis, via each wiki's MediaWiki API | long-form game documentation | CC BY-SA 3.0 |

## Tables (`data/processed/`, DuckDB views in `sga.duckdb`)

| Table | Rows | Notes |
| --- | --- | --- |
| `games_all` | 125,855 | full catalog, canonical columns |
| `games` | 300 | the project subset (below) |
| `reviews` | 417,170 for 299 games | sampled, cleaned, language-tagged |
| `news` | 24,970 items for 295 games (11,044 flagged patch notes) | 2015-02 to 2026-10 |
| `wiki` | 156,356 sections from 10,077 pages in 133 wikis (136 games) | ~152 M characters |
| `review_volume` | 11,364 game-months (6.52 M cleaned reviews) | monthly review / positive counts over **every** cleaned review of each subset game (the population the sample is drawn from) |
| `review_monthly`, `game_summary` | derived | `review_monthly` = `review_volume` + `n_sampled`; for the trend and catalog tools |

Index (Qdrant `sga_chunks`, bge-m3 dense + BM25 sparse vectors, 448-token sentence-packed
chunks): 434,297 review, 61,737 news and 217,046 wiki chunks (2026-10-07).

## Subset selection (`configs/pipeline.yaml`)

- Released 2020-01-01 to 2024-12-31, at least 500 reviews available (any language), at
  least one genre and no non-game software genre (Utilities, Animation & Modeling, …).
- At most 40 games per primary (first-listed) genre, most-reviewed first.
- `must_include` (11 titles with rich patch histories or named in the PRD use cases) wins
  over every rule: Red Dead Redemption 2 (released 2019) and Hades II (catalog release
  2025-09-25; early-access reviews from 2024) sit outside the release window.
- Result: 300 games; primary genres Action 39, Casual 32, Adventure 32, RPG 31, Racing 30,
  Indie 30, Strategy 30, Simulation 30, Free To Play 24, Massively Multiplayer 16, Sports 4,
  Early Access 2. Median price $7.49; 83 free.

## Reviews

- Sample: up to 1,500 reviews per game (median 1,500, minimum 341), stratified by
  recommended / not recommended and by year, so the sample is **balanced on purpose**:
  57.4% recommended in the sample versus a 88.9% median positive ratio in the catalog
  (games with votes).
  Because the strata are per year, the sample's yearly positive share is flat by
  construction (Cyberpunk 2077: 47.8% in every year 2020–2023, versus 62% → 85% in the full
  files). Use catalog `positive` / `negative` for all-time reception, `review_monthly`
  (population counts) for trends, and the sample only for text, reweighted by
  N(game, year, rating) / n(game, year, rating) — what the `review_sentiment` tool does.
- Dates 2015-01 to 2024-12 (91% in 2021–2024): games re-released or long in early access
  carry older reviews.
- **Languages**: the Mendeley files have no language column, so each sampled review is
  tagged with [lingua](https://github.com/pemistahl/lingua-py); texts under 8 letters are
  `unknown`. All languages are kept (bge-m3 is multilingual): English 52.7%, Chinese 18.0%,
  Russian 6.0%, Spanish 4.0%, German 3.1%, Portuguese 2.9%, French 2.8%, Turkish 2.3%,
  Korean 2.1%, Polish 1.4%.
- `playtime_forever_min` is Mendeley's hours on record at scrape time × 60, not playtime at
  review time. `votes_up` is Mendeley's `helpfulness`. 19.5% are early-access reviews.
- Cleaning: HTML stripped, 20–4,000 characters, duplicates per game dropped.

## Wiki

- Each game's `<slug>.fandom.com` is guessed from its name and kept when the wiki's site
  name matches the game; the 300 largest main-namespace pages (≥ 1,500 bytes) are fetched
  and rendered from wikitext to sections.
- 12 wrong matches (TV series, real-world or story wikis sharing a generic word, film
  franchise wikis) were found by sampling page titles and are pinned to "no wiki" in
  `configs/wikis.yaml`. Single-word game names remain the weakest point of the matching.
- Franchise wikis (Borderlands, XCOM, Resident Evil, Total War, Hades, ...) also document
  games outside the subset. Taking their largest pages filed Borderlands 4, XCOM 2 or RE6
  pages under Borderlands 3, Chimera Squad or RE3 (2026-10-07: 7 wikis, hundreds of pages).
  Their games now take only pages from their own categories (`configs/wiki_categories.yaml`:
  category-name prefixes plus direct subcategories; `exclude` drops a sibling's pages), and
  each page carries the games whose categories hold it. Games on a franchise wiki with no
  per-game category (Football Manager, MotoGP, Frostpunk 2, Jurassic World Evolution 2,
  Blood Bowl 3, Crusader Kings III, Uncharted, Total War: WARHAMMER III) get no wiki pages.
- Trade-off: some games now have few pages (Baldur's Gate III 8, Total War: ROME REMASTERED
  18, Age of Wonders 4 18) and pages a wiki files only in generic categories are missed.
  Shared categories still leak a little (Hades II update notes under Hades).

## Known gaps and biases

- 41 subset games (37 free-to-play, e.g. Overwatch 2, THE FINALS, MARVEL SNAP) and
  42,899 catalog apps have **no votes** in the FronkonGames snapshot: `positive`,
  `negative` and `positive_ratio` are null (unknown), not 0. Exclude them when modelling
  reception.
- Starfield and Red Dead Redemption 2 have **no Mendeley reviews**; PRD use case U7
  (Starfield) cannot be answered from this data as is.
- 136 of 300 games have a wiki; wiki coverage favours popular titles.
- 5 games have no Steam news (one app answered HTTP 403).
- The subset over-represents well-reviewed, popular games (most reviews first per genre).
- The catalog's `genres` order decides the primary genre ("Indie", "Free To Play" can win
  over the gameplay genre).

## Privacy

- No author ids or names: the Mendeley `user` column is never read into the tables;
  Steam profile links inside review text are replaced by `[profile link]` in the index.
- Review ids are hashes of (appid, text, date). Review text itself is public Steam content
  and may still contain self-identifying text written by its author.

## Evaluation data

- `data/eval/synthetic_questions.jsonl`: 96 LLM-written questions (Qwen 2.5 7B), one per
  sampled chunk, **not yet hand-checked**. Each has a frozen `dev`/`test` split from a hash
  of its id (50 / 46). Tune on `dev` only; `test` runs once with `--final`.
