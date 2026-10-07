# Steam Game Analyst

A fully local LLM agent that answers analytical questions about PC games by combining
retrieval over Steam reviews, patch notes and wiki pages with data-science tools
(SQL over the catalog, a review sentiment classifier, a reception predictor, trend tests).
Runs on one Linux box with an NVIDIA GPU via Docker Compose. No paid API.

> Status: **W3 — ML tools + agent loop (MVP)**. Data pipeline (W1), dense retrieval with
> metadata filters and cited answers (W2), and the LangGraph agent with four data-science
> tools (W3) work end to end; hybrid retrieval, DistilBERT and the gold set (W4), end-to-end
> evaluation and the Streamlit UI (W5) follow. See `docs/PRD.md` for the plan.

## Stack

| Layer | Choice |
| --- | --- |
| Storage | DuckDB over Parquet (`data/processed/`) |
| LLM | Qwen 2.5 14B / 7B Instruct via Ollama (GPU) |
| Embeddings / reranker | `BAAI/bge-m3`, `BAAI/bge-reranker-base` (sentence-transformers) |
| Vector store | Qdrant |
| Agent | LangGraph |
| ML | scikit-learn, LightGBM + SHAP, DistilBERT |
| UI | Streamlit |

## Data sources (all free; join key `appid`)

| Source | Role | License |
| --- | --- | --- |
| [FronkonGames Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) | catalog, structured features | MIT |
| [Steam Games Metadata and Player Reviews 2020–2024 (Mendeley)](https://data.mendeley.com/datasets/jxy85cr3th/2) | 31M reviews | see dataset page |
| [Steam Web API — GetNewsForApp](https://steamcommunity.com/dev/apiterms) | patch notes / announcements | free key, attribution |
| [Fandom](https://www.fandom.com) wikis, via each wiki's MediaWiki API | wiki pages (largest 300 per game wiki) | CC BY-SA 3.0 |

Raw data is downloaded at build time and never committed. Author/Steam ids are dropped
during cleaning; only derived artefacts (aggregates, embeddings, models) are published.

Coverage, sampling choices, known gaps and privacy: [docs/DATA_CARD.md](docs/DATA_CARD.md).
Exploratory analysis: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

## Quick start (host with Docker + NVIDIA Container Toolkit)

```bash
cp .env.example .env            # add your STEAM_API_KEY
make build                      # build the app/pipeline image
make up                         # ollama + qdrant + app (http://localhost:8501)
make pull-models                # pull the LLMs into the ollama volume
make data                       # download sources, select the subset, build Parquet + DuckDB
docker compose run --rm pipeline data stats
make index                      # chunk + embed (bge-m3, GPU) into Qdrant
docker compose run --rm pipeline ask "How does the crafting system work in Valheim?"
```

GPUs are requested through CDI (`nvidia.com/gpu=all`), so after installing the NVIDIA
Container Toolkit generate the spec once (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`);
no Docker restart is needed.

`make data` downloads the catalog (~hundreds of MB) and the Mendeley review archive
(several GB), then builds:

```
data/processed/
  games_all.parquet      full catalog, canonical columns
  review_counts.parquet  appid -> number of reviews available
  games.parquet          the project subset (configs/pipeline.yaml: 300 games, 2020-2024, genre-balanced)
  reviews.parquet        cleaned + stratified sample (1,500/game, all languages tagged with
                         lingua, no author ids)
  news.parquet           patch notes / announcements for the subset
  wiki.parquet           Fandom wiki sections for the subset (configs/wikis.yaml fixes bad matches)
  review_volume.parquet  monthly review / positive counts over every cleaned review
                         (the population behind the stratified sample)
  sga.duckdb             views over the parquet files + review_monthly + game_summary
```

Tune the subset with `SGA_SUBSET_GAMES` / `SGA_REVIEWS_PER_GAME` in `.env` or edit
`configs/pipeline.yaml`, then `make subset`.

## Retrieval and answers (W2)

`sga index build` chunks reviews (one per review), patch notes/news (by paragraph) and wiki
pages (by section), embeds them with `bge-m3` and loads one Qdrant collection whose payload
carries the filterable fields (game, doc type, date, language, recommended, playtime).
Each chunk also gets a BM25 sparse vector, computed on the CPU (Qdrant supplies the IDF);
search fuses the dense and BM25 rankings with RRF, so exact names, version numbers and
jargon ("DLSS", "2.0") are found even when the embedding misses them. Collections built
before W4 lack these vectors: rebuild with `sga index build --recreate`.
`sga ask` matches game names in the question, parses simple filters ("50+ hours",
"mentioning X", "since 2023"; flags `--game`, `--doc-type`, `--min-hours`, `--language`
override), retrieves, and answers with Qwen 2.5 14B via Ollama with numbered citations.
`--no-llm` shows the retrieved sources only.

On a 10 GB card the 14B model (~9 GB) and bge-m3 do not fit together: indexing embeds on
the GPU (stop Ollama's models first), query-time embedding runs on the CPU.

`sga eval synth` starts the retrieval gold set: one LLM-written question per sampled chunk,
each assigned a frozen `dev`/`test` split from a hash of its id. Tune only on `dev`; report
`test` once. `sga eval retrieval --mode hybrid|dense|bm25` compares the
retrieval legs on the same questions.

## ML tools and agent (W3)

```bash
uv run sga train sentiment      # or run notebooks/02_sentiment.ipynb  -> models/
uv run sga train reception      # or run notebooks/03_reception.ipynb  -> models/
uv run sga agent "Did performance complaints drop after patch 2.0 of Cyberpunk 2077?"
uv run sga agent --no-retrieval "Which roguelikes released in 2024 have the best review ratio among games under €20?"
```

`sga agent` runs a LangGraph state graph: **router** (JSON plan, scope check, refusal path)
→ **retrieve** (the W2 retriever, filtered to the games named in the question) → **agent**
(Qwen 2.5 via `ChatOllama` with the four tools bound) ⇄ **tools** (pydantic-validated calls;
invalid arguments, refused SQL and unseen appids go back to the model on the retry edge;
at most 6 rounds and 5 calls per round) → **answer** (cites tool results `[T1]` and documents
`[1]`; uncited answers are rewritten once). Every run is appended to `data/traces/agent.jsonl`.

| Tool | What it does |
| --- | --- |
| `query_catalog(sql)` | one read-only SELECT over allow-listed views (`n_votes` and a Wilson lower bound `positive_ratio_lower95` added for ranking); parser check (no table functions, no qualified names) plus a sandboxed DuckDB connection (files outside `data/processed/` blocked, config locked, 1 GB memory), 200-row and 500-char cell caps, 20 s timeout, "did you mean" tag names on empty results |
| `review_sentiment(appid, from, to, split_date?)` | weighted positive share and aspect mentions / complaints (performance, bugs, gameplay, story, price, controls) on the English sample, reweighted to the population; before/after test per aspect |
| `predict_reception(features)` | bucket probabilities + TreeSHAP top features for a hypothetical launch |
| `review_trend(appid, granularity, event_date?)` | monthly / quarterly volume and positive share from population counts, WLS trend, change point or before/after test |

Held-out results (`models/*.metrics.json`):

| Model | Split | Metric | Model | Majority baseline |
| --- | --- | --- | --- | --- |
| Sentiment, TF-IDF + LR | 55 unseen games, 38.4k English reviews | F1 (positive) / macro-F1 / ROC-AUC | 0.884 / 0.872 / 0.944 | 0.700 / 0.350 / 0.5 |
| Sentiment, DistilBERT (fine-tuned, 2 epochs) | same 55 games, same reviews | F1 (positive) / macro-F1 / ROC-AUC | 0.913 [0.904, 0.923] / 0.905 / 0.968 | 0.700 / 0.350 / 0.5 |
| Aspects, keyword lexicon | 300 reviews of held-out games, gold labels | macro-F1 | 0.730 [0.667, 0.781] | — |
| Aspects, DistilBERT (qwen2.5 14B silver labels, 12k reviews) | same 300 reviews | macro-F1 | 0.736 [0.681, 0.780] | — |
| Aspects, DistilBERT (Claude labels, 3k reviews, 10 epochs) | same 300 reviews | macro-F1 | 0.707 [0.652, 0.751] | — |
| Reception, LightGBM | released 2023-01 to 2024-05, ≥ 12 months of reviews (3,951 games, ≥ 50 votes) | accuracy [95% CI] / macro-F1 / log loss / ECE | 0.661 [0.647, 0.674] / 0.428 / 0.791 / 0.007 | 0.645 / 0.261 / 0.950 / 0.145 |

The aspect model ties the lexicon (+0.006 macro-F1, 95% CI [−0.057, +0.068]): it finds far more
gameplay talk (F1 0.82 vs 0.57) but is worse on performance and story. Its teacher caps it: the
qwen2.5 14B silver labels themselves score 0.743 against the gold set, over-tagging controls,
story and performance. A blind human check of 35 gold reviews agreed with the gold labels at
macro-F1 0.79, which is about as close as two careful readers get.
Retraining on 3,000 Claude-labelled reviews (cleaner: qwen's controls and performance tags
agree with Claude's only 45-48% of the time) scored lower, 0.707: a quarter of the data left
about 150 examples each for performance and controls. Three configurations have now been
scored on the gold set, so no more are tried against it; the review_sentiment tool keeps the
lexicon.

DistilBERT beats TF-IDF + LR by +0.029 F1 (95% CI [+0.026, +0.033], resampling held-out
games) and scores higher on 53 of the 55 test games; its hyperparameters were fixed in
advance and only the epoch was picked on validation games.
`review_sentiment` uses it (`SGA_SENTIMENT_MODEL=tfidf` switches back). Live on the CPU it
takes 15-20 s per call, so `sga train sentiment-scores` scores every review and aspect sentence
once on the GPU (342k texts, a few minutes) and the tool looks them up (0.1 s); text missing
from the table is scored live.

The reception model beats always predicting "very positive" by **+1.6 points** of bucket
accuracy (95% CI [+0.6, +2.8]) on games released from January 2023 to May 2024, and gives much
better probabilities (log loss 0.79 vs 0.95, calibration error 0.007 vs 0.145). The test is
limited to games with at least a year of reviews before the catalog snapshot (about June 2025:
vote counts collapse for later releases), because early ratings run high: very positive games
are 67% of all 2023+ games but 64.5% of the mature ones. On the previous definition (every
2023+ game) the gain was +0.7 [−0.1, +1.4]. The maturity rule was adopted after that result,
so this test number is not fully blind. Before it, five variants were compared on the 2022
validation year only (recency weighting, rescaling to the latest year's class mix, dropping
release-date features, developer/publisher track-record shape); none beat the existing model
(+3.7 points there), so the model itself is unchanged. Its strongest features are the
developer's and publisher's earlier ratings, release year and price. Catalog fields are
snapshot values, not the exact launch state, so scores may be slightly optimistic (see
`caveats` in the metrics file).

Known limits: the review corpus ends in December 2024 and has no Starfield or RDR2 reviews,
so U7 gets an honest "no data" answer; aspects are keyword weak labels (DistilBERT and a
hand-labelled check come in W4); aspect tests are underpowered in short windows (~1,500
sampled reviews per game across all years); the agent sometimes searches the 300-game
subset where the full catalog was meant; the 14B model partly spills out of 10 GB VRAM, so
answers take 10–90 s.

## Local development (no Docker)

```bash
uv sync --all-extras          # needs Python 3.11/3.12 (uv will fetch one)
uv run pytest                 # unit tests on a synthetic fixture, no network
uv run ruff check src tests
uv run sga --help
```

## Layout

```
src/steam_game_analyst/
  config.py        settings (env SGA_*) + configs/pipeline.yaml
  cli.py           `sga data|index|train|ask|agent|eval`
  data/            W1: download, clean, transform, subset, build (DuckDB)
  retrieval/       W2: chunking, embeddings, Qdrant, hybrid + rerank
  ml/              sentiment (TF-IDF + LR, DistilBERT), keyword and DistilBERT aspects, reception (LightGBM + SHAP)
  tools/           W3: query_catalog, review_sentiment, predict_reception, review_trend
  agent/           W3: LangGraph graph (router, retrieve, agent, tools, answer)
  eval/            W4-5: gold set, retrieval + answer metrics
  ui/              W5: Streamlit app
configs/           pipeline.yaml
docker/            app.Dockerfile
notebooks/         EDA and model notebooks (they write models/)
models/            trained models + metrics JSON (not committed)
tests/             pytest (fixture-based)
```

## License

Code: MIT. Data: see each source above.
