# Steam Game Analyst

A fully local LLM agent that answers analytical questions about PC games by combining
retrieval over Steam reviews, patch notes and wiki pages with data-science tools
(SQL over the catalog, a review sentiment classifier, a reception predictor, trend tests).
Runs on one Linux box with an NVIDIA GPU via Docker Compose. No paid API.

> Status: **W2 — index + baseline RAG**. Data pipeline (W1) and dense retrieval with
> metadata filters + a cited-answer CLI (W2) work; ML tools + LangGraph agent (W3),
> evaluation (W4–W5) and the Streamlit UI (W5) follow. See `docs/PRD.md` for the plan.

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
  sga.duckdb             views over the parquet files + review_monthly + game_summary
```

Tune the subset with `SGA_SUBSET_GAMES` / `SGA_REVIEWS_PER_GAME` in `.env` or edit
`configs/pipeline.yaml`, then `make subset`.

## Retrieval and answers (W2)

`sga index build` chunks reviews (one per review), patch notes/news (by paragraph) and wiki
pages (by section), embeds them with `bge-m3` and loads one Qdrant collection whose payload
carries the filterable fields (game, doc type, date, language, recommended, playtime).
`sga ask` matches game names in the question, parses simple filters ("50+ hours",
"mentioning X", "since 2023"; flags `--game`, `--doc-type`, `--min-hours`, `--language`
override), retrieves, and answers with Qwen 2.5 14B via Ollama with numbered citations.
`--no-llm` shows the retrieved sources only.

On a 10 GB card the 14B model (~9 GB) and bge-m3 do not fit together: indexing embeds on
the GPU (stop Ollama's models first), query-time embedding runs on the CPU.

`sga eval synth` starts the retrieval gold set: one LLM-written question per sampled chunk,
each assigned a frozen `dev`/`test` split from a hash of its id. Tune only on `dev`; report
`test` once.

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
  cli.py           `sga data|index|eval`
  data/            W1: download, clean, transform, subset, build (DuckDB)
  retrieval/       W2: chunking, embeddings, Qdrant, hybrid + rerank
  tools/           W3: query_catalog, review_sentiment, predict_reception, review_trend
  agent/           W3: LangGraph graph
  eval/            W4-5: gold set, retrieval + answer metrics
  ui/              W5: Streamlit app
configs/           pipeline.yaml
docker/            app.Dockerfile
notebooks/         EDA and model notebooks
tests/             pytest (fixture-based)
```

## License

Code: MIT. Data: see each source above.
