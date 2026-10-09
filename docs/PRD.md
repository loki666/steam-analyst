<!-- Exported from the Claude Docs plan (the living copy, with its two diagrams):
     https://claude.ai/code/artifact/f42caff0-0638-4c83-a765-4b1c74f52102 -->

# Steam Game Analyst Agent — PRD & Plan

Oct 6, 2026 · @Philippe

## Overview

**Steam Game Analyst** is an open-source agent that answers analytical questions about PC games by combining retrieval over player reviews, patch notes and wiki pages with calls to data-science tools (pandas queries, trained ML models, time-series stats) built on Steam data. It runs two ways at €0: fully local on one desktop (RTX 3080 10 GB VRAM, 32 GB system RAM) with Ollama, or hosted on a free Oracle Cloud ARM VM with a free hosted LLM (Cloudflare Workers AI).

The project exists to demonstrate, in one artifact, the two skill sets Philippe has trained: the IBM data-science track (EDA, feature engineering, supervised models, evaluation) and the *LLM Engineer's Handbook* RAG track (ingestion, chunking, embeddings, vector search, reranking, agentic orchestration, evaluation). The accent is on LLM engineering: the data-science models are exposed as **tools** the agent decides to call, so the LLM is the centre of the system rather than a chat layer on top of a notebook.

Example interaction: *"Did performance complaints about Cyberpunk 2077 drop after patch 2.0?"* → the agent retrieves the patch note, pulls reviews before/after the date, runs the sentiment classifier and a before/after comparison, and answers with cited review excerpts and numbers.

## Goals and non-goals

The project succeeds if a recruiter or hiring manager can open the repo, run the demo, and see both RAG engineering and applied ML working together, with measured quality.

**Goals**

1. A working agent that answers game-analytics questions with cited evidence (reviews, patch notes, wiki) and computed numbers (tool calls).
2. At least two data-science models exposed as agent tools: a review sentiment / aspect classifier and a game-success predictor (positive-review ratio or owner bucket from launch metadata).
3. A RAG evaluation suite with retrieval metrics (recall@k, MRR) and answer quality (LLM-as-judge, faithfulness), with at least one ablation (chunking or hybrid vs dense retrieval).
4. A public GitHub repo where `docker compose up` brings up the whole stack and one command rebuilds the index from raw data, with architecture diagram, eval results and README.
5. A deployed demo that costs nothing to host.
6. Total spend: €0. Every component is open source or free tier.

**Non-goals**

- Covering all 140k Steam games: a curated subset (roughly 300–1,000 games) keeps indexing and demo latency tractable.
- Real-time data: snapshots refreshed manually via the Steam Web API are enough.
- Recommender system or personalised advice: possible later extension, not in scope.
- Fine-tuning an LLM: embeddings and small classifiers are trained; the generator stays an off-the-shelf open model.
- Production hardening (auth, multi-user, scaling).

## Users and use cases

The primary user is a **game-industry analyst** (publisher, studio, market researcher) who wants evidence-backed answers without writing SQL. The secondary audience is the **portfolio reviewer**, who cares about how the system is built and measured.

Each use case names the capability it exercises, so the plan can be checked against it.

| # | Question the agent must answer | Needs |
| --- | --- | --- |
| U1 | "What do players complain about most in *Hades II*?" | RAG over reviews + aspect clustering |
| U2 | "Did performance complaints drop after patch 2.0 of *Cyberpunk 2077*?" | Patch-note retrieval + sentiment tool + before/after stats |
| U3 | "Which roguelikes released in 2024 have the best review ratio among games under €20?" | SQL/pandas tool over catalog |
| U4 | "How does *Palworld* compare with *Valheim* on reviews, playtime and price?" | Multi-entity structured comparison + review summarisation |
| U5 | "A co-op survival game at €25 with these tags launches next month: how is it likely to be received?" | ML prediction tool (success model) with feature explanation |
| U6 | "How does the crafting system work in *Valheim*?" | RAG over wiki pages (long-form documents) |
| U7 | "Are reviews for *Starfield* trending up or down over the last 12 months?" | Time-series aggregation tool + chart |
| U8 | "Show me reviews mentioning Steam Deck for *Elden Ring* from players with 50+ hours." | Metadata-filtered retrieval |

Out of scope questions (e.g. "what should I play tonight?") get a polite refusal that names what the agent can do.

## Data sources

All sources are free, and every table joins on the Steam `appid`. Wiki pages join through the game name resolved to a Wikidata entry (property P1733 = Steam appid) or a fuzzy title match.

| Source | Role | Volume | Key fields | License / terms |
| --- | --- | --- | --- | --- |
| [FronkonGames Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) | Catalog + structured features | \~124k games | appID, name, release\_date, price, genres, tags, positive/negative, estimated\_owners, playtime, peak\_ccu, metacritic\_score, detailed\_description | MIT |
| [Steam Games Metadata and Player Reviews 2020–2024 (Mendeley)](https://data.mendeley.com/datasets/jxy85cr3th/2) | Review text corpus | 31M+ reviews, 23,107 games | review text, rating, helpfulness votes, playtime, one CSV per AppID (no language column: languages detected with lingua) | Academic, Mendeley Data (confirm CC BY 4.0 on page) |
| [Steam Web API – GetNewsForApp](https://steamcommunity.com/dev/apiterms) | Patch notes and news per game | Pulled for the subset | appid, date, title, contents | Free key, 100k calls/day, attribution required |
| [Fandom](https://www.fandom.com) wikis, via each wiki's MediaWiki API | Long-form wiki documents | Largest 300 pages per game wiki (the nyuuzyou/fandom dump has no per-page wiki id, so it cannot be filtered) | title, text, categories, sections | CC BY-SA 3.0 |
| [Steam Dataset 2025](https://github.com/vintagedon/steam-dataset-2025) | Optional: reference schema + 1M reviews with BGE-M3 embeddings | 239k apps | relational dump, pgvector | CC BY 4.0 |
| [SteamSpy API](https://steamspy.com/api.php) | Optional refresh of owners/playtime | 1 req/s | owners, ccu, playtime | Free |
| [ChronoPlay-QA (HF)](https://huggingface.co/datasets/leoner24/ChronoPlay-QA) | Optional RAG eval seed | 232 QA pairs | question, answer, source doc | Other |

**Subset strategy.** As built: 300 games released 2020–2024 with at least 500 reviews in any language, at most 40 per primary genre, non-game software excluded, plus 11 must-include titles; 1,500 reviews per game stratified by rating and year, every review language-tagged and all languages kept. About 0.8M chunks in total, embedded on the 3080 in roughly 75 minutes. Details and known gaps are in docs/DATA\_CARD.md.

**Data handling rules.** Strip `author_steamid` and any user identifiers before indexing. Publish aggregates, embeddings and models, never a raw re-dump of review text. Keep Fandom-derived text share-alike. Show Valve attribution in the demo. Descriptions and wiki text need HTML/wikitext cleaning.

## Functional requirements

Requirements are tagged Must (MVP) or Should (stretch within the 6 weekends).

**Ingestion and storage**

- Must: one reproducible pipeline (`make data`) that downloads sources, selects the subset, cleans text, and writes Parquet tables: `games`, `reviews`, `news`, `wiki_pages`.
- Must: DuckDB as the analytical store (single file, zero setup, fast on Parquet). Must: the whole system starts with `docker compose up` on a Linux host with an NVIDIA GPU, with data and models in volumes so a rebuild never re-downloads.
- Must: chunking per document type: reviews as one chunk each (with metadata), patch notes by section, wiki pages by heading with a 256–512-token window.

**Retrieval (RAG)**

- Must: dense retrieval with a local embedding model (e.g. `bge-m3` or `nomic-embed-text` via Ollama / sentence-transformers) in a local vector store (Qdrant in Docker, or LanceDB/Chroma embedded).
- Must: metadata filters: appid, document type, language, date range, review rating, playtime at review.
- Should: hybrid retrieval (BM25 + dense) with reciprocal rank fusion, and a local cross-encoder reranker (e.g. `bge-reranker-base`).
- Should: query rewriting / multi-query for vague questions.

**Data-science tools (callable by the agent)**

- Must: `query_catalog(sql)` — guarded SQL over DuckDB for filtering and ranking games (U3, U4).
- Must: `review_sentiment(appid, from, to)` — a trained classifier (TF-IDF + logistic regression baseline, then a fine-tuned small transformer such as DistilBERT) returning sentiment and aspect distribution (performance, bugs, gameplay, story, price, controls) for a time window (U1, U2).
- Must: `predict_reception(features)` — a gradient-boosted model trained on catalog metadata to predict positive-review ratio bucket at launch, with SHAP top features (U5).
- Must: `review_trend(appid, granularity)` — monthly review volume and positive share, with a change-point or before/after test around a date (U2, U7).
- Should: `compare_games(appids)` — structured comparison table.

**Agent**

- Must: a LangGraph state graph with a local model via Ollama (ChatOllama with tool binding): nodes for router/planner, retrieval, tool execution, verification and answer; typed state (question, plan, retrieved chunks, tool results, citations); max N iterations; and a final answer that cites sources (document ids → rendered as review excerpts, patch titles, wiki sections).
- Must: refusal path for out-of-scope questions.
- Should: a verification node that checks numbers in the draft answer against tool outputs and loops back once if unsupported; LangGraph checkpointing (SQLite) so the UI can show and replay each step.

**Interface**

- Must: chat UI (Streamlit or Gradio) with answer, citations panel, and tool-call trace visible (shows the agent's reasoning to reviewers).
- Should: charts rendered inline for trend answers.

**Observability**

- Must: every run logs query, tool calls, retrieved chunks, latency, and tokens to a local trace (e.g. Langfuse self-hosted, or a simple SQLite log) for the evaluation section.

## Architecture and tech stack

Everything runs on one machine at zero cost: Python, DuckDB, a local vector store, and Ollama for the LLM. The stack is chosen so each layer can be swapped (a different embedding model, a hosted LLM) without touching the others.

**Target hardware: RTX 3080 (10 GB VRAM), 32 GB system RAM.** This sizes every choice below. A 14B instruct model at Q4 (about 9 GB) fits in VRAM with a few thousand tokens of context; an 8B at Q4 (about 5 GB) leaves room to co-load the embedder and reranker. Embedding 1M chunks with `bge-m3` takes well under an hour on this GPU, and DistilBERT fine-tuning on 30k reviews runs in minutes per epoch, so Colab is not needed. 32 GB of system RAM is ample for DuckDB over the 31M-row review file.

**Deployment: Docker Compose on native Linux.** The full solution is a `docker-compose.yml` with GPU passthrough via the NVIDIA Container Toolkit:

| Service | Image | Role |
| --- | --- | --- |
| `ollama` | `ollama/ollama` (GPU) | Serves the LLM; models pulled on first start into a named volume |
| `qdrant` | `qdrant/qdrant` | Vector store with payload filters; data in a named volume |
| `app` | own Dockerfile (Python 3.11) | Agent, tools, Streamlit UI; mounts `./data` and `./models` |
| `pipeline` | same image, `profiles: [build]` | One-shot `make data` / `make index` / `make eval`, run with `docker compose run pipeline` |
| `langfuse` + `postgres` | official images, `profiles: [trace]` | Optional tracing dashboard |

The `app` image uses the GPU only for the embedder and reranker (sentence-transformers with CUDA); the LLM is reached over HTTP at `ollama:11434`. As built, the hosted demo runs a CPU-only, multi-arch build of the same app on an Oracle A1 arm64 VM, with the LLM on Cloudflare Workers AI through an OpenAI-compatible client (`SGA_LLM_PROVIDER=openai`).

[Diagram: system architecture · UI, agent, retrieval, tools, data — see the living copy linked above]

The user talks to the agent only; the agent chooses between retrieving evidence and computing numbers, and both paths read the same DuckDB/Parquet tables built by the ingestion pipeline.

| Layer | Choice | Why (and the free alternative) |
| --- | --- | --- |
| Language / env | Python 3.11, uv or poetry, Makefile | Reproducible setup; the same Dockerfile serves dev, pipeline and demo |
| Storage | DuckDB + Parquet | Single file, SQL over 30M rows on a laptop, no server |
| Embeddings | `bge-m3` or `nomic-embed-text` (sentence-transformers / Ollama) | Open weights, multilingual; \~1M chunks embed in under an hour on the 3080 |
| Vector store | Qdrant (Docker service) | Free, fits the Compose setup, rich payload filtering, matches what the handbook teaches |
| Sparse retrieval | `bm25s` or `rank_bm25` | Hybrid baseline for the ablation |
| Reranker | `BAAI/bge-reranker-base` | Open, runs on CPU for top-20 |
| LLM | Qwen 2.5 14B Instruct (Q4, \~9 GB) for final runs; Llama 3.1 8B / Qwen 2.5 7B (Q4, \~5 GB) for iteration, via Ollama | 14B fits 10 GB VRAM and calls tools far more reliably than 7–8B; the hosted demo uses Qwen3-30B-A3B on Cloudflare Workers AI instead |
| Agent framework | LangGraph | Explicit state graph (plan → retrieve / tools → verify → answer), built-in tool-calling nodes, checkpointing and a visual graph for the README; LangChain integrations for Ollama and Qdrant |
| ML | scikit-learn, LightGBM, SHAP, Hugging Face `transformers` for DistilBERT | Standard, explainable |
| Eval | Ragas or custom metrics + a local judge model | Zero cost; judge = the same local LLM or a second one for independence |
| Tracing | Langfuse (self-hosted, Docker) or SQLite log | Free |
| UI | Streamlit | Fastest path to a shareable demo |
| Demo hosting | As built: Oracle Cloud Always Free A1 VM (2 OCPU / 12 GB, arm64) for the app, Qdrant and bge-m3, Caddy for HTTPS, and the LLM on Cloudflare Workers AI (10,000 free neurons a day, about 95 questions) | €0; see Risks for latency |
| Repo | GitHub public, GitHub Actions for lint + tests | Free |

## Evaluation and success metrics

Evaluation is a first-class deliverable: it is what turns a demo into a data-science project. Three layers are measured separately, so a weak answer can be traced to retrieval, to the tools, or to the generator.

**1. Retrieval quality.** Build a gold set of 150–200 question → relevant-chunk pairs: half synthetic (generate questions from sampled chunks with the local LLM, then hand-check), half written by hand against U1–U8. Report recall@5, recall@20 and MRR for dense, BM25, hybrid, and hybrid + reranker, with bootstrap 95% confidence intervals. The ablation table is the headline figure of the README.

**2. Tool / model quality.** Classic ML metrics on held-out data: F1 and ROC-AUC for the sentiment classifier against `voted_up`; macro-F1 for aspect labels on a 300-review hand-labelled sample; RMSE / bucket accuracy and calibration for the reception predictor, with SHAP summary plots. Baselines (TF-IDF + logistic regression, majority class) are reported next to the final models.

**3. End-to-end answer quality.** On a 60-question set (U1–U8 variants) score faithfulness (every claim supported by a retrieved chunk or tool output), answer relevance, and citation correctness, using a local judge model and a manual check on 20 questions to calibrate the judge. Also measure tool-routing accuracy (did the agent call the right tool?) and median latency per question.

**Targets for the MVP**

| Metric | Target |
| --- | --- |
| Recall@20, hybrid + rerank | ≥ 0.85 |
| Sentiment F1 (held-out) | ≥ 0.90 |
| Reception predictor bucket accuracy | clearly above majority baseline, reported with CI |
| Faithfulness (judge) | ≥ 0.85 |
| Tool-routing accuracy | ≥ 0.90 |
| Median latency on the 3080 (14B model) | ≤ 20 s |

All eval runs are scripted (`make eval`) and write a results table that the README embeds, so the numbers are reproducible by anyone who clones the repo.

**Protocol against overfitting.** Each gold question gets a frozen dev or test split from a hash of its id. Every tuning decision (chunking, k, prompts, router rules, models) is judged on dev only; test runs once at the end (`sga eval retrieval --split test --final`). Retrieval is scored per source document, so labels survive chunking changes, and each run keeps per-question ranks for paired comparisons. U1–U8 are demos, never tuning targets.

## Risks and mitigations

The zero-budget, fully local constraint is the main source of risk: compute, not data, is the bottleneck.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| VRAM pressure: LLM, embedder and reranker together exceed 10 GB | Weekends lost waiting | Embed and rerank in separate passes (unload the LLM with Ollama `keep_alive=0` during indexing); keep the reranker on CPU for top-20 if needed; use the 8B model while developing and the 14B only for final eval runs. Watch `nvidia-smi` and cap context length if a model spills into system RAM. |
| Free demo host cannot run a 7B model with acceptable latency | Demo feels broken | As built: the LLM moved to a free hosted endpoint (Workers AI Qwen3-30B-A3B with reasoning on), which matches the local 14B on the test split; daily question caps (20 per visitor, 90 in total) keep it inside the free allowance. |
| Small local models call tools unreliably | Wrong or no tool calls | Constrain to JSON schema (Ollama structured outputs / grammar), few-shot examples per tool, a validation-and-retry step, and a routing-accuracy metric to track it. |
| Review text is noisy, multilingual, review-bombed | Poor retrieval and skewed sentiment | Filter to English, min length, dedupe; flag review-bomb windows via volume spikes; keep `language` as a filter. |
| Mendeley license or Kaggle terms unclear for redistribution | Legal exposure for the public repo | Redistribute only derived artefacts (embeddings, aggregates, models); the pipeline downloads raw data from the source at build time. |
| Steam news API returns sparse or marketing-only posts for some games | Patch-note RAG weak for U2 | Choose subset games with rich update histories (live-service and early-access titles); fall back to wiki "Patch" pages from Fandom. |
| Wiki pages are hard to match to appids | Missing U6 coverage | Resolve via Wikidata P1733 first, fuzzy title match second, manual map for the top 50 games. |
| Scope creep across RAG, ML, agent, UI, eval | Nothing finished in 6 weekends | Strict MVP line (Must items), each weekend ends with something runnable and committed; stretch items only after weekend 5. |
| Local judge model is biased or weak for end-to-end eval | Inflated quality numbers | Calibrate judge against 20 manually scored answers; report agreement; use a different model family as judge than as generator. |

## Delivery plan

Six working weekends, one every two weeks from 10 October to 20 December 2026, with a runnable MVP after weekend 3 so the last three weekends improve quality rather than chase a first result. Dates are a proposal; shift them to your calendar.

[Diagram: delivery timeline · 6 weekends, 2 milestones — see the living copy linked above]

Each weekend ends with a commit that runs from a clean clone. The weekday evenings in between are for long-running jobs (embedding, fine-tuning on Colab) that need no attention.

1. **W1, Data pipeline and EDA (Oct 10–11).** Repo skeleton with Dockerfile and `docker-compose.yml` (ollama, qdrant, app, pipeline profile); `make data`: download catalog and Mendeley reviews, pick the subset, clean, write Parquet, load into DuckDB. Pull patch notes for the subset via the Steam News API and filter Fandom pages. One EDA notebook: review volume over time, rating distribution, price vs reception, language mix. *Done when:* `docker compose up` starts the stack, tables exist, EDA notebook committed, data card written.
2. **W2, Index and baseline RAG (Oct 24–25).** Chunking per document type, embeddings with a small local model, vector store, dense retrieval with metadata filters, a plain RAG answer with citations in a CLI. Start the synthetic question set. *Done when:* U1, U6 and U8 answer from the CLI with cited chunks.
3. **W3, ML tools and agent loop (Nov 7–8).** Train sentiment baseline and reception predictor (notebooks → `models/`), wrap them plus `query_catalog` and `review_trend` as LangChain tools, build the LangGraph graph (router, retrieve, tools, answer) with schema-validated tool calls and a retry edge. *Done when:* U2, U3, U5 and U7 answer end to end. **MVP milestone.**
4. **W4, Hybrid retrieval, reranker, gold set (Nov 21–22).** Add BM25 + RRF and the cross-encoder, finish the 150–200 pair gold set (half hand-checked), run the retrieval ablation with confidence intervals. Fine-tune DistilBERT for sentiment and aspects locally on the 3080 during the week before (minutes per epoch). Before any W4 number: rebuild the review sample, which was not reproducible until the W3 fix (make subset, then make index and make train), so every result comes from data a clean clone reproduces; review questions in the gold set that point at reviews no longer sampled are regenerated. DistilBERT must reach the 0.90 sentiment F1 target (the W3 TF-IDF baseline scores 0.886). Revisit the reception predictor's target and features until it beats the majority baseline on bucket accuracy (W3: +0.7 points, CI −0.1 to +1.4), judging every change on the 2022 validation year only and re-scoring the 2023+ test years once. *Done when:* DistilBERT sentiment F1 ≥ 0.90 on held-out games, reception bucket accuracy above the majority baseline on the 2023+ test years (bootstrap 95% CI of the gain above 0), and the ablation table produced by `make eval`.
5. **W5, End-to-end eval and UI (Dec 5–6).** 60-question answer set, judge model scoring calibrated on 20 manual scores, tool-routing accuracy and latency; Streamlit UI with citations panel, tool trace and trend charts. *Done when:* results tables and UI screenshots in the README.
6. **W6, Deploy and write-up (Dec 19–20).** A $0 cloud deploy picked from the October 2026 hosting research (no free machine runs the whole stack, so the LLM moves to a free hosted model): A = Streamlit Cloud + Cloudflare Workers AI (Qwen3-30B-A3B) + slimmed Qdrant Cloud, B = Oracle free ARM VM (app, Qdrant, bge-m3) + hosted LLM, C = Qwen 2.5 14B on a ZeroGPU Space with a Gradio UI. Swap ChatOllama for an OpenAI-compatible client, re-run the dev answer eval on the hosted model and one test run on what ships; `docker compose` for the full local version (Qwen 2.5 14B on a 10 GB GPU), README with architecture diagram, metrics, limitations and data/licence notes, short demo video. *Done when:* a stranger can use the demo link and reproduce the numbers from the repo.

W6 hosting options, limits and sources: Free cloud hosting for Steam Game Analyst. W5 results that feed W6: on the frozen test split the local 14B passes 40% of answers \[95% CI 25-55%\], routing is exact on 82%, p95 latency 37 s; weakest areas are catalog SQL, patch event dates and SHAP wording (README, End-to-end evaluation).

W6 hosting decision (2026-10-08): setup B. The project's Oracle account keeps the pre-cut Always Free allowance (4 OCPU / 24 GB), and 2 OCPU / \~12 GB are unused, enough for the app, Qdrant, bge-m3 and DuckDB (\~7-8 GB). The LLM moves to a free hosted endpoint. W6 also needs an arm64 image, a keep-alive against Oracle's idle reclaim, and a reproducible deploy (compose file + data download script) in case the allowance changes.

## Status (2026-10-09)

All six weekends' deliverables exist except the demo video; the demo is live and the repo is still private.

| Week | State | Notes |
| --- | --- | --- |
| W1 Data pipeline and EDA | Done | 300 games, 417k sampled reviews, data card |
| W2 Index and baseline RAG | Done | 713k chunks in Qdrant, cited answers |
| W3 ML tools and agent | Done | four tools, LangGraph agent |
| W4 Hybrid, rerank, gold set | Done, with gaps | 195 LLM-written gold questions, none hand-checked; no hand-written U1–U8 questions |
| W5 Answer eval and UI | Done | local 14B test pass 0.40 \[0.25, 0.55\] |
| W6 Deploy and write-up | Done except the video | hosted Qwen3-30B test pass 0.35 (same as the 14B), grounding 0.72 vs 0.84; README updated |

Against the MVP targets: sentiment F1 0.913 and reception above the majority baseline are met; faithfulness (grounded 0.72–0.84 vs 0.85) and tool routing (82–85% vs 90%) are missed; median latency is about 20 s; retrieval hit@10 is 0.927 with rerank on the test split (recall@20 not measured as specified).

After the test runs, four general fixes were made and checked on dev only: grouped citations, review-ratio reporting, a patch-note date lookup, and widening thin before/after windows. None moved the dev scores measurably.

Open:

- [ ] Going public: confirm the Mendeley licence and trim review text from the committed eval files
- [ ] Record the demo video
- [ ] Try a verification node that checks the answer's numbers against tool outputs (the weakest metric is grounding)
- [ ] Grow dev with user-reviewed catalog and patch-window questions, so those fixes can be measured
- [ ] Optional: a health alert for the VM, persistent daily question counters

Stretch, only after W5 is done: query rewriting, `compare_games` tool, Langfuse tracing dashboard, a blog post.

## Open questions and decisions

These decide the W1 setup and should be settled before 10 October.

- [ ] **Hardware.** Settled: native Linux, RTX 3080 10 GB VRAM + 32 GB system RAM, Docker Compose with NVIDIA Container Toolkit (see Architecture).
- [ ] **Vector store.** Settled: Qdrant as a Compose service.
- [ ] **Agent framework.** Settled: LangGraph.
- [ ] **Subset size.** Settled: 300 games, 1,500 reviews each (see Data sources).
- [ ] **Demo host.** Settled in W6: Oracle A1 VM + Cloudflare Workers AI, live at https://92-5-85-76.sslip.io.
- [ ] **Mendeley license.** Still open: confirm the licence on the dataset page before the repo goes public, and decide what to do with committed eval files that quote review text (retrieval pools, judging batches, answer runs).
- [ ] **Repo name and README tone.** Portfolio-first (narrative, results up top) or library-first (install, API)?
- [ ] **Second judge model.** Settled: Claude as judge (a different family from the Qwen generators), a second Claude judge for agreement, and a blind human check on 20 answers.

### Sources

[FronkonGames Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) · [Steam Games Metadata and Player Reviews 2020–2024](https://data.mendeley.com/datasets/jxy85cr3th/2) · [Steam Dataset 2025 data card](https://github.com/vintagedon/steam-dataset-2025/blob/main/steam-dataset-2025-v1/DATASET_CARD.md) · [Steam Web API terms](https://steamcommunity.com/dev/apiterms) · [SteamSpy API](https://steamspy.com/api.php) · [nyuuzyou/fandom](https://huggingface.co/datasets/nyuuzyou/fandom) · [ChronoPlay-QA](https://huggingface.co/datasets/leoner24/ChronoPlay-QA)
