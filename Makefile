.DEFAULT_GOAL := help
COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm pipeline

.PHONY: help setup lint test up down logs build data subset index eval shell pull-models clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------- local dev (outside Docker) ----------
setup: ## Create venv with all extras + dev tools (uv)
	uv sync --all-extras
	uv run pre-commit install || true

lint: ## Ruff + mypy
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

test: ## Unit tests (fast, fixture-based, no network)
	uv run pytest

# ---------- docker ----------
build: ## Build the app/pipeline image
	$(COMPOSE) build

up: ## Start ollama + qdrant + app
	$(COMPOSE) up -d

down: ## Stop everything
	$(COMPOSE) down

logs: ## Tail logs
	$(COMPOSE) logs -f --tail=100

pull-models: ## Pull the LLMs into the ollama volume
	docker exec sga-ollama ollama pull $${SGA_LLM_MODEL:-qwen2.5:14b-instruct-q4_K_M}
	docker exec sga-ollama ollama pull $${SGA_LLM_DEV_MODEL:-qwen2.5:7b-instruct-q4_K_M}

# ---------- pipeline (runs in the pipeline container) ----------
data: ## Download raw sources + build processed Parquet + DuckDB (W1)
	$(RUN) data all

subset: ## Re-run only the subset selection + table build (no download)
	$(RUN) data build

index: ## Chunk + embed + load into Qdrant (W2)
	$(RUN) index build

eval: ## Run the evaluation suite (W4/W5)
	$(RUN) eval all

shell: ## Shell inside the pipeline container
	$(COMPOSE) run --rm --entrypoint bash pipeline

clean: ## Remove processed data (keeps raw downloads)
	rm -rf data/processed/* && touch data/processed/.gitkeep
