# App + pipeline image. CUDA runtime so sentence-transformers / torch can use the GPU
# for embeddings, reranking and DistilBERT fine-tuning. The LLM itself runs in the
# separate `ollama` service.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs; it also fetches a managed Python 3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /root/.local/bin/
RUN uv python install 3.12

WORKDIR /app

# Install dependencies first (layer cache), then the project itself.
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --all-extras --no-dev --python 3.12 || \
    uv sync --no-install-project --all-extras --no-dev --python 3.12

COPY src ./src
COPY configs ./configs
COPY .streamlit ./.streamlit
COPY README.md ./
RUN uv sync --all-extras --no-dev --python 3.12

EXPOSE 8501
CMD ["sga", "--help"]
