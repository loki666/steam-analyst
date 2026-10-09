# App image for the hosted deploy: CPU only, multi-arch (built on the arm64 Oracle VM).
# The LLM is a hosted endpoint (SGA_LLM_PROVIDER=openai); bge-m3 query embeddings run on
# the CPU. torch comes from PyTorch's CPU index on arm64 (see [tool.uv.sources]).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# a compiler only for the odd dependency without an arm64 wheel
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# dependencies first (layer cache), then the project; no notebooks or dev tools
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra rag --extra ml --extra ui

COPY src ./src
COPY configs ./configs
COPY .streamlit ./.streamlit
COPY README.md ./
RUN uv sync --frozen --no-dev --extra rag --extra ml --extra ui

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fs http://127.0.0.1:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/steam_game_analyst/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
