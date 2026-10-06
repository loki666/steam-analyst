"""Settings: environment variables (prefix SGA_) layered over configs/pipeline.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    """Repo root: $SGA_ROOT, else the directory holding pyproject.toml above this file."""
    env = os.environ.get("SGA_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


class Settings(BaseSettings):
    """Runtime settings from the environment (.env is read by docker compose / uv run)."""

    model_config = SettingsConfigDict(env_prefix="SGA_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default_factory=lambda: project_root() / "data")
    models_dir: Path = Field(default_factory=lambda: project_root() / "models")
    configs_dir: Path = Field(default_factory=lambda: project_root() / "configs")

    llm_model: str = "qwen2.5:14b-instruct-q4_K_M"
    llm_dev_model: str = "qwen2.5:7b-instruct-q4_K_M"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-base"

    subset_games: int | None = None
    reviews_per_game: int | None = None

    # Not SGA_-prefixed on purpose: these are conventional names.
    steam_api_key: str | None = Field(default=None, validation_alias="STEAM_API_KEY")
    ollama_host: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_HOST")
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def duckdb_path(self) -> Path:
        return self.processed_dir / "sga.duckdb"


class SubsetConfig(BaseModel):
    n_games: int = 300
    reviews_per_game: int = 500
    min_english_reviews: int = 500
    release_from: str = "2020-01-01"
    release_to: str = "2024-12-31"
    languages: list[str] = ["english"]
    genre_quota: int = 40
    must_include: list[int] = []
    # Steam lists software under genres too; any of these disqualifies an app.
    exclude_genres: list[str] = []


class CleaningConfig(BaseModel):
    min_review_chars: int = 20
    max_review_chars: int = 4000
    drop_duplicates_on: list[str] = ["appid", "review_text"]
    strip_html: bool = True


class NewsConfig(BaseModel):
    count: int = 200
    maxlength: int = 0
    feeds: str = "steam_community_announcements"
    rate_limit_per_sec: float = 2.0


class PipelineConfig(BaseModel):
    sources: dict[str, Any] = {}
    subset: SubsetConfig = SubsetConfig()
    cleaning: CleaningConfig = CleaningConfig()

    @property
    def news(self) -> NewsConfig:
        return NewsConfig(**self.sources.get("news", {}))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    """Load configs/pipeline.yaml and apply SGA_SUBSET_GAMES / SGA_REVIEWS_PER_GAME overrides."""
    settings = get_settings()
    path = path or settings.configs_dir / "pipeline.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = PipelineConfig(**raw)
    if settings.subset_games:
        cfg.subset.n_games = settings.subset_games
    if settings.reviews_per_game:
        cfg.subset.reviews_per_game = settings.reviews_per_game
    return cfg
