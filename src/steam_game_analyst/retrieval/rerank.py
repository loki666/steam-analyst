"""Cross-encoder reranking (bge-reranker-v2-m3 by default) via sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage; higher is more relevant."""
        ...


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder. fp16 on CUDA (~1.1 GB VRAM for
    bge-reranker-v2-m3), fp32 on CPU, where 50 candidates take a few seconds."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
        cache_dir: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        model_kwargs = {"torch_dtype": "float16"} if device.startswith("cuda") else None
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            cache_folder=cache_dir,
            model_kwargs=model_kwargs,
        )
        self._batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages], batch_size=self._batch_size, convert_to_numpy=True
        )
        return [float(s) for s in scores]
