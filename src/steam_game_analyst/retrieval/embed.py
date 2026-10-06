"""Text embeddings (bge-m3 by default) via sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised float32 vectors, one row per text."""
        ...


class SentenceEmbedder:
    """Lazy sentence-transformers model. fp16 on CUDA (~1.2 GB VRAM for bge-m3), fp32 on CPU."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 64,
        max_seq_length: int = 512,
        cache_dir: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"torch_dtype": "float16"} if device.startswith("cuda") else None
        self._model = SentenceTransformer(
            model_name, device=device, cache_folder=cache_dir, model_kwargs=model_kwargs
        )
        self._model.max_seq_length = max_seq_length
        self._batch_size = batch_size
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension  # sentence-transformers < 6
        )
        self.dim = int(get_dim() or 0)

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)
