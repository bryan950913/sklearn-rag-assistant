"""Sentence-Transformers embedding service with normalized vectors."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        self.load()
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            values,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        self.load()
        return int(self._model.get_sentence_embedding_dimension())

