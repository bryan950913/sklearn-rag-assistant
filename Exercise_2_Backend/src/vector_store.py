"""Persistent FAISS inner-product index with JSON metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class FaissVectorStore:
    INDEX_FILENAME = "knowledge.index"
    METADATA_FILENAME = "metadata.json"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.metadata: list[dict[str, object]] = []

    def build_index(
        self, embeddings: np.ndarray, metadata: list[dict[str, object]]
    ) -> "FaissVectorStore":
        import faiss

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2D matrix")
        if matrix.shape[0] != len(metadata):
            raise ValueError("metadata length must equal embedding rows")
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self.metadata = list(metadata)
        return self

    def save_index(self) -> dict[str, object]:
        import faiss

        if self.index is None:
            raise RuntimeError("Build or load an index before saving")
        # Byte serialization avoids FAISS's narrow Windows path API, which can fail
        # when a user profile contains accented characters.
        serialized = faiss.serialize_index(self.index)
        (self.storage_path / self.INDEX_FILENAME).write_bytes(serialized.tobytes())
        (self.storage_path / self.METADATA_FILENAME).write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "vectors": int(self.index.ntotal),
            "dimension": int(self.index.d),
            "metric": "cosine via normalized inner product",
        }
        (self.storage_path / self.MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    def load_index(self) -> "FaissVectorStore":
        import faiss

        index_path = self.storage_path / self.INDEX_FILENAME
        metadata_path = self.storage_path / self.METADATA_FILENAME
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Persistent FAISS index or metadata is missing")
        serialized = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        self.index = faiss.deserialize_index(serialized)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS vector count and metadata count differ")
        return self

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[dict[str, object]]:
        import faiss

        if self.index is None:
            raise RuntimeError("Load or build the index before search")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(int(top_k), self.index.ntotal))
        results: list[dict[str, object]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            item = dict(self.metadata[int(index)])
            item["score"] = float(score)
            results.append(item)
        return results
