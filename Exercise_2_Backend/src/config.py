"""Configuration loading with project-relative, portable paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    llm_model: str
    embedding_model: str
    temperature: float
    max_new_tokens: int
    retrieval_top_k: int
    retrieval_min_score: float
    chunk_size: int
    chunk_overlap: int
    memory_max_turns: int
    memory_max_characters: int
    database_path: Path
    vector_store_path: Path
    knowledge_base_path: Path
    log_path: Path
    log_level: str
    embedding_batch_size: int
    max_query_characters: int
    generation_timeout_seconds: int
    random_seed: int

    def ensure_runtime_directories(self) -> None:
        """Create only the local runtime directories declared in the configuration."""
        for path in (
            self.database_path.parent,
            self.vector_store_path,
            self.knowledge_base_path / "raw",
            self.knowledge_base_path / "processed",
            self.log_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load YAML configuration and resolve every data path from the project root."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    config_path = Path(config_path).resolve()
    project_root = config_path.parent
    with config_path.open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)

    config = AppConfig(
        project_root=project_root,
        llm_model=str(raw["llm_model"]),
        embedding_model=str(raw["embedding_model"]),
        temperature=float(raw["temperature"]),
        max_new_tokens=int(raw["max_new_tokens"]),
        retrieval_top_k=int(raw["retrieval_top_k"]),
        retrieval_min_score=float(raw["retrieval_min_score"]),
        chunk_size=int(raw["chunk_size"]),
        chunk_overlap=int(raw["chunk_overlap"]),
        memory_max_turns=int(raw["memory_max_turns"]),
        memory_max_characters=int(raw["memory_max_characters"]),
        database_path=_resolve(project_root, raw["database_path"]),
        vector_store_path=_resolve(project_root, raw["vector_store_path"]),
        knowledge_base_path=_resolve(project_root, raw["knowledge_base_path"]),
        log_path=_resolve(project_root, raw["log_path"]),
        log_level=str(raw["log_level"]),
        embedding_batch_size=int(raw["embedding_batch_size"]),
        max_query_characters=int(raw["max_query_characters"]),
        generation_timeout_seconds=int(raw["generation_timeout_seconds"]),
        random_seed=int(raw["random_seed"]),
    )
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    config.ensure_runtime_directories()
    return config

