"""Structured JSONL interaction logging without raw questions or identifiers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .safety import sanitize_for_logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class StructuredInteractionLogger:
    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self,
        *,
        anonymous_session_id: str,
        query_length: int,
        retrieval_top_score: float | None,
        retrieved_source_ids: list[str],
        fallback_used: bool,
        latency_ms: float,
        error_type: str | None = None,
        feedback: str | None = None,
        generation_backend: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "anonymous_session_id": sanitize_for_logging(anonymous_session_id),
            "query_length": int(query_length),
            "retrieval_top_score": None if retrieval_top_score is None else float(retrieval_top_score),
            "retrieved_source_ids": [sanitize_for_logging(item) for item in retrieved_source_ids],
            "fallback_used": bool(fallback_used),
            "latency_ms": float(latency_ms),
            "error_type": None if error_type is None else sanitize_for_logging(error_type),
            "feedback": None if feedback is None else sanitize_for_logging(feedback),
            "generation_backend": generation_backend,
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

