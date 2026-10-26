"""Reproducible analytics derived only from structured interaction logs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_interaction_logs(log_path: str | Path) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


def summarize_logs(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "number_of_queries": 0,
            "fallback_rate": 0.0,
            "average_latency_ms": 0.0,
            "most_frequently_retrieved_sources": {},
            "common_failure_categories": {},
        }
    source_counts: Counter[str] = Counter()
    for values in frame["retrieved_source_ids"]:
        source_counts.update(values or [])
    failures = frame["error_type"].dropna().astype(str).value_counts().to_dict()
    return {
        "number_of_queries": int(len(frame)),
        "fallback_rate": float(frame["fallback_used"].astype(bool).mean()),
        "average_latency_ms": float(frame["latency_ms"].mean()),
        "most_frequently_retrieved_sources": dict(source_counts.most_common(10)),
        "common_failure_categories": failures,
    }


def generate_analytics_outputs(log_path: str | Path, output_directory: str | Path) -> dict[str, object]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame = load_interaction_logs(log_path)
    summary = summarize_logs(frame)
    (output_directory / "analytics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if frame.empty:
        return summary

    frame.to_csv(output_directory / "interaction_analytics.csv", index=False)
    fallback_counts = frame["fallback_used"].map({True: "Fallback", False: "Grounded answer"}).value_counts()
    ax = fallback_counts.plot(kind="bar", color=["#D97706", "#2563EB"], figsize=(7, 4))
    ax.set_title("Interaction Outcomes from Actual Logs")
    ax.set_ylabel("Queries")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(output_directory / "fallback_outcomes.png", dpi=180, bbox_inches="tight")
    plt.close()

    source_counts = pd.Series(summary["most_frequently_retrieved_sources"], dtype=float).sort_values()
    if not source_counts.empty:
        ax = source_counts.plot(kind="barh", color="#0F766E", figsize=(8, 5))
        ax.set_title("Most Frequently Retrieved Official Sources")
        ax.set_xlabel("Retrieval occurrences")
        plt.tight_layout()
        plt.savefig(output_directory / "retrieved_sources.png", dpi=180, bbox_inches="tight")
        plt.close()
    return summary
