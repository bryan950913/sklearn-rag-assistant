"""Official scikit-learn source acquisition, extraction, and metadata-aware chunking."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


ALLOWED_HOSTS = {"scikit-learn.org", "www.scikit-learn.org"}


def read_source_catalog(catalog_path: str | Path) -> list[dict[str, str]]:
    with Path(catalog_path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _official_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS


def download_sources(
    catalog_path: str | Path,
    raw_directory: str | Path,
    timeout_seconds: int = 30,
) -> list[dict[str, object]]:
    """Download only allowlisted official URLs and record hashes of actual bytes."""
    raw_directory = Path(raw_directory)
    raw_directory.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    headers = {"User-Agent": "UNAD-Phase4-Educational-RAG/1.0"}
    for source in read_source_catalog(catalog_path):
        url = source["source_url"]
        if not _official_url(url):
            raise ValueError(f"Non-official or non-HTTPS source blocked: {url}")
        response = requests.get(url, timeout=timeout_seconds, headers=headers)
        response.raise_for_status()
        destination = raw_directory / f"{source['source_id']}.html"
        destination.write_bytes(response.content)
        manifest.append(
            {
                "source_id": source["source_id"],
                "url": url,
                "path": str(destination),
                "status_code": response.status_code,
                "bytes": len(response.content),
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "retrieval_date": date.today().isoformat(),
            }
        )
    return manifest


def _clean_element_text(element: object) -> str:
    text = element.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def ingest_downloaded_documents(
    catalog_path: str | Path,
    raw_directory: str | Path,
) -> list[dict[str, str]]:
    """Extract heading-aware records from locally downloaded HTML only."""
    catalog = {row["source_id"]: row for row in read_source_catalog(catalog_path)}
    documents: list[dict[str, str]] = []
    for source_id, source in catalog.items():
        path = Path(raw_directory) / f"{source_id}.html"
        if not path.exists():
            raise FileNotFoundError(f"Downloaded source missing: {path}")
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        main = soup.find("main") or soup.find("article") or soup.body
        if main is None:
            raise ValueError(f"No readable HTML body in {path}")
        for node in main.find_all(["script", "style", "nav", "footer", "noscript"]):
            node.decompose()
        section = source["title"]
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            text = "\n".join(item for item in buffer if item).strip()
            if len(text) >= 80:
                documents.append(
                    {
                        "source_id": source_id,
                        "title": source["title"],
                        "source_url": source["source_url"],
                        "source_path": str(path),
                        "section": section,
                        "text": text,
                    }
                )
            buffer = []

        for element in main.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
            if element.name in {"h1", "h2", "h3"}:
                flush()
                section = _clean_element_text(element) or source["title"]
            else:
                value = _clean_element_text(element)
                if value and (not buffer or value != buffer[-1]):
                    buffer.append(value)
        flush()
    if not documents:
        raise ValueError("No document sections were extracted")
    return documents


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 1000,
    chunk_overlap: int = 180,
) -> list[dict[str, object]]:
    """Create overlapping character chunks without losing source and section metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[dict[str, object]] = []
    section_counts: dict[str, int] = {}
    for document in documents:
        source_id = document["source_id"]
        section_index = section_counts.get(source_id, 0)
        section_counts[source_id] = section_index + 1
        text = re.sub(r"\s+", " ", document["text"]).strip()
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(". ", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary + 1
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{source_id}-S{section_index:04d}-C{chunk_index:04d}",
                        "source_id": source_id,
                        "title": document["title"],
                        "source_url": document["source_url"],
                        "source_path": document["source_path"],
                        "section": document["section"],
                        "text": chunk_text,
                        "character_count": len(chunk_text),
                    }
                )
                chunk_index += 1
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise AssertionError("Chunk IDs must be globally unique")
    return chunks


def write_jsonl(records: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def chunk_statistics(
    source_catalog: list[dict[str, str]],
    documents: list[dict[str, str]],
    chunks: list[dict[str, object]],
) -> dict[str, float | int]:
    lengths = [int(chunk["character_count"]) for chunk in chunks]
    return {
        "source_documents": len(source_catalog),
        "extracted_sections": len(documents),
        "chunks": len(chunks),
        "average_chunk_characters": round(sum(lengths) / len(lengths), 2),
        "minimum_chunk_characters": min(lengths),
        "maximum_chunk_characters": max(lengths),
    }
