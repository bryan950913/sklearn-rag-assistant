"""Input preprocessing that preserves scikit-learn identifiers and code tokens."""

from __future__ import annotations

import re
import unicodedata


CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace without lowercasing or stripping technical punctuation."""
    return WHITESPACE.sub(" ", text).strip()


def sanitize_query(text: str, max_characters: int = 1000) -> str:
    """Normalize Unicode and controls while preserving names such as GridSearchCV."""
    if not isinstance(text, str):
        raise TypeError("Query must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARACTERS.sub(" ", text)
    text = normalize_whitespace(text)
    return text[:max_characters]


def validate_query(text: str, max_characters: int = 1000) -> dict[str, object]:
    """Return a reusable validation record rather than silently accepting empty input."""
    cleaned = sanitize_query(text, max_characters=max_characters)
    return {
        "query": cleaned,
        "is_empty": not bool(cleaned),
        "was_truncated": len(normalize_whitespace(text)) > max_characters,
        "character_count": len(cleaned),
    }


def preprocess_for_retrieval(text: str, max_characters: int = 1000) -> str:
    """Retrieval preprocessing deliberately retains case-sensitive API identifiers."""
    return sanitize_query(text, max_characters=max_characters)

