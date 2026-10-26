"""Grounding, ambiguity, privacy, and secret-sanitization helpers."""

from __future__ import annotations

import re


EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
TOKEN = re.compile(r"\b(?:hf_[A-Za-z0-9]{15,}|sk-[A-Za-z0-9_-]{15,})\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")

DOMAIN_TERMS = {
    "sklearn", "scikit-learn", "model", "classifier", "regression", "clustering",
    "pipeline", "preprocessing", "standardscaler", "encoder", "imputer", "missing",
    "train_test_split", "cross-validation", "cross validation", "gridsearchcv",
    "hyperparameter", "accuracy", "precision", "recall", "f1", "roc", "metric",
    "fit", "predict", "transform", "feature", "random forest", "logistic",
}
# Deterministic rejection is intentionally limited to explicit unsupported intent.
# Other wording is sent to retrieval, whose score decides whether evidence is sufficient.
OUT_OF_DOMAIN_PHRASES = {
    "medical diagnosis", "diagnose me", "diagnose my", "prescribe medication",
    "treatment for my", "legal advice", "should i sue", "file a lawsuit",
    "investment advice", "stock price", "stock prices", "buy this stock",
    "sell this stock", "weather forecast", "weather tomorrow", "will it rain",
    "recipe for", "cooking recipe", "how do i cook", "write malware",
    "create malware", "steal passwords", "steal a password", "crack a password",
    "phishing attack", "ransomware code",
}
AMBIGUOUS_PHRASES = {
    "it does not work", "this fails", "why is it bad", "which one", "help me",
    "what about that", "fix it", "why", "how", "error",
}


def sanitize_for_logging(value: str) -> str:
    value = EMAIL.sub("[REDACTED_EMAIL]", str(value))
    value = TOKEN.sub("[REDACTED_TOKEN]", value)
    return PHONE.sub("[REDACTED_PHONE]", value)


def contains_secret(value: str) -> bool:
    return bool(TOKEN.search(str(value)))


def is_out_of_domain(query: str) -> bool:
    lowered = query.casefold()
    return any(phrase in lowered for phrase in OUT_OF_DOMAIN_PHRASES)


def is_ambiguous(query: str, has_conversation_context: bool = False) -> bool:
    if has_conversation_context:
        return False
    lowered = query.casefold().strip(" ?.!")
    if lowered in AMBIGUOUS_PHRASES:
        return True
    tokens = lowered.split()
    return len(tokens) < 3 and not any(term in lowered for term in DOMAIN_TERMS)


def grounding_warning(answer: str, retrieved_source_ids: list[str]) -> str | None:
    if not answer.strip():
        return "empty_answer"
    if retrieved_source_ids and not any(f"[{source_id}]" in answer for source_id in retrieved_source_ids):
        return "model_omitted_inline_source_id"
    return None
