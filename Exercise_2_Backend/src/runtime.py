"""Single factory used by notebook, evaluation, tests, and Gradio."""

from __future__ import annotations

from pathlib import Path

from .config import AppConfig, load_config
from .embeddings import EmbeddingService
from .llm_service import LLMService
from .logging_utils import StructuredInteractionLogger, configure_logging
from .long_term_memory import SQLiteConversationMemory
from .rag_pipeline import RAGPipeline
from .retriever import Retriever
from .vector_store import FaissVectorStore


def build_pipeline(config_path: str | Path | None = None) -> tuple[RAGPipeline, AppConfig]:
    config = load_config(config_path)
    configure_logging(config.log_level)
    embeddings = EmbeddingService(config.embedding_model, config.embedding_batch_size)
    store = FaissVectorStore(config.vector_store_path).load_index()
    retriever = Retriever(embeddings, store)
    llm = LLMService(
        config.llm_model,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        timeout_seconds=config.generation_timeout_seconds,
        random_seed=config.random_seed,
    )
    memory = SQLiteConversationMemory(config.database_path)
    logger = StructuredInteractionLogger(config.log_path)
    return RAGPipeline(config, retriever, llm, memory, logger), config

