"""End-to-end grounded RAG pipeline with memory, safety, sources, and logging."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from .config import AppConfig
from .llm_service import LLMService
from .logging_utils import StructuredInteractionLogger
from .long_term_memory import SQLiteConversationMemory
from .preprocessing import validate_query
from .prompts import CLARIFICATION_PROMPT, FALLBACK_PROMPT, build_rag_prompt
from .retriever import Retriever
from .safety import grounding_warning, is_ambiguous, is_out_of_domain, sanitize_for_logging
from .short_term_memory import RollingConversationBuffer


class RAGPipeline:
    def __init__(
        self,
        config: AppConfig,
        retriever: Retriever,
        llm_service: LLMService,
        long_term_memory: SQLiteConversationMemory,
        interaction_logger: StructuredInteractionLogger,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.llm_service = llm_service
        self.long_term_memory = long_term_memory
        self.interaction_logger = interaction_logger
        self._buffers: dict[str, RollingConversationBuffer] = defaultdict(
            lambda: RollingConversationBuffer(
                max_turns=config.memory_max_turns,
                max_characters=config.memory_max_characters,
            )
        )

    def _hydrate_buffer(self, session_id: str) -> RollingConversationBuffer:
        buffer = self._buffers[session_id]
        if not buffer.messages():
            for item in self.long_term_memory.load_recent_messages(
                session_id, limit=self.config.memory_max_turns * 2
            ):
                buffer.add(item["role"], item["message"], timestamp=item["created_at"])
        return buffer

    @staticmethod
    def _source_records(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        seen: set[str] = set()
        for chunk in chunks:
            source_id = str(chunk["source_id"])
            if source_id in seen:
                continue
            seen.add(source_id)
            sources.append(
                {
                    "source_id": source_id,
                    "title": str(chunk["title"]),
                    "url": str(chunk["source_url"]),
                    "section": str(chunk["section"]),
                }
            )
        return sources

    @staticmethod
    def _format_context(chunks: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{chunk['source_id']}] {chunk['title']} - {chunk['section']}\n{chunk['text']}"
            for chunk in chunks
        )

    @staticmethod
    def _adapt_detail(answer: str, preference: str) -> str:
        if preference == "short":
            sentences = answer.split(". ")
            return ". ".join(sentences[:3]).strip()
        if preference == "detailed":
            return answer + "\n\nDetail preference: include rationale and a minimal practical example where supported."
        return answer

    def _record_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        safe_user = sanitize_for_logging(user_message)
        safe_assistant = sanitize_for_logging(assistant_message)
        buffer = self._buffers[session_id]
        buffer.add("user", safe_user)
        buffer.add("assistant", safe_assistant)
        self.long_term_memory.save_message(session_id, "user", safe_user)
        self.long_term_memory.save_message(session_id, "assistant", safe_assistant)

    def _fallback_result(
        self,
        *,
        session_id: str,
        query: str,
        answer: str,
        error_type: str,
        started: float,
        retrieved_chunks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        chunks = retrieved_chunks or []
        latency_ms = (time.perf_counter() - started) * 1000
        self._record_turn(session_id, query, answer)
        self.interaction_logger.log_interaction(
            anonymous_session_id=session_id,
            query_length=len(query),
            retrieval_top_score=float(chunks[0]["score"]) if chunks else None,
            retrieved_source_ids=[str(item["source_id"]) for item in chunks],
            fallback_used=True,
            latency_ms=latency_ms,
            error_type=error_type,
            generation_backend="deterministic_safety_fallback",
        )
        return {
            "answer": answer,
            "sources": self._source_records(chunks),
            "retrieved_chunks": chunks,
            "retrieval_scores": [float(item["score"]) for item in chunks],
            "fallback_used": True,
            "latency_ms": latency_ms,
            "error_type": error_type,
            "generation_backend": "deterministic_safety_fallback",
        }

    def ask(
        self,
        query_text: str,
        *,
        session_id: str | None = None,
        response_detail: str | None = None,
        experience_level: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        session_id = self.long_term_memory.ensure_session(session_id)
        buffer = self._hydrate_buffer(session_id)
        validation = validate_query(
            query_text, max_characters=self.config.max_query_characters
        )
        query = str(validation["query"])
        if validation["is_empty"]:
            return self._fallback_result(
                session_id=session_id,
                query=query,
                answer="Please enter a non-empty scikit-learn question.",
                error_type="empty_input",
                started=started,
            )
        if response_detail or experience_level:
            current = self.long_term_memory.load_preferences(session_id)
            self.long_term_memory.save_preferences(
                session_id,
                response_detail=response_detail or current["response_detail"],
                experience_level=experience_level or current["experience_level"],
            )
        preferences = self.long_term_memory.load_preferences(session_id)
        has_context = bool(buffer.messages())
        if is_ambiguous(query, has_conversation_context=has_context):
            return self._fallback_result(
                session_id=session_id,
                query=query,
                answer=CLARIFICATION_PROMPT,
                error_type="ambiguous_query",
                started=started,
            )
        if is_out_of_domain(query):
            return self._fallback_result(
                session_id=session_id,
                query=query,
                answer=FALLBACK_PROMPT,
                error_type="out_of_domain",
                started=started,
            )

        retrieval_query = f"{buffer.as_prompt_text()}\nCURRENT QUESTION: {query}" if has_context else query
        chunks = self.retriever.retrieve(retrieval_query, top_k=self.config.retrieval_top_k)
        if not chunks or float(chunks[0]["score"]) < self.config.retrieval_min_score:
            return self._fallback_result(
                session_id=session_id,
                query=query,
                answer=FALLBACK_PROMPT,
                error_type="weak_retrieval",
                started=started,
                retrieved_chunks=chunks,
            )

        sources = self._source_records(chunks)
        prompt = build_rag_prompt(
            retrieved_context=self._format_context(chunks),
            conversation_memory=buffer.as_prompt_text(),
            user_question=(
                f"Experience level: {preferences['experience_level']}. "
                f"Response detail: {preferences['response_detail']}. {query}"
            ),
        )
        generated = self.llm_service.generate(prompt)
        answer = self._adapt_detail(generated.text, preferences["response_detail"])
        warning = grounding_warning(answer, [item["source_id"] for item in sources])
        source_lines = "\n".join(
            f"- [{item['source_id']}] {item['title']}: {item['url']}" for item in sources
        )
        if source_lines:
            answer = f"{answer}\n\nSources\n{source_lines}"
        latency_ms = (time.perf_counter() - started) * 1000
        self._record_turn(session_id, query, answer)
        self.interaction_logger.log_interaction(
            anonymous_session_id=session_id,
            query_length=len(query),
            retrieval_top_score=float(chunks[0]["score"]),
            retrieved_source_ids=[item["source_id"] for item in sources],
            fallback_used=False,
            latency_ms=latency_ms,
            error_type=warning,
            generation_backend=generated.backend,
        )
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
            "retrieval_scores": [float(item["score"]) for item in chunks],
            "fallback_used": False,
            "latency_ms": latency_ms,
            "error_type": warning,
            "generation_backend": generated.backend,
        }

    def clear_session(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)

    def delete_session(self, session_id: str) -> None:
        """Clear rolling memory and permanently delete the persisted session."""
        self._buffers.pop(session_id, None)
        self.long_term_memory.delete_session(session_id)
