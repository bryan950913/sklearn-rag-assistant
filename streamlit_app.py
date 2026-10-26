"""Streamlit frontend for the validated open-source scikit-learn RAG backend."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = PROJECT_ROOT / "Exercise_2_Backend"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.runtime import build_pipeline
from src.safety import sanitize_for_logging


@st.cache_resource(show_spinner="Loading the validated RAG backend...")
def get_runtime() -> tuple[Any, RLock]:
    """Load shared model/index resources once and serialize shared inference access."""

    pipeline, _ = build_pipeline(CONFIG_PATH)
    return pipeline, RLock()


def new_session_id() -> str:
    """Create a non-identifying browser-session identifier."""

    return f"session-{uuid.uuid4().hex}"


def initialize_session_state() -> None:
    defaults = {
        "session_id": new_session_id(),
        "messages": [],
        "pipeline_initialized": False,
        "last_status": "Ready. The backend loads on the first request.",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No source was asserted; deterministic clarification or fallback was used.")
        return

    with st.expander("Retrieved sources", expanded=False):
        for source in sources:
            source_id = str(source.get("source_id", "Source"))
            title = str(source.get("title", "Official scikit-learn documentation"))
            url = str(source.get("url", ""))
            section = str(source.get("section", "")).strip()
            suffix = f" — {section}" if section else ""
            st.markdown(f"- **[{source_id}]** [{title}]({url}){suffix}")


def clear_conversation() -> None:
    current_session = str(st.session_state.session_id)
    if st.session_state.pipeline_initialized:
        pipeline, runtime_lock = get_runtime()
        with runtime_lock:
            pipeline.clear_session(current_session)
    st.session_state.session_id = new_session_id()
    st.session_state.messages = []
    st.session_state.last_status = "Session reset; persistent data was not deleted."


def delete_persistent_session() -> None:
    current_session = str(st.session_state.session_id)
    pipeline, runtime_lock = get_runtime()
    st.session_state.pipeline_initialized = True
    with runtime_lock:
        pipeline.delete_session(current_session)
    st.session_state.session_id = new_session_id()
    st.session_state.messages = []
    st.session_state.last_status = "Persistent session data deleted."


def ask_backend(prompt: str, detail: str, experience: str) -> dict[str, Any]:
    pipeline, runtime_lock = get_runtime()
    st.session_state.pipeline_initialized = True
    with runtime_lock:
        return pipeline.ask(
            prompt,
            session_id=str(st.session_state.session_id),
            response_detail=detail,
            experience_level=experience,
        )


st.set_page_config(
    page_title="Open-Source scikit-learn Support Assistant",
    page_icon="🧠",
    layout="wide",
)
initialize_session_state()

st.title("Open-Source scikit-learn Technical Support Assistant")
st.write(
    "Ask natural-language questions about common scikit-learn concepts and workflows. "
    "Answers are grounded in the packaged index of official scikit-learn documentation."
)
st.warning(
    "Privacy: redaction is best-effort and cannot detect every personal identifier. "
    "Do not submit personal, confidential, or sensitive information."
)

with st.sidebar:
    st.header("Response settings")
    detail_level = st.selectbox(
        "Detail level", ["short", "standard", "detailed"], index=1
    )
    experience_level = st.selectbox(
        "Experience level", ["beginner", "intermediate"], index=0
    )

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    confirm_delete = st.checkbox("Confirm permanent deletion for this session")
    if st.button(
        "Delete persistent session",
        type="secondary",
        disabled=not confirm_delete,
        use_container_width=True,
    ):
        delete_persistent_session()
        st.rerun()

    st.divider()
    st.subheader("Response status")
    st.caption(str(st.session_state.last_status))
    resource_status = "initialized" if st.session_state.pipeline_initialized else "not loaded"
    st.caption(f"Session runtime: {resource_status}")

with st.expander("Capabilities"):
    st.markdown(
        "- Grounded conceptual explanations\n"
        "- Workflow guidance and common-error troubleshooting\n"
        "- Multi-turn follow-up questions\n"
        "- References to indexed official scikit-learn sources"
    )

with st.expander("Limitations and responsible use"):
    st.markdown(
        "- The 0.5B local model can make mistakes or omit requested inline source IDs.\n"
        "- Deterministic source records are shown separately from semantic answer quality.\n"
        "- Knowledge is limited to the packaged index and may not match every installed version.\n"
        "- Weak, ambiguous, or unsupported requests may trigger clarification or fallback.\n"
        "- Verify important code and decisions against the linked official documentation."
    )

with st.expander("Recommended prompts"):
    st.markdown(
        "1. Why should I fit `StandardScaler` only on training data?\n"
        "2. How do I combine imputation and one-hot encoding?\n"
        "3. When should I use cross-validation?\n"
        "4. Which metric fits an imbalanced classifier?\n"
        "5. How does `Pipeline` reduce data leakage?"
    )

st.info(
    "Hallucination and error disclaimer: this educational prototype is not a substitute "
    "for official documentation or expert review. Confirm consequential guidance independently."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            if message.get("status"):
                st.caption(message["status"])

prompt = st.chat_input("Ask a scikit-learn question...")
if prompt and prompt.strip():
    cleaned_prompt = prompt.strip()
    user_message = {"role": "user", "content": cleaned_prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(cleaned_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating a grounded response..."):
            try:
                result = ask_backend(cleaned_prompt, detail_level, experience_level)
                status = (
                    f"Backend: {result['generation_backend']} | "
                    f"Fallback: {result['fallback_used']} | "
                    f"Latency: {result['latency_ms']:.0f} ms"
                )
                assistant_message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "status": status,
                }
                st.session_state.messages.append(assistant_message)
                st.session_state.last_status = status
                st.markdown(result["answer"])
                render_sources(result["sources"])
                st.caption(status)
            except Exception as exc:
                safe_error = sanitize_for_logging(type(exc).__name__)
                status = f"Local error: {safe_error}"
                assistant_message = {
                    "role": "assistant",
                    "content": (
                        "The local backend could not complete this request. "
                        "Check the deployment setup and runtime logs."
                    ),
                    "sources": [],
                    "status": status,
                }
                st.session_state.messages.append(assistant_message)
                st.session_state.last_status = status
                st.error(assistant_message["content"])
                st.caption(status)

