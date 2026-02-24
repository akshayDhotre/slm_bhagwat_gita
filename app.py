import logging

import streamlit as st

from src import config
from src.rag import GitaRAG

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bhagavad Gita Wisdom Companion",
    page_icon="🕉",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_sources(citation_list: list) -> list[str]:
    """Return deduplicated sloka citation strings (chapter summaries excluded)."""
    seen: set[str] = set()
    lines: list[str] = []
    for meta in citation_list:
        if meta.get("type") == "chapter_summary":
            continue
        key = f"ch{meta.get('chapter')}v{meta.get('verse')}"
        if key not in seen:
            seen.add(key)
            lines.append(
                f"Chapter {meta.get('chapter')}: {meta.get('chapter_name', '')}, "
                f"Verse {meta.get('verse', '?')}"
            )
    return lines


@st.cache_resource(show_spinner="Loading model...")
def load_rag(provider: str, model: str) -> GitaRAG | str:
    """Construct and cache a GitaRAG instance. Returns error string on failure."""
    try:
        if provider == "ollama":
            return GitaRAG(model_provider="ollama", ollama_model=model)
        else:
            return GitaRAG(model_provider="mlx", mlx_model_path=model)
    except (FileNotFoundError, ImportError, Exception) as e:
        return str(e)


# ---------------------------------------------------------------------------
# Sidebar — model configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model Settings")

    provider = st.radio(
        "Provider",
        ["ollama", "mlx"],
        index=0,
        help="'ollama' requires Ollama running locally. 'mlx' requires mlx-lm and a fine-tuned model.",
    )

    if provider == "ollama":
        model_input = st.text_input("Ollama model", value=config.DEFAULT_OLLAMA_MODEL)
    else:
        model_input = st.text_input("MLX model path", value=config.DEFAULT_MLX_MODEL_PATH)

    if st.button("Load model", use_container_width=True):
        # Clear the cache so the next call rebuilds with new settings
        load_rag.clear()
        st.session_state.pop("rag_error", None)

    st.divider()
    st.caption("Powered by ChromaDB · Sentence Transformers · Ollama / MLX")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": str, "content": str}
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []      # list of formatted source strings

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🕉 Bhagavad Gita Wisdom Companion")
st.caption("Ask any question and receive guidance grounded in the Bhagavad Gita.")

# ---------------------------------------------------------------------------
# Load RAG (or show error)
# ---------------------------------------------------------------------------

rag_or_error = load_rag(provider, model_input)
rag_ready = isinstance(rag_or_error, GitaRAG)

if not rag_ready:
    st.error(
        f"**Could not load model:** {rag_or_error}\n\n"
        "Make sure Ollama is running (`ollama serve`) and the model is pulled, "
        "or check the MLX model path. Then click **Load model** in the sidebar."
    )

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Sources panel (from last response)
# ---------------------------------------------------------------------------

if st.session_state.last_sources:
    with st.expander("Sources", expanded=True):
        for src in st.session_state.last_sources:
            st.markdown(f"- {src}")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question...", disabled=not rag_ready):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the assistant response
    with st.chat_message("assistant"):
        try:
            chunks, citations, _ = rag_or_error.stream_answer(prompt)
            answer = st.write_stream(chunks)
        except Exception as e:
            answer = f"Error: {e}"
            st.error(answer)

    # Persist answer and update sources
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = _format_sources(citations)
    st.rerun()
