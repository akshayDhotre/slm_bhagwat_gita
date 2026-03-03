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
    show_pipeline = st.checkbox("Show retrieval pipeline", value=False)
    st.caption("Powered by ChromaDB · Sentence Transformers · Ollama / MLX")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": str, "content": str}
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []      # list of formatted source strings
if "last_pipeline" not in st.session_state:
    st.session_state.last_pipeline = None   # dict with debug_info + llm_prompt

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
# Pipeline panel (from last response) — persisted via session state after rerun
# ---------------------------------------------------------------------------

if show_pipeline and st.session_state.last_pipeline:
    p = st.session_state.last_pipeline
    debug = p["debug_info"]

    with st.expander("Retrieval Pipeline", expanded=True):
        # Stage 1
        st.markdown("**Stage 1** — Chapter Discovery")
        for m in {m["chapter"]: m for m in debug.get("chapter_metas", [])}.values():
            st.write(f"  › Ch. {m.get('chapter')}: {m.get('chapter_name', '')}")
        if not debug.get("chapter_metas"):
            st.write("  › No specific chapters matched — global search only")

        st.divider()

        # Stage 2
        st.markdown("**Stage 2** — Candidate Pool")
        st.write(
            f"  › {debug.get('primary_sloka_count', 0)} from target chapters  "
            f"+ {debug.get('global_sloka_count', 0)} global supplement"
        )
        if debug.get("new_chapters"):
            st.write(
                f"  › Cross-chapter additions: "
                f"Ch. {', '.join(str(c) for c in debug['new_chapters'])}"
            )

        st.divider()

        # Stage 3
        st.markdown("**Stage 3** — Re-ranked Verses")
        st.write(
            f"  › Top {len(debug.get('top_verses', []))} selected from "
            f"{debug.get('merged_count', 0)} unique candidates"
        )
        for v in debug.get("top_verses", []):
            snippet = v["text_snippet"][:90]
            st.caption(
                f"[{v['distance']}] Ch. {v['chapter']} · Verse {v['verse']} — {snippet}…"
            )

        st.divider()

        # Stage 4
        st.markdown("**Stage 4** — System Prompt")
        with st.expander("View system prompt"):
            st.code(p["llm_prompt"], language="text")

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

    with st.chat_message("assistant"):
        try:
            if show_pipeline:
                # ── Live pipeline display ──────────────────────────────────
                with st.status("Running retrieval pipeline...", expanded=True) as pipeline_status:

                    # Stage 1: Chapter discovery
                    st.write("**Stage 1** — Searching chapter themes...")
                    chapter_docs, chapter_metas, found_chapters = rag_or_error.query_chapters(prompt)
                    if found_chapters:
                        seen_ch: set = set()
                        for m in chapter_metas:
                            ch = m.get("chapter")
                            if ch not in seen_ch:
                                seen_ch.add(ch)
                                st.write(f"  › Ch. {ch}: {m.get('chapter_name', '')}")
                    else:
                        st.write("  › No specific chapters matched — using global search only")

                    # Stage 2: Sloka retrieval
                    st.write("**Stage 2** — Retrieving candidate verses...")
                    (primary_docs, primary_metas, primary_distances,
                     global_docs,  global_metas,  global_distances) = rag_or_error.query_slokas(prompt, found_chapters)
                    st.write(
                        f"  › {len(primary_docs)} from target chapters  "
                        f"+ {len(global_docs)} global supplement"
                    )

                    # Stage 3: Re-ranking
                    st.write("**Stage 3** — Re-ranking by embedding distance...")
                    final_docs, final_metas, top_debug, new_chapters, merged_count = (
                        rag_or_error.merge_and_rerank(
                            primary_docs, primary_metas, primary_distances,
                            global_docs,  global_metas,  global_distances,
                            found_chapters,
                        )
                    )
                    st.write(
                        f"  › Top {len(top_debug)} verses selected from "
                        f"{merged_count} unique candidates"
                    )
                    if new_chapters:
                        st.write(
                            f"  › Cross-chapter additions: "
                            f"Ch. {', '.join(str(c) for c in new_chapters)}"
                        )
                    for v in top_debug:
                        snippet = v["text_snippet"][:90]
                        st.caption(
                            f"[{v['distance']}] Ch. {v['chapter']} · "
                            f"Verse {v['verse']} — {snippet}…"
                        )

                    # Build retrieved dict and get chunk iterator + prompt
                    retrieved = {
                        "chapters": {"docs": chapter_docs, "metas": chapter_metas},
                        "slokas":   {"docs": final_docs,   "metas": final_metas},
                    }
                    citations = []   # will be populated by stream_from_retrieved
                    chunks, citations, _, llm_prompt = rag_or_error.stream_from_retrieved(retrieved, prompt)
                    debug_info = {
                        "chapter_metas": chapter_metas,
                        "primary_sloka_count": len(primary_docs),
                        "global_sloka_count": len(global_docs),
                        "merged_count": merged_count,
                        "new_chapters": new_chapters,
                        "top_verses": top_debug,
                    }

                    # Stage 4: show the system prompt before collapsing
                    st.write("**Stage 4** — Prompt assembled. Streaming response...")
                    with st.expander("View system prompt"):
                        st.code(llm_prompt, language="text")

                    pipeline_status.update(
                        label="Retrieval complete — streaming response...",
                        state="complete",
                        expanded=True,
                    )

                # Stream the response below the (still-open) status container
                answer = st.write_stream(chunks)

            else:
                # ── Standard path (no pipeline display) ───────────────────
                chunks, citations, _, debug_info, llm_prompt = rag_or_error.stream_answer(prompt)
                answer = st.write_stream(chunks)

        except Exception as e:
            answer = f"Error: {e}"
            st.error(answer)
            citations, debug_info, llm_prompt = [], {}, ""

    # Persist answer, sources, and pipeline snapshot
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = _format_sources(citations)
    st.session_state.last_pipeline = {"debug_info": debug_info, "llm_prompt": llm_prompt}
    st.rerun()
