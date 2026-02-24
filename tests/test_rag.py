"""
Unit tests for GitaRAG.

Run with:
    pytest tests/test_rag.py -v

All external dependencies (ChromaDB, CrossEncoder, Ollama, MLX) are mocked
so these tests run offline with no GPU or model files required.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Patch heavy imports before GitaRAG is loaded
# ---------------------------------------------------------------------------
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())
sys.modules.setdefault("ollama", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())

from src.rag import GitaRAG  # noqa: E402 — must come after mocking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_QUERY = {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def _make_chapter_result(chapters: list[dict]) -> dict:
    return {
        "documents": [[f"Chapter {c['chapter']} summary" for c in chapters]],
        "metadatas": [chapters],
        "distances": [[i * 0.1 for i in range(len(chapters))]],
    }


def _make_sloka_result(slokas: list[dict], distances: list[float] | None = None) -> dict:
    if distances is None:
        distances = [i * 0.1 for i in range(len(slokas))]
    return {
        "documents": [[f"Sloka ch{s['chapter']} v{s['verse']}" for s in slokas]],
        "metadatas": [slokas],
        "distances": [distances],
    }


def _make_rag() -> GitaRAG:
    """Construct a GitaRAG with all external dependencies mocked."""
    with patch("chromadb.PersistentClient"):
        return GitaRAG(model_provider="ollama")


def _set_query_side_effect(rag, chapter_result, primary_sloka_result, global_result=None):
    """
    Wire up the 3 collection.query calls:
      1. chapter summary search
      2. chapter-filtered sloka search  (only called when chapters found)
      3. global supplement search        (always called)
    """
    if global_result is None:
        global_result = _EMPTY_QUERY
    rag.collection.query.side_effect = [
        chapter_result,
        primary_sloka_result,
        global_result,
    ]


# ---------------------------------------------------------------------------
# Tests: retrieve_hierarchical
# ---------------------------------------------------------------------------

class TestRetrieveHierarchical:

    def test_returns_expected_keys(self):
        rag = _make_rag()
        chapters = [{"type": "chapter_summary", "chapter": 2, "chapter_name": "Sankhya Yoga"}]
        slokas = [
            {"type": "sloka", "chapter": 2, "verse": 47},
            {"type": "sloka", "chapter": 2, "verse": 20},
            {"type": "sloka", "chapter": 2, "verse": 19},
        ]
        _set_query_side_effect(rag, _make_chapter_result(chapters), _make_sloka_result(slokas))

        result = rag.retrieve_hierarchical("What is duty?")

        assert "chapters" in result
        assert "slokas" in result
        assert "docs" in result["chapters"]
        assert "metas" in result["chapters"]

    def test_chapter_filter_applied_to_sloka_query(self):
        rag = _make_rag()
        chapters = [{"type": "chapter_summary", "chapter": 3, "chapter_name": "Karma Yoga"}]
        slokas = [{"type": "sloka", "chapter": 3, "verse": 19}]
        _set_query_side_effect(rag, _make_chapter_result(chapters), _make_sloka_result(slokas))

        rag.retrieve_hierarchical("How do I act without desire?")

        # Call 2 (index 1) is the chapter-filtered sloka search
        second_call_kwargs = rag.collection.query.call_args_list[1].kwargs
        where = second_call_kwargs["where"]
        assert "$and" in where
        assert {"type": "sloka"} in where["$and"]

    def test_empty_chapters_skips_filtered_search_but_runs_global(self):
        rag = _make_rag()
        # No chapters found → no filtered sloka query, but global supplement still runs
        rag.collection.query.side_effect = [
            _EMPTY_QUERY,   # call 1: chapter summaries → empty
            _EMPTY_QUERY,   # call 2: global supplement → empty
        ]

        result = rag.retrieve_hierarchical("some obscure query")

        assert result["chapters"]["docs"] == []
        assert result["slokas"]["docs"] == []
        # Chapter search + global supplement = 2 calls (filtered skipped)
        assert rag.collection.query.call_count == 2

    def test_reranker_selects_top_n(self):
        from src import config
        n_primary = config.N_SLOKA_CANDIDATES
        n_top = config.N_TOP_RESULTS

        rag = _make_rag()
        chapters = [{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]
        slokas = [{"type": "sloka", "chapter": 2, "verse": i} for i in range(1, n_primary + 1)]
        # Distances auto-assigned as [0.0, 0.1, ..., (n_primary-1)*0.1]; top N = lowest distances
        _set_query_side_effect(rag, _make_chapter_result(chapters), _make_sloka_result(slokas))

        result = rag.retrieve_hierarchical("test query")

        assert len(result["slokas"]["docs"]) == n_top

    def test_global_supplement_adds_cross_chapter_verses(self):
        """Verses from chapters not in the primary filter should appear after merge+sort."""
        rag = _make_rag()
        chapters = [{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]
        primary = [{"type": "sloka", "chapter": 2, "verse": i} for i in range(1, 4)]
        # Global supplement includes verses from chapters 13 and 15 (not in primary filter).
        # Assign low distances so they rank inside top-N after merge.
        global_slokas = [
            {"type": "sloka", "chapter": 13, "verse": 1},
            {"type": "sloka", "chapter": 13, "verse": 2},
            {"type": "sloka", "chapter": 15, "verse": 1},
            {"type": "sloka", "chapter": 2,  "verse": 1},  # duplicate — should be deduped
        ]
        _set_query_side_effect(
            rag,
            _make_chapter_result(chapters),
            _make_sloka_result(primary),
            _make_sloka_result(global_slokas, distances=[0.05, 0.15, 0.25, 0.35]),
        )

        result = rag.retrieve_hierarchical("What is the self?")

        # Chapters present in results should include cross-chapter supplement
        result_chapters = {m["chapter"] for m in result["slokas"]["metas"]}
        assert len(result_chapters) > 1  # more than just chapter 2

    def test_duplicate_verses_deduped_across_primary_and_global(self):
        rag = _make_rag()
        chapters = [{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]
        # Primary and global return the same 3 verses
        same_slokas = [
            {"type": "sloka", "chapter": 2, "verse": 47},
            {"type": "sloka", "chapter": 2, "verse": 20},
            {"type": "sloka", "chapter": 2, "verse": 19},
        ]
        _set_query_side_effect(
            rag,
            _make_chapter_result(chapters),
            _make_sloka_result(same_slokas),
            _make_sloka_result(same_slokas),  # exact duplicates
        )

        result = rag.retrieve_hierarchical("test")

        # Should still only have 3 unique slokas, not 6
        result_keys = [(m["chapter"], m["verse"]) for m in result["slokas"]["metas"]]
        assert len(result_keys) == len(set(result_keys))


# ---------------------------------------------------------------------------
# Tests: generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:

    def test_returns_three_tuple(self):
        rag = _make_rag()
        _set_query_side_effect(
            rag,
            _make_chapter_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        )
        rag._generate_with_ollama = MagicMock(return_value="Mock wisdom answer")

        result = rag.generate_answer("What is dharma?")

        assert isinstance(result, tuple) and len(result) == 3
        answer, citations, context = result
        assert isinstance(answer, str)
        assert isinstance(citations, list)
        assert isinstance(context, str)

    def test_context_contains_retrieved_text(self):
        rag = _make_rag()
        _set_query_side_effect(
            rag,
            _make_chapter_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Sankhya Yoga"}]),
            _make_sloka_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        )
        rag._generate_with_ollama = MagicMock(return_value="Answer")

        _, _, context = rag.generate_answer("test")

        assert "RELEVANT CHAPTER CONTEXT" in context
        assert "RELEVANT VERSES" in context

    def test_ollama_provider_calls_ollama_not_mlx(self):
        rag = _make_rag()
        _set_query_side_effect(
            rag,
            _make_chapter_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        )
        rag._generate_with_ollama = MagicMock(return_value="Ollama response")
        rag._generate_with_mlx = MagicMock(return_value="MLX response")

        rag.generate_answer("test")

        rag._generate_with_ollama.assert_called_once()
        rag._generate_with_mlx.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: MLX error handling
# ---------------------------------------------------------------------------

class TestMLXErrorHandling:

    def test_missing_model_path_raises_file_not_found(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent_model")
        with patch("chromadb.PersistentClient"), \
             patch("src.rag.MLX_AVAILABLE", True), \
             patch("src.rag._mlx_load") as mock_load:

            with pytest.raises(FileNotFoundError, match="MLX model not found"):
                GitaRAG(model_provider="mlx", mlx_model_path=fake_path)

            mock_load.assert_not_called()

    def test_mlx_unavailable_raises_import_error(self):
        with patch("chromadb.PersistentClient"), \
             patch("src.rag.MLX_AVAILABLE", False):

            with pytest.raises(ImportError, match="mlx-lm"):
                GitaRAG(model_provider="mlx")


# ---------------------------------------------------------------------------
# Tests: _build_context and _build_prompt
# ---------------------------------------------------------------------------

class TestBuildContextAndPrompt:

    def test_build_context_empty_retrieved(self):
        rag = _make_rag()
        context, citations = rag._build_context({"chapters": {"docs": [], "metas": []}, "slokas": {"docs": [], "metas": []}})
        assert context == ""
        assert citations == []

    def test_build_context_chapter_section_present(self):
        rag = _make_rag()
        retrieved = {
            "chapters": {
                "docs": ["Chapter 2 summary text"],
                "metas": [{"chapter": 2, "chapter_name": "Sankhya Yoga"}],
            },
            "slokas": {"docs": [], "metas": []},
        }
        context, citations = rag._build_context(retrieved)
        assert "RELEVANT CHAPTER CONTEXT" in context
        assert "Chapter 2" in context
        assert "Sankhya Yoga" in context
        assert len(citations) == 1

    def test_build_context_sloka_section_present(self):
        rag = _make_rag()
        retrieved = {
            "chapters": {"docs": [], "metas": []},
            "slokas": {
                "docs": ["Verse text here"],
                "metas": [{"chapter": 6, "verse": 35}],
            },
        }
        context, citations = rag._build_context(retrieved)
        assert "RELEVANT VERSES" in context
        assert "Chapter 6" in context
        assert "Verse 35" in context
        assert len(citations) == 1

    def test_build_context_both_sections(self):
        rag = _make_rag()
        retrieved = {
            "chapters": {
                "docs": ["Chapter summary"],
                "metas": [{"chapter": 3, "chapter_name": "Karma Yoga"}],
            },
            "slokas": {
                "docs": ["Sloka text"],
                "metas": [{"chapter": 3, "verse": 19}],
            },
        }
        context, citations = rag._build_context(retrieved)
        assert "RELEVANT CHAPTER CONTEXT" in context
        assert "RELEVANT VERSES" in context
        assert len(citations) == 2

    def test_build_prompt_contains_query_and_context(self):
        rag = _make_rag()
        prompt = rag._build_prompt("What is duty?", "Some context here.")
        assert "What is duty?" in prompt
        assert "Some context here." in prompt
        assert "Answer:" in prompt

    def test_build_prompt_wisdom_companion_tone(self):
        rag = _make_rag()
        prompt = rag._build_prompt("Why do I suffer?", "context")
        # Prompt must instruct not to quote verses
        assert "Do NOT quote verses" in prompt
        assert "wisdom companion" in prompt.lower() or "Bhagavad Gita" in prompt


# ---------------------------------------------------------------------------
# Tests: stream_answer
# ---------------------------------------------------------------------------

class TestStreamAnswer:

    def test_stream_answer_returns_three_tuple(self):
        rag = _make_rag()
        _set_query_side_effect(
            rag,
            _make_chapter_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        )
        rag._stream_with_ollama = MagicMock(return_value=iter(["chunk1", "chunk2"]))

        result = rag.stream_answer("What is karma?")
        assert isinstance(result, tuple) and len(result) == 3
        chunks, citations, context = result
        assert isinstance(citations, list)
        assert isinstance(context, str)

    def test_stream_answer_yields_chunks(self):
        rag = _make_rag()
        _set_query_side_effect(
            rag,
            _make_chapter_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        )
        rag._stream_with_ollama = MagicMock(return_value=iter(["Hello", " world"]))

        chunks, _, _ = rag.stream_answer("test")
        collected = "".join(chunks)
        assert collected == "Hello world"

    def test_stream_answer_uses_ollama_provider(self):
        rag = _make_rag()
        _set_query_side_effect(
            rag,
            _make_chapter_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        )
        rag._stream_with_ollama = MagicMock(return_value=iter(["ok"]))
        rag._stream_with_mlx = MagicMock(return_value=iter(["mlx"]))

        rag.stream_answer("test")

        rag._stream_with_ollama.assert_called_once()
        rag._stream_with_mlx.assert_not_called()

    def test_stream_answer_uses_mlx_provider(self):
        rag = _make_rag()
        rag.provider = "mlx"
        _set_query_side_effect(
            rag,
            _make_chapter_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        )
        rag._stream_with_ollama = MagicMock(return_value=iter(["ollama"]))
        rag._stream_with_mlx = MagicMock(return_value=iter(["mlx chunk"]))

        rag.stream_answer("test")

        rag._stream_with_mlx.assert_called_once()
        rag._stream_with_ollama.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: generate_answer with MLX provider
# ---------------------------------------------------------------------------

class TestGenerateAnswerMLX:

    def test_mlx_provider_calls_mlx_not_ollama(self):
        rag = _make_rag()
        rag.provider = "mlx"
        _set_query_side_effect(
            rag,
            _make_chapter_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        )
        rag._generate_with_ollama = MagicMock(return_value="Ollama response")
        rag._generate_with_mlx = MagicMock(return_value="MLX response")

        answer, _, _ = rag.generate_answer("test")

        rag._generate_with_mlx.assert_called_once()
        rag._generate_with_ollama.assert_not_called()
        assert answer == "MLX response"


# ---------------------------------------------------------------------------
# Tests: _format_sources (from main.py)
# ---------------------------------------------------------------------------

def _get_format_sources():
    """Import _format_sources from main without executing main()."""
    import main
    return main._format_sources


class TestFormatSources:

    def test_empty_citations(self):
        fmt = _get_format_sources()
        assert fmt([]) == ""

    def test_chapter_summaries_excluded(self):
        fmt = _get_format_sources()
        citations = [{"type": "chapter_summary", "chapter": 2, "verse": None}]
        assert fmt(citations) == ""

    def test_sloka_citation_formatted(self):
        fmt = _get_format_sources()
        citations = [{"type": "sloka", "chapter": 6, "verse": 35, "chapter_name": "Dhyana Yoga"}]
        result = fmt(citations)
        assert "Chapter 6" in result
        assert "Verse 35" in result
        assert "Dhyana Yoga" in result

    def test_duplicate_citations_deduped(self):
        fmt = _get_format_sources()
        same = {"type": "sloka", "chapter": 2, "verse": 47, "chapter_name": "Sankhya Yoga"}
        result = fmt([same, same, same])
        # Only one entry for ch2v47
        assert result.count("Verse 47") == 1

    def test_mixed_types_only_slokas_shown(self):
        fmt = _get_format_sources()
        citations = [
            {"type": "chapter_summary", "chapter": 3},
            {"type": "sloka", "chapter": 3, "verse": 19, "chapter_name": "Karma Yoga"},
        ]
        result = fmt(citations)
        assert "Verse 19" in result
        assert result.count("\n") == 0  # single line, no extra entries
