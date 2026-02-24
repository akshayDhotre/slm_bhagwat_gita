"""
Unit tests for GitaRAG.

Run with:
    pytest tests/test_rag.py -v

All external dependencies (ChromaDB, CrossEncoder, Ollama, MLX) are mocked
so these tests run offline with no GPU or model files required.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

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

def _make_chapter_query_result(chapters: list[dict]):
    """Build a fake ChromaDB query result for chapter summaries."""
    return {
        "documents": [[f"Chapter {c['chapter']} summary text" for c in chapters]],
        "metadatas": [chapters],
    }


def _make_sloka_query_result(slokas: list[dict]):
    """Build a fake ChromaDB query result for slokas."""
    return {
        "documents": [[f"Sloka text for chapter {s['chapter']} verse {s['verse']}" for s in slokas]],
        "metadatas": [slokas],
    }


def _make_rag(cross_encoder_scores=None) -> GitaRAG:
    """Construct a GitaRAG with all external dependencies mocked."""
    with patch("chromadb.PersistentClient"), \
         patch("src.rag.CrossEncoder") as MockCE:

        mock_ce_instance = MagicMock()
        MockCE.return_value = mock_ce_instance
        if cross_encoder_scores is not None:
            mock_ce_instance.predict.return_value = cross_encoder_scores

        rag = GitaRAG(model_provider="ollama")
        rag.cross_encoder = mock_ce_instance
        return rag


# ---------------------------------------------------------------------------
# Tests: retrieve_hierarchical
# ---------------------------------------------------------------------------

class TestRetrieveHierarchical:

    def test_returns_expected_keys(self):
        rag = _make_rag(cross_encoder_scores=[0.9, 0.8, 0.7])
        chapters = [{"type": "chapter_summary", "chapter": 2, "chapter_name": "Sankhya Yoga"}]
        slokas = [
            {"type": "sloka", "chapter": 2, "verse": 47},
            {"type": "sloka", "chapter": 2, "verse": 20},
            {"type": "sloka", "chapter": 2, "verse": 19},
        ]
        rag.collection.query.side_effect = [
            _make_chapter_query_result(chapters),
            _make_sloka_query_result(slokas),
        ]

        result = rag.retrieve_hierarchical("What is duty?")

        assert "chapters" in result
        assert "slokas" in result
        assert "docs" in result["chapters"]
        assert "metas" in result["chapters"]

    def test_chapter_filter_applied_to_sloka_query(self):
        rag = _make_rag(cross_encoder_scores=[0.9])
        chapters = [{"type": "chapter_summary", "chapter": 3, "chapter_name": "Karma Yoga"}]
        slokas = [{"type": "sloka", "chapter": 3, "verse": 19}]

        rag.collection.query.side_effect = [
            _make_chapter_query_result(chapters),
            _make_sloka_query_result(slokas),
        ]

        rag.retrieve_hierarchical("How do I act without desire?")

        # Second query call should include chapter filter
        second_call_kwargs = rag.collection.query.call_args_list[1].kwargs
        where = second_call_kwargs["where"]
        assert "$and" in where
        conditions = where["$and"]
        assert {"type": "sloka"} in conditions

    def test_empty_result_when_no_chapters_found(self):
        rag = _make_rag()
        rag.collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

        result = rag.retrieve_hierarchical("some obscure query")

        assert result["chapters"]["docs"] == []
        assert result["slokas"]["docs"] == []
        # Sloka query should NOT be called if no chapters were found
        assert rag.collection.query.call_count == 1

    def test_reranker_selects_top_n(self):
        from src import config
        n_slokas = config.N_SLOKA_CANDIDATES
        n_top = config.N_TOP_RESULTS

        rag = _make_rag(cross_encoder_scores=list(range(n_slokas, 0, -1)))
        chapters = [{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]
        slokas = [{"type": "sloka", "chapter": 2, "verse": i} for i in range(1, n_slokas + 1)]

        rag.collection.query.side_effect = [
            _make_chapter_query_result(chapters),
            _make_sloka_query_result(slokas),
        ]

        result = rag.retrieve_hierarchical("test query")

        assert len(result["slokas"]["docs"]) == n_top


# ---------------------------------------------------------------------------
# Tests: generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:

    def test_returns_three_tuple(self):
        rag = _make_rag(cross_encoder_scores=[0.9])
        rag.collection.query.side_effect = [
            _make_chapter_query_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_query_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        ]
        rag._generate_with_ollama = MagicMock(return_value="Mock wisdom answer")

        result = rag.generate_answer("What is dharma?")

        assert isinstance(result, tuple)
        assert len(result) == 3
        answer, citations, context = result
        assert isinstance(answer, str)
        assert isinstance(citations, list)
        assert isinstance(context, str)

    def test_context_contains_retrieved_text(self):
        rag = _make_rag(cross_encoder_scores=[0.9])
        rag.collection.query.side_effect = [
            _make_chapter_query_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Sankhya Yoga"}]),
            _make_sloka_query_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        ]
        rag._generate_with_ollama = MagicMock(return_value="Answer")

        _, _, context = rag.generate_answer("test")

        assert "RELEVANT CHAPTER CONTEXT" in context
        assert "RELEVANT VERSES" in context

    def test_ollama_provider_calls_ollama(self):
        rag = _make_rag(cross_encoder_scores=[0.9])
        rag.collection.query.side_effect = [
            _make_chapter_query_result([{"type": "chapter_summary", "chapter": 2, "chapter_name": "Yoga"}]),
            _make_sloka_query_result([{"type": "sloka", "chapter": 2, "verse": 47}]),
        ]
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
             patch("src.rag.CrossEncoder"), \
             patch("src.rag.MLX_AVAILABLE", True), \
             patch("src.rag.load") as mock_load:

            with pytest.raises(FileNotFoundError, match="MLX model not found"):
                GitaRAG(model_provider="mlx", mlx_model_path=fake_path)

            mock_load.assert_not_called()

    def test_mlx_unavailable_raises_import_error(self):
        with patch("chromadb.PersistentClient"), \
             patch("src.rag.CrossEncoder"), \
             patch("src.rag.MLX_AVAILABLE", False):

            with pytest.raises(ImportError, match="mlx-lm"):
                GitaRAG(model_provider="mlx")
