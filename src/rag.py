import logging
import os
from typing import Iterator, Optional

import chromadb
from chromadb.utils import embedding_functions
import ollama

from src import config

# MLX imports — kept as module-level callables so pyright knows they're always bound.
_mlx_load = None
_mlx_generate = None
_mlx_stream_generate = None
MLX_AVAILABLE = False
MLX_STREAM_AVAILABLE = False

try:
    from mlx_lm import load as _mlx_load, generate as _mlx_generate  # type: ignore[assignment]
    MLX_AVAILABLE = True
    try:
        from mlx_lm import stream_generate as _mlx_stream_generate  # type: ignore[assignment]
        MLX_STREAM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

logger = logging.getLogger(__name__)


class GitaRAG:
    def __init__(
        self,
        model_provider="ollama",
        ollama_model=config.DEFAULT_OLLAMA_MODEL,
        mlx_model_path=config.DEFAULT_MLX_MODEL_PATH,
    ):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        self.collection = self.client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=self.sentence_transformer_ef,
        )

        self.provider = model_provider
        self.ollama_model = ollama_model
        self.mlx_model_path = mlx_model_path
        self.mlx_model = None
        self.mlx_tokenizer = None

        if self.provider == "mlx":
            if not MLX_AVAILABLE:
                raise ImportError("mlx-lm is not installed. Run: pip install mlx-lm")
            if not os.path.exists(self.mlx_model_path):
                raise FileNotFoundError(
                    f"MLX model not found at '{self.mlx_model_path}'. "
                    "Run the fine-tuning notebook first, or pass a valid --model path."
                )
            logger.info("Loading MLX model from %s ...", self.mlx_model_path)
            self.mlx_model, self.mlx_tokenizer = _mlx_load(self.mlx_model_path)  # type: ignore[misc]
            logger.info("MLX model loaded successfully.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_hierarchical(self, query: str) -> dict:
        """
        Two-stage retrieval with cross-chapter fallback.

        Stage 1: Find the most relevant chapter summaries (broad theme search).
        Stage 2: Retrieve slokas from those chapters (primary), then supplement
                 with a small global search to catch cross-chapter wisdom.
        Stage 3: Merge unique candidates, sort by embedding distance, return top-N.

        Re-ranking uses ChromaDB's embedding distances directly — no separate
        cross-encoder model needed.  Lower distance = more similar to the query.
        """
        # --- Stage 1: chapter summary search ---
        chapter_results = self.collection.query(
            query_texts=[query],
            n_results=config.N_CHAPTER_RESULTS,
            where={"type": "chapter_summary"},
        )

        chapter_docs, chapter_metas, found_chapters = [], [], []
        if chapter_results["documents"] and chapter_results["documents"][0]:
            chapter_docs = chapter_results["documents"][0]
            chapter_metas = chapter_results["metadatas"][0]
            found_chapters = list({m["chapter"] for m in chapter_metas})

        logger.debug("Primary chapters: %s", found_chapters)

        # --- Stage 2a: chapter-filtered sloka search ---
        primary_docs, primary_metas, primary_distances = [], [], []
        if found_chapters:
            where_clause = {
                "$and": [
                    {"type": "sloka"},
                    {"chapter": {"$in": found_chapters}},
                ]
            }
            res = self.collection.query(
                query_texts=[query],
                n_results=config.N_SLOKA_CANDIDATES,
                where=where_clause,
            )
            primary_docs      = (res["documents"]  or [[]])[0]
            primary_metas     = (res["metadatas"]  or [[]])[0]
            primary_distances = (res["distances"]  or [[]])[0]

        # --- Stage 2b: global supplement (cross-chapter fallback) ---
        global_res = self.collection.query(
            query_texts=[query],
            n_results=config.N_GLOBAL_SUPPLEMENT,
            where={"type": "sloka"},
        )
        global_docs      = (global_res["documents"]  or [[]])[0]
        global_metas     = (global_res["metadatas"]  or [[]])[0]
        global_distances = (global_res["distances"]  or [[]])[0]

        # --- Merge: deduplicate by (chapter, verse), first seen wins ---
        seen_keys = set()
        merged_docs, merged_metas, merged_distances = [], [], []
        for doc, meta, dist in (
            list(zip(primary_docs, primary_metas, primary_distances))
            + list(zip(global_docs, global_metas, global_distances))
        ):
            key = (meta.get("chapter"), meta.get("verse"))
            if key not in seen_keys:
                seen_keys.add(key)
                merged_docs.append(doc)
                merged_metas.append(meta)
                merged_distances.append(dist)

        # Log chapters actually present in the merged pool
        merged_chapters = sorted({m.get("chapter") for m in merged_metas})
        new_chapters = sorted(set(merged_chapters) - set(found_chapters))
        if new_chapters:
            logger.debug("Cross-chapter supplement added chapters: %s", new_chapters)

        # --- Stage 3: sort by embedding distance (ascending = most similar first) ---
        final_docs, final_metas = [], []
        if merged_docs:
            scored = sorted(
                zip(merged_distances, merged_docs, merged_metas),
                key=lambda x: x[0],
            )
            top = scored[: config.N_TOP_RESULTS]
            final_docs  = [x[1] for x in top]
            final_metas = [x[2] for x in top]

        return {
            "chapters": {"docs": chapter_docs, "metas": chapter_metas},
            "slokas":   {"docs": final_docs,   "metas": final_metas},
        }

    # ------------------------------------------------------------------
    # Context & prompt builders (shared by generate and stream)
    # ------------------------------------------------------------------

    def _build_context(self, retrieved: dict) -> tuple[str, list]:
        context_str = ""
        citation_list = []

        if retrieved["chapters"]["docs"]:
            context_str += "=== RELEVANT CHAPTER CONTEXT ===\n"
            for i, text in enumerate(retrieved["chapters"]["docs"]):
                meta = retrieved["chapters"]["metas"][i]
                context_str += (
                    f"[Chapter {meta.get('chapter')}: "
                    f"{meta.get('chapter_name', 'Summary')}]\n{text}\n\n"
                )
                citation_list.append(meta)

        if retrieved["slokas"]["docs"]:
            context_str += "=== RELEVANT VERSES (SLOKAS) ===\n"
            for i, text in enumerate(retrieved["slokas"]["docs"]):
                meta = retrieved["slokas"]["metas"][i]
                context_str += (
                    f"[Chapter {meta.get('chapter')}, "
                    f"Verse {meta.get('verse')}]\n{text}\n\n"
                )
                citation_list.append(meta)

        return context_str, citation_list

    def _build_prompt(self, query: str, context_str: str) -> str:
        return f"""You are a wisdom companion grounded in the teachings of the Bhagavad Gita.

Your task is to answer the user's personal question using the provided context, which contains relevant passages or interpretations from the Gita.
Base your response primarily on this context. Do not introduce ideas that clearly contradict it.

Guidelines:
- Express the wisdom in your own natural, contemporary language.
- Do NOT quote verses, mention chapter or verse numbers, or sound like a scripture reference.
- Speak as a fellow human sharing lived understanding—calm, grounded, and compassionate.
- Keep the guidance practical and applicable to modern life.
- Avoid moralizing or preaching; offer clarity, not instruction.

If the context does not directly address the user's question:
- Say so honestly and briefly.
- Then offer a related insight consistent with the broader spirit of the Bhagavad Gita
  (duty, detachment, self-awareness, equanimity, devotion, or disciplined action).

Use only the information in the context and generally accepted themes of the Gita.
Do not invent specific teachings or attribute ideas explicitly to Krishna or Arjuna.

Context:
{context_str}

User Question:
{query}

Answer:"""

    # ------------------------------------------------------------------
    # Generation (non-streaming)
    # ------------------------------------------------------------------

    def _generate_with_ollama(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            return (
                f"Error calling Ollama: {e}. "
                f"Make sure Ollama is running and '{self.ollama_model}' is pulled."
            )

    def _generate_with_mlx(self, prompt: str) -> str:
        if self.mlx_model is None or self.mlx_tokenizer is None:
            raise RuntimeError("MLX model is not loaded.")
        if _mlx_generate is None:
            raise RuntimeError("mlx_lm.generate not available.")
        if hasattr(self.mlx_tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.mlx_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        return _mlx_generate(  # type: ignore[misc]
            self.mlx_model, self.mlx_tokenizer,
            prompt=formatted_prompt, verbose=False, max_tokens=config.MLX_MAX_TOKENS,
        )

    def generate_answer(self, query: str) -> tuple[str, list, str]:
        """Return (answer_str, citation_list, context_str)."""
        retrieved = self.retrieve_hierarchical(query)
        context_str, citation_list = self._build_context(retrieved)
        prompt = self._build_prompt(query, context_str)

        if self.provider == "mlx":
            answer = self._generate_with_mlx(prompt)
        else:
            answer = self._generate_with_ollama(prompt)

        return answer, citation_list, context_str

    # ------------------------------------------------------------------
    # Generation (streaming)
    # ------------------------------------------------------------------

    def _stream_with_ollama(self, prompt: str) -> Iterator[str]:
        """Yield text chunks from Ollama's streaming API."""
        try:
            stream = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield token
        except Exception as e:
            logger.error("Ollama streaming failed: %s", e)
            yield (
                f"\nError calling Ollama: {e}. "
                f"Make sure Ollama is running and '{self.ollama_model}' is pulled."
            )

    def _stream_with_mlx(self, prompt: str) -> Iterator[str]:
        """Yield text chunks from MLX. Falls back to full generate if stream_generate unavailable."""
        if self.mlx_model is None or self.mlx_tokenizer is None:
            raise RuntimeError("MLX model is not loaded.")
        if hasattr(self.mlx_tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.mlx_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        if MLX_STREAM_AVAILABLE and _mlx_stream_generate is not None:
            for response in _mlx_stream_generate(  # type: ignore[misc]
                self.mlx_model, self.mlx_tokenizer,
                prompt=formatted_prompt, max_tokens=config.MLX_MAX_TOKENS,
            ):
                yield response.text
        else:
            # Older mlx_lm — generate full response and yield as one block
            if _mlx_generate is None:
                raise RuntimeError("mlx_lm.generate not available.")
            yield _mlx_generate(  # type: ignore[misc]
                self.mlx_model, self.mlx_tokenizer,
                prompt=formatted_prompt, verbose=False, max_tokens=config.MLX_MAX_TOKENS,
            )

    def stream_answer(self, query: str) -> tuple[Iterator[str], list, str]:
        """
        Stream-friendly variant of generate_answer.
        Returns (chunk_iterator, citation_list, context_str).
        Caller must consume chunk_iterator to receive the full answer.
        """
        retrieved = self.retrieve_hierarchical(query)
        context_str, citation_list = self._build_context(retrieved)
        prompt = self._build_prompt(query, context_str)

        if self.provider == "mlx":
            chunks = self._stream_with_mlx(prompt)
        else:
            chunks = self._stream_with_ollama(prompt)

        return chunks, citation_list, context_str
