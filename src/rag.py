import logging
import os

import chromadb
from chromadb.utils import embedding_functions
import ollama
from sentence_transformers import CrossEncoder

from src import config

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

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
        self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)

        self.provider = model_provider
        self.ollama_model = ollama_model
        self.mlx_model_path = mlx_model_path
        self.mlx_model = None
        self.mlx_tokenizer = None

        if self.provider == "mlx":
            if not MLX_AVAILABLE:
                raise ImportError(
                    "mlx-lm is not installed. Run: pip install mlx-lm"
                )
            if not os.path.exists(self.mlx_model_path):
                raise FileNotFoundError(
                    f"MLX model not found at '{self.mlx_model_path}'. "
                    "Run the fine-tuning notebook first, or pass a valid --model path."
                )
            logger.info("Loading MLX model from %s ...", self.mlx_model_path)
            self.mlx_model, self.mlx_tokenizer = load(self.mlx_model_path)
            logger.info("MLX model loaded successfully.")

    def retrieve_hierarchical(self, query):
        # Step 1: Broad search — find relevant chapter summaries to pin down the theme
        chapter_results = self.collection.query(
            query_texts=[query],
            n_results=config.N_CHAPTER_RESULTS,
            where={"type": "chapter_summary"},
        )

        found_chapters = []
        chapter_docs = []
        chapter_metas = []

        if chapter_results["documents"] and chapter_results["documents"][0]:
            chapter_docs = chapter_results["documents"][0]
            chapter_metas = chapter_results["metadatas"][0]
            found_chapters = list({m["chapter"] for m in chapter_metas})

        logger.debug("Focused on chapters: %s", found_chapters)

        # Step 2: Focused search — retrieve slokas only from the identified chapters
        sloka_results = {"documents": [[]], "metadatas": [[]]}

        if found_chapters:
            where_clause = {
                "$and": [
                    {"type": "sloka"},
                    {"chapter": {"$in": found_chapters}},
                ]
            }
            sloka_results = self.collection.query(
                query_texts=[query],
                n_results=config.N_SLOKA_CANDIDATES,
                where=where_clause,
            )

            docs = sloka_results["documents"][0]
            metas = sloka_results["metadatas"][0]

            if docs:
                pairs = [[query, doc] for doc in docs]
                scores = self.cross_encoder.predict(pairs)
                scored = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)
                top = scored[: config.N_TOP_RESULTS]
                sloka_results["documents"][0] = [x[1] for x in top]
                sloka_results["metadatas"][0] = [x[2] for x in top]

        return {
            "chapters": {"docs": chapter_docs, "metas": chapter_metas},
            "slokas": {
                "docs": sloka_results["documents"][0],
                "metas": sloka_results["metadatas"][0],
            },
        }

    def _generate_with_ollama(self, prompt):
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

    def _generate_with_mlx(self, prompt):
        if hasattr(self.mlx_tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.mlx_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        return generate(
            self.mlx_model,
            self.mlx_tokenizer,
            prompt=formatted_prompt,
            verbose=False,
            max_tokens=1024,
        )

    def generate_answer(self, query):
        retrieved = self.retrieve_hierarchical(query)

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

        prompt = f"""You are a wisdom companion grounded in the teachings of the Bhagavad Gita.

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
- Then offer a related insight that is consistent with the broader philosophical spirit of the Bhagavad Gita (duty, detachment, self-awareness, equanimity, devotion, or disciplined action).

Use only the information in the context and generally accepted themes of the Gita.
Do not invent specific teachings or attribute ideas explicitly to Krishna or Arjuna.

Context:
{context_str}

User Question:
{query}

Answer:"""

        if self.provider == "mlx":
            answer = self._generate_with_mlx(prompt)
        else:
            answer = self._generate_with_ollama(prompt)

        return answer, citation_list, context_str


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Use MLX provider — update path if you trained the 8B model
    rag = GitaRAG(model_provider="mlx", mlx_model_path="models/gita-llama-3.1-8b-fused")

    q = "Why am I always distracted?"
    print(f"Question: {q}")
    answer, sources, context_str = rag.generate_answer(q)
    print(f"\nAnswer: {answer}")
    print(f"\nSources: {sources}")
