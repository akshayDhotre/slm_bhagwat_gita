# Central configuration for the Gita RAG pipeline.
# All magic numbers, paths, and model names live here.

# --- Vector DB ---
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "bhagavad_gita"

# --- Embedding model ---
# Single model for both indexing and distance-based re-ranking.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval parameters ---
N_CHAPTER_RESULTS   = 4   # broad search: top N chapter summaries
N_SLOKA_CANDIDATES  = 15  # focused search: candidates from matched chapters
N_GLOBAL_SUPPLEMENT = 10  # cross-chapter fallback: extra global candidates
N_TOP_RESULTS       = 5   # final results after re-ranking

# --- Default model configs ---
DEFAULT_OLLAMA_MODEL   = "deepseek-r1:8b"
DEFAULT_MLX_MODEL_PATH = "models/gita-llama-3.1-8b-fused"
