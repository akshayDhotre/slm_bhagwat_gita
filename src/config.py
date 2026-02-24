# Central configuration for the Gita RAG pipeline.
# All magic numbers, paths, and model names live here.

# --- Vector DB ---
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "bhagavad_gita"

# --- Embedding & re-ranking models ---
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Retrieval parameters ---
N_CHAPTER_RESULTS   = 4   # broad search: top N chapter summaries
N_SLOKA_CANDIDATES  = 20  # focused search: candidates before re-ranking
N_TOP_RESULTS       = 5   # final results after re-ranking

# --- Default model configs ---
DEFAULT_OLLAMA_MODEL   = "gemma3:12b"
DEFAULT_MLX_MODEL_PATH = "models/gita-llama-3.1-8b-fused"
