import json
import logging
import os

import chromadb
from chromadb.utils import embedding_functions

from src import config

logger = logging.getLogger(__name__)


def ingest_data():
    input_file = "data/processed/gita_knowledge_base.jsonl"

    if not os.path.exists(input_file):
        logger.error("%s not found. Please run src/data_ingestion.py first.", input_file)
        return

    logger.info("Loading dataset from %s ...", input_file)

    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

    try:
        client.delete_collection(config.COLLECTION_NAME)
        logger.info("Deleted existing collection to refresh schema.")
    except ValueError:
        pass  # Collection didn't exist

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
    )

    documents = []
    metadatas = []
    ids = []
    count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                doc_data = json.loads(line)

                text = doc_data.get("content", "")

                meta = {
                    "source": doc_data.get("metadata", {}).get("source", "unknown"),
                    "type": doc_data.get("type", "unknown"),
                    "chapter": doc_data.get("chapter", 0) or 0,
                    "verse": doc_data.get("verse", 0) or 0,
                    "chapter_name": doc_data.get("metadata", {}).get("chapter_name", ""),
                }

                if doc_data.get("type") == "sloka":
                    meta["original_id"] = str(
                        doc_data.get("metadata", {}).get("original_id", "")
                    )

                documents.append(text)
                metadatas.append(meta)
                ids.append(str(doc_data.get("id", f"doc_{count}")))
                count += 1

                if len(documents) >= 100:
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    documents = []
                    metadatas = []
                    ids = []

            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line: %.50s...", line)
            except Exception as e:
                logger.error("Error processing line: %s", e)

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    logger.info("Ingestion complete. Total documents: %d", collection.count())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ingest_data()
