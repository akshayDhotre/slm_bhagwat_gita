import json
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path

def ingest_data():
    input_file = "data/processed/gita_knowledge_base.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run src/data_ingestion.py first.")
        return

    print(f"Loading dataset from {input_file}...")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="data/chroma_db")
    
    # Use a standard sentence transformer model for embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Get or create collection - we might want to reset it or check if we should delete old one
    # For now, let's assume we want to overwrite/update. 
    # To be safe, let's delete the old collection if it exists to clean up schema changes
    try:
        client.delete_collection("bhagavad_gita")
        print("Deleted existing collection to refresh schema.")
    except ValueError:
        pass # Collection didn't exist

    collection = client.create_collection(
        name="bhagavad_gita",
        embedding_function=sentence_transformer_ef
    )
    
    documents = []
    metadatas = []
    ids = []
    
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                doc_data = json.loads(line)
                
                # Content is already pre-formatted in data_ingestion.py
                text = doc_data.get('content', '')
                
                # Prepare metadata
                # ChromaDB metadata must be int, float, str, or bool. No dicts/lists.
                meta = {
                    "source": doc_data.get('metadata', {}).get('source', 'unknown'),
                    "type": doc_data.get('type', 'unknown'),
                    "chapter": doc_data.get('chapter', 0) or 0, # Ensure int
                    "verse": doc_data.get('verse', 0) or 0,     # Ensure int
                    "chapter_name": doc_data.get('metadata', {}).get('chapter_name', ''), # Add chapter name
                }
                
                # Add extra fields if they exist and are simple types
                if doc_data.get('type') == 'sloka':
                    meta['original_id'] = str(doc_data.get('metadata', {}).get('original_id', ''))
                
                documents.append(text)
                metadatas.append(meta)
                ids.append(str(doc_data.get('id', f"doc_{count}")))
                
                count += 1
                
                # Batch add
                if len(documents) >= 100:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    documents = []
                    metadatas = []
                    ids = []
                    
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")
            except Exception as e:
                print(f"Error processing line: {e}")

    # Add remaining
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    print("Ingestion complete.")
    print(f"Total documents ingested: {collection.count()}")

if __name__ == "__main__":
    ingest_data()
