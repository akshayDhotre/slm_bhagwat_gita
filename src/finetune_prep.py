import os
import json
from datasets import load_dataset

# Configuration
DATASET_ID = "JDhruv14/Bhagavad-Gita-QA"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gita_qa_finetune.jsonl")

def ensure_directories():
    """Create necessary directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_qa_dataset():
    """Download and process the QA dataset."""
    print(f"Downloading dataset: {DATASET_ID}...")
    
    try:
        # This dataset is likely a standard HF dataset
        dataset = load_dataset(DATASET_ID)
        print(f"Dataset loaded. Keys: {dataset.keys()}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    processed_examples = []
    
    # Iterate through splits (usually 'train')
    for split in dataset.keys():
        print(f"Processing split: {split}")
        data = dataset[split]
        
        for entry in data:
            # Inspect columns. Based on description: Question, Answer, etc.
            # We'll try to find the relevant columns.
            
            # Potential columns: 'question', 'answer', 'context' (maybe), 'language'
            # The description mentioned English, Hindi, Gujarati.
            
            question = entry.get('question') or entry.get('Question')
            answer = entry.get('answer') or entry.get('Answer')
            
            if not question or not answer:
                continue
                
            # Format for Fine-tuning (Alpaca style)
            # {
            #   "instruction": "You are a helpful assistant...",
            #   "input": "Question...",
            #   "output": "Answer..."
            # }
            
            # We can add a system prompt or just use the question as instruction.
            # Let's use a standard format.
            
            example = {
                "instruction": "Answer the following question based on the Bhagavad Gita.",
                "input": question,
                "output": answer,
                "metadata": {
                    "source": DATASET_ID,
                    "language": entry.get('language', 'unknown'),
                    "verse": entry.get('verse', '') # if available
                }
            }
            processed_examples.append(example)

    print(f"Processed {len(processed_examples)} QA pairs.")
    
    # Save to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in processed_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"Saved fine-tuning data to {OUTPUT_FILE}")

if __name__ == "__main__":
    ensure_directories()
    process_qa_dataset()
