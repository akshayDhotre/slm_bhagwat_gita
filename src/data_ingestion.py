import os
import json
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Configuration
DATASET_ID = "XenArcAI/Bhagwat-Gita-Infinity"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gita_knowledge_base.jsonl")

def ensure_directories():
    """Create necessary directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

from huggingface_hub import snapshot_download
import glob

def process_dataset():
    """Download and process the dataset."""
    print(f"Downloading dataset: {DATASET_ID}...")
    
    try:
        # Download the entire dataset repository
        local_dir = snapshot_download(repo_id=DATASET_ID, repo_type="dataset")
        print(f"Dataset downloaded to: {local_dir}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    processed_docs = []
    chapter_map = {}

    # 1. Process Chapters
    # Look for chapters folder
    chapter_files = glob.glob(os.path.join(local_dir, "chapters", "*.json"))
    if not chapter_files:
        # Try singular 'chapter'
        chapter_files = glob.glob(os.path.join(local_dir, "chapter", "*.json"))
    
    print(f"Found {len(chapter_files)} chapter files.")
    
    for file_path in chapter_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                
            doc = {
                'type': 'chapter_summary',
                'id': entry.get('chapter_number'),
                'chapter': int(entry.get('chapter_number', 0)),
                'chapter_name': f"{entry.get('translation')} - {entry.get('meaning', {}).get('en', '')}",
                'content': f"{entry.get('translation')} - {entry.get('meaning', {}).get('en', '')}\nSummary:{entry.get('summary', {}).get('en', '') if isinstance(entry.get('summary'), dict) else str(entry.get('summary', ''))}",
                'metadata': {
                    'source': DATASET_ID,
                    'type': 'summary',
                    'file': os.path.basename(file_path)
                }
            }
            processed_docs.append(doc)
            
            # Populate chapter map
            if entry.get('chapter_number'):
                chapter_map[int(entry.get('chapter_number'))] = doc['chapter_name']
        except Exception as e:
            print(f"Error processing chapter file {file_path}: {e}")

    # 2. Process Slokas
    # Look for sloks folder
    sloka_files = glob.glob(os.path.join(local_dir, "sloks", "*.json"))
    if not sloka_files:
        # Try singular 'slok'
        sloka_files = glob.glob(os.path.join(local_dir, "slok", "*.json"))
        
    print(f"Found {len(sloka_files)} sloka files.")
    
    for file_path in sloka_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                        
            # Find best English translation and meaning
            # The dataset has translations under specific author keys.
            # Common keys: 'siva' (Sivananda), 'prabhu' (Prabhupada), 'gambhirananda', 'purohit', 'rams', etc.
            # We will prioritize them in a specific order.
            
            author_priority = ['siva', 'prabhu', 'gambhirananda', 'purohit', 'rams', 'adi', 'chinmay', 'san', 'tej']
            
            trans_en = ""
            comment_en = ""
            selected_author = ""
            
            # Check explicit top-level fields first (just in case)
            trans_en = entry.get('translation_en') or entry.get('english_translation')
            
            if not trans_en:
                for author in author_priority:
                     if author in entry and isinstance(entry[author], dict):
                         # 'et' is English Translation
                         candidate_trans = entry[author].get('et')
                         if candidate_trans:
                             trans_en = candidate_trans
                             comment_en = entry[author].get('ec', "")
                             selected_author = author
                             break
            
            # If still no translation, try to find ANY 'et'
            if not trans_en:
                for key, value in entry.items():
                    if isinstance(value, dict) and 'et' in value:
                        trans_en = value['et']
                        comment_en = value.get('ec', "")
                        selected_author = key
                        break
            
            if not trans_en:
                trans_en = ""
            
            if not trans_en:
                continue

            # ID
            slok_id = entry.get('slok_id') or entry.get('_id')
            
            doc = {
                'type': 'sloka',
                'id': slok_id,
                'chapter': entry.get('chapter', ''),
                'verse': entry.get('verse', ''),
                'content': f"{trans_en.strip()}\n\nCommentary:\n{comment_en.strip()}" if comment_en else trans_en.strip(),
                'author': selected_author,
                'metadata': {
                    'source': DATASET_ID,
                    'original_id': slok_id,
                    'file': os.path.basename(file_path)
                }
            }
            
            # Try to parse chapter/verse
            try:
                if slok_id:
                    # Handle BG17.25 format
                    clean_id = str(slok_id).replace('BG', '').strip()
                    if '.' in clean_id:
                        parts = clean_id.split('.')
                    elif '_' in str(slok_id):
                        parts = str(slok_id).split('_')
                    else:
                        parts = []
                        
                    nums = [int(p) for p in parts if p.isdigit()]
                    if len(nums) >= 2:
                        doc['chapter'] = nums[0]
                        doc['verse'] = nums[1]
            except:
                pass
            
            # Injects Chapter Name into Metadata if available
            if doc['chapter'] in chapter_map:
                doc['metadata']['chapter_name'] = chapter_map[doc['chapter']]
            
            processed_docs.append(doc)
            
        except Exception as e:
            print(f"Error processing sloka file {file_path}: {e}")

    print(f"Processed {len(processed_docs)} documents.")
    
    # Save to JSONL
    if processed_docs:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for doc in processed_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        print(f"Saved processed data to {OUTPUT_FILE}")
    else:
        print("No documents processed.")

if __name__ == "__main__":
    ensure_directories()
    process_dataset()
