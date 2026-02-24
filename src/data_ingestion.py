import glob
import json
import logging
import os

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Configuration
DATASET_ID = "XenArcAI/Bhagwat-Gita-Infinity"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gita_knowledge_base.jsonl")


def ensure_directories():
    """Create necessary directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_dataset():
    """Download and process the dataset."""
    logger.info("Downloading dataset: %s ...", DATASET_ID)

    try:
        local_dir = snapshot_download(repo_id=DATASET_ID, repo_type="dataset")
        logger.info("Dataset downloaded to: %s", local_dir)
    except Exception as e:
        logger.error("Failed to download dataset: %s", e)
        return

    processed_docs = []
    chapter_map = {}

    # 1. Process Chapters
    chapter_files = glob.glob(os.path.join(local_dir, "chapters", "*.json"))
    if not chapter_files:
        chapter_files = glob.glob(os.path.join(local_dir, "chapter", "*.json"))

    logger.info("Found %d chapter files.", len(chapter_files))

    for file_path in chapter_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entry = json.load(f)

            chapter_name = f"{entry.get('translation')} - {entry.get('meaning', {}).get('en', '')}"
            doc = {
                "type": "chapter_summary",
                "id": entry.get("chapter_number"),
                "chapter": int(entry.get("chapter_number", 0)),
                "chapter_name": chapter_name,
                "content": (
                    f"{chapter_name}\nSummary:"
                    f"{entry.get('summary', {}).get('en', '') if isinstance(entry.get('summary'), dict) else str(entry.get('summary', ''))}"
                ),
                "metadata": {
                    "source": DATASET_ID,
                    "type": "summary",
                    "chapter_name": chapter_name,
                    "file": os.path.basename(file_path),
                },
            }
            processed_docs.append(doc)

            if entry.get("chapter_number"):
                chapter_map[int(entry.get("chapter_number"))] = doc["chapter_name"]
        except Exception as e:
            logger.error("Error processing chapter file %s: %s", file_path, e)

    # 2. Process Slokas
    sloka_files = glob.glob(os.path.join(local_dir, "sloks", "*.json"))
    if not sloka_files:
        sloka_files = glob.glob(os.path.join(local_dir, "slok", "*.json"))

    logger.info("Found %d sloka files.", len(sloka_files))

    for file_path in sloka_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entry = json.load(f)

            # Find best English translation and commentary.
            # Priority order: siva (Sivananda), prabhu (Prabhupada), gambhirananda, etc.
            author_priority = ["siva", "prabhu", "gambhirananda", "purohit", "rams", "adi", "chinmay", "san", "tej"]

            trans_en = entry.get("translation_en") or entry.get("english_translation") or ""
            comment_en = ""
            selected_author = ""

            if not trans_en:
                for author in author_priority:
                    if author in entry and isinstance(entry[author], dict):
                        candidate = entry[author].get("et")
                        if candidate:
                            trans_en = candidate
                            comment_en = entry[author].get("ec", "")
                            selected_author = author
                            break

            # Fallback: find any 'et' key
            if not trans_en:
                for key, value in entry.items():
                    if isinstance(value, dict) and "et" in value:
                        trans_en = value["et"]
                        comment_en = value.get("ec", "")
                        selected_author = key
                        break

            if not trans_en:
                continue

            slok_id = entry.get("slok_id") or entry.get("_id")

            doc = {
                "type": "sloka",
                "id": slok_id,
                "chapter": entry.get("chapter", ""),
                "verse": entry.get("verse", ""),
                "content": (
                    f"{trans_en.strip()}\n\nCommentary:\n{comment_en.strip()}"
                    if comment_en
                    else trans_en.strip()
                ),
                "author": selected_author,
                "metadata": {
                    "source": DATASET_ID,
                    "original_id": slok_id,
                    "file": os.path.basename(file_path),
                },
            }

            # Parse chapter/verse from ID (e.g. BG17.25 or 17_25)
            try:
                if slok_id:
                    clean_id = str(slok_id).replace("BG", "").strip()
                    if "." in clean_id:
                        parts = clean_id.split(".")
                    elif "_" in str(slok_id):
                        parts = str(slok_id).split("_")
                    else:
                        parts = []

                    nums = [int(p) for p in parts if p.isdigit()]
                    if len(nums) >= 2:
                        doc["chapter"] = nums[0]
                        doc["verse"] = nums[1]
            except (ValueError, AttributeError):
                pass

            if doc["chapter"] in chapter_map:
                doc["metadata"]["chapter_name"] = chapter_map[doc["chapter"]]

            processed_docs.append(doc)

        except Exception as e:
            logger.error("Error processing sloka file %s: %s", file_path, e)

    logger.info("Processed %d documents.", len(processed_docs))

    if processed_docs:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for doc in processed_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        logger.info("Saved processed data to %s", OUTPUT_FILE)
    else:
        logger.warning("No documents processed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_directories()
    process_dataset()
