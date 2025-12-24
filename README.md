# Bhagavad Gita AI 🕉️

A high-performance "Wisdom RAG" application that combines **Hierarchical Retrieval**, **Semantic Re-ranking**, and **Deep Commentary** to answer life's questions using the Bhagavad Gita.

Unlike standard RAG pipelines that only match similar words, this system mimics a knowledgeable teacher: passing through the broad context of a chapter, drilling down to specific verses, and validating the relevance using a Cross-Encoder model.

## 🚀 Key Features

*   **Hierarchical Retrieval**: "Forest & Trees" approach. First identifies the most relevant *Theme* (Chapter), then searches for *Specifics* (Slokas) within that context.
*   **Semantic Re-ranking**: Uses a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to strictly grade the top 20 retrieved verses and select only the Top 5 most relevant ones.
*   **Commentary-Enriched**: doesn't just index the translation; it indexes the deep philosophical *Purports/Commentaries* (by Swami Sivananda, Prabhupada, etc.) to capture abstract concepts.
*   **Context-Aware Citations**: Answers are grounded in reality with citations like `[Chapter 6: Dhyana Yoga, Verse 34]`.
*   **Privacy-First**: Runs 100% locally using **Ollama** and **ChromaDB**.

## 🏗️ Architecture

1.  **Ingestion with Enrichment**:
    *   Raw data (JSON) is processed to extract translations AND commentaries.
    *   Chapter Summaries are created as separate "Anchor" documents.
2.  **Two-Step Search**:
    *   *Step 1*: Find Top 2 relevant Chapter Summaries.
    *   *Step 2*: Fetch Top 20 Slokas *only* from those chapters.
3.  **Precision Filtering**:
    *   The 20 candidates are re-scored by a Cross-Encoder.
    *   The Top 5 are sent to the LLM.

## 🛠️ Setup

### Prerequisites
*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) (Recommended) or pip
*   [Ollama](https://ollama.com/)

### 1. Installation

Clone the repo and install dependencies:

```bash
uv sync
# OR
pip install -r requirements.txt
```

### 2. Model Setup

Pull the reasoning model used by the system:

```bash
ollama pull deepseek-r1:8b
```

### 3. Build Knowledge Base

Download the dataset and build the enriched vector database:

```bash
uv run src/data_ingestion.py
uv run src/ingest.py
```
*(This will process ~740 documents including Chapters and Slokas)*

## 🏃‍♂️ Usage

Start the interactive chat session:

```bash
uv run main.py
```

**Example Interaction:**
```text
You: Why am I always distracted?
Thinking...
Debug: Focused on Chapters: [6]

Bot: Staying focused is difficult because the mind by nature is restless...
...
--- Sources ---
Chapter 6: Dhyana Yoga, Verse 35
Chapter 6: Dhyana Yoga, Verse 26
---------------
```

## 📂 Structure

*   `src/data_ingestion.py`: cleaning logic, commentary extraction, and structure normalization.
*   `src/ingest.py`: ChromaDB indexing and schema management.
*   `src/rag.py`: The Core Engine (Hierarchical Search + Re-ranking Logic).
*   `main.py`: Interactive CLI entry point.
*   `data/`: Stores the local vector database.
