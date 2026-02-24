import argparse
import logging
import sys

from src import config
from src.rag import GitaRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _format_sources(citation_list: list) -> str:
    """Format sloka citations for display. Chapter summaries are retrieval
    routing artifacts and are excluded — only verse-level sources are shown."""
    seen = set()
    lines = []
    for meta in citation_list:
        if meta.get("type") == "chapter_summary":
            continue
        key = f"ch{meta.get('chapter')}v{meta.get('verse')}"
        label = (
            f"Chapter {meta.get('chapter')}: {meta.get('chapter_name', '')}, "
            f"Verse {meta.get('verse', '?')}"
        )
        if key not in seen:
            seen.add(key)
            lines.append(label)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Bhagavad Gita Chatbot RAG")
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "mlx"],
        help="Model provider: 'ollama' or 'mlx'",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Ollama model name or MLX fused model path",
    )
    args = parser.parse_args()

    logger.info("Initializing Gita Chatbot with provider=%s ...", args.provider)

    try:
        if args.provider == "ollama":
            rag = GitaRAG(
                model_provider="ollama",
                ollama_model=args.model or config.DEFAULT_OLLAMA_MODEL,
            )
        else:
            rag = GitaRAG(
                model_provider="mlx",
                mlx_model_path=args.model or config.DEFAULT_MLX_MODEL_PATH,
            )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error initializing RAG: %s", e)
        logger.error("Did you run 'python -m src.ingest' first?")
        sys.exit(1)

    print("\nNamaste! I am your Bhagavad Gita companion. Ask me anything.")
    print("Type 'exit', 'quit', or 'bye' to end the session.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Dhanyavad! Goodbye.")
                break
            if not user_input.strip():
                continue

            print("Thinking...")
            answer, sources, _ = rag.generate_answer(user_input)

            print(f"\nWisdom Bot: {answer}\n")

            if sources:
                print("--- Sources ---")
                print(_format_sources(sources))
                print("---------------\n")

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            logger.error("Unexpected error: %s", e)


if __name__ == "__main__":
    main()
