import argparse
import sys
from src.rag import GitaRAG

def main():
    parser = argparse.ArgumentParser(description="Bhagavad Gita Chatbot RAG")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "mlx"], help="Model provider: 'ollama' or 'mlx'")
    parser.add_argument("--model", type=str, help="Model name (for Ollama) or path (for MLX)")
    
    args = parser.parse_args()
    
    print(f"Initializing Gita Chatbot with {args.provider}...")
    
    try:
        if args.provider == "ollama":
            model_name = args.model if args.model else "gemma3:12b"
            rag = GitaRAG(model_provider="ollama", ollama_model=model_name)
        else: # mlx
            model_path = args.model if args.model else "models/gita-llama-3.2-3b-fused"
            rag = GitaRAG(model_provider="mlx", mlx_model_path=model_path)
            
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        print("Did you run 'src/ingest.py' first?")
        sys.exit(1)
        
    print("\nNamaste! I am your Bhagavad Gita companion. Ask me anything.")
    print("Type 'exit' or 'quit' or 'bye' to end the session.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Dhanyavad! Goodbye.")
                break
            
            if not user_input.strip():
                continue
                
            print("Thinking...")
            answer, sources, _ = rag.generate_answer(user_input)
            
            print(f"\nWisdom Bot: {answer}\n")
            
            if sources:
                print("--- Sources ---")
                for meta in sources:
                    chapter_info = f"Chapter {meta['chapter']}"
                    if meta.get('chapter_name'):
                        chapter_info += f": {meta.get('chapter_name')}"
                    print(f"{chapter_info}, Verse {meta['verse']}")
                print("---------------\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
