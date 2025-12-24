from src.rag import GitaRAG
import sys

def main():
    print("Initializing Gita Chatbot...")
    try:
        rag = GitaRAG()
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
            
            print(f"\nBot: {answer}\n")
            
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
