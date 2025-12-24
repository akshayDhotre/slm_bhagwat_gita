import chromadb
from chromadb.utils import embedding_functions
import ollama
from sentence_transformers import CrossEncoder

class GitaRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_collection(
            name="bhagavad_gita",
            embedding_function=self.sentence_transformer_ef
        )
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def retrieve_hierarchical(self, query):
        # Step 1: Broad Search - Find relevant Chapter Summaries
        # This helps us pinpoint the "context" or "theme" (e.g., Karma Yoga, Devotion)
        chapter_results = self.collection.query(
            query_texts=[query],
            n_results=4, # Get top 4 chapters
            where={"type": "chapter_summary"}
        )
        
        found_chapters = []
        chapter_docs = []
        chapter_metas = []
        
        if chapter_results['documents'] and chapter_results['documents'][0]:
            chapter_docs = chapter_results['documents'][0]
            chapter_metas = chapter_results['metadatas'][0]
            # Extract unique chapter IDs
            for meta in chapter_metas:
                found_chapters.append(meta['chapter'])
        
        found_chapters = list(set(found_chapters))
        print(f"Debug: Focused on Chapters: {found_chapters}")
        
        # Step 2: Focused Search - Find Slokas within those Chapters
        # We assume that if the Chapter Summary is relevant, the answer lies in its verses.
        sloka_results = {'documents': [[]], 'metadatas': [[]]}
        
        if found_chapters:
             # Construct filter: (type == sloka) AND (chapter IN found_chapters)
            where_clause = {
                "$and": [
                    {"type": "sloka"},
                    {"chapter": {"$in": found_chapters}}
                ]
            }
            
            sloka_results = self.collection.query(
                query_texts=[query],
                n_results=20, # Get top 20 candidates for reranking
                where=where_clause
            )
            
            # Reranking Logic
            docs = sloka_results['documents'][0]
            metas = sloka_results['metadatas'][0]
            
            if docs:
                # Create pairs of (Query, Document)
                pairs = [[query, doc] for doc in docs]
                
                # Score pairs
                scores = self.cross_encoder.predict(pairs)
                
                # Zip together: (Score, Doc, Meta)
                scored_results = list(zip(scores, docs, metas))
                
                # Sort by score descending
                scored_results.sort(key=lambda x: x[0], reverse=True)
                
                # Take top 5
                top_results = scored_results[:5]
                
                # Unzip back into lists for return structure
                sloka_results['documents'][0] = [x[1] for x in top_results]
                sloka_results['metadatas'][0] = [x[2] for x in top_results]
            
        return {
            "chapters": {"docs": chapter_docs, "metas": chapter_metas},
            "slokas": {"docs": sloka_results['documents'][0], "metas": sloka_results['metadatas'][0]}
        }
    
    def generate_answer(self, query):
        # 1. Retrieve relevant documents using new strategy
        retrieved = self.retrieve_hierarchical(query)
        
        context_str = ""
        citation_list = []
        
        # Add Chapter Summaries to Context
        if retrieved['chapters']['docs']:
            context_str += "=== RELEVANT CHAPTER CONTEXT ===\n"
            for i, text in enumerate(retrieved['chapters']['docs']):
                meta = retrieved['chapters']['metas'][i]
                context_str += f"[Chapter {meta.get('chapter')}: {meta.get('chapter_name', 'Summary')}]\n{text}\n\n"
                citation_list.append(meta)

        # Add Slokas to Context
        if retrieved['slokas']['docs']:
            context_str += "=== RELEVANT VERSES (SLOKAS) ===\n"
            for i, text in enumerate(retrieved['slokas']['docs']):
                meta = retrieved['slokas']['metas'][i]
                context_str += f"[Chapter {meta.get('chapter')}, Verse {meta.get('verse')}]\n{text}\n\n"
                citation_list.append(meta)

        # 3. Construct Prompt
        prompt = f"""You are a wisdom companion grounded in the teachings of the Bhagavad Gita.

Your task is to answer the user’s personal question using the provided context, which contains relevant passages or interpretations from the Gita. 
Base your response primarily on this context. Do not introduce ideas that clearly contradict it.

Guidelines:
- Express the wisdom in your own natural, contemporary language.
- Do NOT quote verses, mention chapter or verse numbers, or sound like a scripture reference.
- Speak as a fellow human sharing lived understanding—calm, grounded, and compassionate.
- Keep the guidance practical and applicable to modern life.
- Avoid moralizing or preaching; offer clarity, not instruction.

If the context does not directly address the user’s question:
- Say so honestly and briefly.
- Then offer a related insight that is consistent with the broader philosophical spirit of the Bhagavad Gita (duty, detachment, self-awareness, equanimity, devotion, or disciplined action).

Use only the information in the context and generally accepted themes of the Gita.
Do not invent specific teachings or attribute ideas explicitly to Krishna or Arjuna.

Context:
{context_str}

User Question:
{query}

Answer:"""

        # 4. Call Ollama
        model = "gemma3:12b"
        try:
            response = ollama.chat(model=model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content'], citation_list, context_str
        except Exception as e:
            return f"Error calling Ollama: {e}. Make sure Ollama is running and '{model}' is pulled.", [], ""

if __name__ == "__main__":
    rag = GitaRAG()
    q = "Why I am always distracted?"
    print(f"Question: {q}")
    answer, sources, context_str = rag.generate_answer(q)
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    print(f"\nContext: {context_str}")
