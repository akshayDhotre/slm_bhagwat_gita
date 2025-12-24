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
            n_results=2, # Get top 2 chapters
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
        prompt = f"""Answer personal questions with wisdom drawn from the Bhagavad Gita, expressed in your own natural words. 
        Reflect on relevant teachings without citing verse numbers or sounding like a reference. 
        If the exact answer isn’t found in the Gita, acknowledge that honestly, while offering related insight from its broader wisdom. 
        Keep the response practical for modern life, calm and simple in tone, and speak as a fellow human sharing lived wisdom—not as an assistant or commentator.

        Context:
        {context_str}

        User Question: {query}

        Answer:"""

        # 4. Call Ollama
        model = "deepseek-r1:8b"
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
