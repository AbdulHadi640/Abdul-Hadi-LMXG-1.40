import numpy as np
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from database import SessionLocal
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # Forces it to use the locally cached model without checking online
def search_news(query_text, top_k=5):
    print(f"🔍 Searching for: '{query_text}'...\n")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query_text)
    
    session = SessionLocal()
    try:
        results = session.execute(
            text("""
                SELECT e.news_id, e.embedding, a.title, a.source_name, a.first_paragraph 
                FROM news_embeddings e
                JOIN articles a ON e.news_id = a.id
            """)
        ).fetchall()
        
        if not results:
            print("⚠️ No embeddings found in the database.")
            return
            
        scored_results = []
        for row in results:
            news_id, embedding_str, title, source_name, first_paragraph = row
            stored_vector = np.array(eval(embedding_str), dtype=float)
            
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), 
                stored_vector.reshape(1, -1)
            )[0][0]
            
            scored_results.append({
                "news_id": news_id,
                "title": title,
                "source_name": source_name,
                "first_paragraph": first_paragraph,
                "score": float(similarity)
            })
            
        scored_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)
        
        print(f"--- Top {top_k} Matching Articles ---")
        for i, item in enumerate(scored_results[:top_k], 1):
            print(f"{i}. [Score: {item['score']:.4f}] {item['title']}")
            print(f"   Source: {item['source_name']}")
            print(f"   Summary: {item['first_paragraph'][:120]}...\n")
            
    finally:
        session.close()