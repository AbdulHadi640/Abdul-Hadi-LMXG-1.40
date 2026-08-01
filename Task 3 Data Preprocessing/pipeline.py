import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from database import engine, SessionLocal, CleanedArticle, NewsEmbedding, articles_table
import json

def clean_text_data(text: str) -> str:
    """Utility function to standardize and clean raw text strings."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags, special symbols, or excessive whitespace
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_news_pipeline():
    session = SessionLocal()
    try:
        # --- STEP 1: Ingest Raw Data & Clean It Into a Separate Table ---
        print("📥 Fetching raw articles from PostgreSQL...")
        with engine.connect() as conn:
            df = pd.read_sql(articles_table.select(), conn)
        
        if df.empty:
            print("⚠️ No articles found in the database.")
            return

        print(f"🧹 Cleaning and standardizing {len(df)} articles...")
        for _, row in df.iterrows():
            art_id = row['id']
            
            # Combine fields and apply text cleaning rules
            raw_combined = f"{row.get('source_name', '')} {row.get('title', '')} {row.get('first_paragraph', '')}"
            cleaned_string = clean_text_data(raw_combined)
            
            # Save or update in the 'cleaned_articles' table
            existing_clean = session.query(CleanedArticle).filter_by(article_id=art_id).first()
            if existing_clean:
                existing_clean.clean_text = cleaned_string
            else:
                session.add(CleanedArticle(article_id=art_id, clean_text=cleaned_string))
        
        session.commit()
        print("✅ Data cleaning complete and saved to 'cleaned_articles' table.")

        # --- STEP 2: Generate Sentence Embeddings from Cleaned Data ---
        print("🤖 Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Query the clean text table instead of raw data
        clean_records = session.query(CleanedArticle).all()
        clean_texts = [rec.clean_text for rec in clean_records]
        clean_ids = [rec.article_id for rec in clean_records]
        
        print("🔢 Generating vector embeddings from clean data...")
        embeddings = model.encode(clean_texts, show_progress_bar=True)
        
        print("💾 Saving embeddings to database...")
        for idx, art_id in enumerate(clean_ids):
            existing_emb = session.query(NewsEmbedding).filter_by(article_id=art_id).first()
            vector_str = json.dumps(embeddings[idx].tolist())
            
            if existing_emb:
                existing_emb.embedding = vector_str
            else:
                session.add(NewsEmbedding(article_id=art_id, embedding=vector_str))
                
        session.commit()
        print("✅ Pipeline execution completed successfully with separated clean storage!")

    except Exception as e:
        session.rollback()
        print(f"❌ Error in pipeline: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    run_news_pipeline()