import pandas as pd
from sentence_transformers import SentenceTransformer
from database import engine, SessionLocal, Base, NewsEmbedding

def run_news_pipeline():
    print("🗄️ Ensuring database tables exist...")
    Base.metadata.create_all(bind=engine)

    print("📦 Fetching raw news data from 'articles'...")
    query = "SELECT * FROM articles;"
    df = pd.read_sql(query, con=engine)
    print(f"📊 Fetched {len(df)} records.")

    print("🧹 Cleaning and building contextual documents...")
    target_text_cols = ['source_name', 'title', 'first_paragraph']
    
    def create_contextual_document(row):
        parts = []
        for col in target_text_cols:
            if col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{col.replace('_', ' ').capitalize()}: {str(val).strip()}")
        return " | ".join(parts) if parts else "No content"

    df['contextual_document'] = df.apply(create_contextual_document, axis=1)
    df['contextual_document'] = df['contextual_document'].fillna("").astype(str).str.strip()

    print("🤖 Loading sentence transformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"🔄 Generating vector embeddings for {len(df)} records...")
    texts = df['contextual_document'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    print("💾 Storing embeddings into PostgreSQL...")
    session = SessionLocal()
    try:
        session.query(NewsEmbedding).delete() # Clear old embeddings
        
        for idx, row in df.iterrows():
            record_id = row['id']
            vector_str = str(embeddings[idx].tolist())
            
            entry = NewsEmbedding(news_id=record_id, embedding=vector_str)
            session.add(entry)
            
        session.commit()
        print("🎉 Pipeline executed successfully!")
    except Exception as e:
        session.rollback()
        print(f"❌ Error in pipeline: {e}")
        raise
    finally:
        session.close()