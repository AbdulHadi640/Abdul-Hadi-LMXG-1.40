import os
from sqlalchemy import create_engine, Column, Integer, String, Text, Table, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
load_dotenv()  
DB_USER = os.getenv("PG_USER", "postgres")
DB_PASSWORD = os.getenv("PG_PASSWORD", "")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("PG_NAME", "news_scraper")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

# Reflect raw source table
articles_table = Table('articles', metadata, autoload_with=engine)

# 1. Cleaned Data Table (The Staging/Clean Layer)
class CleanedArticle(Base):
    __tablename__ = 'cleaned_articles'
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, unique=True, index=True) # Links back to raw article
    clean_text = Column(Text)                            # Stores standardized, cleaned text

# 2. Vector Embeddings Table
class NewsEmbedding(Base):
    __tablename__ = 'news_embeddings'
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, unique=True, index=True)
    embedding = Column(Text)  # Stored as serialized vector representation

Base.metadata.create_all(bind=engine)
