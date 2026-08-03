# Task 3: News Data Preprocessing & Semantic Vector Search Pipeline

A production-ready data engineering and Natural Language Processing (NLP) pipeline that implements best practices by separating raw data ingestion, text cleaning/staging, dense vector generation, and semantic similarity search using PostgreSQL and Sentence Transformers.

---

## Architecture & Data Flow

1. **Raw Ingestion Layer (`articles` table):** Captures and stores raw scraped news records.
2. **Cleaned Staging Layer (`cleaned_articles` table):** Standardizes, cleans, and isolates textual features (HTML stripping, whitespace normalization) away from the raw source of truth.
3. **Vector Embedding Layer (`news_embeddings` table):** Feeds pristine text from the clean layer into the Hugging Face transformer model to generate 384-dimensional dense vectors additively.
4. **Vector Similarity Search (`search.py`):** Encodes natural language user queries and calculates cosine similarity scores via Scikit-Learn to retrieve the most contextually relevant news articles.

---

## Project Structure

```text
Task 3 Data Preprocessing/
│
├── database.py       # Manages PostgreSQL engine, sessions, and SQLAlchemy ORM models (cleaned_articles & news_embeddings)
├── pipeline.py       # Two-step pipeline: (1) Cleans raw text into staging table, (2) Generates & stores vector embeddings
├── search.py         # Encodes queries and computes sklearn cosine similarity against stored vectors
└── main.py           # Unified entry point to run ingestion/cleaning pipelines or execute searches
