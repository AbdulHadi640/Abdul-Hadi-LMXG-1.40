# Task 3: News Data Preprocessing & Semantic Search Pipeline

## Project Overview & Explanation

This project is a production-ready, modular data engineering and Natural Language Processing (NLP) pipeline built to turn raw, unstructured scraped news articles into searchable semantic vectors. 

In traditional keyword-based search systems, looking up "artificial intelligence breakthroughs" would fail if an article uses terms like "machine learning milestones" without mentioning the exact keywords. This pipeline solves that limitation by converting article texts into high-dimensional numerical vectors (embeddings) that capture true semantic meaning, allowing users to search by concept rather than exact keywords.

---

## How the System Works (Architecture)

The codebase is broken down into four distinct, modular Python files to separate concerns and maintain clean code structure:

1. **`database.py` (The Data Layer):**
   * Manages the connection to the local PostgreSQL database (`news_scraper`) using SQLAlchemy and Psycopg2.
   * Automatically reflects and interacts with the raw scraped `articles` source table.
   * Defines the ORM schema for the `news_embeddings` target table, utilizing an additive and non-destructive approach so existing embeddings are safely updated or preserved without wiping historical data.

2. **`pipeline.py` (The Ingestion & Embedding Engine):**
   * Connects to PostgreSQL using Pandas to fetch raw records from the `articles` table.
   * Concatenates relevant textual features (`source_name`, `title`, and `first_paragraph`) to form a rich text representation for each news item.
   * Loads the Hugging Face `sentence-transformers` model (`all-MiniLM-L6-v2`) to encode the combined text documents into 384-dimensional dense numerical vectors.
   * Serializes the vectors into JSON format and bulk-loads them securely back into the `news_embeddings` table.

3. **`search.py` (The Vector Similarity Search Engine):**
   * Encodes a natural language user query string into a 384-dimensional vector using the same transformer model.
   * Fetches stored article embeddings from the database, reconstructs them into a NumPy matrix, and pulls matching metadata from the `articles` table.
   * Utilizes Scikit-Learn's `cosine_similarity` algorithm to mathematically measure the semantic angle between the user query vector and every stored article vector.
   * Sorts and outputs the top-$K$ most contextually relevant news articles complete with similarity confidence scores, sources, and text summaries.

4. **`main.py` (The Unified Entry Point):**
   * Acts as a clean execution switch, letting you seamlessly toggle between running the backend embedding ingestion pipeline or executing natural language searches.

---
