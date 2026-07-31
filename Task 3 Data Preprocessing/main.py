from pipeline import run_news_pipeline
from search import search_news

if __name__ == "__main__":
    # 1. Run the pipeline if you need to update/regenerate embeddings
    run_news_pipeline()
    
    # 2. Test semantic search
    # search_news("artificial intelligence breakthroughs", top_k=3)