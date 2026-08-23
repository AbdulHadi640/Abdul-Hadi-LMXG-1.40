"""
Feature Engineer - Extract Advanced Features (FIXED)
=====================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import config

class FeatureEngineer:
    """Extract text, temporal, and source features"""
    
    def extract_all_features(self, df):
        """Extract all features from news data"""
        print("\n" + "="*60)
        print("[5/8] FEATURE ENGINEERING")
        print("="*60)
        
        # Text length features
        print("  - Text length features")
        df['title_length'] = df['title'].fillna('').str.len()
        df['desc_length'] = df['description'].fillna('').str.len()
        df['title_words'] = df['title'].fillna('').str.split().str.len()
        df['desc_words'] = df['description'].fillna('').str.split().str.len()
        
        # Urgency score
        print("  - Urgency score")
        urgency_pattern = '|'.join(config.URGENCY_WORDS)
        df['urgency_score'] = (
            df['title'].fillna('').str.lower().str.count(urgency_pattern) +
            df['description'].fillna('').str.lower().str.count(urgency_pattern)
        )
        
        # Sentiment indicators
        print("  - Sentiment indicators")
        pos_pattern = '|'.join(config.POSITIVE_WORDS)
        neg_pattern = '|'.join(config.NEGATIVE_WORDS)
        
        df['positive_score'] = (
            df['title'].fillna('').str.lower().str.count(pos_pattern) +
            df['description'].fillna('').str.lower().str.count(pos_pattern)
        )
        df['negative_score'] = (
            df['title'].fillna('').str.lower().str.count(neg_pattern) +
            df['description'].fillna('').str.lower().str.count(neg_pattern)
        )
        df['sentiment_balance'] = df['positive_score'] - df['negative_score']
        
        # Temporal features (cyclical encoding)
        print("  - Temporal features (cyclical)")
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_ny'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_ny'] / 24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        
        # Source credibility
        print("  - Source credibility")
        df['source_credibility'] = df['source'].map(config.SOURCE_CREDIBILITY).fillna(0.5)
        df['is_major_outlet'] = df['source'].isin(config.MAJOR_OUTLETS).astype(int)
        
        print(f"\n✓ Total features extracted: 15+")
        
        return df
    
    def create_embeddings(self, df):
        """Create text embeddings and combine all features"""
        print("\n" + "="*60)
        print("[6/8] CREATING EMBEDDINGS")
        print("="*60)
        
        # FIX: Ensure we have a 'text_cleaned' column regardless of cleaner version
        if 'text_cleaned' not in df.columns:
            if 'text_normalized' in df.columns:
                print("  Using 'text_normalized' column from optimized cleaner...")
                df['text_cleaned'] = df['text_normalized']
            else:
                print("  Creating 'text_cleaned' column on the fly...")
                df['text_cleaned'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        
        # Sentence-BERT embeddings
        print("Generating Sentence-BERT embeddings (this will take a few minutes)...")
        embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        embeddings = embedder.encode(
            df['text_cleaned'].tolist(),
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"✓ Embeddings shape: {embeddings.shape}")
        
        # TF-IDF features
        print("Creating TF-IDF features...")
        tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            stop_words='english',
            max_df=0.9
        tfidf_features = tfidf.fit_transform(df['text_cleaned']).toarray()
        
        # SAVE TFIDF FOR LIVE INFERENCE
        import joblib
        tfidf_path = config.OUTPUT_DIR / 'tfidf_vectorizer.joblib'
        joblib.dump(tfidf, tfidf_path)
        
        print(f"✓ TF-IDF shape: {tfidf_features.shape}")
        
        # Numeric features
        numeric_features = df[[
            'title_length', 'desc_length', 'title_words', 'desc_words',
            'urgency_score', 'positive_score', 'negative_score', 'sentiment_balance',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'source_credibility', 'is_major_outlet'
        ]].values
        
        # Combine all features
        all_features = np.hstack([embeddings, tfidf_features, numeric_features])
        print(f"\n✓ Combined features shape: {all_features.shape}")
        
        return all_features