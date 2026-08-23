import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import config

print("Saving missing artifacts for Streamlit...")

# 1. Load the original checkpoint dataframe to fit TF-IDF
print("Loading checkpoint dataframe...")
df = pd.read_pickle(config.OUTPUT_DIR / 'df_step6.pkl')

if 'text_cleaned' not in df.columns:
    df['text_cleaned'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

print("Fitting TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(
    max_features=config.TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    stop_words='english',
    max_df=0.9
)
tfidf.fit(df['text_cleaned'])
joblib.dump(tfidf, config.OUTPUT_DIR / 'tfidf_vectorizer.joblib')
print(f"✓ Saved tfidf_vectorizer.joblib (Vocabulary size: {len(tfidf.vocabulary_)})")

# 2. Fit Scaler on the features array
print("\nLoading checkpoint features...")
features = np.load(config.OUTPUT_DIR / 'features_step6.npy')
print("Fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(features)
joblib.dump(scaler, config.OUTPUT_DIR / 'scaler.joblib')
print("✓ Saved scaler.joblib")

print("\n🎉 ALL ARTIFACTS SAVED SUCCESSFULLY! You are ready for Streamlit!")
