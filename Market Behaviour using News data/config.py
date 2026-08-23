"""
Configuration Settings
======================
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_FOLDER = BASE_DIR / "filtered_candidates"
SPY_DATA_PATH = DATA_FOLDER / "SPY_1min.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "saved_models"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Time Settings
TIMEZONE_NY = "America/New_York"

# Duplicate Removal
SIMILARITY_THRESHOLD = 0.95  # 95% similar = duplicate

# Market Sessions
MARKET_SESSIONS = {
    'Pre-Market': (240, 570),    # 4:00 AM - 9:30 AM (in minutes)
    'Regular': (570, 960),       # 9:30 AM - 4:00 PM
    'After-Hours': (960, 1200),  # 4:00 PM - 8:00 PM
    'Closed': None               # Everything else
}

# Clustering
N_CLUSTERS = 3
CLUSTER_LABELS = {0: 'Crash', 1: 'Neutral', 2: 'Spike'}

# Model Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50

# Feature Engineering
TFIDF_MAX_FEATURES = 500
EMBEDDING_MODEL = 'paraphrase-MiniLM-L6-v2'

# Source Credibility
SOURCE_CREDIBILITY = {
    'BBC': 0.95,
    'NPR': 0.92,
    'The Guardian': 0.90,
    'CNBC': 0.88,
    'CBS News': 0.85,
    'NBC News': 0.83,
    'CNN': 0.80,
    'Al Jazeera': 0.78,
    'Deutsche Welle': 0.82,
    'Euronews': 0.80
}

MAJOR_OUTLETS = ['BBC', 'CNN', 'CNBC', 'NPR', 'The Guardian']

# Urgency & Sentiment Words
URGENCY_WORDS = ['breaking', 'urgent', 'alert', 'crisis', 'crash', 'shock', 'emergency']
POSITIVE_WORDS = ['gain', 'rise', 'grow', 'profit', 'up', 'surge', 'bull']
NEGATIVE_WORDS = ['loss', 'fall', 'drop', 'crash', 'decline', 'down', 'bear']