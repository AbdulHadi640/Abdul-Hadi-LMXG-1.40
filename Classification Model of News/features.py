from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# ============================================================
# TEXT FEATURES
# ============================================================

def text_features(char_max=100000):
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.995,
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                min_df=2,
                max_features=char_max,
                sublinear_tf=True,
            ),
        ),
    ])
