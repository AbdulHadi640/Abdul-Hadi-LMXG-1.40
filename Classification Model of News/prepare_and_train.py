import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

from config import INPUT_FILE, RANDOM_STATE, CATEGORIES
from utils import normalize
from rules import trusted_title_label, url_taxonomy
from features import text_features

# ============================================================
# FINAL TARGETED LABEL FIX
#
# Input:
#   news.csv
#
# Outputs only:
#   corrected_news.csv
#   news_classifier.pkl
#
# No Cleanlab
# No SentenceTransformer
# No sample weights
# No augmentation
# ============================================================


# ============================================================
# LOAD + CLEAN
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 78)
print("FINAL TARGETED LABEL FIX + TF-IDF + LINEARSVC")
print("=" * 78)

print("\nOriginal rows:", len(df))

if "Title" not in df.columns or "Category" not in df.columns:
    raise ValueError("news.csv must contain Title and Category columns.")

if "URL" not in df.columns:
    df["URL"] = ""

df = df.dropna(subset=["Title", "Category"]).copy()

df["Title"] = (
    df["Title"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df["Category"] = (
    df["Category"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["Title"] != "")
    & df["Category"].isin(CATEGORIES)
].copy()

df["Original_Category"] = df["Category"]
df["Title_Normalized"] = df["Title"].map(normalize)
df["Original_Row_ID"] = np.arange(len(df))

print("Valid rows:", len(df))


# ============================================================
# DUPLICATE / CONFLICT CONSENSUS
# ============================================================

majority = (
    df.groupby("Title_Normalized")["Original_Category"]
    .agg(lambda s: s.value_counts().index[0])
    .rename("Original_Majority")
)

label_count = (
    df.groupby("Title_Normalized")["Original_Category"]
    .nunique()
    .rename("Original_Label_Count")
)

unique = (
    df.sort_values("Original_Row_ID")
    .drop_duplicates("Title_Normalized")
    [["Title_Normalized", "Title", "URL"]]
    .merge(majority, on="Title_Normalized")
    .merge(label_count, on="Title_Normalized")
)

print("Unique headlines:", len(unique))
print(
    "Conflicting headlines:",
    int((unique["Original_Label_Count"] > 1).sum())
)


# ============================================================
# PASS 1: TRUSTED RULE / URL LABELS
# ============================================================

trusted_labels = []
label_sources = []

for row in unique.itertuples(index=False):
    title_label = trusted_title_label(row.Title)
    url_label = url_taxonomy(row.URL)

    if title_label is not None:
        trusted_labels.append(title_label)

        if url_label == title_label:
            label_sources.append("TITLE+URL")
        else:
            label_sources.append("TITLE")

    elif url_label is not None:
        trusted_labels.append(url_label)
        label_sources.append("URL")

    else:
        trusted_labels.append(None)
        label_sources.append("UNRESOLVED")


unique["Trusted_Label"] = trusted_labels
unique["Label_Source"] = label_sources

trusted = unique[
    unique["Trusted_Label"].notna()
].copy()

unresolved_mask = (
    unique["Trusted_Label"].isna()
)

print("\nTrusted labels:", len(trusted))
print(
    "Unresolved before pass 2:",
    int(unresolved_mask.sum())
)

print("\nTrusted distribution:")
print(trusted["Trusted_Label"].value_counts())


# ============================================================
# PASS 2: SIMPLE LABEL CLEANER
#
# Only unresolved headlines are filled here.
# ============================================================

label_cleaner = Pipeline([
    (
        "features",
        text_features(char_max=100000),
    ),
    (
        "classifier",
        LogisticRegression(
            C=3.0,
            max_iter=2500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
    ),
])

print("\nTraining label cleaner...")

label_cleaner.fit(
    trusted["Title"],
    trusted["Trusted_Label"],
)

unique["Final_Category"] = unique["Trusted_Label"]

if unresolved_mask.any():
    unresolved_pred = label_cleaner.predict(
        unique.loc[
            unresolved_mask,
            "Title",
        ]
    )

    unique.loc[
        unresolved_mask,
        "Final_Category",
    ] = unresolved_pred

    unique.loc[
        unresolved_mask,
        "Label_Source",
    ] = "AUTO_LABELER"


print("\nFinal corrected distribution:")
print(unique["Final_Category"].value_counts())

print(
    "\nLabels different from original majority:",
    int(
        (
            unique["Final_Category"]
            != unique["Original_Majority"]
        ).sum()
    ),
)


# ============================================================
# PROPAGATE TO ALL VALID ROWS
# ============================================================

df = df.merge(
    unique[
        [
            "Title_Normalized",
            "Original_Majority",
            "Final_Category",
            "Label_Source",
        ]
    ],
    on="Title_Normalized",
    how="left",
)


# ============================================================
# UNIQUE TITLE 80/20 SPLIT
# ============================================================

train_unique, test_unique = train_test_split(
    unique,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=unique["Final_Category"],
)

train_titles = set(
    train_unique["Title_Normalized"]
)

train_df = df[
    df["Title_Normalized"].isin(train_titles)
].copy()

test_df = df[
    ~df["Title_Normalized"].isin(train_titles)
].copy()

test_eval = (
    test_df
    .sort_values("Original_Row_ID")
    .drop_duplicates("Title_Normalized")
    .copy()
)

print("\nTraining rows:", len(train_df))
print("Testing rows :", len(test_df))
print("Unique test  :", len(test_eval))
print("Unused rows  : 0")


# ============================================================
# FINAL MODEL
# Simple TF-IDF + LinearSVC
# ============================================================

model = Pipeline([
    (
        "features",
        text_features(char_max=120000),
    ),
    (
        "classifier",
        LinearSVC(
            C=0.10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])

print("\nTraining final TF-IDF + LinearSVC...")

model.fit(
    train_df["Title"],
    train_df["Final_Category"],
)

pred = model.predict(
    test_eval["Title"]
)


# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(
    test_eval["Final_Category"],
    pred,
)

bal = balanced_accuracy_score(
    test_eval["Final_Category"],
    pred,
)

macro = f1_score(
    test_eval["Final_Category"],
    pred,
    average="macro",
)

weighted = f1_score(
    test_eval["Final_Category"],
    pred,
    average="weighted",
)

print("\n" + "=" * 78)
print("FINAL TEST - UNIQUE UNSEEN HEADLINES")
print("=" * 78)

print(f"Accuracy          : {acc*100:.2f}%")
print(f"Balanced Accuracy : {bal*100:.2f}%")
print(f"Macro F1          : {macro*100:.2f}%")
print(f"Weighted F1       : {weighted*100:.2f}%")

print("\nClassification Report:\n")
print(
    classification_report(
        test_eval["Final_Category"],
        pred,
        digits=4,
    )
)


# ============================================================
# SAVE ONLY TWO FILES
# ============================================================

df.to_csv(
    "corrected_news.csv",
    index=False,
)

joblib.dump(
    model,
    "news_classifier.pkl",
)

print("\nSaved only:")
print("1. corrected_news.csv")
print("2. news_classifier.pkl")

print(
    "\nNOTE: Test labels are automatically corrected weak labels, "
    "not human-verified gold labels."
)


if __name__ == "__main__":
    pass