import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# DATA
# =========================

df = pd.read_csv("news.csv")

df = df[
    ["Title", "URL", "Final_Category"]
].dropna(subset=["Title"]).reset_index(drop=True)


# =========================
# USERS
# =========================

users = {
    "user1": ["Artificial Intelligence"],
    "user2": ["Stock Market", "Crypto"],
    "user3": ["Health"],
    "user4": ["Politics", "International"],
    "user5": ["Energy"],
    "user6": ["Robotics", "Technology"],
    "user7": ["Business"],
    "user8": ["Cybersecurity", "Artificial Intelligence"]
}

user_names = list(users.keys())

topics = sorted(set(
    topic
    for interests in users.values()
    for topic in interests
))


# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

news_vectors = vectorizer.fit_transform(df["Title"])

topic_vectors = vectorizer.transform(topics)

similarities = cosine_similarity(
    topic_vectors,
    news_vectors
)


# =========================
# GET NEWS
# =========================

def get_article(topic, seen_articles):

    topic_index = topics.index(topic)

    scores = similarities[topic_index]

    ranked = np.argsort(scores)[::-1]

    for index in ranked:

        if index not in seen_articles:

            seen_articles.add(index)

            return df.iloc[index]

    return None


# =========================
# REWARD
# =========================

def get_reward():

    feedback = input("Like / Dislike (l/d): ").lower()

    while feedback not in ["l", "d"]:
        feedback = input("Enter l or d: ").lower()

    if feedback == "l":
        return 5

    return -5