import random
import db
import feature_extraction as fe


def main():
    df = db.fetch_articles()
    if df.empty:
        raise ValueError("Query returned 0 rows. Check the 'articles' table has data.")
    if "title" not in df.columns:
        raise ValueError(f"Expected a 'title' column, got: {list(df.columns)}")

    df["title"] = df["title"].astype(str)
    # NOTE: placeholder clicks — no real relationship to title content.
    # Replace with real click data before training/evaluating any model.
    df["clicks"] = [random.randint(5000, 10000) for _ in range(len(df))]

    print("\n🏷️  Building features...")
    df = fe.build_features(df)

    final_cols = [
        "id", "title", "clicks", "word_count",
        "category", "subcategory", "trending_topic", "is_trending",
        "tone", "headline_style", "urgency", "story_scope", "story_type",
        "region", "mentioned_entities", "involves_children",
        "contains_quote", "contains_statistic", "severity", "audience",
    ]
    df_out = df[final_cols]

    print(f"\nGenerated {len(final_cols) - 3} feature columns.")
    print("\nSample output:")
    print(df_out.head(10).to_string())

    db.save_features(df_out)       # writes to the 'article_features' table
    db.save_features_csv(df_out)   # also keeps a local CSV copy


if __name__ == "__main__":
    main()