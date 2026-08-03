"""
db.py — everything related to loading from and saving to Postgres.
"""
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

# -------------------------------------------------------------------
# Connection setup
# -------------------------------------------------------------------
db_name = os.getenv("PG_DATABASE", "news_scraper")
host = os.getenv("PG_HOST", "localhost")
password = os.getenv("PG_PASSWORD")
port = os.getenv("PG_PORT", "5432")
user = os.getenv("PG_USER", "postgres")

if not password:
    raise EnvironmentError(
        "PG_PASSWORD is not set. Check that a .env file exists in this folder."
    )

DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

FEATURES_TABLE_NAME = "article_features"


# -------------------------------------------------------------------
# Load
# -------------------------------------------------------------------
def fetch_articles() -> pd.DataFrame:
    """Pull id and title from the raw articles table."""
    print("🔌 Connecting to database and loading articles...")
    session = SessionLocal()
    try:
        query = text("SELECT id, title FROM articles;")
        df = pd.read_sql(query, session.bind)
        print(f"✅ Successfully loaded {len(df)} records!")
        return df
    except Exception as e:
        print(f"❌ Error while fetching data: {e}")
        raise
    finally:
        session.close()


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------
def save_features(df: pd.DataFrame, table_name: str = FEATURES_TABLE_NAME,
                   if_exists: str = "replace") -> None:
    """
    Write the feature-enriched dataframe to a new database table.

    if_exists:
        "replace" - drop and recreate the table each run (default)
        "append"  - add rows to an existing table
        "fail"    - raise an error if the table already exists
    """
    print(f"💾 Writing {len(df)} rows to table '{table_name}' (if_exists='{if_exists}')...")
    try:
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        print(f"✅ Saved to database table '{table_name}'.")
    except Exception as e:
        print(f"❌ Error while saving to database: {e}")
        raise


def save_features_csv(df: pd.DataFrame, path: str = "articles_with_features.csv") -> None:
    """Also keep a CSV copy for quick inspection outside the DB."""
    df.to_csv(path, index=False)
    print(f"💾 Also saved CSV -> {path}")