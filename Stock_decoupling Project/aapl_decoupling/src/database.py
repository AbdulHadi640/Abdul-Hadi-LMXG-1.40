import psycopg
from psycopg import sql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

try:
    from .config import (
        PG_HOST, PG_PORT, PG_DATABASE,
        PG_USER, PG_PASSWORD,
    )
except ImportError:
    from config import (
        PG_HOST, PG_PORT, PG_DATABASE,
        PG_USER, PG_PASSWORD,
    )
# CONNECTIONS
def get_connection(database=None, autocommit=False):
    """Return a psycopg connection."""
    return psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=database or PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD,
        autocommit=autocommit,
    )


def get_engine():
    """Return SQLAlchemy engine."""
    url = URL.create(
        "postgresql+psycopg",
        username=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
    )
    return create_engine(url, pool_pre_ping=True)

# DATABASE SETUP

def create_database_if_not_exists():
    """Create project database when missing."""

    with get_connection(
        database="postgres",
        autocommit=True,
    ) as conn:

        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (PG_DATABASE,),
            )

            if cur.fetchone():
                print(f"✅ Database exists: {PG_DATABASE}")
                return

            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(PG_DATABASE)
                )
            )

            print(f"✅ Database created: {PG_DATABASE}")


def test_connection():
    """Verify PostgreSQL connection."""

    engine = get_engine()

    try:
        with engine.connect() as conn:
            name = conn.execute(
                text("SELECT current_database()")
            ).scalar()

            print(f"✅ PostgreSQL connected: {name}")

    finally:
        engine.dispose()

# TABLE SCHEMA

TABLES = [
    """
    CREATE TABLE IF NOT EXISTS market_bars (
        id BIGSERIAL PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        timestamp_utc TIMESTAMPTZ NOT NULL,
        datetime_ny TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        trading_date DATE NOT NULL,
        open DOUBLE PRECISION NOT NULL,
        high DOUBLE PRECISION NOT NULL,
        low DOUBLE PRECISION NOT NULL,
        close DOUBLE PRECISION NOT NULL,
        volume BIGINT,
        vwap DOUBLE PRECISION,
        transactions INTEGER,
        source VARCHAR(30) DEFAULT 'massive',
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, timestamp_utc)
    );
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_market_bars_date
    ON market_bars(trading_date);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_market_bars_ticker_date
    ON market_bars(ticker, trading_date);
    """,

    """
    CREATE TABLE IF NOT EXISTS historical_dataset (
        id BIGSERIAL PRIMARY KEY,
        datetime_ny TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        trading_date DATE NOT NULL,
        split VARCHAR(10) NOT NULL,
        target SMALLINT,
        features JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        CHECK (target IS NULL OR target IN (0, 1)),
        UNIQUE(datetime_ny)
    );
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_historical_split
    ON historical_dataset(split);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_historical_date
    ON historical_dataset(trading_date);
    """,

    """
    CREATE TABLE IF NOT EXISTS predictions (
        id BIGSERIAL PRIMARY KEY,
        datetime_ny TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        trading_date DATE NOT NULL,
        model_version VARCHAR(100) NOT NULL,
        probability_decoupled DOUBLE PRECISION NOT NULL,
        prediction SMALLINT NOT NULL,
        actual_label SMALLINT,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        CHECK (prediction IN (0, 1)),
        CHECK (actual_label IS NULL OR actual_label IN (0, 1)),
        CHECK (
            probability_decoupled >= 0
            AND probability_decoupled <= 1
        ),
        UNIQUE(datetime_ny, model_version)
    );
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_predictions_date
    ON predictions(trading_date);
    """,

    """
    CREATE TABLE IF NOT EXISTS model_results (
        id BIGSERIAL PRIMARY KEY,
        model_version VARCHAR(100) NOT NULL,
        evaluation_set VARCHAR(30) NOT NULL,
        start_date DATE,
        end_date DATE,
        n_rows INTEGER,
        coverage DOUBLE PRECISION,
        accuracy DOUBLE PRECISION,
        majority_accuracy DOUBLE PRECISION,
        balanced_accuracy DOUBLE PRECISION,
        roc_auc DOUBLE PRECISION,
        pr_auc DOUBLE PRECISION,
        precision_decoupled DOUBLE PRECISION,
        recall_decoupled DOUBLE PRECISION,
        f1_decoupled DOUBLE PRECISION,
        metrics_json JSONB,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """,
]


def create_tables():
    """Create all project tables and indexes."""

    engine = get_engine()

    try:
        with engine.begin() as conn:
            for statement in TABLES:
                conn.execute(text(statement))

        print("✅ Tables ready")

    finally:
        engine.dispose()


def show_tables():
    """Display project tables."""

    query = text("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)

    engine = get_engine()

    try:
        with engine.connect() as conn:
            rows = conn.execute(query).fetchall()

        print("\nPostgreSQL tables:")
        for row in rows:
            print(f"  ✅ {row[0]}")

    finally:
        engine.dispose()

# INITIALIZATION

def initialize_database():
    print("\nAAPL DECOUPLING DATABASE SETUP")

    create_database_if_not_exists()
    test_connection()
    create_tables()
    show_tables()

    print("\n✅ Database initialization completed")


if __name__ == "__main__":
    initialize_database()
