import sys
import time

import pandas as pd
import requests

try:
    from .config import (
        TICKERS,
        MASSIVE_API_KEY,
        TIMEZONE,
        MARKET_OPEN,
        MARKET_CLOSE,
    )
    from .database import get_connection

except ImportError:
    from config import (
        TICKERS,
        MASSIVE_API_KEY,
        TIMEZONE,
        MARKET_OPEN,
        MARKET_CLOSE,
    )
    from database import get_connection

# API SETTINGS

REQUEST_DELAY_SECONDS = 13
MAX_RETRIES = 8


def _response_detail(response):
    """Safely read API error without exposing request URL/API key."""
    try:
        return response.json()
    except Exception:
        return response.text[:500]

# MASSIVE API
def download_ticker(
    ticker,
    date,
    max_retries=MAX_RETRIES,
):
    """
    Download one ticker's 1-minute bars for one day
    and keep regular New York session only.
    """

    if not MASSIVE_API_KEY:
        raise ValueError(
            "MASSIVE_API_KEY missing from .env"
        )

    url = (
        "https://api.massive.com/v2/aggs/ticker/"
        f"{ticker}/range/1/minute/{date}/{date}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": MASSIVE_API_KEY,
    }

    rows = []
    retries = 0

    while url:
        # REQUEST
        try:
            response = requests.get(
                url,
                params=params,
                timeout=60,
            )

        except requests.RequestException as exc:

            retries += 1

            if retries > max_retries:
                raise RuntimeError(
                    f"{ticker}: network retries exceeded"
                ) from exc

            wait = min(
                5 * (2 ** (retries - 1)),
                60,
            )

            print(
                f"   ⚠ Network error. "
                f"Retrying in {wait}s..."
            )

            time.sleep(wait)
            continue
        # RATE LIMIT
        if response.status_code == 429:

            retries += 1

            if retries > max_retries:
                raise RuntimeError(
                    f"{ticker}: rate-limit retries exceeded"
                )

            retry_after = response.headers.get(
                "Retry-After"
            )

            try:
                wait = (
                    float(retry_after)
                    if retry_after
                    else 15 * (2 ** (retries - 1))
                )
            except (TypeError, ValueError):
                wait = 15 * (2 ** (retries - 1))

            wait = min(wait, 120)

            print(
                f"   ⚠ Rate limit reached. "
                f"Waiting {wait:.0f}s..."
            )

            time.sleep(wait)
            continue
        # AUTH / ACCESS
        if response.status_code == 401:
            raise RuntimeError(
                f"{ticker}: API authentication failed\n"
                f"Massive response: "
                f"{_response_detail(response)}"
            )

        if response.status_code == 403:
            raise RuntimeError(
                f"{ticker}: API access denied\n"
                f"Massive response: "
                f"{_response_detail(response)}"
            )
        # OTHER API ERRORS
        if not response.ok:
            raise RuntimeError(
                f"{ticker}: Massive API HTTP "
                f"{response.status_code}\n"
                f"Massive response: "
                f"{_response_detail(response)}"
            )

        retries = 0
        # READ RESPONSE
        payload = response.json()

        rows.extend(
            payload.get("results", [])
        )

        # Pagination
        url = payload.get("next_url")

        if url:
            params = {
                "apiKey": MASSIVE_API_KEY
            }
    # NO DATA
    if not rows:
        return pd.DataFrame()
    # CONVERT TO DATAFRAME

    df = pd.DataFrame({
        "open": [
            row.get("o")
            for row in rows
        ],
        "high": [
            row.get("h")
            for row in rows
        ],
        "low": [
            row.get("l")
            for row in rows
        ],
        "close": [
            row.get("c")
            for row in rows
        ],
        "volume": [
            row.get("v")
            for row in rows
        ],
        "vwap": [
            row.get("vw")
            for row in rows
        ],
        "transactions": [
            row.get("n")
            for row in rows
        ],
        "timestamp_utc": pd.to_datetime(
            [
                row.get("t")
                for row in rows
            ],
            unit="ms",
            utc=True,
        ),
    })
    # UTC → NEW YORK
    ny_time = (
        df["timestamp_utc"]
        .dt.tz_convert(TIMEZONE)
    )

    df["datetime_ny"] = (
        ny_time.dt.tz_localize(None)
    )

    df["trading_date"] = (
        df["datetime_ny"].dt.date
    )

    df["ticker"] = ticker
    # REGULAR SESSION ONLY
    # 09:30 <= time < 16:00

    market_open = pd.Timestamp(
        MARKET_OPEN
    ).time()

    market_close = pd.Timestamp(
        MARKET_CLOSE
    ).time()

    clock = (
        df["datetime_ny"].dt.time
    )

    df = df[
        (clock >= market_open)
        & (clock < market_close)
    ]

    return (
        df.sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc")
        .reset_index(drop=True)
    )
# SAVE MARKET DATA
def save_market_bars(df):
    """Insert/update bars in PostgreSQL."""

    if df.empty:
        return 0

    query = """
        INSERT INTO market_bars (
            ticker,
            timestamp_utc,
            datetime_ny,
            trading_date,
            open,
            high,
            low,
            close,
            volume,
            vwap,
            transactions,
            source
        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s
        )

        ON CONFLICT (
            ticker,
            timestamp_utc
        )

        DO UPDATE SET
            datetime_ny = EXCLUDED.datetime_ny,
            trading_date = EXCLUDED.trading_date,
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            vwap = EXCLUDED.vwap,
            transactions = EXCLUDED.transactions;
    """

    def optional(value, cast):
        if pd.isna(value):
            return None
        return cast(value)

    records = []

    for row in df.itertuples(index=False):

        records.append(
            (
                row.ticker,

                pd.Timestamp(
                    row.timestamp_utc
                ).to_pydatetime(),

                pd.Timestamp(
                    row.datetime_ny
                ).to_pydatetime(),

                row.trading_date,

                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),

                optional(
                    row.volume,
                    int,
                ),

                optional(
                    row.vwap,
                    float,
                ),

                optional(
                    row.transactions,
                    int,
                ),

                "massive",
            )
        )

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.executemany(
                query,
                records,
            )

    return len(records)

# LOAD ONE DAY
def load_day_from_database(date):
    """Load all stored tickers for one trading day."""

    query = """
        SELECT
            ticker,
            timestamp_utc,
            datetime_ny,
            trading_date,
            open,
            high,
            low,
            close,
            volume,
            vwap,
            transactions

        FROM market_bars

        WHERE trading_date = %s

        ORDER BY
            ticker,
            datetime_ny;
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                (date,),
            )

            rows = cur.fetchall()

            columns = [
                column.name
                for column in cur.description
            ]

    if not rows:
        raise ValueError(
            f"No market data found for {date}"
        )

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    df["datetime_ny"] = pd.to_datetime(
        df["datetime_ny"]
    )

    return df

# DAY SUMMARY
def show_day_summary(date):
    """Show number of stored bars per ticker."""

    query = """
        SELECT
            ticker,
            COUNT(*),
            MIN(datetime_ny),
            MAX(datetime_ny)

        FROM market_bars

        WHERE trading_date = %s

        GROUP BY ticker

        ORDER BY ticker;
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                (date,),
            )

            rows = cur.fetchall()

    print(
        f"\nPOSTGRESQL SUMMARY — {date}\n"
    )

    if not rows:
        print(
            f"⚠ No market bars stored for {date}"
        )
        return

    for ticker, count, start, end in rows:

        print(
            f"{ticker:5} | "
            f"bars={count:3} | "
            f"{start} to {end}"
        )
# EXACT 12-WAY ALIGNMENT
def build_aligned_master(date):
    """
    Keep only exact timestamps shared by
    all 12 instruments.
    """

    bars = load_day_from_database(date)

    available = set(
        bars["ticker"].unique()
    )

    missing = [
        ticker
        for ticker in TICKERS
        if ticker not in available
    ]

    if missing:
        raise ValueError(
            "Missing tickers for "
            f"{date}: "
            + ", ".join(missing)
        )
    # COMMON TIMESTAMPS

    common_times = None

    for ticker in TICKERS:

        times = set(
            bars.loc[
                bars["ticker"] == ticker,
                "datetime_ny",
            ]
        )

        if common_times is None:
            common_times = times

        else:
            common_times &= times

    if not common_times:
        raise ValueError(
            "No common timestamps across "
            "all 12 instruments."
        )

    master = pd.DataFrame({
        "Datetime": sorted(common_times)
    })

    raw_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
    ]
    # MERGE EACH TICKER
    for ticker in TICKERS:

        temp = bars.loc[
            bars["ticker"] == ticker,
            ["datetime_ny"] + raw_columns,
        ].copy()

        temp = temp.rename(
            columns={
                "datetime_ny":
                    "Datetime",

                "open":
                    f"{ticker}_Open",

                "high":
                    f"{ticker}_High",

                "low":
                    f"{ticker}_Low",

                "close":
                    f"{ticker}_Close",

                "volume":
                    f"{ticker}_Volume",

                "vwap":
                    f"{ticker}_VWAP",

                "transactions":
                    f"{ticker}_Transactions",
            }
        )

        master = master.merge(
            temp,
            on="Datetime",
            how="inner",
            validate="one_to_one",
        )

    master["TradingDate"] = (
        master["Datetime"].dt.date
    )

    master = (
        master
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

    print(
        f"✅ Exact aligned rows: "
        f"{len(master):,}"
    )

    print(
        f"✅ Columns: "
        f"{len(master.columns)}"
    )

    print(
        f"✅ Range: "
        f"{master['Datetime'].min()} "
        f"to "
        f"{master['Datetime'].max()}"
    )

    return master
# FETCH FULL DAY
def fetch_day(date):
    """
    Download all project instruments
    and save them to PostgreSQL.
    """

    print(
        f"\nMASSIVE → POSTGRESQL — {date}"
    )

    total = 0

    for number, ticker in enumerate(
        TICKERS,
        start=1,
    ):

        print(
            f"[{number:02d}/{len(TICKERS)}] "
            f"{ticker}"
        )

        df = download_ticker(
            ticker,
            date,
        )

        if df.empty:

            print(
                "   ⚠ No bars returned"
            )

        else:

            saved = save_market_bars(df)

            total += saved

            print(
                f"   ✅ {saved:,} "
                f"regular-session bars"
            )
        # IMPORTANT:
        # Keep requests below API rate limit.

        if number < len(TICKERS):

            print(
                f"   ⏳ Waiting "
                f"{REQUEST_DELAY_SECONDS}s..."
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    print(
        f"\n✅ Total processed: "
        f"{total:,}"
    )

    show_day_summary(date)

# COMMAND LINE

def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "  python src/data.py fetch YYYY-MM-DD\n"
            "  python src/data.py summary YYYY-MM-DD\n"
            "  python src/data.py align YYYY-MM-DD"
        )

        raise SystemExit(1)

    command = (
        sys.argv[1]
        .strip()
        .lower()
    )

    date = sys.argv[2].strip()

    if command == "fetch":

        fetch_day(date)

    elif command == "summary":

        show_day_summary(date)

    elif command == "align":

        master = build_aligned_master(
            date
        )

        print(
            master[
                [
                    "Datetime",
                    "AAPL_Close",
                    "QQQ_Close",
                    "SPY_Close",
                ]
            ]
            .head()
            .to_string(index=False)
        )

    else:

        raise ValueError(
            f"Unknown command: {command}"
        )


if __name__ == "__main__":
    main()
