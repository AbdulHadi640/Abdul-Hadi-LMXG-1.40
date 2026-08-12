import sys
import warnings

import joblib
import numpy as np
import pandas as pd

try:
    from .config import (
        TICKERS,
        TECH_TICKERS,
        SUPPLY_TICKERS,
        TIMEZONE,
        FEATURES_91,
        PREPROCESSING_BUNDLE_FILE,
        HISTORICAL_DATA_FILE,
    )
except ImportError:
    from config import (
        TICKERS,
        TECH_TICKERS,
        SUPPLY_TICKERS,
        TIMEZONE,
        FEATURES_91,
        PREPROCESSING_BUNDLE_FILE,
        HISTORICAL_DATA_FILE,
    )

# ============================================================
# EXACT CLOCK / ROLLING HELPERS
# ============================================================

def exact_clock_value(frame, value_column, minutes):
    left = frame[["TradingDate", "Datetime"]].copy()
    left["_order"] = np.arange(len(frame))
    left["_lookup"] = (
        left["Datetime"] - pd.to_timedelta(minutes, unit="min")
    )

    right = frame[["TradingDate", "Datetime", value_column]].rename(
        columns={"Datetime": "_lookup", value_column: "_value"}
    )

    merged = left.merge(
        right,
        on=["TradingDate", "_lookup"],
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_order")

    return pd.Series(
        merged["_value"].to_numpy(),
        index=frame.index,
        dtype=float,
    )


def exact_return(frame, ticker, minutes):
    previous = exact_clock_value(frame, f"{ticker}_Close", minutes)
    return frame[f"{ticker}_Close"] / previous - 1.0


def rolling_by_day(frame, value_column, window, min_periods, closed, method):
    result = pd.Series(np.nan, index=frame.index, dtype=float)

    for _, group in frame.groupby("TradingDate", sort=False):
        group = group.sort_values("Datetime")
        series = pd.Series(
            group[value_column].to_numpy(),
            index=pd.DatetimeIndex(group["Datetime"]),
        )
        rolled = series.rolling(
            window=window,
            min_periods=min_periods,
            closed=closed,
        )
        result.loc[group.index] = getattr(rolled, method)().to_numpy()

    return result


def rolling_std_by_day(frame, value_column, window, min_periods, closed):
    return rolling_by_day(
        frame, value_column, window, min_periods, closed, "std"
    )


def rolling_mean_by_day(frame, value_column, window, min_periods, closed):
    return rolling_by_day(
        frame, value_column, window, min_periods, closed, "mean"
    )


# ============================================================
# INPUT / PREPROCESSING
# ============================================================

def required_raw_columns():
    columns = [
        "Datetime",
        "AAPL_High",
        "AAPL_Low",
        "AAPL_VWAP",
    ]

    columns += [
        f"{ticker}_Close"
        for ticker in TICKERS
    ]

    return list(dict.fromkeys(columns))


def load_preprocessing_bundle():
    if not PREPROCESSING_BUNDLE_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessing bundle not found:\n"
            f"{PREPROCESSING_BUNDLE_FILE}"
        )

    prep = joblib.load(
        PREPROCESSING_BUNDLE_FILE
    )

    required = [
        "tickers",
        "tech_tickers",
        "supply_tickers",
        "timezone",
        "relationship_features",
        "relationship_coefficients",
        "relationship_intercept",
        "normalized_decoupling_threshold",
        "window_threshold",
        "classifier_features",
        "classifier_feature_count",
    ]

    missing = [
        key for key in required
        if key not in prep
    ]

    if missing:
        raise ValueError(
            "Preprocessing bundle missing:\n"
            + "\n".join(missing)
        )

    if list(prep["tickers"]) != list(TICKERS):
        raise ValueError(
            "Ticker universe does not match bundle."
        )

    if list(prep["classifier_features"]) != list(FEATURES_91):
        raise ValueError(
            "91-feature order does not match bundle."
        )

    if int(prep["classifier_feature_count"]) != 91:
        raise ValueError(
            "Bundle feature count is not 91."
        )

    return prep


def prepare_master_frame(master):
    df = master.copy()

    if (
        "Datetime" not in df.columns
        and "datetime_ny" in df.columns
    ):
        df = df.rename(
            columns={"datetime_ny": "Datetime"}
        )

    if "Datetime" not in df.columns:
        raise ValueError(
            "Master requires Datetime column."
        )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        errors="raise",
    )

    if df["Datetime"].dt.tz is not None:
        df["Datetime"] = (
            df["Datetime"]
            .dt.tz_convert(TIMEZONE)
        )

    df = (
        df.sort_values("Datetime")
        .drop_duplicates("Datetime")
        .reset_index(drop=True)
    )

    df["TradingDate"] = (
        df["Datetime"].dt.date
    )

    return df


def validate_raw_columns(df):
    missing = [
        col
        for col in required_raw_columns()
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing raw columns:\n"
            + "\n".join(missing)
        )


# ============================================================
# FEATURE BLOCKS
# ============================================================

def add_basic_features(df):
    """Returns, candle, groups and relative relationships."""

    for ticker in TICKERS:
        df[f"{ticker}_Ret_1m"] = exact_return(
            df,
            ticker,
            1,
        )

    df["AAPL_Range_Pct"] = (
        (df["AAPL_High"] - df["AAPL_Low"])
        / df["AAPL_Close"]
    )

    df["AAPL_Close_VWAP_Gap"] = (
        (df["AAPL_Close"] - df["AAPL_VWAP"])
        / df["AAPL_VWAP"]
    )

    tech_cols = [
        f"{ticker}_Ret_1m"
        for ticker in TECH_TICKERS
    ]

    supply_cols = [
        f"{ticker}_Ret_1m"
        for ticker in SUPPLY_TICKERS
    ]

    df["Tech_Group_Ret_1m"] = (
        df[tech_cols].mean(axis=1)
    )

    df["Supply_Group_Ret_1m"] = (
        df[supply_cols].mean(axis=1)
    )

    df["AAPL_vs_Tech"] = (
        df["AAPL_Ret_1m"]
        - df["Tech_Group_Ret_1m"]
    )

    df["AAPL_vs_Supply"] = (
        df["AAPL_Ret_1m"]
        - df["Supply_Group_Ret_1m"]
    )

    df["AAPL_vs_QQQ"] = (
        df["AAPL_Ret_1m"]
        - df["QQQ_Ret_1m"]
    )

    df["AAPL_vs_SPY"] = (
        df["AAPL_Ret_1m"]
        - df["SPY_Ret_1m"]
    )

    df["Tech_vs_Supply"] = (
        df["Tech_Group_Ret_1m"]
        - df["Supply_Group_Ret_1m"]
    )

    df["Tech_Dispersion"] = (
        df[tech_cols].std(axis=1, ddof=1)
    )

    df["Supply_Dispersion"] = (
        df[supply_cols].std(axis=1, ddof=1)
    )

    return df


def add_residual_features(df, prep):
    """Expected AAPL return + normalized residual."""

    relationship_features = list(
        prep["relationship_features"]
    )

    coefficients = np.asarray(
        prep["relationship_coefficients"],
        dtype=float,
    )

    if len(relationship_features) != len(coefficients):
        raise ValueError(
            "Relationship feature/coefficient mismatch."
        )

    intercept = float(
        prep["relationship_intercept"]
    )

    df["Expected_AAPL_Ret_1m"] = (
        df[relationship_features]
        .to_numpy(dtype=float)
        @ coefficients
        + intercept
    )

    df["AAPL_Residual_1m"] = (
        df["AAPL_Ret_1m"]
        - df["Expected_AAPL_Ret_1m"]
    )

    df["Residual_Scale_60m"] = rolling_std_by_day(
        df,
        "AAPL_Residual_1m",
        prep.get(
            "residual_volatility_window",
            "60min",
        ),
        int(
            prep.get(
                "residual_volatility_min_periods",
                20,
            )
        ),
        prep.get(
            "residual_volatility_closed",
            "left",
        ),
    )

    df["Normalized_Decoupling_Score"] = (
        df["AAPL_Residual_1m"].abs()
        / df["Residual_Scale_60m"]
    )

    threshold = float(
        prep["normalized_decoupling_threshold"]
    )

    valid = (
        df["Normalized_Decoupling_Score"]
        .notna()
    )

    df["Decoupled_Now_Norm"] = np.nan

    df.loc[
        valid,
        "Decoupled_Now_Norm",
    ] = (
        df.loc[
            valid,
            "Normalized_Decoupling_Score",
        ]
        >= threshold
    ).astype(int)

    return df


def add_lag_features(df, prep):
    """Historical exact-clock lag features."""

    score_lags = prep.get(
        "score_lags",
        [1, 2, 3, 5, 10, 15],
    )

    for lag in score_lags:
        for source, prefix in [
            (
                "Normalized_Decoupling_Score",
                "NormScore",
            ),
            (
                "AAPL_Residual_1m",
                "Residual",
            ),
            (
                "Decoupled_Now_Norm",
                "State",
            ),
        ]:
            df[f"{prefix}_Lag_{lag}m"] = (
                exact_clock_value(
                    df,
                    source,
                    lag,
                )
            )

    for lag in (1, 3, 5, 10):
        df[f"TechRet_Lag_{lag}m"] = (
            exact_clock_value(
                df,
                "Tech_Group_Ret_1m",
                lag,
            )
        )

        df[f"SupplyRet_Lag_{lag}m"] = (
            exact_clock_value(
                df,
                "Supply_Group_Ret_1m",
                lag,
            )
        )

    return df


def add_time_window_features(df, prep):
    """Time-of-day and sustained 10-minute features."""

    df["Minutes_Since_Open"] = (
        df["Datetime"].dt.hour * 60
        + df["Datetime"].dt.minute
        - 570
    )

    angle = (
        2.0
        * np.pi
        * df["Minutes_Since_Open"]
        / 390.0
    )

    df["Time_Sin"] = np.sin(angle)
    df["Time_Cos"] = np.cos(angle)

    window_minutes = int(
        prep.get("window_minutes", 10)
    )

    df["WindowScore_10m"] = rolling_mean_by_day(
        df,
        "Normalized_Decoupling_Score",
        f"{window_minutes}min",
        5,
        "right",
    )

    threshold = float(
        prep["window_threshold"]
    )

    valid = df["WindowScore_10m"].notna()

    df["WindowDecoupled_10m"] = np.nan

    df.loc[
        valid,
        "WindowDecoupled_10m",
    ] = (
        df.loc[
            valid,
            "WindowScore_10m",
        ]
        >= threshold
    ).astype(int)

    window_lags = prep.get(
        "window_lags",
        [1, 3, 5, 10],
    )

    for lag in window_lags:
        for source in [
            "WindowScore_10m",
            "WindowDecoupled_10m",
        ]:
            df[f"{source}_Lag_{lag}m"] = (
                exact_clock_value(
                    df,
                    source,
                    lag,
                )
            )

    return df


def add_multi_horizon_features(df, prep):
    """3m, 5m, 10m and 15m ecosystem relationships."""

    horizons = prep.get(
        "multi_return_horizons",
        [3, 5, 10, 15],
    )

    for horizon in horizons:

        for ticker in TICKERS:
            df[f"{ticker}_Ret_{horizon}m"] = (
                exact_return(
                    df,
                    ticker,
                    horizon,
                )
            )

        tech_cols = [
            f"{ticker}_Ret_{horizon}m"
            for ticker in TECH_TICKERS
        ]

        supply_cols = [
            f"{ticker}_Ret_{horizon}m"
            for ticker in SUPPLY_TICKERS
        ]

        tech = f"Tech_Group_Ret_{horizon}m"
        supply = f"Supply_Group_Ret_{horizon}m"

        df[tech] = df[tech_cols].mean(axis=1)
        df[supply] = df[supply_cols].mean(axis=1)

        aapl = f"AAPL_Ret_{horizon}m"

        df[f"AAPL_vs_Tech_{horizon}m"] = (
            df[aapl] - df[tech]
        )

        df[f"AAPL_vs_Supply_{horizon}m"] = (
            df[aapl] - df[supply]
        )

        df[f"AAPL_vs_QQQ_{horizon}m"] = (
            df[aapl]
            - df[f"QQQ_Ret_{horizon}m"]
        )

        df[f"AAPL_vs_SPY_{horizon}m"] = (
            df[aapl]
            - df[f"SPY_Ret_{horizon}m"]
        )

    return df


# ============================================================
# MAIN FEATURE BUILDER
# ============================================================

def build_features(
    master,
    prep=None,
    verbose=True,
):
    warnings.filterwarnings(
        "ignore",
        category=pd.errors.PerformanceWarning,
    )

    prep = prep or load_preprocessing_bundle()

    df = prepare_master_frame(master)
    validate_raw_columns(df)

    df = add_basic_features(df)
    df = add_residual_features(df, prep)
    df = add_lag_features(df, prep)
    df = add_time_window_features(df, prep)
    df = add_multi_horizon_features(df, prep)

    missing = [
        feature
        for feature in FEATURES_91
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing final features:\n"
            + "\n".join(missing)
        )

    if verbose:
        complete = (
            df[FEATURES_91]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .shape[0]
        )

        print(
            f"✅ Final features: "
            f"{len(FEATURES_91)}/91"
        )

        print(
            f"✅ Complete rows: {complete:,}"
        )

    return df


def get_model_ready_frame(feature_frame):
    missing = [
        feature
        for feature in FEATURES_91
        if feature not in feature_frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing)
        )

    result = feature_frame[
        ["Datetime", "TradingDate"]
        + FEATURES_91
    ].copy()

    return (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(subset=FEATURES_91)
        .reset_index(drop=True)
    )


# ============================================================
# HISTORICAL 91-FEATURE AUDIT
# ============================================================

def audit_historical_features(last_n_days=5):

    if not HISTORICAL_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Historical dataset not found:\n"
            f"{HISTORICAL_DATA_FILE}"
        )

    columns = list(
        dict.fromkeys(
            required_raw_columns()
            + FEATURES_91
        )
    )

    historical = pd.read_parquet(
        HISTORICAL_DATA_FILE,
        columns=columns,
    )

    historical["Datetime"] = pd.to_datetime(
        historical["Datetime"]
    )

    dates = historical["Datetime"].dt.date
    selected = sorted(
        dates.unique()
    )[-last_n_days:]

    audit_df = (
        historical[
            dates.isin(selected)
        ]
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

    reference = (
        audit_df[FEATURES_91]
        .reset_index(drop=True)
    )

    raw = audit_df[
        required_raw_columns()
    ].copy()

    rebuilt = build_features(
        raw,
        verbose=False,
    )[FEATURES_91].reset_index(drop=True)

    passed = 0
    failed = []

    for feature in FEATURES_91:

        expected = pd.to_numeric(
            reference[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

        actual = pd.to_numeric(
            rebuilt[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

        same = np.allclose(
            actual,
            expected,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )

        if same:
            passed += 1
        else:
            failed.append(feature)

    print("\nHISTORICAL FEATURE RECONSTRUCTION AUDIT")
    print(f"Audit days      : {len(selected)}")
    print(f"Audit rows      : {len(audit_df):,}")
    print(f"Passed features : {passed}/91")
    print(f"Failed features : {len(failed)}/91")

    if failed:
        print("\nFailed:")
        for feature in failed:
            print(f" - {feature}")

        raise ValueError(
            "Feature reconstruction failed."
        )

    print(
        "\n✅ PASS: 91/91 features "
        "reconstructed successfully"
    )

    return True


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python src/features.py audit"
        )
        raise SystemExit(1)

    if sys.argv[1].lower() == "audit":
        audit_historical_features(
            last_n_days=5
        )
    else:
        raise ValueError(
            f"Unknown command: {sys.argv[1]}"
        )