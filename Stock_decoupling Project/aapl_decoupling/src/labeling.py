import sys

import numpy as np
import pandas as pd

try:
    from .config import (
        HISTORICAL_DATA_FILE, TARGET_COLUMN,
        ENTER_INTENSITY, EXIT_INTENSITY,
        ENTER_PERSISTENCE, EXIT_PERSISTENCE,
        V4_MIN_RUN, V4_EDGE_MARGIN,
    )
except ImportError:
    from config import (
        HISTORICAL_DATA_FILE, TARGET_COLUMN,
        ENTER_INTENSITY, EXIT_INTENSITY,
        ENTER_PERSISTENCE, EXIT_PERSISTENCE,
        V4_MIN_RUN, V4_EDGE_MARGIN,
    )


def exact_future(df, column, minutes=10):
    """Exact value at t + minutes."""

    left = df[["TradingDate", "Datetime"]].copy()
    left["_i"] = np.arange(len(df))
    left["_time"] = left["Datetime"] + pd.Timedelta(minutes=minutes)

    right = df[
        ["TradingDate", "Datetime", column]
    ].rename(
        columns={
            "Datetime": "_time",
            column: "_value",
        }
    )

    out = left.merge(
        right,
        on=["TradingDate", "_time"],
        how="left",
        validate="many_to_one",
    ).sort_values("_i")

    return pd.Series(
        out["_value"].to_numpy(),
        index=df.index,
        dtype=float,
    )


def persistence_10m(df):
    """10-minute rolling mean per trading day."""

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for _, g in df.groupby("TradingDate", sort=False):
        g = g.sort_values("Datetime")

        s = pd.Series(
            g["Decoupled_Now_Norm"].to_numpy(),
            index=pd.DatetimeIndex(g["Datetime"]),
        )

        result.loc[g.index] = (
            s.rolling(
                "10min",
                min_periods=5,
                closed="right",
            )
            .mean()
            .to_numpy()
        )

    return result


def hysteresis(group):
    """Frozen V3 hysteresis."""

    labels = pd.Series(
        np.nan,
        index=group.index,
        dtype=float,
    )

    state = 0

    for idx, row in group.sort_values("Datetime").iterrows():

        intensity = row["Future_WindowScore_10m_V2"]
        persistence = row["Future_Persistence_10m"]

        if pd.isna(intensity) or pd.isna(persistence):
            continue

        if state == 0:
            if (
                intensity >= ENTER_INTENSITY
                and persistence >= ENTER_PERSISTENCE
            ):
                state = 1

        elif (
            intensity <= EXIT_INTENSITY
            and persistence <= EXIT_PERSISTENCE
        ):
            state = 0

        labels.loc[idx] = state

    return labels


def add_runs(df):
    """Create contiguous 1-minute V3 runs."""

    state = df["Target_V3_Hysteresis_Raw"]

    prev_state = (
        df.groupby("TradingDate")[state.name]
        .shift()
    )

    prev_time = (
        df.groupby("TradingDate")["Datetime"]
        .shift()
    )

    valid = state.notna()

    new_run = valid & (
        prev_state.isna()
        | state.ne(prev_state)
        | (df["Datetime"] - prev_time).ne(
            pd.Timedelta(minutes=1)
        )
    )

    df["V4_Run_ID"] = (
        new_run.cumsum()
        .astype(float)
        .where(valid)
    )

    groups = df.groupby(
        "V4_Run_ID",
        dropna=True,
    )

    df["V4_Run_Length"] = (
        groups["Datetime"].transform("size")
    )

    df["V4_Pos_From_Start"] = groups.cumcount()

    df["V4_Pos_To_End"] = (
        df["V4_Run_Length"]
        - df["V4_Pos_From_Start"]
        - 1
    )

    return df


def build_v4_target(feature_frame, verbose=True):
    """Build Target_V4_Run8_Margin1."""

    df = feature_frame.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    if "TradingDate" not in df:
        df["TradingDate"] = df["Datetime"].dt.date

    df = (
        df.sort_values(["TradingDate", "Datetime"])
        .reset_index(drop=True)
    )

    required = {
        "Decoupled_Now_Norm",
        "WindowScore_10m",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing labeling columns: {sorted(missing)}"
        )

    df["Persistence_10m"] = persistence_10m(df)

    df["Future_WindowScore_10m_V2"] = exact_future(
        df,
        "WindowScore_10m",
    )

    df["Future_Persistence_10m"] = exact_future(
        df,
        "Persistence_10m",
    )

    df["Target_V3_Hysteresis_Raw"] = np.nan

    for _, group in df.groupby(
        "TradingDate",
        sort=False,
    ):
        labels = hysteresis(group)

        df.loc[
            labels.index,
            "Target_V3_Hysteresis_Raw",
        ] = labels

    df = add_runs(df)

    clear = (
        df["Target_V3_Hysteresis_Raw"].notna()
        & (df["V4_Run_Length"] >= V4_MIN_RUN)
        & (df["V4_Pos_From_Start"] >= V4_EDGE_MARGIN)
        & (df["V4_Pos_To_End"] >= V4_EDGE_MARGIN)
    )

    df[TARGET_COLUMN] = np.nan

    df.loc[
        clear,
        TARGET_COLUMN,
    ] = df.loc[
        clear,
        "Target_V3_Hysteresis_Raw",
    ]

    if verbose:
        possible = df[
            "Target_V3_Hysteresis_Raw"
        ].notna().sum()

        labeled = df[TARGET_COLUMN].notna().sum()

        print("\nV4 GROUND TRUTH")
        print(f"Possible : {possible:,}")
        print(f"Clear    : {labeled:,}")
        print(
            f"Coverage : "
            f"{labeled / possible:.2%}"
        )
        print(
            f"ALIGNED  : "
            f"{(df[TARGET_COLUMN] == 0).sum():,}"
        )
        print(
            f"DECOUPLED: "
            f"{(df[TARGET_COLUMN] == 1).sum():,}"
        )

    return df


def audit_historical_labels(last_n_days=5):
    """Verify reconstructed V4 target."""

    cols = [
        "Datetime",
        "TradingDate",
        "Decoupled_Now_Norm",
        "WindowScore_10m",
        TARGET_COLUMN,
    ]

    data = pd.read_parquet(
        HISTORICAL_DATA_FILE,
        columns=cols,
    )

    data["Datetime"] = pd.to_datetime(
        data["Datetime"]
    )

    days = sorted(
        data["TradingDate"].unique()
    )[-last_n_days:]

    audit = (
        data[data["TradingDate"].isin(days)]
        .sort_values(["TradingDate", "Datetime"])
        .reset_index(drop=True)
    )

    expected = audit[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    actual = build_v4_target(
        audit[
            [
                "Datetime",
                "TradingDate",
                "Decoupled_Now_Norm",
                "WindowScore_10m",
            ]
        ],
        verbose=False,
    )[TARGET_COLUMN].to_numpy(dtype=float)

    nan_mismatch = np.sum(
        np.isnan(expected) != np.isnan(actual)
    )

    valid = (
        np.isfinite(expected)
        & np.isfinite(actual)
    )

    value_mismatch = np.sum(
        expected[valid] != actual[valid]
    )

    print("\nHISTORICAL V4 LABEL AUDIT")
    print(f"Audit days      : {len(days)}")
    print(f"Audit rows      : {len(audit):,}")
    print(f"NaN mismatches  : {nan_mismatch}")
    print(f"Value mismatches: {value_mismatch}")

    if nan_mismatch or value_mismatch:
        raise ValueError("V4 label audit failed.")

    print("\n✅ PASS: labels reconstructed exactly")


if __name__ == "__main__":

    if len(sys.argv) == 2 and sys.argv[1].lower() == "audit":
        audit_historical_labels()

    else:
        print("Usage: python src/labeling.py audit")