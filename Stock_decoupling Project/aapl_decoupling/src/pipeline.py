import sys

import joblib
import numpy as np
import pandas as pd

try:
    from .config import (
        FEATURES_91,
        DEPLOYMENT_MODEL_FILE,
        OUTPUTS_DIR,
    )
    from .database import get_connection
    from .data import build_aligned_master
    from .features import (
        build_features,
        get_model_ready_frame,
    )

except ImportError:
    from config import (
        FEATURES_91,
        DEPLOYMENT_MODEL_FILE,
        OUTPUTS_DIR,
    )
    from database import get_connection
    from data import build_aligned_master
    from features import (
        build_features,
        get_model_ready_frame,
    )


def load_model():
    """Load and validate frozen deployment model."""

    if not DEPLOYMENT_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {DEPLOYMENT_MODEL_FILE}"
        )

    bundle = joblib.load(DEPLOYMENT_MODEL_FILE)

    required = {
        "model",
        "feature_names",
        "threshold",
        "model_version",
    }

    missing = required - set(bundle)

    if missing:
        raise ValueError(
            f"Model bundle missing: {sorted(missing)}"
        )

    if list(bundle["feature_names"]) != FEATURES_91:
        raise ValueError(
            "Model feature order does not match FEATURES_91."
        )

    return (
        bundle["model"],
        float(bundle["threshold"]),
        str(bundle["model_version"]),
    )


def save_predictions(df, model_version):
    """Insert/update predictions in PostgreSQL."""

    query = """
        INSERT INTO predictions (
            datetime_ny,
            trading_date,
            model_version,
            probability_decoupled,
            prediction,
            actual_label
        )
        VALUES (%s, %s, %s, %s, %s, NULL)

        ON CONFLICT (datetime_ny, model_version)
        DO UPDATE SET
            trading_date = EXCLUDED.trading_date,
            probability_decoupled =
                EXCLUDED.probability_decoupled,
            prediction = EXCLUDED.prediction,
            created_at = CURRENT_TIMESTAMP;
    """

    records = [
        (
            pd.Timestamp(row.Datetime).to_pydatetime(),
            row.TradingDate,
            model_version,
            float(row.Probability_Decoupled),
            int(row.Prediction),
        )
        for row in df.itertuples(index=False)
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, records)

    return len(records)


def run_prediction_pipeline(date):
    """Run frozen V4 prediction pipeline for one day."""

    print(f"\nAAPL DECOUPLING PREDICTION — {date}")

    model, threshold, version = load_model()

    master = build_aligned_master(date)

    features = build_features(
        master,
        verbose=False,
    )

    ready = get_model_ready_frame(features)

    # Future t+10 target must remain inside market session.
    cutoff = pd.Timestamp("15:49").time()

    ready = (
        ready[
            ready["Datetime"].dt.time <= cutoff
        ]
        .copy()
        .reset_index(drop=True)
    )

    if ready.empty:
        raise ValueError("No prediction-ready rows.")

    probabilities = model.predict_proba(
        ready[FEATURES_91]
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = pd.DataFrame({
        "Datetime": ready["Datetime"],
        "TradingDate": ready["TradingDate"],
        "Probability_Decoupled": probabilities,
        "Prediction": predictions,
    })

    result["Prediction_Label"] = np.where(
        result["Prediction"] == 1,
        "DECOUPLED",
        "ALIGNED",
    )

    saved = save_predictions(
        result,
        version,
    )

    output_file = (
        OUTPUTS_DIR
        / f"predictions_{date}.parquet"
    )

    result.to_parquet(
        output_file,
        index=False,
    )

    latest = result.iloc[-1]

    print(f"✅ Model      : {version}")
    print(f"✅ Rows       : {len(result):,}")
    print(f"✅ Threshold  : {threshold}")
    print(
        f"✅ Latest     : {latest['Datetime']} "
        f"→ {latest['Prediction_Label']}"
    )
    print(
        f"✅ Probability: "
        f"{latest['Probability_Decoupled']:.6f}"
    )
    print(
        f"✅ ALIGNED    : "
        f"{(result['Prediction'] == 0).sum():,}"
    )
    print(
        f"✅ DECOUPLED  : "
        f"{(result['Prediction'] == 1).sum():,}"
    )
    print(f"✅ DB saved   : {saved:,}")
    print(f"✅ File       : {output_file}")

    return result


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python src/pipeline.py YYYY-MM-DD"
        )
        raise SystemExit(1)

    run_prediction_pipeline(sys.argv[1])