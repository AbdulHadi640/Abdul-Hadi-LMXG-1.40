import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

try:
    from .config import (
        FEATURES_91, TARGET_COLUMN, MODEL_THRESHOLD, MODEL_PARAMS,
        HISTORICAL_TEST_MODEL_FILE, DEPLOYMENT_MODEL_FILE,
        TEST_PREDICTIONS_FILE, TEST_METRICS_FILE,
    )
    from .database import get_connection
    from .data import build_aligned_master
    from .features import build_features
    from .labeling import build_v4_target
except ImportError:
    from config import (
        FEATURES_91, TARGET_COLUMN, MODEL_THRESHOLD, MODEL_PARAMS,
        HISTORICAL_TEST_MODEL_FILE, DEPLOYMENT_MODEL_FILE,
        TEST_PREDICTIONS_FILE, TEST_METRICS_FILE,
    )
    from database import get_connection
    from data import build_aligned_master
    from features import build_features
    from labeling import build_v4_target

MODEL_VERSION = "v4_base91_xgb_depth3_threshold043"
OUTPUTS_DIR = Path(TEST_METRICS_FILE).parent
SUMMARY_FILE = OUTPUTS_DIR / "model_accuracy_summary.csv"

# DATA

def get_split_summary():
    sql = """
        SELECT split, COUNT(*), COUNT(target)
        FROM historical_dataset
        WHERE split IN ('TRAIN','VAL','TEST')
        GROUP BY split;
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return {
        split: {"total": int(total), "labeled": int(labeled)}
        for split, total, labeled in rows
    }


def load_historical_data():
    sql = """
        SELECT datetime_ny, trading_date, split, target, features
        FROM historical_dataset
        WHERE split IN ('TRAIN','VAL','TEST')
          AND target IS NOT NULL
        ORDER BY datetime_ny;
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    if not rows:
        raise ValueError("No historical labeled data found.")

    df = pd.DataFrame(
        rows,
        columns=["datetime_ny", "trading_date", "split", "target", "features"],
    )
    feature_df = pd.DataFrame(df["features"].tolist())
    missing = [f for f in FEATURES_91 if f not in feature_df.columns]
    if missing:
        raise ValueError("Missing stored features: " + ", ".join(missing))

    meta = df[["datetime_ny", "trading_date", "split", "target"]].reset_index(drop=True)
    result = pd.concat([meta, feature_df[FEATURES_91].reset_index(drop=True)], axis=1)
    result["target"] = result["target"].astype(int)
    return result


def prepare_usable_rows(df):
    clean = df.dropna(subset=FEATURES_91).copy().reset_index(drop=True)
    if clean.empty:
        raise ValueError("No usable model rows.")
    return clean

# MODEL + METRICS
def create_model():
    return XGBClassifier(**MODEL_PARAMS)


def calculate_metrics(y_true, scores, preds):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = np.asarray(preds, dtype=int)

    acc = accuracy_score(y_true, preds)
    majority = max(np.mean(y_true == 0), np.mean(y_true == 1))
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    m = {
        "rows": int(len(y_true)),
        "accuracy": float(acc),
        "majority_accuracy": float(majority),
        "accuracy_gain": float(acc - majority),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "aligned_recall": float(recall_score(y_true, preds, pos_label=0, zero_division=0)),
        "precision_decoupled": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
        "recall_decoupled": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "f1_decoupled": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, scores))
        m["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        m["roc_auc"] = None
        m["pr_auc"] = None
    return m


def score_frame(model, df):
    scores = model.predict_proba(df[FEATURES_91])[:, 1]
    preds = (scores >= MODEL_THRESHOLD).astype(int)
    metrics = calculate_metrics(df["target"], scores, preds)
    out = pd.DataFrame({
        "datetime_ny": df["datetime_ny"].values,
        "trading_date": df["trading_date"].values,
        "actual_label": df["target"].values,
        "probability_decoupled": scores,
        "prediction": preds,
    })
    return out, metrics


def print_metrics(title, m):
    print(f"\n{title}\n" + "-" * 60)
    print(f"Rows               : {m['rows']:,}")
    print(f"Accuracy           : {m['accuracy']:.4%}")
    print(f"Majority baseline  : {m['majority_accuracy']:.4%}")
    print(f"Gain               : {m['accuracy_gain']:+.4%}")
    print(f"Balanced accuracy  : {m['balanced_accuracy']:.4%}")
    print(f"ALIGNED recall     : {m['aligned_recall']:.4%}")
    print(f"DECOUPLED precision: {m['precision_decoupled']:.4%}")
    print(f"DECOUPLED recall   : {m['recall_decoupled']:.4%}")
    print(f"DECOUPLED F1       : {m['f1_decoupled']:.4f}")
    print(f"Macro F1           : {m['macro_f1']:.4f}")
    if m["roc_auc"] is not None:
        print(f"ROC-AUC            : {m['roc_auc']:.4f}")
        print(f"PR-AUC             : {m['pr_auc']:.4f}")
    print(f"Confusion matrix   : [[{m['tn']}, {m['fp']}], [{m['fn']}, {m['tp']}]]")

# SAVE HELPERS

def save_model_bundle(model, path, splits, data):
    bundle = {
        "model": model,
        "feature_names": FEATURES_91,
        "feature_count": len(FEATURES_91),
        "threshold": MODEL_THRESHOLD,
        "target": TARGET_COLUMN,
        "model_version": MODEL_VERSION,
        "model_params": MODEL_PARAMS,
        "trained_splits": splits,
        "training_rows": int(len(data)),
        "training_start": str(data["datetime_ny"].min()),
        "training_end": str(data["datetime_ny"].max()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_description": "Future stable sustained AAPL decoupling regime",
    }
    joblib.dump(bundle, path)
    print(f"✅ Model saved: {path}")


def save_json(data, path):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_summary_row(name, date, start, end, m, coverage=1.0, model_version=MODEL_VERSION):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "evaluation_set": name,
        "date": str(date),
        "start_date": str(start),
        "end_date": str(end),
        "rows": m["rows"],
        "coverage": float(coverage),
        "accuracy": m["accuracy"],
        "majority_accuracy": m["majority_accuracy"],
        "accuracy_gain": m["accuracy_gain"],
        "balanced_accuracy": m["balanced_accuracy"],
        "aligned_recall": m["aligned_recall"],
        "precision_decoupled": m["precision_decoupled"],
        "recall_decoupled": m["recall_decoupled"],
        "f1_decoupled": m["f1_decoupled"],
        "macro_f1": m["macro_f1"],
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        "model_version": model_version,
    }
    new = pd.DataFrame([row])

    if SUMMARY_FILE.exists():
        old = pd.read_csv(SUMMARY_FILE)
        keep = ~(
            old["evaluation_set"].astype(str).eq(str(name))
            & old["date"].astype(str).eq(str(date))
        )
        new = pd.concat([old.loc[keep], new], ignore_index=True)

    new.to_csv(SUMMARY_FILE, index=False)


def save_model_result(name, start, end, m, coverage, model_version=MODEL_VERSION):
    sql = """
        INSERT INTO model_results (
            model_version, evaluation_set, start_date, end_date,
            n_rows, coverage, accuracy, majority_accuracy,
            balanced_accuracy, roc_auc, pr_auc,
            precision_decoupled, recall_decoupled,
            f1_decoupled, metrics_json
        ) VALUES (
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
        );
    """
    values = (
        model_version, name, start, end, m["rows"], float(coverage),
        m["accuracy"], m["majority_accuracy"], m["balanced_accuracy"],
        m["roc_auc"], m["pr_auc"], m["precision_decoupled"],
        m["recall_decoupled"], m["f1_decoupled"], Jsonb(m),
    )

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM model_results WHERE model_version=%s AND evaluation_set=%s;",
            (model_version, name),
        )
        cur.execute(sql, values)


# ============================================================
# FROZEN HISTORICAL TEST
# ============================================================

def run_frozen_test():
    print("\nFROZEN V4 HISTORICAL TEST")

    summary = get_split_summary()
    data = prepare_usable_rows(load_historical_data())
    train_val = data[data["split"].isin(["TRAIN", "VAL"])].copy()
    test = data[data["split"] == "TEST"].copy()

    if train_val.empty or test.empty:
        raise ValueError("TRAIN+VAL or TEST is empty.")

    print(f"TRAIN+VAL rows : {len(train_val):,}")
    print(f"TEST rows      : {len(test):,}")
    print(f"Features       : {len(FEATURES_91)}")
    print(f"Threshold      : {MODEL_THRESHOLD}")

    model = create_model()
    model.fit(train_val[FEATURES_91], train_val["target"])
    save_model_bundle(model, HISTORICAL_TEST_MODEL_FILE, ["TRAIN", "VAL"], train_val)

    pred_df, m = score_frame(model, test)
    total = summary["TEST"]["total"]
    labeled = summary["TEST"]["labeled"]
    coverage = len(test) / total
    m.update({
        "test_total_rows": int(total),
        "test_labeled_rows": int(labeled),
        "test_usable_rows": int(len(test)),
        "target_coverage": float(labeled / total),
        "effective_coverage": float(coverage),
        "threshold": float(MODEL_THRESHOLD),
        "model_version": MODEL_VERSION,
    })

    print_metrics("RESULT", m)
    pred_df.to_parquet(TEST_PREDICTIONS_FILE, index=False)
    save_json(m, TEST_METRICS_FILE)
    save_model_result("HISTORICAL_TEST", test["trading_date"].min(), test["trading_date"].max(), m, coverage)
    save_summary_row("TEST", "historical", test["trading_date"].min(), test["trading_date"].max(), m, coverage)

    print("\n✅ Historical TEST complete")
    print("⚠️ Do not tune using TEST results.")
    return m

# TRAIN / VAL / TEST ACCURACY

def evaluate_historical_splits():
    print("\n" + "=" * 70)
    print("HISTORICAL TRAIN / VAL / TEST EVALUATION")
    print("=" * 70)

    data = prepare_usable_rows(load_historical_data())
    train = data[data["split"] == "TRAIN"].copy()
    val = data[data["split"] == "VAL"].copy()
    test = data[data["split"] == "TEST"].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("TRAIN, VAL or TEST split is empty.")

    # TRAIN model -> TRAIN diagnostic and VAL holdout
    dev_model = create_model()
    dev_model.fit(train[FEATURES_91], train["target"])
    train_pred, train_m = score_frame(dev_model, train)
    val_pred, val_m = score_frame(dev_model, val)

    # TRAIN+VAL -> untouched TEST
    train_val = pd.concat([train, val], ignore_index=True)
    test_model = create_model()
    test_model.fit(train_val[FEATURES_91], train_val["target"])
    test_pred, test_m = score_frame(test_model, test)

    print_metrics("TRAIN — IN-SAMPLE DIAGNOSTIC", train_m)
    print_metrics("VAL — DEVELOPMENT HOLDOUT", val_m)
    print_metrics("TEST — UNTOUCHED HISTORICAL HOLDOUT", test_m)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    train_pred.to_parquet(OUTPUTS_DIR / "historical_train_predictions.parquet", index=False)
    val_pred.to_parquet(OUTPUTS_DIR / "historical_val_predictions.parquet", index=False)
    test_pred.to_parquet(OUTPUTS_DIR / "historical_test_predictions.parquet", index=False)

    rows = []
    for name, frame, m in [
        ("TRAIN_DIAGNOSTIC", train, train_m),
        ("VAL", val, val_m),
        ("TEST", test, test_m),
    ]:
        rows.append({
            "evaluation_set": name,
            "start_date": str(frame["trading_date"].min()),
            "end_date": str(frame["trading_date"].max()),
            **m,
        })
        save_summary_row(name, "historical", frame["trading_date"].min(), frame["trading_date"].max(), m)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUTS_DIR / "historical_evaluation_summary.csv", index=False)
    save_json(rows, OUTPUTS_DIR / "historical_evaluation_summary.json")

    print("\n✅ Historical evaluation saved")
    print(f"   {OUTPUTS_DIR / 'historical_evaluation_summary.csv'}")
    print(f"   {OUTPUTS_DIR / 'historical_evaluation_summary.json'}")
    print(f"   {SUMMARY_FILE}")
    print("\nNOTE: TRAIN is diagnostic; VAL and TEST are holdout results.")
    return summary

# DEPLOYMENT MODEL

def train_deployment_model():
    data = prepare_usable_rows(load_historical_data())
    deployment = data[data["split"].isin(["TRAIN", "VAL", "TEST"])].copy()

    if deployment.empty:
        raise ValueError("Deployment dataset is empty.")

    print("\nFINAL DEPLOYMENT MODEL")
    print(f"Training rows : {len(deployment):,}")
    print(f"Features      : {len(FEATURES_91)}")

    model = create_model()
    model.fit(deployment[FEATURES_91], deployment["target"])
    save_model_bundle(model, DEPLOYMENT_MODEL_FILE, ["TRAIN", "VAL", "TEST"], deployment)
    print("✅ Deployment model ready")
    return model

# LIVE / FRESH FORWARD ACCURACY

def load_frozen_predictions(date, model_version):
    sql = """
        SELECT datetime_ny, trading_date, probability_decoupled, prediction
        FROM predictions
        WHERE trading_date=%s AND model_version=%s
        ORDER BY datetime_ny;
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (date, model_version))
        rows = cur.fetchall()

    df = pd.DataFrame(
        rows,
        columns=["Datetime", "TradingDate", "Probability_Decoupled", "Prediction"],
    )
    if df.empty:
        raise ValueError(
            f"No saved predictions found for {date}. "
            f"Run: python main.py predict {date}"
        )
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df


def update_prediction_actuals(evaluation, date, model_version):
    records = [
        (int(r.Actual), pd.Timestamp(r.Datetime).to_pydatetime(), model_version)
        for r in evaluation.itertuples(index=False)
    ]
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE predictions SET actual_label=NULL WHERE trading_date=%s AND model_version=%s;",
            (date, model_version),
        )
        cur.executemany(
            "UPDATE predictions SET actual_label=%s WHERE datetime_ny=%s AND model_version=%s;",
            records,
        )


def evaluate_live_day(date):
    """Evaluate SAVED predictions after the full day is available."""

    print("\n" + "=" * 70)
    print("LIVE / FRESH FORWARD V4 EVALUATION")
    print("=" * 70)
    print(f"Trading date: {date}")

    bundle = joblib.load(DEPLOYMENT_MODEL_FILE)
    model_version = str(bundle["model_version"])
    predictions = load_frozen_predictions(date, model_version)
    print(f"✅ Saved predictions: {len(predictions):,}")

    master = build_aligned_master(date)
    features = build_features(master, verbose=False)
    labels = build_v4_target(features, verbose=False)
    actuals = labels[["Datetime", "TradingDate", TARGET_COLUMN]].rename(
        columns={TARGET_COLUMN: "Actual"}
    )

    evaluation = predictions.merge(
        actuals,
        on=["Datetime", "TradingDate"],
        how="left",
        validate="one_to_one",
    )
    prediction_rows = len(evaluation)
    evaluation = evaluation[evaluation["Actual"].notna()].copy().reset_index(drop=True)

    if evaluation.empty:
        raise ValueError(
            "No clear V4 labels overlap saved predictions. "
            "Make sure the full trading day is available."
        )

    evaluation["Actual"] = evaluation["Actual"].astype(int)
    m = calculate_metrics(
        evaluation["Actual"],
        evaluation["Probability_Decoupled"],
        evaluation["Prediction"],
    )
    coverage = len(evaluation) / prediction_rows
    m.update({
        "prediction_rows": int(prediction_rows),
        "clear_evaluable_rows": int(len(evaluation)),
        "effective_coverage": float(coverage),
        "trading_date": str(date),
        "model_version": model_version,
    })

    print_metrics(f"LIVE RESULT — {date}", m)
    print(f"Effective coverage : {coverage:.2%}")

    evaluation["Correct"] = evaluation["Actual"] == evaluation["Prediction"]
    evaluation_file = OUTPUTS_DIR / f"live_evaluation_{date}.parquet"
    metrics_file = OUTPUTS_DIR / f"live_metrics_{date}.json"
    evaluation.to_parquet(evaluation_file, index=False)
    save_json(m, metrics_file)

    update_prediction_actuals(evaluation, date, model_version)
    save_model_result(f"FRESH_FORWARD_{date}", date, date, m, coverage, model_version)
    save_summary_row("LIVE", date, date, date, m, coverage, model_version)

    print("\n✅ Live evaluation saved")
    print(f"   {evaluation_file}")
    print(f"   {metrics_file}")
    print(f"   {SUMMARY_FILE}")
    return evaluation, m

# CLI

def main():
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python src/model.py test\n"
            "  python src/model.py history-eval\n"
            "  python src/model.py live-eval YYYY-MM-DD\n"
            "  python src/model.py deploy"
        )
        raise SystemExit(1)

    command = sys.argv[1].lower()

    if command == "test":
        run_frozen_test()
    elif command == "history-eval":
        evaluate_historical_splits()
    elif command == "live-eval":
        if len(sys.argv) != 3:
            raise ValueError("Date required: python src/model.py live-eval YYYY-MM-DD")
        evaluate_live_day(sys.argv[2])
    elif command == "deploy":
        train_deployment_model()
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
