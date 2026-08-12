import os
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=True)

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"

for folder in (ARTIFACTS_DIR, OUTPUTS_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATABASE + API
# ============================================================

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE", "")
PG_USER = os.getenv("PG_USER", "")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "")


# ============================================================
# SOURCE DATA
# ============================================================

SOURCE_DATA_PATH = Path(
    os.getenv(
        "SOURCE_DATA_PATH",
        "E:/stock_decoupling_source/stock_decoupling",
    )
)

PROCESSED_DATA_PATH = SOURCE_DATA_PATH / "data" / "processed"

HISTORICAL_DATA_FILE = (
    PROCESSED_DATA_PATH
    / "label_v4_stable_regime_experiment.parquet"
)

PREPROCESSING_BUNDLE_FILE = (
    SOURCE_DATA_PATH
    / "final_live_preprocessing_bundle.joblib"
)


# ============================================================
# STOCK UNIVERSE
# ============================================================

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "TSM", "AVGO", "TXN", "SWKS", "AMKR",
    "QQQ", "SPY",
]

TECH_TICKERS = ["MSFT", "NVDA", "GOOGL", "META"]
SUPPLY_TICKERS = ["TSM", "AVGO", "TXN", "SWKS", "AMKR"]

TIMEZONE = "America/New_York"
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"


# ============================================================
# FROZEN TARGET SETTINGS
# ============================================================

TARGET_COLUMN = "Target_V4_Run8_Margin1"

ENTER_INTENSITY = 0.9171388638811381
EXIT_INTENSITY = 0.718725187843583
ENTER_PERSISTENCE = 0.40
EXIT_PERSISTENCE = 0.20

V4_MIN_RUN = 8
V4_EDGE_MARGIN = 1


# ============================================================
# FINAL XGBOOST
# ============================================================

MODEL_THRESHOLD = 0.43

MODEL_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.025,
    "max_depth": 3,
    "min_child_weight": 10,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.30,
    "reg_lambda": 6.0,
    "gamma": 0.05,
    "scale_pos_weight": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# FROZEN 91 FEATURES — ORDER MUST NOT CHANGE
# ============================================================

FEATURES_91 = [
    "AAPL_Ret_1m",
    "AAPL_Range_Pct",
    "AAPL_Close_VWAP_Gap",
    "Tech_Group_Ret_1m",
    "Supply_Group_Ret_1m",
    "QQQ_Ret_1m",
    "SPY_Ret_1m",
    "MSFT_Ret_1m",
    "NVDA_Ret_1m",
    "GOOGL_Ret_1m",
    "META_Ret_1m",
    "TSM_Ret_1m",
    "AVGO_Ret_1m",
    "TXN_Ret_1m",
    "SWKS_Ret_1m",
    "AMKR_Ret_1m",
    "AAPL_vs_Tech",
    "AAPL_vs_Supply",
    "AAPL_vs_QQQ",
    "AAPL_vs_SPY",
    "Tech_vs_Supply",
    "Tech_Dispersion",
    "Supply_Dispersion",
    "Expected_AAPL_Ret_1m",
    "AAPL_Residual_1m",
    "Residual_Scale_60m",
    "Normalized_Decoupling_Score",
    "Decoupled_Now_Norm",
]

FEATURES_91 += [
    f"NormScore_Lag_{m}m"
    for m in (1, 2, 3, 5, 10, 15)
]

FEATURES_91 += [
    f"Residual_Lag_{m}m"
    for m in (1, 2, 3, 5, 10, 15)
]

FEATURES_91 += [
    f"State_Lag_{m}m"
    for m in (1, 2, 3, 5, 10, 15)
]

FEATURES_91 += [
    f"TechRet_Lag_{m}m"
    for m in (1, 3, 5, 10)
]

FEATURES_91 += [
    f"SupplyRet_Lag_{m}m"
    for m in (1, 3, 5, 10)
]

FEATURES_91 += [
    "Minutes_Since_Open",
    "Time_Sin",
    "Time_Cos",
    "WindowScore_10m",
    "WindowDecoupled_10m",
]

FEATURES_91 += [
    f"WindowScore_10m_Lag_{m}m"
    for m in (1, 3, 5, 10)
]

FEATURES_91 += [
    f"WindowDecoupled_10m_Lag_{m}m"
    for m in (1, 3, 5, 10)
]

for horizon in (3, 5, 10, 15):
    FEATURES_91 += [
        f"Tech_Group_Ret_{horizon}m",
        f"Supply_Group_Ret_{horizon}m",
        f"AAPL_vs_Tech_{horizon}m",
        f"AAPL_vs_Supply_{horizon}m",
        f"AAPL_vs_QQQ_{horizon}m",
        f"AAPL_vs_SPY_{horizon}m",
    ]


# ============================================================
# MODEL / OUTPUT FILES
# ============================================================

HISTORICAL_TEST_MODEL_FILE = (
    ARTIFACTS_DIR / "v4_historical_test_model.joblib"
)

DEPLOYMENT_MODEL_FILE = (
    ARTIFACTS_DIR / "v4_deployment_model.joblib"
)

TEST_PREDICTIONS_FILE = (
    OUTPUTS_DIR / "v4_test_predictions.parquet"
)

TEST_METRICS_FILE = (
    OUTPUTS_DIR / "v4_test_metrics.json"
)


# ============================================================
# QUICK VALIDATION
# ============================================================

def validate_config():
    missing_db = [
        name
        for name, value in {
            "PG_DATABASE": PG_DATABASE,
            "PG_USER": PG_USER,
            "PG_PASSWORD": PG_PASSWORD,
        }.items()
        if not value
    ]

    if missing_db:
        raise ValueError(
            f"Missing database settings: {', '.join(missing_db)}"
        )

    if len(TICKERS) != 12:
        raise ValueError("Expected exactly 12 tickers.")

    if len(FEATURES_91) != 91:
        raise ValueError(
            f"Expected 91 features, found {len(FEATURES_91)}."
        )

    if len(set(FEATURES_91)) != 91:
        raise ValueError("Duplicate feature names found.")

    print("✅ Config OK")
    print(f"✅ Tickers: {len(TICKERS)}")
    print(f"✅ Features: {len(FEATURES_91)}")
    print(f"✅ Target: {TARGET_COLUMN}")
    print(f"✅ Threshold: {MODEL_THRESHOLD}")


if __name__ == "__main__":
    validate_config()