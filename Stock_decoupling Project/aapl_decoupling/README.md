# AAPL Decoupling Prediction System

Machine-learning system for predicting whether **Apple (AAPL)** is moving normally with its market ecosystem or entering a **future stable decoupling regime**.

The project uses 1-minute market data from AAPL, technology companies, Apple-related supply-chain stocks, and broad-market ETFs to generate a frozen set of **91 features** and classify each valid timestamp as:

- `0` → **ALIGNED**
- `1` → **DECOUPLED**

> This project predicts market **decoupling behavior**, not stock-price direction.

---

## Project Overview

AAPL normally moves with broader technology, supply-chain, and market signals.

This system estimates whether AAPL's movement is consistent with those signals or unusually different from them.

### High-Level Flow

```text
Massive API
    ↓
1-Minute Market Data
    ↓
PostgreSQL
    ↓
Exact 12-Instrument Alignment
    ↓
91 Feature Engineering
    ↓
XGBoost Classifier
    ↓
Probability of Decoupling
    ↓
Threshold = 0.43
    ↓
ALIGNED / DECOUPLED
    ↓
Predictions + Evaluation Results
```

---

## Instruments Used

The project uses **12 instruments**.

### Target

```text
AAPL
```

### Technology Group

```text
MSFT
NVDA
GOOGL
META
```

### Supply-Chain Group

```text
TSM
AVGO
TXN
SWKS
AMKR
```

### Market Benchmarks

```text
QQQ
SPY
```

Only timestamps available across **all 12 instruments** are retained.

---

## Model

Final classifier:

```text
XGBoost Classifier
```

Model version:

```text
v4_base91_xgb_depth3_threshold043
```

Decision rule:

```text
Probability >= 0.43  → DECOUPLED
Probability <  0.43  → ALIGNED
```

The model uses exactly:

```text
91 frozen features
```

---

## Feature Engineering

The feature set contains signals such as:

- 1-minute stock returns
- AAPL range percentage
- AAPL VWAP gap
- Technology-group returns
- Supply-chain-group returns
- AAPL vs Technology
- AAPL vs Supply Chain
- AAPL vs QQQ
- AAPL vs SPY
- Group dispersion
- Expected AAPL return
- AAPL residual
- Residual volatility
- Normalized decoupling score
- Decoupling persistence
- Exact-clock lag features
- Time-of-day features
- Rolling 10-minute features
- 3-minute relationships
- 5-minute relationships
- 10-minute relationships
- 15-minute relationships

The project uses **exact clock-time lookups**, not blind row shifting.

For example:

```text
10:15 with 5-minute lag
→ looks specifically for 10:10
```

---

## Ground Truth

The final target is:

```text
Target_V4_Run8_Margin1
```

The V4 labeling system uses:

- future decoupling intensity
- future persistence
- hysteresis
- stable contiguous runs
- minimum run length
- edge-margin filtering

Frozen settings:

```text
Minimum Run Length = 8
Edge Margin        = 1
```

Some uncertain transition rows intentionally remain unlabeled.

---

## Project Structure

```text
aapl_decoupling/
│
├── main.py
├── .env
│
├── src/
│   ├── config.py
│   ├── database.py
│   ├── data.py
│   ├── features.py
│   ├── labeling.py
│   ├── model.py
│   └── pipeline.py
│
├── artifacts/
│   ├── v4_historical_test_model.joblib
│   └── v4_deployment_model.joblib
│
├── data/
├── outputs/
└── logs/
```

### File Responsibilities

| File | Purpose |
|---|---|
| `main.py` | Main CLI controller |
| `config.py` | Frozen project configuration |
| `database.py` | PostgreSQL setup and connections |
| `data.py` | API download, storage, and exact alignment |
| `features.py` | 91-feature reconstruction |
| `labeling.py` | V4 ground-truth generation |
| `model.py` | Training and evaluation |
| `pipeline.py` | Deployment prediction pipeline |

---

# Setup

## 1. Create Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

---

## 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost psycopg[binary] sqlalchemy joblib requests pyarrow python-dotenv
```

---

## 3. Configure `.env`

Create a `.env` file in the project root:

```env
MASSIVE_API_KEY=YOUR_API_KEY

PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=stock_decoupling
PG_USER=postgres
PG_PASSWORD=YOUR_PASSWORD
```

Never commit `.env` to GitHub.

Recommended `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
*.log
```

---

# Database Setup

Initialize PostgreSQL:

```bash
python main.py init-db
```

Main database tables:

```text
market_bars
historical_dataset
predictions
model_results
```

---

# Running the Project

## 1. Download Market Data

```bash
python main.py fetch YYYY-MM-DD
```

Example:

```bash
python main.py fetch 2026-08-10
```

The downloader fetches all 12 instruments and stores them in PostgreSQL.

A request delay is used to avoid API rate-limit errors.

---

## 2. Check Downloaded Data

```bash
python main.py summary YYYY-MM-DD
```

Example:

```bash
python main.py summary 2026-08-10
```

---

## 3. Check Exact Alignment

```bash
python main.py align YYYY-MM-DD
```

Example:

```bash
python main.py align 2026-08-10
```

Verified example:

```text
Exact aligned rows: 388
Columns: 86
```

---

## 4. Generate Predictions

```bash
python main.py predict YYYY-MM-DD
```

Example:

```bash
python main.py predict 2026-08-10
```

Verified output:

```text
Model       : v4_base91_xgb_depth3_threshold043
Rows        : 322
Threshold   : 0.43
Latest      : 15:49 → DECOUPLED
Probability : 0.838842

ALIGNED     : 247
DECOUPLED   : 75
```

Predictions are saved to:

```text
outputs/predictions_YYYY-MM-DD.parquet
```

and PostgreSQL.

---

# Regression Audits

## Feature Audit

```bash
python main.py features-audit
```

Expected:

```text
Passed features : 91/91
Failed features : 0/91
```

---

## Label Audit

```bash
python main.py labels-audit
```

Expected:

```text
NaN mismatches   : 0
Value mismatches : 0
```

These commands should be run after modifying feature or labeling logic.

---

# Historical Evaluation

## Frozen Historical TEST

```bash
python main.py test
```

The model is trained on:

```text
TRAIN + VAL
```

and evaluated on the untouched:

```text
TEST
```

### Verified TEST Result

| Metric | Result |
|---|---:|
| Accuracy | **82.3638%** |
| Majority Baseline | 79.0606% |
| Accuracy Gain | +3.3032 pp |
| Balanced Accuracy | 64.4292% |
| ALIGNED Recall | 95.2864% |
| DECOUPLED Precision | 65.3547% |
| DECOUPLED Recall | 33.5720% |
| DECOUPLED F1 | 0.4436 |
| Macro F1 | 0.6694 |
| ROC-AUC | 0.8204 |
| PR-AUC | 0.5676 |

Confusion matrix:

```text
[[17082, 845],
 [3154, 1594]]
```

The TEST set must not be used for model or threshold tuning.

---
# High-Confidence Prediction Mode

In addition to the standard classification threshold, the project was also evaluated using a **selective high-confidence prediction strategy**.

The idea is simple:

```text
Higher confidence requirement
        ↓
Fewer predictions accepted
        ↓
More uncertain predictions rejected
        ↓
Higher accuracy on the selected subset
```

Instead of forcing the model to classify every available row, predictions with uncertain probabilities can be excluded.

This creates a trade-off between:

```text
Accuracy ↔ Coverage
```

## Confidence Threshold Analysis

Validation experiments showed the following behavior:

| Confidence Cutoff | Accuracy | Coverage |
|---|---:|---:|
| 0.75 | 90.35% | 27.33% |
| 0.80 | 92.60% | 20.73% |
| 0.85 | 94.40% | 15.73% |
| **0.90** | **96.39%** | **11.43%** |

At the strongest confidence level:

```text
Confidence Cutoff = 0.90
Accuracy          = 96.39%
Coverage          = 11.43%
```

Therefore, the system demonstrated that **95%+ accuracy is achievable on highly confident predictions**, although only a smaller subset of predictions is accepted.

### Important Interpretation

This does **not** mean that the model has 96% accuracy across the complete dataset.

The primary full-coverage historical model result remains:

```text
Historical TEST Accuracy = 82.3638%
```

High-confidence filtering is a **selective prediction mode** where the model is allowed to avoid uncertain cases.

The trade-off is:

```text
Lower confidence threshold
→ More predictions
→ Higher coverage
→ Lower accuracy

Higher confidence threshold
→ Fewer predictions
→ Lower coverage
→ Higher accuracy
```

This experiment demonstrates that model confidence contains useful information and can be used when prediction quality is more important than prediction coverage.
# TRAIN / VAL / TEST Evaluation

Run:

```bash
python main.py history-eval
```

Current results:

| Split | Accuracy | Balanced Accuracy | F1 Decoupled | ROC-AUC |
|---|---:|---:|---:|---:|
| TRAIN Diagnostic | 86.1809% | 76.2298% | 0.6440 | 0.8922 |
| VAL | 82.2626% | 67.8980% | 0.5082 | 0.8262 |
| TEST | **82.3638%** | 64.4292% | 0.4436 | 0.8204 |

> TRAIN accuracy is an in-sample diagnostic and should not be presented as final model accuracy.

The main historical generalization result is the **untouched TEST accuracy**.

---

# Live / Fresh Forward Evaluation

After predictions have already been saved for a complete trading day:

```bash
python main.py live-eval YYYY-MM-DD
```

Example:

```bash
python main.py live-eval 2026-08-10
```

### Verified Fresh Result — 2026-08-10

| Metric | Result |
|---|---:|
| Saved Predictions | 322 |
| Evaluable Rows | 308 |
| Effective Coverage | 95.65% |
| Accuracy | **90.2597%** |
| Majority Baseline | 74.3506% |
| Accuracy Gain | +15.9091 pp |
| Balanced Accuracy | 85.9875% |
| ALIGNED Recall | 94.7598% |
| DECOUPLED Precision | 83.5616% |
| DECOUPLED Recall | 77.2152% |
| DECOUPLED F1 | 0.8026 |
| ROC-AUC | 0.9547 |
| PR-AUC | 0.8971 |

Confusion matrix:

```text
[[217, 12],
 [18, 61]]
```

Fresh forward results are supplemental evidence and should not replace the historical TEST result.

---

# Recommended Daily Workflow

For a new complete trading day:

```bash
python main.py fetch YYYY-MM-DD
python main.py summary YYYY-MM-DD
python main.py predict YYYY-MM-DD
python main.py live-eval YYYY-MM-DD
```

Flow:

```text
FETCH
  ↓
SUMMARY
  ↓
PREDICT
  ↓
LIVE-EVAL
```

`live-eval` should only be run after enough future/full-day data exists because V4 ground truth uses future information.

---

# Main Commands

| Command | Purpose |
|---|---|
| `python main.py init-db` | Initialize PostgreSQL |
| `python main.py fetch DATE` | Download 12 instruments |
| `python main.py summary DATE` | Show stored data |
| `python main.py align DATE` | Check exact alignment |
| `python main.py predict DATE` | Generate predictions |
| `python main.py features-audit` | Verify 91 features |
| `python main.py labels-audit` | Verify V4 labels |
| `python main.py test` | Frozen historical TEST |
| `python main.py history-eval` | TRAIN / VAL / TEST evaluation |
| `python main.py live-eval DATE` | Fresh forward evaluation |
| `python main.py deploy` | Retrain deployment model |

---

# Output Files

Important generated files:

```text
outputs/
│
├── predictions_YYYY-MM-DD.parquet
│
├── historical_train_predictions.parquet
├── historical_val_predictions.parquet
├── historical_test_predictions.parquet
│
├── historical_evaluation_summary.csv
├── historical_evaluation_summary.json
│
├── live_evaluation_YYYY-MM-DD.parquet
├── live_metrics_YYYY-MM-DD.json
│
└── model_accuracy_summary.csv
```

The main evaluation summary is:

```text
outputs/model_accuracy_summary.csv
```

It can be opened directly in Excel.

---

# Important Notes

### Why are aligned rows greater than prediction rows?

Some features require historical context, including:

- lagged returns
- 60-minute residual volatility
- rolling windows

Therefore early-session rows do not yet contain all 91 usable features.

---

### Why are live evaluation rows fewer than predictions?

The V4 labeling system intentionally leaves uncertain transition rows unlabeled.

Example:

```text
Predictions = 322
Evaluated   = 308
```

This is expected.

---

### API Error `429`

Means:

```text
Too Many Requests
```

The downloader already waits between ticker requests and retries when necessary.

---

### API Error `403`

Means the API provider denied access to the requested data.

It may be related to:

- data availability,
- account entitlement,
- requested data recency.

---

# Deployment Warning

Running:

```bash
python main.py deploy
```

re-trains and overwrites the deployment model.

Do not run this command during normal daily prediction unless you intentionally want to rebuild the deployment model.

---

# Tech Stack

```text
Python
Pandas
NumPy
scikit-learn
XGBoost
PostgreSQL
Psycopg
SQLAlchemy
Requests
Joblib
Parquet / PyArrow
Massive API
```

---

# Project Status

```text
Database                ✅
API Data Collection     ✅
Rate-Limit Handling     ✅
Exact 12-Way Alignment  ✅
91 Feature Engineering  ✅
Feature Audit            ✅
V4 Ground Truth          ✅
Label Audit              ✅
Historical Evaluation    ✅
Deployment Prediction    ✅
Live Evaluation          ✅
CSV / JSON Saving        ✅
PostgreSQL Saving        ✅
```

---

## Project Summary

**Project:** Machine Learning-Based AAPL Decoupling Prediction Using Supply-Chain and Market Signals  
**Target:** Future Stable Sustained AAPL Decoupling Regime  
**Model:** XGBoost Classifier  
**Features:** 91  
**Decision Threshold:** 0.43  
**Data Frequency:** 1 Minute  
**Timezone:** America/New_York  
**Historical TEST Accuracy:** 82.3638%  
**Fresh Forward Accuracy (2026-08-10):** 90.2597%

---

## Disclaimer

This project is developed for **academic and machine-learning research purposes**. It is not intended to provide financial or investment advice.