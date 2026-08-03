# ML Pipeline

Pulls enriched feature data (editorial features + 384-dimensional text embeddings) directly from a Postgres database, evaluates multicollinearity, and benchmarks several regression models on the dataset. It exports the benchmark results and full test set predictions to a single local CSV.

## Project structure

```
news_features_simple/
├── ml_pipeline.py           # entry point — reads from DB, runs preprocessing, PCA, and LazyPredict
├── db.py                    # database connection and load logic
├── .env                     # your database credentials (create this yourself)
└── lazypredict_results.csv  # combined performance metrics and test set predictions
```

## Requirements

```bash
pip install pandas numpy scikit-learn lazypredict matplotlib seaborn statsmodels sqlalchemy psycopg2-binary python-dotenv sentence-transformers torch
```

## Setup

1. Create a `.env` file in this folder with your database credentials:

   ```
   PG_DATABASE=news_scraper
   PG_HOST=localhost
   PG_PORT=5432
   PG_USER=postgres
   PG_PASSWORD=your_password_here
   ```

   Only `PG_PASSWORD` is required — the others default to the values
   shown above if omitted.

2. The pipeline will attempt to load an `article_features_with_embeddings` table. If it doesn't exist, it gracefully falls back to `article_features` and generates the embeddings on the fly.

## Running

Activate your virtual environment and execute the pipeline:

```bash
python ml_pipeline.py
```

This script will run sequentially:
1. **Data Extraction**: Connects to Postgres and tries to load the `article_features_with_embeddings` table (falling back to `article_features` if needed).
2. **Preprocessing**: Automatically parses single vector columns (if any), generates embeddings on the fly (if missing), and encodes categorical features.
3. **Multicollinearity**: Automatically drops tabular features with high correlation (`> 0.95`) and runs a Variance Inflation Factor (VIF) check (skipping embeddings to preserve vector space and prevent singular matrix errors).
4. **PCA**: Splits the data into Train/Val/Test splits and performs Principal Component Analysis keeping 95% of the variance.
5. **Model Benchmarking**: Passes the PCA-reduced test set to `LazyPredict` to evaluate dozens of baseline models simultaneously.

### Understanding the Output

The script outputs to a single file: `lazypredict_results.csv`. This file has two sections:

* **Top Section (The Metrics):** A scoreboard of all tested models, automatically sorted by **R-Squared** in descending order (best models at the top). It includes a custom **Accuracy (%)** metric, which is calculated as `100 - (MAPE * 100)`. Because the target variable (`clicks`) is a continuous number, traditional classification accuracy cannot be used.
* **Bottom Section (The Predictions):** Below the `=== TEST SET PREDICTIONS ===` divider, you will find the exact numeric predictions that every single model made for every row in the hidden test set.


## Important notes

- **`clicks` is placeholder data.** It's filled with random numbers
  (`random.randint(5000, 10000)`) so the pipeline runs end to end.
  It has **no relationship to the actual title content** — don't
  train or evaluate any model against it until you swap in real
  click data. As a result, you should expect models to score a negative R-Squared (performing worse than a Dummy baseline).
- **`trending_topic` scales with dataset size.** The minimum number
  of repeats needed for a word to count as "trending" is calculated
  automatically (`max(3, 0.05% of total rows)`), so it behaves
  sensibly whether you're testing on 10 rows or running on 16,000.