import pandas as pd
import numpy as np
import warnings
import db
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings('ignore')

def main():
    print("Starting End-to-End Machine Learning Pipeline...")
    
    # ---------------------------------------------------------
    # PART 1: DATA EXTRACTION (FROM DB)
    # ---------------------------------------------------------
    print("\n[PART 1] Extracting enriched features and embeddings from database...")
    df_enriched = db.fetch_enriched_features()
    
    if df_enriched.empty:
        raise ValueError("Query returned 0 rows. Check the 'article_features_with_embeddings' table.")
    
    target_col = 'clicks'
    if target_col not in df_enriched.columns:
        raise ValueError(f"Expected target column '{target_col}' not found.")

    # ---------------------------------------------------------
    # PART 2: PREPROCESSING FOR ML
    # ---------------------------------------------------------
    print("\n[PART 2] Preprocessing for Machine Learning...")
    # Separate features and target (ignore id and title)
    cols_to_drop = ['id', 'title', target_col]
    X_raw = df_enriched.drop(columns=[col for col in cols_to_drop if col in df_enriched.columns])
    y = df_enriched[target_col]
    
    # Encode categorical tabular features
    print("Encoding categorical tabular features...")
    # Identify embedding columns vs tabular columns
    emb_cols = [col for col in X_raw.columns if str(col).startswith('emb_')]
    tabular_cols = [col for col in X_raw.columns if col not in emb_cols]
    
    X_tabular = X_raw[tabular_cols]
    X_encoded = pd.get_dummies(X_tabular, drop_first=True)
    
    # Combine encoded tabular features with embeddings
    emb_df = X_raw[emb_cols]
    X_combined = pd.concat([X_encoded, emb_df], axis=1)
    X_combined = X_combined.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"Combined feature matrix shape: {X_combined.shape}")

    # ---------------------------------------------------------
    # PART 3: MULTICOLLINEARITY (Correlation & VIF)
    # ---------------------------------------------------------
    print("\n[PART 3] Multicollinearity Checking...")
    print("3a. Calculating correlation matrix (Dropping > 0.95)...")
    corr_matrix = X_combined.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    X_filtered = X_combined.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated features.")
    print(f"Features remaining: {X_filtered.shape[1]}")

    print("\n3b. Calculating Variance Inflation Factor (VIF)...")
    print("Note: This might take a few moments for high-dimensional data.")
    try:
        X_vif = add_constant(X_filtered)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_filtered.columns
        vif_vals = [variance_inflation_factor(X_vif.values, i) for i in range(1, len(X_vif.columns))]
        vif_data["VIF"] = vif_vals
        
        high_vif = vif_data[vif_data['VIF'] > 10]
        print(f"Found {len(high_vif)} features with VIF > 10 (High Multicollinearity).")
    except Exception as e:
        print(f"VIF calculation encountered an error (likely singular matrix): {e}")

    # ---------------------------------------------------------
    # PART 4: DATA SPLIT, SCALING, & PCA
    # ---------------------------------------------------------
    print("\n[PART 4] Data Splitting, Scaling, and PCA...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_filtered, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"PCA reduced feature dimensions from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]} while retaining 95% variance.")

    # ---------------------------------------------------------
    # PART 5: MODEL BENCHMARKING (LAZYPREDICT)
    # ---------------------------------------------------------
    print("\n[PART 5] Benchmarking models with LazyPredict on Test Set...")
    
    # Custom metric to satisfy the "Accuracy" requirement for numerical targets
    # We use MAPE (Mean Absolute Percentage Error) translated into an accuracy percentage
    def accuracy_metric(y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        # Cap at 0% minimum accuracy
        return max(0.0, 100.0 - (mape * 100))
        
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=accuracy_metric, predictions=True)
    
    models, predictions = reg.fit(X_train_pca, X_test_pca, y_train, y_test)
    
    # Rename the custom metric column to "Accuracy (%)"
    if "accuracy_metric" in models.columns:
        models.rename(columns={"accuracy_metric": "Accuracy (%)"}, inplace=True)
        
    # Sort models so the best (highest R-Squared) are at the top
    if "R-Squared" in models.columns:
        models.sort_values(by="R-Squared", ascending=False, inplace=True)
    
    print("\n--- Top 10 LazyPredict Results on Test Data ---")
    print(models.head(10))
    
    models.to_csv("lazypredict_results.csv")
    with open("lazypredict_results.csv", "a", encoding="utf-8", newline="") as f:
        f.write("\n=== TEST SET PREDICTIONS ===\n")
        predictions.to_csv(f)
        
    print("\nPipeline complete! Results and predictions saved to a single 'lazypredict_results.csv'.")

if __name__ == "__main__":
    main()
