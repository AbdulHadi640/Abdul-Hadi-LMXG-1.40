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
import ast

warnings.filterwarnings('ignore')

def main():
    print("Starting End-to-End Machine Learning Pipeline...")
    
    # ---------------------------------------------------------
    # PART 1: DATA EXTRACTION (FROM DB)
    # ---------------------------------------------------------
    print("\n[PART 1] Extracting enriched features and embeddings from database...")
    df_enriched = db.fetch_enriched_features()
    
    if df_enriched.empty:
        raise ValueError("Query returned 0 rows. Check the database tables.")
    
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
    
    # ---------------------------------------------------------
    # DYNAMIC EMBEDDING LOGIC
    # ---------------------------------------------------------
    emb_cols = [col for col in X_raw.columns if str(col).startswith('emb_')]
    
    if 'embedding' in X_raw.columns:
        print("Found single 'embedding' vector column. Unpacking...")
        if isinstance(X_raw['embedding'].iloc[0], str):
            X_raw['embedding'] = X_raw['embedding'].apply(ast.literal_eval)
        
        emb_df_temp = pd.DataFrame(X_raw['embedding'].tolist(), index=X_raw.index)
        emb_cols = [f'emb_{i}' for i in range(emb_df_temp.shape[1])]
        emb_df_temp.columns = emb_cols
        X_raw = pd.concat([X_raw.drop(columns=['embedding']), emb_df_temp], axis=1)
        
    elif len(emb_cols) == 0:
        print("No embeddings found in database. Generating on the fly from engineered features...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Combine all engineered features into a single text string per row
        def combine_features(row):
            return " | ".join([f"{col}: {val}" for col, val in row.items()])
            
        combined_text = X_raw.apply(combine_features, axis=1)
        
        print("Encoding combined features...")
        embeddings = model.encode(combined_text.tolist(), show_progress_bar=True)
        emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
        emb_df_temp = pd.DataFrame(embeddings, columns=emb_cols, index=X_raw.index)
        
        # Save this back to the database!
        print("Saving newly generated embeddings to the database...")
        df_to_save = pd.concat([df_enriched, emb_df_temp], axis=1)
        db.save_features(df_to_save, table_name="article_features_with_embeddings", if_exists="replace")
        
        X_raw = pd.concat([X_raw, emb_df_temp], axis=1)
    else:
        print("Found pre-computed 'emb_' columns in the database.")
    
    # Encode categorical tabular features
    print("Encoding categorical tabular features...")
    # Identify embedding columns vs tabular columns
    tabular_cols = [col for col in X_raw.columns if not str(col).startswith('emb_')]
    
    X_tabular = X_raw[tabular_cols]
    X_encoded = pd.get_dummies(X_tabular, drop_first=True, dtype=float)
    
    # Ensure tabular features are strictly numeric floats
    X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    print(f"Encoded tabular feature matrix shape: {X_encoded.shape}")

    # ---------------------------------------------------------
    # PART 3: MULTICOLLINEARITY (Correlation & VIF) - TABULAR ONLY
    # ---------------------------------------------------------
    print("\n[PART 3] Multicollinearity Checking (Tabular Features Only)...")
    print("3a. Calculating correlation matrix (Dropping > 0.95)...")
    corr_matrix = X_encoded.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    X_encoded_filtered = X_encoded.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated tabular features.")
    print(f"Tabular features remaining: {X_encoded_filtered.shape[1]}")

    print("\n3b. Calculating Variance Inflation Factor (VIF)...")
    try:
        X_vif = add_constant(X_encoded_filtered)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_encoded_filtered.columns
        vif_vals = [variance_inflation_factor(X_vif.values, i) for i in range(1, len(X_vif.columns))]
        vif_data["VIF"] = vif_vals
        
        high_vif = vif_data[vif_data['VIF'] > 10]
        to_drop_vif = high_vif['feature'].tolist()
        X_encoded_filtered = X_encoded_filtered.drop(columns=to_drop_vif, errors='ignore')
        print(f"Found and dropped {len(high_vif)} tabular features with VIF > 10.")
    except Exception as e:
        print(f"VIF calculation encountered an error: {e}")

    # Combine filtered tabular features with untouched embeddings
    emb_df = X_raw[emb_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_filtered = pd.concat([X_encoded_filtered, emb_df], axis=1)
    print(f"Combined final feature matrix shape (Tabular + Embeddings): {X_filtered.shape}")

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
