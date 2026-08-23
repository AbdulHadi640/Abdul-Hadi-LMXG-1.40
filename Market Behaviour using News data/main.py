"""
Main Pipeline Execution (WITH AUTO-SAVE CHECKPOINT)
====================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import config
import os

from data_loader import DataLoader
from data_cleaner import DataCleaner
from market_session import MarketSessionClassifier
from feature_engineer import FeatureEngineer
from spy_aligner import SPYAligner
from clusterer import MarketImpactClusterer
from models import ModelTrainer
from visualizer import plot_confusion_matrix, plot_model_comparison

def run_pipeline():
    """Execute complete pipeline with Checkpointing"""
    
    # Checkpoint paths
    checkpoint_df = config.OUTPUT_DIR / "df_step6.pkl"
    checkpoint_feat = config.OUTPUT_DIR / "features_step6.npy"
    
    # CHECK IF CHECKPOINT EXISTS
    if checkpoint_df.exists() and checkpoint_feat.exists():
        print("\n" + "="*80)
        print("✅ CHECKPOINT FOUND! Loading saved data (Skipping Steps 1-6 in 2 seconds...)")
        print("="*80)
        df = pd.read_pickle(checkpoint_df)
        features = np.load(checkpoint_feat)
        print(f"✓ Loaded DataFrame: {len(df)} rows")
        print(f"✓ Loaded Features shape: {features.shape}")
        print("Resuming from Step 7...")
        
    else:
        print("\n" + "="*80)
        print("FINANCIAL NEWS MARKET IMPACT PREDICTION PIPELINE (Fresh Run)")
        print("="*80)
        
        # PHASE 1: Load Data
        loader = DataLoader()
        df = loader.load_all_csvs()
        df = loader.convert_to_ny_time(df)
        
        # PHASE 2: Clean Data
        cleaner = DataCleaner()
        df = cleaner.remove_duplicates(df)
        
        # PHASE 3: Market Session Classification
        session_cls = MarketSessionClassifier()
        df = session_cls.classify_all(df)
        
        # PHASE 4: Feature Engineering
        fe = FeatureEngineer()
        df = fe.extract_all_features(df)
        
        # PHASE 5: Create Embeddings
        features = fe.create_embeddings(df)
        
        # 💾 AUTO-SAVE CHECKPOINT HERE 💾
        print("\n" + "="*60)
        print("💾 SAVING CHECKPOINT TO DISK...")
        print("="*60)
        df.to_pickle(checkpoint_df)
        np.save(checkpoint_feat, features)
        print(f"✅ SUCCESS! Saved to {checkpoint_df}")
        print(f"✅ SUCCESS! Saved to {checkpoint_feat}")
        print("🚀 If the pipeline crashes now, it will load this in 2 seconds next time!")
        print("="*60)

    # PHASE 6: SPY Data Alignment (Will run fresh or after checkpoint)
    spy = SPYAligner(config.SPY_DATA_PATH)
    if spy.spy_data is None:
        spy.create_synthetic_spy(df)
    df = spy.align_and_calculate_returns(df)
    
    # PHASE 7: Market Impact Clustering
    clusterer = MarketImpactClusterer()
    df, kmeans = clusterer.cluster(df)
    
    # PHASE 8: Model Training
    labels = df['cluster_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels
    )
    
    trainer = ModelTrainer()
    
    # Train Ensemble
    acc_ensemble, y_pred_ensemble = trainer.train_ensemble(
        X_train, X_test, y_train, y_test
    )
    
    # Train LSTM
    acc_lstm = trainer.train_lstm(
        X_train, X_test, y_train, y_test
    )
    
    # PHASE 9: Visualization
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    plot_confusion_matrix(y_test, y_pred_ensemble, 'Ensemble Model')
    
    accuracies = {
        'Ensemble': acc_ensemble,
        'LSTM': acc_lstm
    }
    plot_model_comparison(accuracies)
    
    # Final Results
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✓ Ensemble Accuracy: {acc_ensemble:.4f}")
    print(f"✓ LSTM Accuracy: {acc_lstm:.4f}")
    print(f"✓ Best Model: {'Ensemble' if acc_ensemble > acc_lstm else 'LSTM'}")
    
    # Save final results
    output_file = config.OUTPUT_DIR / 'processed_news_final.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Final Results saved to {output_file}")
    
    # Save trained models for fast inference in the future
    print("\n" + "="*60)
    print("SAVING MODELS TO DISK...")
    print("="*60)
    
    ensemble_path = config.OUTPUT_DIR / 'ensemble_model.joblib'
    joblib.dump(trainer.models['ensemble'], ensemble_path)
    print(f"✓ Ensemble model saved to {ensemble_path}")
    
    lstm_path = config.OUTPUT_DIR / 'lstm_model.keras'
    trainer.models['lstm'].save(lstm_path)
    print(f"✓ LSTM model saved to {lstm_path}")
    
    return df, trainer


if __name__ == "__main__":
    df, trainer = run_pipeline()