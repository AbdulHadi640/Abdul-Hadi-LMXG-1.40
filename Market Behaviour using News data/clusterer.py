"""
Market Impact Clusterer - K-Means Clustering (SAFE)
====================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import config

class MarketImpactClusterer:
    """Cluster market impact using K-Means"""
    
    def __init__(self):
        self.kmeans = None
        self.scaler = StandardScaler()
    
    def cluster(self, df):
        """Standard Deviation thresholding for market impact"""
        print("\n" + "="*60)
        print("MARKET IMPACT CLASSIFICATION (Std Dev Thresholds)")
        print("="*60)
        
        # SAFETY CHECK: Ensure no NaNs exist
        if df['corrected_return'].isna().any():
            print("  ⚠️ Warning: NaNs detected in corrected_return. Filling with 0.0...")
            df['corrected_return'] = df['corrected_return'].fillna(0.0)
            
        returns = df['corrected_return']
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # Define Thresholds
        crash_threshold = mean_ret - (1.5 * std_ret)
        spike_threshold = mean_ret + (1.5 * std_ret)
        
        print(f"  - Mean Return: {mean_ret:.6f}")
        print(f"  - Std Dev: {std_ret:.6f}")
        print(f"  - Crash Threshold: <= {crash_threshold:.6f}")
        print(f"  - Spike Threshold: >= {spike_threshold:.6f}")
        
        # Assign Labels
        conditions = [
            (returns <= crash_threshold),
            (returns >= spike_threshold)
        ]
        choices = [0, 2] # 0: Crash, 2: Spike
        # Default is 1 (Neutral)
        df['cluster_label'] = np.select(conditions, choices, default=1)
        
        cluster_names = {0: 'Crash', 1: 'Neutral', 2: 'Spike'}
        df['impact_category'] = df['cluster_label'].map(cluster_names)
        
        print("\nClass distribution:")
        print(df['impact_category'].value_counts())
        
        print("\nClass characteristics:")
        for cluster in [0, 1, 2]:
            mask = df['cluster_label'] == cluster
            print(f"\n  Class {cluster} ({cluster_names[cluster]}):")
            print(f"    Count: {mask.sum()}")
            if mask.sum() > 0:
                print(f"    Mean return: {df.loc[mask, 'corrected_return'].mean():.6f}")
                print(f"    Std return: {df.loc[mask, 'corrected_return'].std():.6f}")
        
        return df, None # Return None for kmeans model as it's no longer used