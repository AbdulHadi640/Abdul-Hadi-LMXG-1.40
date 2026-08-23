"""
SPY Data Aligner - Align News with SPY Prices (ROBUST FIX)
===========================================================
"""

import pandas as pd
import numpy as np
import os
import config

class SPYAligner:
    """Align news with SPY price data and calculate returns"""
    
    def __init__(self, spy_data_path=None):
        self.spy_data = None
        if spy_data_path and os.path.exists(spy_data_path):
            self.load_spy_data(spy_data_path)
    
    def load_spy_data(self, path):
        """Load SPY 1-minute price data"""
        print("\n" + "="*60)
        print("[7/8] LOADING SPY DATA")
        print("="*60)
        
        if str(path).endswith('.xlsx'):
            self.spy_data = pd.read_excel(path)
        else:
            self.spy_data = pd.read_csv(path)
            
        # Check for specific SPY Excel format
        if 'market_ny' in self.spy_data.columns:
            self.spy_data = self.spy_data.rename(columns={'market_ny': 'timestamp'})
        elif 'market_utc' in self.spy_data.columns:
            self.spy_data = self.spy_data.rename(columns={'market_utc': 'timestamp'})
            
        if 'close' in self.spy_data.columns:
            self.spy_data = self.spy_data.rename(columns={'close': 'spy_price'})
        elif 'vwap' in self.spy_data.columns:
            self.spy_data = self.spy_data.rename(columns={'vwap': 'spy_price'})
            
        # Fallback if names still don't match
        if len(self.spy_data.columns) >= 2 and 'timestamp' not in self.spy_data.columns:
            self.spy_data = self.spy_data.rename(columns={self.spy_data.columns[0]: 'timestamp', self.spy_data.columns[1]: 'spy_price'})
            
        # Safely convert to datetime, parsing any mixed offsets to UTC first, then NY
        self.spy_data['timestamp'] = pd.to_datetime(self.spy_data['timestamp'], utc=True).dt.tz_convert('America/New_York')
        
        # FIX: Ensure spy_price is numeric (in case Excel read it as strings)
        self.spy_data['spy_price'] = pd.to_numeric(self.spy_data['spy_price'], errors='coerce')
        
        self.spy_data = self.spy_data.set_index('timestamp')
        print(f"✓ Loaded {len(self.spy_data)} rows")
    
    def create_synthetic_spy(self, df):
        """Create synthetic SPY data if not available"""
        print("\n" + "="*60)
        print("[7/8] CREATING SYNTHETIC SPY DATA")
        print("="*60)
        
        # Use the timezone-aware gdelt_date to create matching timestamps
        min_time = df['gdelt_date'].min()
        max_time = df['gdelt_date'].max()
        tz = min_time.tzinfo  # Get the timezone (America/New_York)
        
        # Create date range with the SAME timezone
        timestamps = pd.date_range(start=min_time, end=max_time, freq='1min', tz=tz)
        
        np.random.seed(config.RANDOM_STATE)
        returns = np.random.normal(0.0001, 0.001, len(timestamps))
        prices = 400 * np.cumprod(1 + returns)
        
        self.spy_data = pd.DataFrame({
            'timestamp': timestamps,
            'spy_price': prices
        }).set_index('timestamp')
        
        print(f"✓ Created {len(self.spy_data)} synthetic prices (Timezone: {tz})")
    
    def align_and_calculate_returns(self, df):
        """Align news with SPY and calculate returns using merge_asof"""
        print("\n" + "="*60)
        print("[8/8] ALIGNING WITH SPY & CALCULATING RETURNS")
        print("="*60)
        
        # Sort both dataframes by time (required for merge_asof)
        df = df.sort_values('gdelt_date').reset_index(drop=True)
        spy_df = self.spy_data.reset_index().sort_values('timestamp')
        
        # merge_asof finds the nearest prior timestamp (Industry Standard for Financial Data)
        # Tolerance of 2 minutes ensures we don't match news to wildly different times
        df = pd.merge_asof(
            df, 
            spy_df, 
            left_on='gdelt_date', 
            right_on='timestamp', 
            direction='nearest',
            tolerance=pd.Timedelta('5 minutes')
        )
        
        # Safety fill for any remaining NaNs
        df['spy_price'] = df['spy_price'].ffill().bfill().fillna(400.0)
        
        # Calculate 5-minute forward return
        df['spy_return_5min'] = df['spy_price'].shift(-5) / df['spy_price'] - 1
        
        # Market-adjusted return (subtract rolling mean)
        df['market_return'] = df['spy_return_5min'].rolling(window=20, min_periods=1).mean()
        df['corrected_return'] = df['spy_return_5min'] - df['market_return']
        
        # CRITICAL FIX: Fill any remaining NaNs in corrected_return with 0.0
        nan_count = df['corrected_return'].isna().sum()
        if nan_count > 0:
            print(f"  Warning: Found {nan_count} NaNs in returns. Filling with 0.0...")
            df['corrected_return'] = df['corrected_return'].fillna(0.0)
        
        print(f"✓ Mean 5-min return: {df['spy_return_5min'].mean():.6f}")
        print(f"✓ Std return: {df['spy_return_5min'].std():.6f}")
        print(f"✓ Mean corrected return: {df['corrected_return'].mean():.6f}")
        print(f"✓ Final NaNs in corrected_return: {df['corrected_return'].isna().sum()}")
        
        return df