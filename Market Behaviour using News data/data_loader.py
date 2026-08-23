"""
Data Loader - Load and Convert Time (SIMPLE FIX)
=================================================
"""

import pandas as pd
import pytz
from pathlib import Path
import config

class DataLoader:
    """Load GDELT data from CSV files"""
    
    def __init__(self):
        self.timezone_ny = pytz.timezone(config.TIMEZONE_NY)
        
    def load_all_csvs(self, data_folder=None):
        """Load all CSV files from folder"""
        if data_folder is None:
            data_folder = config.DATA_FOLDER
        
        data_folder = Path(data_folder)
        csv_files = list(data_folder.glob("*.csv"))
        
        print("\n" + "="*60)
        print("[1/8] LOADING DATA")
        print("="*60)
        print(f"Found {len(csv_files)} CSV files in {data_folder}")
        
        all_data = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
                df['source_file'] = file.name
                all_data.append(df)
                print(f"  ✓ {file.name}: {len(df)} rows")
            except Exception as e:
                print(f"  ✗ Error loading {file.name}: {e}")
        
        if not all_data:
            raise ValueError("No CSV files found or all failed to load!")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal: {len(combined_df)} rows loaded")
        
        return combined_df
    
    def convert_to_ny_time(self, df, date_column='gdelt_date'):
        """Convert UTC timestamps to New York Time (SIMPLE CHECK)"""
        print("\n" + "="*60)
        print("[2/8] TIME CONVERSION (UTC → New York)")
        print("="*60)
        
        # 1. Parse datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # 2. Check if timezone already exists
        tz_info = df[date_column].dt.tz
        print(f"  Current timezone info: {tz_info}")
        
        if tz_info is not None:
            # If it already has a timezone, convert it to UTC directly
            print("  Timezone exists. Converting to UTC...")
            df[date_column] = df[date_column].dt.tz_convert('UTC')
        else:
            # If no timezone, localize to UTC
            print("  No timezone. Localizing to UTC...")
            df[date_column] = df[date_column].dt.tz_localize('UTC')
            
        # 3. Convert UTC to New York Time
        df[date_column] = df[date_column].dt.tz_convert(self.timezone_ny)
        print("  Successfully converted to New York Time.")
        
        # 4. Extract time features
        df['hour_ny'] = df[date_column].dt.hour
        df['minute_ny'] = df[date_column].dt.minute
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        print(f"✓ Time range: {df[date_column].min()} to {df[date_column].max()}")
        print(f"✓ Extracted features: hour_ny, minute_ny, day_of_week, is_weekend")
        
        return df