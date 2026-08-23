"""
Market Session Classifier
=========================
"""

import pandas as pd
import config

class MarketSessionClassifier:
    """Classify news into market sessions"""
    
    def classify_session(self, row):
        """
        Classify into Pre-Market/Regular/After-Hours/Closed
        
        NYSE Hours:
        - Pre-Market: 4:00 AM - 9:30 AM EST
        - Regular: 9:30 AM - 4:00 PM EST
        - After-Hours: 4:00 PM - 8:00 PM EST
        - Closed: 8:00 PM - 4:00 AM EST + Weekends
        """
        hour = row['hour_ny']
        minute = row['minute_ny']
        day = row['day_of_week']
        
        # Convert to minutes from midnight
        time_min = hour * 60 + minute
        
        # Weekend
        if day >= 5:
            return 'Closed', 3
        
        # Pre-Market: 4:00 AM - 9:30 AM (240 - 570 minutes)
        if 240 <= time_min < 570:
            return 'Pre-Market', 0
        
        # Regular: 9:30 AM - 4:00 PM (570 - 960 minutes)
        elif 570 <= time_min < 960:
            return 'Regular', 1
        
        # After-Hours: 4:00 PM - 8:00 PM (960 - 1200 minutes)
        elif 960 <= time_min < 1200:
            return 'After-Hours', 2
        
        # Closed
        else:
            return 'Closed', 3
    
    def classify_all(self, df):
        """Classify all rows"""
        print("\n" + "="*60)
        print("[4/8] MARKET SESSION CLASSIFICATION")
        print("="*60)
        
        df[['market_session', 'session_encoded']] = df.apply(
            lambda x: pd.Series(self.classify_session(x)),
            axis=1
        )
        
        counts = df['market_session'].value_counts()
        print("\nSession distribution:")
        for session, count in counts.items():
            print(f"  - {session}: {count} ({count/len(df)*100:.1f}%)")
        
        return df