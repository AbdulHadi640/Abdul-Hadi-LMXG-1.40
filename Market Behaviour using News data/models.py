"""
Model Trainer - Train Ensemble and LSTM Models
===============================================
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Bidirectional, Dropout, 
    MultiHeadAttention, LayerNormalization, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import config

class ModelTrainer:
    """Train multiple models for market impact prediction"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    
    def train_ensemble(self, X_train, X_test, y_train, y_test):
        """Train ensemble of multiple models"""
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE MODEL")
        print("="*60)
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # SAVE SCALER FOR LIVE INFERENCE
        import joblib
        scaler_path = config.OUTPUT_DIR / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Undersample massive Neutral class to prevent OutOfMemory (RAM crash)
        print("  - Undersampling Neutral class to prevent MemoryError...")
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=config.RANDOM_STATE)
        X_train_under, y_train_under = rus.fit_resample(X_train_scaled, y_train)
        
        # SMOTE Oversampling for remaining classes
        print("  - Applying SMOTE to balance remaining classes...")
        smote = SMOTE(sampling_strategy='not majority', random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)
        
        # Model 1: Random Forest
        print("  - Random Forest")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        
        # Model 2: Gradient Boosting
        print("  - Gradient Boosting")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE
        )
        
        # Voting Ensemble
        print("  - Creating ensemble...")
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        # Train
        print("Training ensemble...")
        ensemble.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate
        y_pred = ensemble.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Ensemble Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Crash', 'Neutral', 'Spike']
        ))
        
        self.models['ensemble'] = ensemble
        
        return acc, y_pred
    
    def train_lstm(self, X_train, X_test, y_train, y_test):
        """Train LSTM with Attention model"""
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)
        
        # Undersample massive Neutral class to prevent OutOfMemory (RAM crash)
        print("  - Undersampling Neutral class to prevent MemoryError...")
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=config.RANDOM_STATE)
        X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
        
        # SMOTE Oversampling
        print("  - Applying SMOTE to balance remaining classes for LSTM...")
        smote = SMOTE(sampling_strategy='not majority', random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)
        
        # Reshape for Keras (Standard 2D array for Dense network)
        X_train_dnn = X_train_resampled
        X_test_dnn = X_test
        
        # Build model (Deep Dense Network instead of LSTM for static embeddings)
        print("Building Deep Neural Network architecture...")
        model = Sequential([
            Input(shape=(X_train_dnn.shape[1],)),
            
            Dense(256, activation='relu'),
            LayerNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            LayerNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            
            # Output
            Dense(3, activation='softmax')
        ])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train
        print("Training Deep Neural Network...")
        model.fit(
            X_train_dnn, y_train_resampled,
            validation_split=0.2,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, acc = model.evaluate(X_test_dnn, y_test, verbose=0)
        print(f"\n✓ Deep NN Accuracy: {acc:.4f}")
        
        self.models['lstm'] = model  # Keeping dict key same so it doesn't break main.py
        
        return acc