"""
Configuration for Stock Predictor V2
Improved approach with classification, ensemble, and regime detection
"""

import os
from pathlib import Path

# Project paths
# Fix: Go up to find project root (src/v2/config_v2.py -> project root)
# Current file: src/v2/config_v2.py
# parent: src/v2, parent.parent: src, parent.parent.parent: project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"  # Training data (may have legacy format)
CACHE_DATA_DIR = DATA_DIR / "cache"  # Runtime cache (always standard format)
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
MODEL_RESULTS_DIR = PROJECT_ROOT / "models" / "results" / "v2"

# Create directories
CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data parameters
HORIZONS = [5, 10, 20]  # Prediction horizons in days
DEFAULT_HORIZON = 5  # Default prediction horizon

# Classification threshold
CLASSIFICATION_THRESHOLD = 0.5  # Probability threshold for up/down

# Model parameters
MODEL_PARAMS = {
    'logistic_regression': {
        'C': 0.1,
        'max_iter': 1000,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.1,
        'min_samples_split': 20,
        'random_state': 42
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'probability': True,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'naive_bayes': {
        'var_smoothing': 1e-9
    }
}

# Ensemble parameters
ENSEMBLE_WEIGHTS = {
    'logistic_regression': 0.2,
    'random_forest': 0.3,
    'gradient_boosting': 0.25,
    'svm': 0.15,
    'naive_bayes': 0.1
}

# Regime detection parameters
REGIME_PARAMS = {
    'short_ma': 50,
    'long_ma': 200,
    'volatility_window': 20,
    'volatility_threshold': 0.02  # 2% daily volatility threshold
}

# Feature engineering parameters
FEATURE_PARAMS = {
    'technical_indicators': {
        'ma_periods': [5, 10, 20, 50, 100, 200],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'atr_period': 14,
        'stoch_period': 14
    }
}

# Training parameters
TRAIN_PARAMS = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'cv_folds': 5
}

# Walk-forward validation parameters
WALK_FORWARD_PARAMS = {
    'train_months': 12,
    'test_months': 3,
    'step_months': 3,
    'min_train_samples': 200,
    'min_test_samples': 30
}

# Target encoding
TARGET_COLUMNS = {
    5: 'target_5d',
    10: 'target_10d',
    20: 'target_20d'
}

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc_roc',
    'auc_pr',
    'confusion_matrix'
]

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'