"""
Configuration for Stock Predictor V2.5
Enhanced approach with multi-class classification and detailed thresholds
"""

import os
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DATA_DIR = DATA_DIR / "cache"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_RESULTS_DIR = MODEL_DIR / "results" / "v2_5"
LOG_DIR = PROJECT_ROOT / "src" / "v2_5" / "logs"

# Create directories
CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create log subdirectories
TRAIN_LOG_DIR = LOG_DIR / "training"
PREDICTION_LOG_DIR = LOG_DIR / "prediction"
API_LOG_DIR = LOG_DIR / "api"

TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
API_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Data parameters
HORIZONS = [5, 10, 20, 30]
DEFAULT_HORIZON = 20

# New threshold-based classification (V2.5)
# For each horizon, we have 4 classes: UP, DOWN, UP_DOWN, SIDEWAYS
# UP: max gain > threshold in any day within horizon
# DOWN: max loss > threshold in any day within horizon
# UP_DOWN: both max gain > threshold AND max loss > threshold
# SIDEWAYS: neither max gain > threshold nor max loss > threshold
THRESHOLDS = [0.01, 0.025, 0.05]  # 1%, 2.5%, 5%
THRESHOLD_LABELS = ["1pct", "2_5pct", "5pct"]

# Classification target: 4 classes
# 0: SIDEWAYS, 1: UP, 2: DOWN, 3: UP_DOWN
CLASS_LABELS = ["SIDEWAYS", "UP", "DOWN", "UP_DOWN"]
CLASSIFICATION_THRESHOLD = 0.5

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
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    },
    'catboost': {
        'iterations': 100,
        'depth': 3,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': 0
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
    'logistic_regression': 0.15,
    'random_forest': 0.20,
    'gradient_boosting': 0.20,
    'xgboost': 0.20,
    'catboost': 0.15,
    'svm': 0.05,
    'naive_bayes': 0.05
}

# Regime detection parameters
REGIME_PARAMS = {
    'short_ma': 50,
    'long_ma': 200,
    'volatility_window': 20,
    'volatility_threshold': 0.02,
    'additional_regimes': {
        'high_volatility': 0.03,
        'low_volatility': 0.005
    }
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
    },
    'sentiment_indicators': {
        'news_sentiment': False,  # Placeholder for future
        'social_sentiment': False  # Placeholder for future
    }
}

# Training parameters
TRAIN_PARAMS = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'cv_folds': 5
}

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc_roc',
    'auc_pr',
    'confusion_matrix',
    'balanced_accuracy'
]

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Logging helper function
def get_log_filename(log_type: str) -> str:
    """Generate log filename with date and time."""
    now = datetime.now()
    return f"{log_type}_{now.strftime('%Y%m%d_%H%M%S')}.log"
