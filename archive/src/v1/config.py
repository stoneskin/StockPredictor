"""
Configuration settings for the stock prediction project.
"""

import os
from datetime import datetime, timedelta

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, MODEL_CHECKPOINTS_DIR, ONNX_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== DATA CONFIG ====================
TICKER = "QQQ"
MARKET_BENCHMARK = "SPY"  # For relative strength

# Date range
TRAIN_START = "2020-01-01"
TRAIN_END = "2024-12-31"
VAL_START = "2025-01-01"
VAL_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = datetime.now().strftime("%Y-%m-%d")

# ==================== FEATURE CONFIG ====================
# Vegas Channel parameters (default from Pine Script)
VEGAS_MA1_PERIOD = 144
VEGAS_MA2_PERIOD = 169
VEGAS_MA12_PERIOD = 576
VEGAS_MA22_PERIOD = 676
VEGAS_MA_TYPE = "ema"  # 'sma', 'ema', 'wma'

# Hull MA parameters
HULL_LENGTH = 55
HULL_MODE = 1  # 1=HMA, 2=EHMA, 3=THMA

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# RSI
RSI_LENGTH = 14

# Stochastic
STOCH_K = 3
STOCH_D = 3
STOCH_RSI_LENGTH = 14

# Volume and Volatility
VOLUME_MA_PERIOD = 20
ATR_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_STD = 2

# ==================== TARGET CONFIG ====================
PREDICTION_HORIZON = 15  # Predict 15-day forward return
TARGET_THRESHOLD = 0.0  # For classification (positive return = success)

# Risk metrics
CALCULATE_MAX_DRAWDOWN = True
MAX_DRAWDOWN_WINDOW = PREDICTION_HORIZON

# ==================== MODEL CONFIG ====================
MODEL_TYPE = "lightgbm"  # 'lightgbm', 'xgboost', 'sklearn'
MODEL_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
}

# For multi-task (multiple outputs) - optional
MULTI_TASK = False
TASK_WEIGHTS = {'return': 1.0, 'risk': 0.5}  # Loss weights if multi-task

# ==================== TRAINING CONFIG ====================
BATCH_SIZE = None  # Not used for tree-based models
NUM_EPOCHS = 1  # Tree-based models train in one go
PATIENCE = 50  # Early stopping

# ==================== INFERENCE CONFIG ====================
MODEL_OUTPUT_PATH = os.path.join(MODEL_CHECKPOINTS_DIR, "latest_model.pkl")
ONNX_MODEL_PATH = os.path.join(ONNX_DIR, "model.onnx")
API_HOST = "0.0.0.0"
API_PORT = 8000

# ==================== SAGEMAKER CONFIG (optional) ====================
SAGEMAKER_CONFIG = {
    'role': os.getenv('SAGEMAKER_ROLE_ARN', ''),
    'region': 'us-east-1',
    'instance_type': 'ml.m5.xlarge',
    'instance_count': 1,
    'framework_version': '1.0-1',
    'py_version': 'py3',
}