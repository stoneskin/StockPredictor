# 🏗️ Architecture - V2.5.1

System design and data flow for Stock Predictor V2.5.1.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Stock Predictor V2.5.1                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐  │
│  │   Raw Data   │───▶│   Training   │───▶│   Trained Models       │  │
│  │  (CSV files) │    │  (train.py)  │    │  (XGBoost, RF, etc.)  │  │
│  └──────────────┘    └──────────────┘    └────────────────────────┘  │
│                                                    │                    │
│                                                    ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         API Server (FastAPI)                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │  │
│  │  │  /predict  │  │/predict/multi│  │    /model-info          │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                       │                                 │
│                                       ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        Client Response                            │  │
│  │   {prediction: "UP", probabilities: {UP: 10%, DOWN: 5%, ...}} │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Training Pipeline

```
data/raw/qqq.csv ──▶ Data Preparation ──▶ Feature Engineering ──▶ Training
                            │                    │                    │
                            ▼                    ▼                    ▼
                   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                   │ Target       │    │ 64 Features  │    │ 7 Models     │
                   │ Creation     │    │ (technical   │    │ (XGBoost,   │
                   │ (4 classes) │    │ indicators)  │    │  RF, GB,     │
                   │              │    │              │    │  SVM, etc.)  │
                   └──────────────┘    └──────────────┘    └──────────────┘
```

### 2. Inference Pipeline

```
Client Request ──▶ Load Model ──▶ Compute Features ──▶ Predict ──▶ Response
     │                │                │                  │          │
     ▼                ▼                ▼                  ▼          ▼
  /predict      models/          compute_          model.      JSON with
  endpoint       results/        features()       predict()   probabilities
                 *.pkl
```

---

## Key Components

### Training (`src/train_v2_5.py`)

| Component | Description |
|-----------|-------------|
| `prepare_data()` | Loads raw data, creates targets |
| `add_technical_indicators()` | Computes 47+ features |
| `add_regime_features()` | Adds market regime features |
| `add_market_features()` | Adds SPY correlation features |
| `create_models()` | Initializes all model types |
| `apply_smote()` | Balances class distribution |
| `evaluate_model()` | Computes metrics |

### Data Preparation (`src/data_preparation_v2_5.py`)

```
Input: Raw OHLCV data (QQQ, SPY)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Target Creation (4-class)                               │
│    - UP: max_gain > threshold                               │
│    - DOWN: max_loss > threshold                             │
│    - UP_DOWN: both exceed threshold                         │
│    - SIDEWAYS: neither exceeds                              │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Engineering (64 features)                        │
│    - Moving Averages (MA, EMA) - 12                         │
│    - RSI, MACD, Bollinger Bands - 10                      │
│    - ATR, Stochastic - 6                                    │
│    - Momentum, ROC - 5                                     │
│    - Volume indicators - 4                                 │
│    - Regime features - 27                                  │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
Output: X (features), y (targets), feature_names
```

### API (`src/inference_v2_5.py`)

| Function | Purpose |
|----------|---------|
| `compute_features()` | Computes 64 features for prediction |
| `load_models()` | Loads trained models on startup |
| `get_stock_data()` | Fetches data from cache or Yahoo Finance |
| `predict()` | Single prediction endpoint |
| `predict_multi()` | Multi-horizon/threshold endpoint |

---

## Feature Engineering Details

### Technical Indicators (47 features)

| Category | Features | Count |
|----------|----------|-------|
| Moving Averages | ma_5, ema_5, ma_10, ema_10, ..., ma_200, ema_200 | 12 |
| RSI | rsi | 1 |
| MACD | macd, macd_signal, macd_hist | 3 |
| Bollinger Bands | bb_upper, bb_lower, bb_width, bb_position | 4 |
| ATR | atr, atr_pct | 2 |
| Stochastic | stoch_k, stoch_d | 2 |
| Momentum | momentum_5, momentum_10, momentum_20 | 3 |
| ROC | roc_5, roc_10 | 2 |
| Volume | volume_ma_5, volume_ma_20, volume_ratio | 3 |
| Price Position | price_above_ma50, price_above_ma200 | 2 |

### Regime Features (17 features)

| Category | Features | Count |
|----------|----------|-------|
| MA Cross | ma_cross_bull, ma_cross_bear, ma_cross_sideways | 3 |
| Volatility | volatility, volatility_high, volatility_normal, volatility_low | 4 |
| ATR Ratio | atr_ratio | 1 |
| Trend Strength | trend_strength | 1 |
| Price vs MA | price_vs_ma50, price_vs_ma200 | 2 |
| Volatility Regime | volatility_regime_num | 1 |
| Momentum Regime | momentum_positive, momentum_strong | 2 |
| RSI Regime | rsi_overbought, rsi_oversold, rsi_neutral | 3 |

### Market Features (10 features)

| Features | Description |
|----------|-------------|
| correlation_spy_5d/10d/20d | Rolling correlation with SPY |
| spy_correlation_positive/negative/neutral | Correlation regime |
| spy_momentum | SPY momentum |
| spy_momentum_positive | SPY momentum direction |

---

## Model Architecture

### Supported Models

| Model | Strengths | Default Weight |
|-------|-----------|-----------------|
| **XGBoost** | Best performance, handles imbalance well | 20% |
| Random Forest | Robust, fast, good baseline | 20% |
| Gradient Boosting | Good accuracy, interpretable | 20% |
| CatBoost | Handles categorical features | 15% |
| Logistic Regression | Interpretable, fast | 15% |
| SVM | Good for high-dimensional data | 5% |
| Naive Bayes | Fast, probabilistic | 5% |

### Ensemble Strategy

```
Ensemble = Σ (model.predict_proba() × weight)
Final Prediction = argmax(Ensemble)
```

---

## Class Imbalance Handling

### Problem
- At 5% threshold: ~95% of samples are SIDEWAYS
- At 1% threshold with long horizon: binary class collapse

### Solution: SMOTE

```python
# Before training
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Result: Balanced class distribution
# [414, 414, 414, 414] (was [399, 248, 511, 179])
```

---

## Version Differences

| Feature | V2.5.0 | V2.5.1 |
|---------|---------|---------|
| Classes | 4 | 4 (reordered) |
| Class Order | SIDEWAYS, UP, DOWN, UP_DOWN | UP, DOWN, UP_DOWN, SIDEWAYS |
| Best Model | RandomForest | XGBoost |
| SMOTE | No | Yes |
| Features | ~47 | 64 |
| Regime Features | Basic | Full |
| API Version | 2.5.0 | 2.5.1 |

---

## File Structure

```
v2.5/
├── src/
│   ├── config_v2_5.py          # Configuration
│   ├── data_preparation_v2_5.py # Data processing
│   ├── train_v2_5.py           # Training script
│   ├── inference_v2_5.py       # API server
│   ├── logging_utils.py        # Logging utilities
│   ├── models_v2/              # Model implementations
│   │   ├── base.py
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   └── ...
│   └── regime_v2/              # Regime detection
├── models/
│   └── results/                # Trained models (*.pkl)
├── data/
│   ├── raw/                   # Raw data (qqq.csv, spy.csv)
│   └── cache/                 # Cached data
└── docs/                      # Documentation
```

---

## Performance Metrics

### Training Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | True positives / (TP + FP) |
| Recall | True positives / (TP + FN) |
| F1 | Harmonic mean of precision/recall |
| AUC-ROC | Area under ROC curve (multi-class) |

### Typical Results (V2.5.1)

| Horizon | Threshold | Accuracy | AUC-ROC |
|---------|-----------|----------|---------|
| 5d | 1% | 61% | 0.82 |
| 5d | 2.5% | 81% | 0.91 |
| 10d | 1% | 82% | 0.89 |
| 20d | 2.5% | 93% | 0.99 |
| 30d | 2.5% | 97% | 1.00 |

---

## Configuration

Key parameters in `config_v2_5.py`:

```python
# Prediction parameters
HORIZONS = [5, 10, 20, 30]
THRESHOLDS = [0.01, 0.025, 0.05]
CLASS_LABELS = ["UP", "DOWN", "UP_DOWN", "SIDEWAYS"]

# Training parameters
USE_SMOTE = True
USE_TIMESERIES_CV = False
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| scikit-learn | Base ML models |
| xgboost | XGBoost classifier |
| catboost | CatBoost classifier |
| imbalanced-learn | SMOTE for class imbalance |
| fastapi | API server |
| pandas | Data processing |
| numpy | Numerical operations |
| yfinance | Stock data fetching |

---

## Version History

- **2.5.1** (2026-02-28): Enhanced architecture, SMOTE, 64 features
- **2.5.0** (2026-02-27): Initial 4-class classification
- **2.0** (2025): Binary classification (UP/DOWN)
