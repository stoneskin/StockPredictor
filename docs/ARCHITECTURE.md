# 🏗️ System Architecture

A detailed explanation of how the Stock Predictor system is designed and works.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│   User Request (prediction for QQQ, horizon=5 days)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Fetch Historical Data       │
        │  (200+ days from Yahoo)      │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Calculate Technical         │
        │  Indicators (25+ features)   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Detect Market Regime        │
        │  (Bull/Bear/Sideways)        │
        └──────────────┬───────────────┘
                       │
                       ▼
     ┌─────────────────────────────────────────┐
     │  Make Predictions (5 Base Models)       │
     ├───────────────────┬───────────────────┤
     │ Logistic (20%)    │ Random Forest(30%)│
     │ SVM (15%)         │ Gradient Boost(25%)
     │ Naive Bayes (10%) │                   │
     └───────────┬───────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────────┐
        │  Ensemble Voting             │
        │  (Weighted Average)          │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Adjust by Horizon           │
        │  (5/10/20/30 day factors)    │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Return Prediction           │
        │  - Direction (UP/DOWN)       │
        │  - Probabilities             │
        │  - Confidence Level          │
        └──────────────────────────────┘
```

---

## Component Breakdown

### 1. Data Layer (`src/v2/data_preparation_v2.py`)

**Responsibility**: Load, prepare, and feature-engineer raw market data

**Process**:
1. **Fetch Data**: Download OHLCV (Open, High, Low, Close, Volume) from Yahoo Finance
2. **Validate**: Check data quality and handle missing values
3. **Feature Engineering**: Calculate technical indicators
4. **Label Generation**: Create classification labels (UP/DOWN)
5. **Data Splitting**: Train/validation/test sets with time-series logic

**Key Functions**:
```python
def prepare_data(symbol: str) -> Tuple[pd.DataFrame, np.ndarray]:
    # 1. Load historical data
    # 2. Calculate indicators
    # 3. Create labels
    # 4. Return features and targets
    pass
```

**Features Generated** (25+):
- **Moving Averages**: MA5, MA10, MA20, MA50, MA200, EMA5, EMA20, etc.
- **Momentum**: RSI (14), MACD, MACD Signal, STOCH, ROC
- **Volatility**: ATR, Bollinger Bands width, std deviation
- **Volume**: VWAP, Volume MA
- **Market Regime**: SPY correlation, VIX-like features

**Output**: DataFrame with columns:
```
date, open, high, low, close, volume, 
ma_5, ma_20, rsi, macd, atr, ... (25+ features),
target (0/1)
```

---

### 2. Training Layer (`src/v2/train_v2.py`)

**Responsibility**: Train all models and save them

**Process**:
```
1. Load prepared data
2. For each horizon (5, 10, 20, 30 days):
   a. Train Logistic Regression
   b. Train Random Forest
   c. Train Gradient Boosting
   d. Train SVM
   e. Train Naive Bayes
   f. Evaluate all on validation set
   g. Create ensemble with weights
3. Save all models
4. Generate performance report
```

**Model Storage**:
```
models/results/v2/
├── horizon_5/
│   ├── logistic.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── svm.pkl
│   ├── naive_bayes.pkl
│   └── ensemble_weights.json
├── horizon_10/
├── horizon_20/
├── horizon_30/
└── horizon_comparison.json  # Overall results
```

**Output**: Performance metrics
```json
{
  "horizon_5": {
    "accuracy": 0.54,
    "precision": 0.52,
    "recall": 0.48,
    "auc_roc": 0.56,
    "f1_score": 0.50,
    "models": {
      "logistic": {"accuracy": 0.53, ...},
      "random_forest": {"accuracy": 0.55, ...},
      ...
    }
  },
  ...
}
```

---

### 3. Inference Layer (`src/v2/inference_v2.py`)

**Responsibility**: Serve predictions via FastAPI API

**Architecture**:
```python
# 1. Initialize
load_model(horizon=5)  # Load ensemble models

# 2. On Request
data = fetch_data(symbol="QQQ")
features = compute_features_v2(data)
probabilities = model.predict_proba(features)
adjusted_proba = adjust_by_horizon(probabilities, data, horizon=5)

# 3. Format Response
response = {
    "symbol": "QQQ",
    "date": today,
    "predictions": [
        {
            "horizon": 5,
            "prediction": "UP" if proba[1] > 0.5 else "DOWN",
            "probability_up": proba[1],
            "probability_down": proba[0],
            "confidence": max(proba)
        },
        ...
    ]
}
```

**Key Endpoints**:
- `GET /` - Info
- `POST /predict/simple` - Quick prediction
- `POST /predict` - Full prediction
- `GET /health` - Health check
- `GET /docs` - Interactive API docs

---

### 4. Models Layer (`src/v2/models_v2/`)

**5 Base Models with Characteristics**:

| Model | Type | Pros | Cons | Training Time |
|-------|------|------|------|---------|
| **Logistic Regression** | Linear | Fast, interpretable | Low complexity | <1s |
| **Random Forest** | Ensemble | Non-linear, robust | Black box | 5-10s |
| **Gradient Boosting** | Sequential | Strong performance | Prone to overfit | 10-30s |
| **SVM (RBF)** | Kernel | Good separation | Slow for large data | 20-60s |
| **Naive Bayes** | Probabilistic | Fast, handles uncertainty | Assumes independence | <1s |

**Base Model Interface** (all inherit):
```python
class BaseModel:
    def fit(self, X_train, y_train):
        """Train the model"""
        pass
    
    def predict(self, X):
        """Return binary predictions (0/1)"""
        pass
    
    def predict_proba(self, X):
        """Return probability predictions [p_down, p_up]"""
        pass
    
    def evaluate(self, X_test, y_test):
        """Return metrics dict"""
        pass
```

---

### 5. Ensemble Layer (`src/v2/models_v2/ensemble.py`)

**Purpose**: Combine 5 models into one strong predictor

**Algorithm**:
```
predictions = []
for model, weight in zip(models, weights):
    pred = model.predict_proba(X)  # [p_down, p_up]
    predictions.append(pred * weight)

final_prediction = sum(predictions) / sum(weights)
```

**Default Weights** (tuned on validation set):
- Random Forest: 30%
- Gradient Boosting: 25%
- Logistic Regression: 20%
- SVM: 15%
- Naive Bayes: 10%

**Why Ensemble?**
- ✅ Reduces single-model bias
- ✅ More stable predictions
- ✅ Better generalization
- ✅ Handles different data patterns

---

### 6. Market Regime Detector (`src/v2/regime_v2/`)

**Purpose**: Identify market type for better interpretation

**Strategies**:

**MA Crossover** (Moving Average):
```python
# Using 50-day and 200-day moving averages
if price > ma_50 > ma_200:
    regime = "BULL"      # Strong uptrend
elif price < ma_50 < ma_200:
    regime = "BEAR"      # Strong downtrend
else:
    regime = "SIDEWAYS"  # No clear trend
```

**Volatility Regime**:
```python
# Using ATR (Average True Range)
if atr > high_threshold:
    volatility = "HIGH"      # Choppy, unpredictable
else:
    volatility = "LOW"       # Stable, predictable
```

**Usage**:
```python
detector = MARegimeDetector()
regime = detector.detect(df)

if regime == "BULL":
    # Be more confident in UP predictions
    confidence *= 1.1
elif regime == "BEAR":
    # Be more confident in DOWN predictions
    confidence *= 1.1
```

---

## Data Flow Diagram

```
Raw Data (Yahoo Finance)
    ↓
    ├─→ QQQ OHLCV data
    ├─→ SPY OHLCV data (for regime detection)
    └─→ Store in data/raw/
    
Preprocessed Data
    ↓
    ├─→ Calculate 25+ features
    ├─→ Create classification labels
    ├─→ Handle missing values
    └─→ Store in data/processed/
    
Training Data
    ↓
    ├─→ 70% train set
    ├─→ 15% validation set
    └─→ 15% test set
    
Model Training (4 horizons × 5 models = 20 models)
    ↓
    ├─→ Logistic Regression (lightweight)
    ├─→ Random Forest (fast ensemble)
    ├─→ Gradient Boosting (powerful)
    ├─→ SVM (separator)
    └─→ Naive Bayes (baseline)
    
Ensemble Creation
    ↓
    └─→ Weighted voting → Final predictions
    
Inference API
    ↓
    ├─→ Load ensemble
    ├─→ Fetch recent data
    ├─→ Compute features
    └─→ Return probability-based prediction
```

---

## Configuration (`src/v2/config_v2.py`)

**Key Parameters**:
```python
# Data
SYMBOL = "QQQ"
TRAIN_YEARS = 5
MIN_HISTORY = 200

# Features
FEATURE_WINDOWS = [5, 10, 20, 50, 200]
INCLUDE_REGIMES = True

# Training
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Models
HORIZONS = [5, 10, 20, 30]
MODEL_WEIGHTS = {
    "random_forest": 0.30,
    "gradient_boosting": 0.25,
    "logistic": 0.20,
    "svm": 0.15,
    "naive_bayes": 0.10
}

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True
```

**How to Configure**:
```python
# Before training/inference, edit config_v2.py:
HORIZONS = [5, 10]  # Only predict 5 and 10 days
SYMBOL = "TSLA"  # Switch to Tesla
TRAIN_YEARS = 3  # Use 3 years of data

# Then run training/inference
python src/v2/train_v2.py
python -m uvicorn src.v2.inference_v2:app
```

---

## Key Design Decisions

### ✅ Why Classification Instead of Regression?
- Stock returns are noisy, hard to predict exactly
- Binary up/down is more robust
- Probability outputs are more useful for trading

### ✅ Why Multiple Horizons?
- Different predictions for different strategies
- 5-day for short-term trading
- 20-30 day for medium-term investing
- Allows flexibility

### ✅ Why Ensemble?
- Single models overfit easily with limited data
- Different models catch different patterns
- More stable and better generalization
- Reduces variance

### ✅ Why Walk-Forward Validation?
- Respects time-series structure
- Prevents data leakage
- More realistic evaluation
- Simulates real trading scenario

---

## Performance Metrics

**Classification Metrics Used**:
- **Accuracy**: What % of predictions are correct
  ```
  Accuracy = (TP + TN) / Total
  ```

- **Precision**: Of predicted UPs, how many were correct
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall**: Of actual UPs, how many we caught
  ```
  Recall = TP / (TP + FN)
  ```

- **AUC-ROC**: Area Under Receiver Operating Curve
  - Measures true positive vs false positive rate
  - 0.5 = random, 1.0 = perfect

- **F1-Score**: Harmonic mean of precision and recall
  ```
  F1 = 2 * (Precision * Recall) / (Precision + Recall)
  ```

---

## Typical Results

Example performance across horizons:

```
Horizon 5-day:
  Accuracy: 54%
  AUC-ROC: 0.56
  Best Model: Random Forest (55% accuracy)

Horizon 10-day:
  Accuracy: 52%
  AUC-ROC: 0.54
  Best Model: Gradient Boosting (53% accuracy)

Horizon 20-day:
  Accuracy: 51%
  AUC-ROC: 0.52
  Best Model: Ensemble (51% accuracy)

Horizon 30-day:
  Accuracy: 50%
  AUC-ROC: 0.51
  (Approaching random guessing)
```

**Interpretation**:
- 5-day predictions are best (~54% > 50% random baseline)
- Long-term predictions are weaker (close to 50%)
- Ensemble matches or exceeds individual models

---

## Next Steps

1. **Review specific implementations**:
   - Data: `src/v2/data_preparation_v2.py`
   - Training: `src/v2/train_v2.py`
   - Models: `src/v2/models_v2/`
   - API: `src/v2/inference_v2.py`

2. **Understand the models**:
   - Each model code is well-commented
   - Ensemble voting logic is straightforward
   - Feature engineering uses standard ta-lib

3. **Experiment**:
   - Change hyperparameters in `config_v2.py`
   - Train models: `python src/v2/train_v2.py`
   - Test predictions: `python tests/test_api.py`

---

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start guide
- [V2_CLASSIFICATION.md](V2_CLASSIFICATION.md) - Detailed V2 design
- [API_REFERENCE.md](API_REFERENCE.md) - API endpoints
- Code comments in `src/v2/` - Implementation details