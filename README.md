# 📈 Stock Predictor - ML for Trading

---

**🌐 Language**: [中文版 (Chinese)](README_cn.md)

A complete machine learning system for predicting QQQ stock price movements using classification + ensemble learning. Learn ML fundamentals while building a real trading prediction system.

**Status**: ✅ Fully functional | **Version**: 2.5.2 | **Platform**: Windows/Linux/Mac | **Framework**: scikit-learn + FastAPI

> **Note**: The latest version (V2.5.2) is in the `v2.5/` folder. The V2 version is in `src/v2/`. See [CHANGELOG.md](CHANGELOG.md) for version history.

> **V3.0.2 Progress (Phase 0)**: Leakage-gated shared pipeline work has started in `v3.0/`.

---

## 🎯 Quick Overview

| Aspect | Details |
|--------|---------|
| **Target** | QQQ (Invesco QQQ Trust) |
| **Prediction** | 4-class: UP, DOWN, UP_DOWN, SIDEWAYS |
| **Horizons** | 3, 5, 10, 15, 20, 30 days ahead (configurable) |
| **Thresholds** | 0.75%, 1%, 1.5%, 2.5%, 5% price movement |
| **Models** | 7 ensemble (LR, RF, GB, XGBoost, CatBoost, SVM, NB) |
| **Features** | 64 technical indicators + market regime detection |
| **Speed** | <100ms per prediction |

### ✨ Key Features

- **🤖 7 Ensemble Models**: XGBoost (best), CatBoost, Gradient Boosting, Random Forest, Logistic Regression, SVM, Naive Bayes
- **📊 64 Technical Features**: MA, RSI, MACD, ATR, Bollinger Bands, Trend, Regime, SPY correlation
- **⚡ Real-time API**: FastAPI server with automatic data fetching from Yahoo Finance
- **🔮 Multiple Horizons**: Predict 3, 5, 10, 15, 20, 30 days ahead simultaneously
- **📊 Backtesting**: Historical validation with PNG charts and CSV results (configurable period)
- **🔍 Model Transparency**: Responses include `model_used` field and full class probabilities
- **🎓 Complete Documentation**: Architecture, API guide, troubleshooting
- **🧪 Model Persistence**: Pre-trained models available with feature names
- **📈 Market Regime Detection**: Track bull/bear/sideways + volatility states
- **☁️ Cloud Ready**: Can deploy to AWS SageMaker
- **📝 Date/Time Logging**: Separate logs for training, prediction, API

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[v2.5/README.md](v2.5/README.md)** | Latest version (V2.5.2) - quick start |
| **[v2.5/docs/README.md](v2.5/docs/README.md)** | V2.5 docs index |
| **[v2.5/docs/API_REFERENCE.md](v2.5/docs/API_REFERENCE.md)** | Complete API reference with response explanations |
| **[v2.5/docs/ARCHITECTURE.md](v2.5/docs/ARCHITECTURE.md)** | System design & data flow |
| **[v2.5/docs/API_GUIDE.md](v2.5/docs/API_GUIDE.md)** | API usage guide |
| **[v3.0/IMPROVEMENT_PLAN_V3.0.2.md](v3.0/IMPROVEMENT_PLAN_V3.0.2.md)** | V3.0 execution plan baseline |
| **[v3.0/LEAKAGE_AUDIT_REPORT.md](v3.0/LEAKAGE_AUDIT_REPORT.md)** | Phase 0 leakage controls and gate status |
| **[v3.0/docs/VALIDATION_PROTOCOL.md](v3.0/docs/VALIDATION_PROTOCOL.md)** | V3.0 Phase 0 validation protocol |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history |

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Start API Server (V2.5)
**IMPORTANT**: All commands below must be run from the `v2.5/` folder.

```bash
cd v2.5
python -m uvicorn src.inference_v2_5:app --reload --host 0.0.0.0 --port 8000
```

### 3️⃣ Make Your First Prediction

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "QQQ",
        "horizon": 20,
        "threshold": 0.01
    }
)
print(response.json())
```

**Using curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ", "horizon": 20, "threshold": 0.01}'
```

**See Live API Docs**: http://localhost:8000/docs

---

## 📁 Project Structure

```
StockPredictor/
├── v2.5/                 # Current version (2.5.2) [USE THIS]
│   ├── src/                        # Source code
│   │   ├── config_v2_5.py          # Configuration
│   │   ├── data_preparation_v2_5.py # Data preparation
│   │   ├── inference_v2_5.py       # API server
│   │   ├── train_v2_5.py          # Training script
│   │   ├── logging_utils.py        # Logging utilities
│   │   ├── models_v2/              # 7 ML models
│   │   └── regime_v2/              # Market regime detection
│   ├── tests/                      # Test files
│   ├── docs/                       # V2.5 documentation
│   │   └── API_GUIDE.md           # API endpoints
│   ├── data/                       # Data files
│   └── models/                     # Trained models
│
├── src/v2/                         # Legacy version (2.0)
├── archive/                        # Old versions (v1, v1.5)
├── docs/                           # Documentation
├── CHANGELOG.md                   # Version history
├── README.md                       # This file
└── README_cn.md                    # Chinese version
```


---

## 🧠 Model Architecture

### 7 Base Models (V2.5)

| Model | Strengths | Use Case |
|-------|-----------|----------|
| **Logistic Regression** | Interpretable, simple baseline | Simple patterns |
| **Random Forest** | Handles non-linear, robust, fast | General purpose |
| **Gradient Boosting** | Powerful | Main predictor |
| **XGBoost** | High performance, gradient boosting | High accuracy |
| **CatBoost** | Handles categorical features | Categorical data |
| **SVM (RBF)** | Complex decision boundaries | Non-linear patterns |
| **Naive Bayes** | Very fast | Real-time prediction |

### Ensemble Strategy (V2.5)

Models combined via **weighted voting**:
- **Random Forest**: 20% weight
- **Gradient Boosting**: 20% weight
- **XGBoost**: 20% weight
- **Logistic Regression**: 15% weight
- **CatBoost**: 15% weight
- **SVM**: 5% weight
- **Naive Bayes**: 5% weight

---

## 📊 4-Class Classification (V2.5)

V2.5 introduces 4-class classification:

| Class | Condition |
|-------|----------|
| **UP** | Max gain > threshold, max loss ≤ threshold |
| **DOWN** | Max loss > threshold, max gain ≥ threshold |
| **UP_DOWN** | Both max gain AND max loss > threshold |
| **SIDEWAYS** | Neither exceeds threshold |

Example: 5-day horizon, 1% threshold:
- Price goes up >1% but never down >1%: **UP**
- Price goes down >1% but never up >1%: **DOWN**
- Price goes up >1% AND down >1%: **UP_DOWN**
- Price stays within ±1%: **SIDEWAYS**

---

## 📈 V2.5 Model Performance (February 2026)

### Performance Summary

| Horizon | Threshold | Best Model | Accuracy | AUC-ROC |
|---------|-----------|------------|----------|---------|
| 5d | 1% | XGBoost | 61.19% | 0.820 |
| 5d | 2.5% | XGBoost | 80.97% | 0.913 |
| 10d | 1% | XGBoost | 82.02% | 0.885 |
| 10d | 2.5% | XGBoost | 79.78% | 0.943 |
| 20d | 2.5% | XGBoost | 92.83% | 0.993 |
| 30d | 2.5% | XGBoost | 97.34% | 0.997 |

### Key Improvements (V2.5.1 vs V2.5.0)

- **XGBoost outperforms RandomForest** by 7-22% accuracy
- **10d/1% now viable** - 82% accuracy (was 60%)
- **20d/2.5% improved** - 93% accuracy (was 77-79%)
- **SMOTE handling** - balanced class distribution during training
- **64 features** - added regime detection features

### Recommendations

1. Use **XGBoost** as default model (best performer)
2. Use **2.5% threshold** for best balance of difficulty and accuracy
3. Avoid 1% threshold with short horizons (5d) - inherently hard
4. For risk-averse strategies: use 20d/5% threshold (>98% accuracy)

## 🔄 V2.5.2 Enhancements (February 2026)

### New Features

- **Backtesting API**: `/backtest` endpoint with configurable period, daily predictions, PNG charts, CSV results
- **Model Transparency**: `model_used` field in all prediction responses
- **Multi-Class Handling**: Gracefully handles models with fewer than 4 classes (fills missing with 0 probability)
- **Backtest Module**: `src/backtest_v2_5.py` with full historical analysis

### API Changes

- POST `/predict` and POST `/predict/multi` now return `model_used` in each prediction
- GET `/backtest/{symbol}` and POST `/backtest` added
- All endpoints support TQQQ and any other symbol

---

## 🚀 Training & Running V2.5

**IMPORTANT**: All commands below must be run from the `v2.5/` folder.

### Train Models
```bash
cd v2.5
python src/train_v2_5.py
```

### Start API
```bash
cd v2.5
python -m uvicorn src.inference_v2_5:app --reload --port 8000
```

### Make Prediction
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "QQQ",
        "horizon": 20,
        "threshold": 0.01
    }
)
print(response.json())
```

---

## 📚 Version History

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

| Version | Method | Status |
|---------|--------|--------|
| **2.5** | 4-class classification | ✅ Current (use v2.5/) |
| **2.0** | Binary classification | Legacy (src/v2/) |
| **1.5** | Walk-Forward validation | Archive |
| **1.0** | Regression | Archive |

### Market Regime Detection

**MA Crossover State** (50/200-day moving averages):
- 🟢 **Bull**: Price > MA50 > MA200
- 🔴 **Bear**: Price < MA50 < MA200
- 🟠 **Sideways**: Other conditions

**Volatility State**:
- **High**: Daily volatility > 2%
- **Normal**: 1% - 2%
- **Low**: < 1%

---

## 📊 Performance Results

### Best Horizon: 20-Day Predictions

| Model | Accuracy | ROC-AUC | F1 Score | Precision | Recall |
|-------|----------|---------|----------|-----------|--------|
| **Gradient Boosting** | 88.7% | 0.954 | 0.918 | 0.92 | 0.92 |
| **Random Forest** | 88.5% | 0.945 | 0.911 | 0.90 | 0.92 |
| **Logistic Regression** | 62.5% | 0.512 | 0.760 | 0.75 | 0.77 |
| **SVM** | 61.9% | 0.508 | 0.755 | 0.74 | 0.77 |
| **Naive Bayes** | 55.3% | 0.485 | 0.715 | 0.72 | 0.71 |
| **Ensemble** | 64.8% | 0.578 | 0.775 | 0.79 | 0.76 |

### Horizon Comparison (Ensemble)

| Horizon | Accuracy | ROC-AUC | F1 Score | Notes |
|---------|----------|---------|----------|-------|
| 5-day | 58.2% | 0.536 | 0.726 | Too short, noisy |
| 10-day | 61.7% | 0.541 | 0.750 | Moderate |
| **20-day** | 64.8% | 0.578 | 0.775 | **BEST - Recommended** |
| 30-day | 52.1% | 0.485 | 0.695 | Too long, weak signal |

**Baseline**: 50% (random guessing)  
**Note**: Gradient Boosting single model performs better than ensemble - consider adjusting voting weights

### Top Features by Importance

1. `trend_strength` - (MA50 - MA200) / MA200
2. `distance_ma200` - Distance to 200-day MA (%)
3. `volatility` - Historical volatility (20-day)
4. `correlation_spy_20d` - Correlation with SPY index
5. `rsi` - Relative Strength Index
6. `macd_signal` - MACD Signal line

---

## 💻 Usage Examples

### API Usage (Recommended)

**1. Start the server:**
```bash
cd v2.5
python -m uvicorn src.inference_v2_5:app --reload --port 8000
```

**2. Simple prediction:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "QQQ",
        "date": "2026-02-25",
        "horizon": 20,
        "threshold": 0.01
    }
)
print(response.json())
```

**3. Multi-horizon prediction:**
```python
response = requests.post(
    "http://localhost:8000/predict/multi",
    json={
        "symbol": "QQQ",
        "horizons": [5, 10, 20, 30],
        "thresholds": [0.01, 0.025, 0.05]
    }
)
```

**4. Backtesting (GET example):**
```bash
curl "http://localhost:8000/backtest/TQQQ?horizon=20&threshold=0.01&days_back=180"
```

**5. Backtesting (POST example):**
```python
response = requests.post(
    "http://localhost:8000/backtest",
    json={
        "symbol": "TQQQ",
        "horizon": 20,
        "threshold": 0.01,
        "days_back": 180
    }
)
print(response.json())
```

**6. View API documentation:**
Open `http://localhost:8000/docs` in browser for interactive docs (includes backtest endpoints)
---

**3. Advanced prediction (POST):**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "current": {
            "date": "2026-02-25",
            "open": 520.0,
            "high": 525.0,
            "low": 518.0,
            "close": 522.0,
            "volume": 50000000
        },
        "horizon": 20
    }
)

result = response.json()
print(f"Prediction: {result['prediction']}")  # 'UP' or 'DOWN'
print(f"Probability: {result['probability_up']:.1%}")
```

**4. View API documentation:**
Open `http://localhost:8000/docs` in browser for interactive docs

### Python Script Usage

```python
import sys
sys.path.insert(0, 'src/v2')

from data_preparation_v2 import prepare_data
import joblib

# 1. Prepare data
X, y, feature_names, df = prepare_data(horizon=20)

# 2. Load pre-trained model (Gradient Boosting - best)
model = joblib.load('models/results/v2/gradientboosting_model.pkl')

# 3. Predict
prediction = model.predict(X[-10:])  # Last 10 samples
probability = model.predict_proba(X[-10:])

print(f"Predictions: {prediction}")  # [0, 1, 0, ...]
print(f"Up Probabilities: {probability[:, 1]}")  # [0.45, 0.92, ...]
```

### Loading Pre-trained Models

```python
import joblib

# Load best model (Gradient Boosting - 88.7% accuracy)
model = joblib.load('models/results/v2/gradientboosting_model.pkl')

# Load feature names
with open('models/results/v2/feature_names.txt') as f:
    feature_names = [line.strip() for line in f]

# Feature count is 47
print(f"Total features: {len(feature_names)}")
print(f"First 5 features: {feature_names[:5]}")

# Make predictions
import numpy as np
X_new = np.random.randn(10, 47)  # 10 samples, 47 features
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

---

## 🎓 Training Your Own Model

### Full Training Pipeline

```bash
# 1. Navigate to project root
cd StockPredictor

# 2. Run training (downloads data, engineers features, trains 5 models + ensemble)
python src/v2/train_v2.py

# 3. Check results
cat models/results/v2/results.txt
```

**What training does:**
- Downloads 5 years of QQQ and SPY daily data from Yahoo Finance
- Engineers 47 technical indicators (MA, RSI, MACD, ATR, Bollinger Bands, etc.)
- Creates classification labels (UP=1 if price rises in N days, DOWN=0)
- Trains 5 base models (Logistic, RF, GB, SVM, NB)
- Trains ensemble model with weighted voting
- Evaluates on test set with multiple metrics
- Saves results and pre-trained models

### Configuration

Edit `src/v2/config_v2.py`:

```python
# Stock & data
SYMBOL = "QQQ"              # Stock symbol to predict
TRAIN_YEARS = 5             # Years of historical data

# Prediction horizons
HORIZONS = [5, 10, 20]      # Predict for these many days
DEFAULT_HORIZON = 20        # Default when not specified

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 20,
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
    },
    # ...
}

# Ensemble weights (sum should = 1.0)
ENSEMBLE_WEIGHTS = {
    'GradientBoosting': 0.25,
    'RandomForest': 0.30,
    'LogisticRegression': 0.20,
    'SVM': 0.15,
    'NaiveBayes': 0.10,
}
```

Then retrain:
```bash
python src/v2/train_v2.py
```

---

## ☁️ Deployment

### Option 1: Local FastAPI Server (Recommended for Development)

```bash
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

Access at: http://localhost:8000  
API Docs at: http://localhost:8000/docs

### Option 2: Production FastAPI (with Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 src.v2.inference_v2:app
```

### Option 3: AWS SageMaker (Production Scale)

**Prepare AWS environment:**
```bash
pip install sagemaker boto3
aws configure  # Set AWS credentials
```

**Deploy model:**
```bash
python docs/v2/train_deploy_sagemaker_v2.py --mode deploy --endpoint stock-predictor-v2
```

**Predict with SageMaker:**
```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='stock-predictor-v2',
    ContentType='application/json',
    Body=json.dumps({"features": [0.1, 0.05, -0.02, ...]})
)
result = json.loads(response['Body'].read().decode())
print(f"Prediction: {result['prediction']}")
```

### Option 4: Docker Container

**Create Dockerfile:**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ src/
COPY models/ models/
CMD ["python", "-m", "uvicorn", "src.v2.inference_v2:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t stock-predictor .
docker run -p 8000:8000 stock-predictor
```

---

## ⚙️ Configuration & Tuning

### Prediction Horizons

Adjust in `src/v2/config_v2.py`:
```python
HORIZONS = [5, 10, 20]     # Predict for how many days
DEFAULT_HORIZON = 20        # Default for API
```

**Recommendations**:
- **5-day**: Quick trades, high noise (58% acc) - risky
- **10-day**: Medium term (62% acc) - moderate
- **20-day**: Sweet spot (65% acc) - ← **RECOMMENDED**
- **30+ days**: Weak signal (50% acc) - avoid

### Feature Engineering

Core features in `src/v2/data_preparation_v2.py`:
- **Trend**: Moving averages (10, 20, 50, 200-day)
- **Momentum**: RSI, MACD, Stochastic Oscillator
- **Volatility**: ATR, Bollinger Bands
- **Correlation**: With SPY, VIX if available
- **Market Regime**: MA crossover, volatility state
- **Price Patterns**: Gap, reversals, support/resistance

Modify to add/remove features and retrain.

### Model Hyperparameters

Adjust in `src/v2/config_v2.py` MODEL_PARAMS:
```python
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,    # More trees = better but slower
        'max_depth': 5,         # Limit depth to prevent overfitting
        'min_samples_split': 20,
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,   # Smaller = more accurate but slower
        'max_depth': 3,
    },
    # ... other models
}
```

Then retrain: `python src/v2/train_v2.py`

---

## 📞 Troubleshooting

### API Won't Start

**Error**: `Connection refused` or port already in use
```bash
# Check what's using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Mac/Linux

# Use different port
python -m uvicorn src.v2.inference_v2:app --host 0.0.0.0 --port 8001
```

### ModuleNotFoundError

**Error**: `No module named 'src'` or similar
```bash
# Make sure you're in project root
cd StockPredictor

# Run from project root (not subdirectory)
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

### Models Not Found

**Error**: `FileNotFoundError: models/results/v2/...`
```bash
# Train models first
python src/v2/train_v2.py

# Verify they exist
ls models/results/v2/
```

### Data Download Issues

**Error**: `Can't download data from Yahoo Finance`
```bash
# Check internet connection
# Verify cache exists
ls data/cache/

# Try manually downloading
python -c "import yfinance as yf; yf.download('QQQ', start='2020-01-01', end='2025-12-31').to_csv('data/cache/qqq.csv')"
```

### Requirements Not Installed

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

More troubleshooting: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 📝 Project Details

| Item | Details |
|------|---------|
| **Language** | Python 3.8+ |
| **Core Libraries** | scikit-learn, pandas, numpy, yfinance |
| **API Framework** | FastAPI + uvicorn |
| **ML Models** | 5 ensemble (classification) |
| **Prediction Target** | QQQ price direction (UP/DOWN) |
| **Data Source** | Yahoo Finance (free) |
| **Training Data** | 5 years of daily OHLCV |
| **Features** | 47 technical indicators |
| **Status** | Production ready ✅ |

---

## 📚 Version History

| Version | Approach | Status | Notes |
|---------|----------|--------|-------|
| **V2** | Classification + Ensemble | ✅ Active | Current - **Use this!** Best results |
| **V1.5** | Walk-Forward Validation | ⚠️ Experimental | Research & optimization only |
| **V1** | Regression (continuous) | 📚 Legacy | Learning reference - don't use for trading |

---

## ❓ FAQ

**Q: Can I predict other stocks besides QQQ?**  
A: Yes! Change `SYMBOL = "SPY"` in `src/v2/config_v2.py` and retrain. Works with any stock symbol from Yahoo Finance.

**Q: Is 65% accuracy enough to trade profitably?**  
A: Carefully. After commissions, slippage, and market impact, your actual returns may be lower. Use as signal confirmation, not sole decision.

**Q: How often should I retrain?**  
A: Monthly recommended, or when accuracy drops visibly. Market conditions change constantly.

**Q: What's the minimum data needed?**  
A: At least 200 trading days (~1 year). More is better. Current system uses 5 years for robust training.

**Q: Can I run this on GPU?**  
A: scikit-learn uses CPU by default. For GPU, consider XGBoost-GPU or PyTorch implementations.

**Q: What if I want to add more features?**  
A: Edit `src/v2/data_preparation_v2.py` to add technical indicators, then retrain models.

**Q: Can I use this for options trading?**  
A: Yes, but be aware: implied volatility, time decay, and other factors matter. Use as directional guide only.

**Q: What exchange/timezone does the data use?**  
A: Yahoo Finance uses NYSE hours (EST). Predictions use previous day's close.

---

## 🎯 Recommended Next Steps

1. **Read** [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - detailed quick start guide
2. **Understand** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - system design and data flow
3. **Learn** [docs/V2_CLASSIFICATION.md](docs/V2_CLASSIFICATION.md) - ML approach details
4. **Try** [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - all API endpoints
5. **Troubleshoot** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - when stuck

---

## 💡 Tips for Better Results

1. **Use 20-day horizon** - sweet spot between accuracy and prediction window
2. **Monitor market regime** - predictions more reliable in trending markets
3. **Combine with technical analysis** - don't rely solely on ML model
4. **Backtest before trading** - validate strategy on historical data
5. **Retrain regularly** - market conditions change, model needs updates
6. **Use ensemble prediction** - combine multiple signals for robustness
7. **Check correlation with SPY** - QQQ highly correlated with market
8. **Manage position size** - 65% accuracy doesn't mean guaranteed profits

---

## 📄 License & Disclaimer

Educational project - Free to use for learning and research purposes.

**DISCLAIMER**: This model is for educational purposes only. It is NOT investment advice and does NOT guarantee profits. Trading and investing involve significant risk of loss. Always do your own research and consult a qualified financial advisor before making investment decisions.

---

## 🤝 Contributing

Found a bug or have an improvement? Let me know!

---

**Ready to get started?** 🚀

1. **Install**: `pip install -r requirements.txt`
2. **Train**: `python src/v2/train_v2.py` (5 mins on CPU)
3. **Predict**: `python -m uvicorn src.v2.inference_v2:app --reload --port 8000`
4. **Visit**: http://localhost:8000/docs

**Questions?** Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) or review code comments.

Happy predicting! 📊
