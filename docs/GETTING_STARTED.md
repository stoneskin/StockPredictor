# 🚀 Getting Started with Stock Predictor

A comprehensive machine learning project for predicting QQQ stock price movements using ensemble models and technical analysis.

> **Target Audience**: Beginners and students learning ML + stock market prediction  
> **Complexity**: Beginner to Intermediate  
> **Time to First Prediction**: ~5 minutes

---

## ⚡ Quick Start (5 Minutes)

### 1. Clone & Setup

```bash
# Navigate to project
cd StockPredictor

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Prediction API

```bash
# Run the prediction server
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` to see the interactive API documentation.

### 3.Make Your First Prediction

**Option A: Using curl (Terminal)**
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'
```

**Option B: Using Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/simple",
    json={"symbol": "QQQ"}
)
print(response.json())
```

### 4. Review Results

You'll get predictions for multiple time horizons (5, 10, 20, 30 days):

```json
{
  "symbol": "QQQ",
  "date": "2026-02-24",
  "predictions": [
    {
      "horizon": 5,
      "prediction": "UP",
      "probability_up": 0.62,
      "probability_down": 0.38,
      "confidence": 0.62
    },
    {
      "horizon": 10,
      "prediction": "DOWN",
      "probability_up": 0.45,
      "probability_down": 0.55,
      "confidence": 0.55
    }
    // ... more horizons
  ]
}
```

---

## 📚 Project Overview

### What is This Project?

**Stock Predictor** is a machine learning system that predicts whether the QQQ stock will go up or down in the next 5, 10, 20, or 30 days.

**Key Features:**
- 📊 Uses 5 different ML models (Logistic Regression, Random Forest, SVM, Gradient Boosting, Naive Bayes)
- 🎯 Ensemble voting combines predictions for better accuracy
- 📈 Detects market conditions (bull, bear, sideways)
- 🔄 Multiple prediction horizons (5/10/20/30 days)
- 🚀 FastAPI server for easy integration

### Project Architecture

```
StockPredictor/
├── src/
│   ├── v2/                  # Main prediction system (you'll use this)
│   │   ├── inference_v2.py  # FastAPI server - START HERE
│   │   ├── train_v2.py      # Model training script
│   │   ├── config_v2.py     # Configuration
│   │   ├── models_v2/       # Base models
│   │   │   ├── logistic_model.py
│   │   │   ├── random_forest_model.py
│   │   │   ├── gradient_boosting_model.py
│   │   │   ├── svm_model.py
│   │   │   ├── naive_bayes_model.py
│   │   │   └── ensemble.py
│   │   ├── regime_v2/       # Market detection
│   │   │   ├── detector.py
│   │   │   └── ma_crossover.py
│   │   └── walk_forward/    # Validation
│   │       ├── validation.py
│   │       ├── feature_selector.py
│   │       └── trainer.py
│   └── v1/                  # Legacy regression version (reference)
├── data/
│   ├── raw/                 # Downloaded stock data
│   └── processed/           # Prepared training data
├── models/
│   └── results/v2/          # Trained models & results
├── tests/                   # Test scripts
└── docs/                    # Documentation
```

---

## 🎯 How the Prediction Works

### Step 1: Download Data
- Fetches historical QQQ and SPY data from Yahoo Finance
- Stores locally for reproducibility

### Step 2: Calculate Technical Indicators
The system computes 25+ technical indicators:
- Moving Averages (5, 10, 20, 50, 200 day)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Bollinger Bands
- ATR (Average True Range)
- And many more...

### Step 3: Create Labels
- Determines if the price went UP or DOWN in the next N days (5/10/20/30)

### Step 4: Train Multiple Models
- **Logistic Regression**: Linear model, interpretable
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential tree building
- **SVM**: Support Vector Machine with RBF kernel
- **Naive Bayes**: Probabilistic classifier

### Step 5: Ensemble Voting
Each model votes with weights:
- Random Forest: 30%
- Gradient Boosting: 25%
- Logistic Regression: 20%
- SVM: 15%
- Naive Bayes: 10%

Final prediction = weighted average of all votes

---

## 📖 Learning Path

### For Complete Beginners:
1. **Read**: [GETTING_STARTED.md](GETTING_STARTED.md) ← You are here
2. **Run**: The API and make predictions
3. **Read**: [ARCHITECTURE.md](ARCHITECTURE.md) - understand how it works
4. **Explore**: [API_REFERENCE.md](API_REFERENCE.md) - API endpoints
5. **Look at**: Code in `src/v2/` - comment by comment

### For ML Students:
1. **Read**: [ARCHITECTURE.md](ARCHITECTURE.md) - system design
2. **Review**: [V2_CLASSIFICATION.md](V2_CLASSIFICATION.md) - detailed approach
3. **Study**: Code in `src/v2/models_v2/` - model implementations
4. **Run**: Training: `python src/v2/train_v2.py`
5. **Analyze**: Results in `models/results/v2/`

### For Production:
1. **Setup**: Configure in `src/v2/config_v2.py`
2. **Deploy**: Run API server: `python -m uvicorn src.v2.inference_v2:app`
3. **Monitor**: Check logs and predictions
4. **Retrain**: Run `python src/v2/train_v2.py` monthly

---

## 🔧 Common Tasks

### Make Predictions via Terminal
```bash
# Simple prediction with default settings
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'

# Custom prediction with specific horizons
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "TSLA",
    "horizons": [5, 10],
    "min_history": 200
  }'
```

### Make Predictions via Python
```python
import requests

# Simple prediction
response = requests.post(
    "http://localhost:8000/predict/simple",
    json={"symbol": "QQQ"}
)
result = response.json()
print(f"5-day prediction: {result['predictions'][0]}")

# Full API call
response = requests.post(
    "http://localhost:8000/predict",
    json={"symbol": "TSLA", "horizons": [5, 10, 20]}
)
predictions = response.json()
for pred in predictions['predictions']:
    print(f"{pred['horizon']:2d}d: {pred['prediction']:4s} (confidence: {pred['confidence']:.2%})")
```

### Train Models with New Data
```bash
python src/v2/train_v2.py
```

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_api.py -v

# Or run individual test scripts
python tests/test_qqq_fix.py
```

### View API Documentation
After starting the server, visit: `http://localhost:8000/docs`

Interactive Swagger UI with:
- All endpoints listed
- Request/response examples
- Try it out buttons

---

## 🤔 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Solution**: Make sure you're in the project root directory:
```bash
cd StockPredictor  # Make sure you're here
python -m uvicorn src.v2.inference_v2:app
```

### "Connection refused" when calling API

**Solution**: Make sure the server is running:
```bash
# Terminal 1: Start the server
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Make requests
curl http://localhost:8000/predict/simple -X POST -H "Content-Type: application/json" -d '{"symbol": "QQQ"}'
```

### "No data found for symbol"

**Possible causes:**
- Invalid stock symbol (QQQ, TSLA, SPY are supported)
- Yahoo Finance is temporarily unavailable
- Internet connection issue

**Solution**: Check if symbol is correct and try again after a moment

### Missing trained models

**Solution**: Train models first:
```bash
python src/v2/train_v2.py
```

This creates models in `models/results/v2/`

---

## 📚 Next Steps

1. **Make your first prediction** (Done! ✓)
2. **Read ARCHITECTURE.md** to understand system design
3. **Review API_REFERENCE.md** to explore all endpoints
4. **Study the code** in `src/v2/` folder
5. **Modify and experiment** - change parameters in `src/v2/config_v2.py`
6. **Train your own models** with different hyperparameters
7. **Deploy to production** for live predictions

---

## 📞 Need Help?

- **Setup Issues**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Understanding Code**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Questions**: See [API_REFERENCE.md](API_REFERENCE.md)
- **Deep Learning**: See [V2_CLASSIFICATION.md](V2_CLASSIFICATION.md)

---

## 📄 License

This is an educational project for learning purposes.

Created for: Learning machine learning + stock market prediction  
Difficulty: Beginner to Intermediate  
Time Commitment: 2-4 weeks to understand entire project
