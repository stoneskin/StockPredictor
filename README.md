# 📈 Stock Predictor - ML for Trading

A complete machine learning system for predicting QQQ stock price movements. Learn ML fundamentals while building a real trading prediction system.

**Status**: ✅ Fully functional | **Version**: 2.0 | **Platform**: Windows/Linux/Mac

---

## 🎯 What You Get

```
Your Goal          Your Tool               Success Rate
─────────────────────────────────────────────────────────
Predict if QQQ     5 Ensemble Models      52-54% (vs 50% random)
will go UP/DOWN    + Smart Features       ✅ Better than guessing
in 5-30 days       + Fast API             Takes ~100ms per prediction
```

### ✨ Key Features

- **🤖 5 Ensemble Models**: Logistic Regression, Random Forest, SVM, Gradient Boosting, Naive Bayes
- **📊 25+ Technical Indicators**: All major indicators (MA, RSI, MACD, ATR, Bollinger Bands, etc.)
- **⚡ Real-time Predictions**: Fast API server with <100ms response time
- **🔮 Multiple Horizons**: Predict for 5, 10, 20, 30 days ahead
- **🎓 Well Documented**: Code comments + comprehensive guides
- **🧪 Easy Testing**: Sample test scripts included

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** | Quick start (read this first!) |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design |
| **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** | All API endpoints |
| **[docs/V2_CLASSIFICATION.md](docs/V2_CLASSIFICATION.md)** | Detailed approach |
| **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | Common issues |

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Install
```bash
pip install -r requirements.txt
```

### 2️⃣ Start API Server
```bash
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

### 3️⃣ Make Prediction
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'
```

**Result**: 🎉 You've made your first ML prediction!

---

## 📁 Project Structure

```
StockPredictor/
├── 📚 docs/                        # Documentation
│   ├── GETTING_STARTED.md          # First doc to read
│   ├── ARCHITECTURE.md             # How it works
│   ├── API_REFERENCE.md            # API guide
│   └── ...
├── 🧠 src/
│   ├── v2/                         # Main version
│   │   ├── inference_v2.py         # API server
│   │   ├── train_v2.py             # Training
│   │   ├── models_v2/              # 5 models
│   │   └── ...
│   └── v1/                         # Legacy version
├── 📊 data/                        # Data files
├── 🤖 models/                      # Trained models
├── ✅ tests/                        # Tests
└── 📋 requirements.txt
```

---

## 💻 Common Commands

**Make Prediction** (Python):
```python
import requests
response = requests.post(
    "http://localhost:8000/predict/simple",
    json={"symbol": "QQQ"}
)
print(response.json())
```

**Train Models**:
```bash
python src/v2/train_v2.py
```

**View API Docs**:
Visit `http://localhost:8000/docs`

---

## 📊 Performance

```
5-day:  54% accuracy  ✅ Useful
10-day: 52% accuracy  ⚠️  Marginal
20-day: 51% accuracy  ⚠️  Weak
30-day: 50% accuracy  ❌ Same as guessing
```

Baseline (random guessing): 50%

---

## 🎯 Next Steps

1. **Read [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** - detailed quickstart
2. **Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - how it works
3. **Try [docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - all endpoints
4. **Review [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - when stuck

---

## ⚙️ Configuration

Edit `src/v2/config_v2.py`:

```python
SYMBOL = "QQQ"                # Change stock
HORIZONS = [5, 10, 20, 30]   # Time horizons
TRAIN_YEARS = 5               # Years of data
```

Then run:
```bash
python src/v2/train_v2.py
```

---

## 📞 Troubleshooting

**"ModuleNotFoundError"** → Make sure you're in project root directory
```bash
cd StockPredictor
```

**"Connection refused"** → Start the server in another terminal
```bash
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

**"Models not found"** → Train first
```bash
python src/v2/train_v2.py
```

More help: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 📝 Project Details

| Item | Details |
|------|---------|
| Language | Python 3.8+ |
| Framework | scikit-learn, FastAPI |
| Data Source | Yahoo Finance |
| Prediction | QQQ UP/DOWN |
| Models | 5 ensemble |
| Status | Production ready ✅ |

---

## 📚 Versions

- **V2 (Current)** - Classification (UP/DOWN) - Use this! ✅
- **V1 (Legacy)** - Regression - Learning reference 📚

---

## ❓ FAQ

**Can I predict other stocks?**  
Yes! Change `SYMBOL` in `src/v2/config_v2.py`

**Can I use this to trade?**  
Carefully. 52% accuracy beats guessing but losses are still possible.

**How often to retrain?**  
Monthly recommended.

**Minimum data needed?**  
200 days (~1 year). More is better.

---

## 📄 License

Educational project - Free to use for learning.

---

**Ready? Start with [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** 🚀
