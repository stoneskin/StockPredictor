# Quick Start Guide - Enhanced Stock Prediction API

## TL;DR - Get Started in 30 Seconds

### 1. Start the API
```bash
python -m uvicorn src.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

### 2. Make a Prediction (Any Terminal/Python)
```bash
# Bash/PowerShell
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'
```

```python
# Python
import requests
response = requests.post(
    "http://localhost:8000/predict/simple",
    json={"symbol": "QQQ"}
)
print(response.json())
```

### 3. You Get Back
```json
{
  "symbol": "QQQ",
  "date": "2026-02-20",
  "predictions": [
    {
      "horizon": 5,
      "prediction": "DOWN",
      "probability_up": 0.35,
      "probability_down": 0.65,
      "confidence": 0.65
    },
    // ... more horizons (10, 20, 30 by default)
  ]
}
```

That's it! No need to:
- ❌ Manually download data
- ❌ Prepare 200+ days of history
- ❌ Format market data
- ❌ Handle dates manually

## Common Use Cases

### Predict QQQ (Simplest)
```json
{"symbol": "QQQ"}
```

### Predict Multiple Stocks
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'

curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY"}'
```

### Specific Date & Horizons
```json
{
  "symbol": "TSLA",
  "date": "2026-02-20",
  "horizons": [5, 10]
}
```

### Only Long-Term Forecast (20-30 days)
```json
{
  "symbol": "QQQ",
  "horizons": [20, 30]
}
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict/simple` | POST | **[NEW]** Simplified prediction - Recommended! |
| `/predict` | POST | Legacy prediction (requires manual data) |
| `/health` | GET | Check if API is running |
| `/model-info` | GET | Get model details |

## Response Fields Explained

```json
{
  "symbol": "QQQ",                    // The stock symbol
  "date": "2026-02-20",               // Date used for prediction
  "predictions": [
    {
      "horizon": 5,                   // Days into the future
      "prediction": "UP",             // UP or DOWN
      "probability_up": 0.65,         // 0-1 (0% to 100%)
      "probability_down": 0.35,       // Always sums to 1.0
      "confidence": 0.65              // How sure (max of two probs)
    }
  ]
}
```

## Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Model not loaded" | Server just started | Wait a moment and retry |
| "Invalid date format" | Wrong date format | Use YYYY-MM-DD format |
| "Insufficient data" | Stock not in database | Yahoo Finance doesn't have enough history |
| Connection refused | Server not running | Run: `python -m uvicorn src.inference_v2:app --reload` |

## Data Handling

**First Time:**
1. API checks local file `data/raw/QQQ.csv`
2. File doesn't exist → Fetches from Yahoo Finance (~2-3 seconds)
3. Saves to local file for next time

**Subsequent Times:**
1. Uses cached file → Instant lookup
2. If date is newer than cache → Auto-fetches just missing data

**Result:** First prediction takes ~2-3 seconds, next ones take <100ms!

## Default Behaviors

| Parameter | Default | Example Override |
|-----------|---------|-------------------|
| `symbol` | **Required** | `"QQQ"` |
| `date` | Latest available | `"2026-02-15"` |
| `horizons` | `[5,10,20,30]` | `[5,10]` |

## Example Workflow

```bash
# Start server (do once)
python -m uvicorn src.inference_v2:app --reload &

# Wait for startup...

# Test with different stocks
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" -d '{"symbol": "QQQ"}'

curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" -d '{"symbol": "SPY"}'

curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" -d '{"symbol": "AAPL"}'

# Test health
curl http://localhost:8000/health

# View model info
curl http://localhost:8000/model-info
```

## Python Integration

```python
import requests
import json

def get_prediction(symbol, date=None, horizons=None):
    """Simple wrapper for API"""
    payload = {"symbol": symbol}
    if date:
        payload["date"] = date
    if horizons:
        payload["horizons"] = horizons
    
    response = requests.post(
        "http://localhost:8000/predict/simple",
        json=payload
    )
    return response.json()

# Use it
result = get_prediction("QQQ")
for pred in result["predictions"]:
    print(f"{pred['horizon']}d: {pred['prediction']} ({pred['confidence']:.0%})")

# Output:
# 5d: DOWN (65%)
# 10d: UP (62%)
# 20d: UP (71%)
# 30d: UP (58%)
```

## Key Features

✨ **Completely Automatic** - No manual data handling needed  
⚡ **Fast** - Caches data locally after first use  
🔄 **Updates Automatically** - Fetches new data when needed  
🎯 **Multiple Predictions** - Get forecasts for several timeframes at once  
📊 **Model Features:**
  - 47 technical indicators
  - Market regime detection
  - SPY correlation analysis
  - Ensemble trained on years of data
  - ~58% ROC-AUC performance

## Docs & Files

- 📖 **`API_GUIDE.md`** - Complete API documentation  
- 📝 **`ENHANCEMENT_SUMMARY.md`** - What was added  
- 🧪 **`test_api.py`** - Test script that shows usage  
- 📦 **`src/inference_v2.py`** - Main API implementation  

## Support & Troubleshooting

Check `API_GUIDE.md` for:
- Detailed troubleshooting guide
- Python client examples
- Data management details
- Advanced usage patterns
- Error codes and solutions

## That's It!

You now have a powerful, easy-to-use stock prediction API that:
- Automatically manages data
- Returns multiple time horizon forecasts
- Requires just a stock symbol to work

Enjoy! 🚀
