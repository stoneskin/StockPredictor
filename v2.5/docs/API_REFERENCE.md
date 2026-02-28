# 📡 API Reference - V2.5.2

Complete documentation of all API endpoints for the Stock Predictor V2.5.2 service.

**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)  
**Version**: 2.5.2

---

## Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Get API info |
| `/predict` | POST | Single prediction (recommended) |
| `/predict/multi` | POST | Multiple horizons/thresholds |
| `/backtest` | POST | Run backtest analysis |
| `/backtest/{symbol}` | GET | Run backtest analysis (GET) |
| `/health` | GET | Health check |
| `/model-info` | GET | Model information |

---

## 1. GET `/`

Get basic API information.

### Request
```bash
curl http://localhost:8000/
```

### Response
```json
{
  "message": "Stock Prediction API V2.5",
  "version": "2.5.1",
  "description": "4-class classification: UP, DOWN, UP_DOWN, SIDEWAYS",
  "endpoints": ["/predict", "/predict/multi", "/health", "/model-info"]
}
```

---

## 2. GET `/health`

Check if the API is running and models are loaded.

### Request
```bash
curl http://localhost:8000/health
```

### Response (Success)
```json
{
  "status": "healthy",
  "models_loaded": 12,
  "message": "API is running and ready for predictions"
}
```

---

## 3. POST `/predict`

**Recommended endpoint** - Get prediction for a single horizon and threshold.

### Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ", "horizon": 20, "threshold": 0.025}'
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| symbol | string | Yes | - | Stock symbol (e.g., "QQQ", "SPY") |
| horizon | int | No | 20 | Prediction horizon in days (5, 10, 20, 30) |
| threshold | float | No | 0.01 | Price movement threshold (0.0075=0.75%, 0.01=1%, 0.015=1.5%, 0.025=2.5%, 0.05=5%) |
| date | string | No | latest | Date for prediction (YYYY-MM-DD) |

### Response

```json
{
  "symbol": "QQQ",
  "date": "2026-02-27",
  "predictions": [
    {
      "horizon": 20,
      "threshold": 0.025,
      "prediction": "SIDEWAYS",
      "probabilities": {
        "UP": 0.0066,
        "DOWN": 0.0878,
        "UP_DOWN": 0.0377,
        "SIDEWAYS": 0.8679
      },
      "confidence": 0.8679
    }
  ]
}
```

### Response Explanation

| Field | Type | Description |
|-------|------|-------------|
| symbol | string | The stock symbol you requested |
| date | string | Date of prediction (or latest available) |
| predictions[].horizon | int | Prediction horizon in days |
| predictions[].threshold | float | Threshold used for classification |
| **prediction** | string | **Predicted class** - UP, DOWN, UP_DOWN, or SIDEWAYS |
| **probabilities** | dict | **Class probabilities** - shows likelihood of each class |
| **confidence** | float | **Highest probability** - confidence in the prediction |

#### Class Probabilities Explained

- **UP** (0.66%): Probability that price will go up more than threshold without going down more than threshold
- **DOWN** (8.78%): Probability that price will go down more than threshold without going up more than threshold  
- **UP_DOWN** (3.77%): Probability that price will BOTH go up AND down more than threshold (volatile)
- **SIDEWAYS** (86.79%): Probability that price will stay within ±threshold (no significant movement)

#### How to Use Probabilities

```python
# Example: Get probability of significant movement (not SIDEWAYS)
prob_significant_movement = 1 - response['predictions'][0]['probabilities']['SIDEWAYS']
# In this case: 1 - 0.8679 = 0.1321 (13.21% chance of significant movement)

# Example: Get probability of UP movement
prob_up = response['predictions'][0]['probabilities']['UP']
```

---

## 4. POST `/predict/multi`

Get predictions for multiple horizons and thresholds in one request.

### Request
```bash
curl -X POST http://localhost:8000/predict/multi \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "QQQ",
    "horizons": [5, 10, 20, 30],
    "thresholds": [0.01, 0.025, 0.05]
  }'
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| symbol | string | Yes | - | Stock symbol |
| horizons | array | No | [5, 10, 20, 30] | List of horizons |
| thresholds | array | No | [0.01, 0.025, 0.05] | List of thresholds |
| date | string | No | latest | Date for prediction |

### Response

```json
{
  "symbol": "QQQ",
  "date": "2026-02-27",
  "predictions": [
    {"horizon": 5, "threshold": 0.01, "prediction": "UP", ...},
    {"horizon": 10, "threshold": 0.01, "prediction": "SIDEWAYS", ...},
    {"horizon": 20, "threshold": 0.01, "prediction": "SIDEWAYS", ...},
    ...
   ]
}
```

---

## 5. POST `/backtest`

Run comprehensive backtest for a symbol over a historical period.

### Request
```bash
curl -X POST "http://localhost:8000/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "TQQQ",
    "horizon": 20,
    "threshold": 0.01,
    "days_back": 180,
    "end_date": null
  }'
```

### Request Parameters (JSON body)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| symbol | string | Yes | - | Stock symbol (e.g., "TQQQ", "QQQ") |
| horizon | int | No | 20 | Prediction horizon in days (3, 5, 10, 15, 20, 30) |
| threshold | float | No | 0.01 | Threshold (0.0075, 0.01, 0.015, 0.025, 0.05) |
| days_back | int | No | 180 | Number of trading days to backtest |
| end_date | string | No | today | End date (YYYY-MM-DD) |

### Response
```json
{
  "symbol": "TQQQ",
  "horizon": 20,
  "threshold": 0.01,
  "days_back": 180,
  "summary": {
    "total_predictions": 41,
    "correct_predictions": 37,
    "accuracy_pct": 90.24,
    "by_class": {
      "UP": {"count": 5, "correct": 4, "accuracy_pct": 80.0},
      "DOWN": {"count": 2, "correct": 1, "accuracy_pct": 50.0},
      "UP_DOWN": {"count": 34, "correct": 32, "accuracy_pct": 94.12}
    },
    "by_prediction": {
      "UP_DOWN": {"count": 35, "actual_correct": 33, "precision_pct": 94.29}
    }
  },
  "chart_path": "v2.5/data/backtest/charts/backtest_TQQQ_20260228_111020.png",
  "results_file": "v2.5/data/backtest/backtest_TQQQ_h20_t10d_20260228_111020.csv"
}
```

### Response Explanation

| Field | Type | Description |
|-------|------|-------------|
| symbol | string | Symbol tested |
| horizon | int | Prediction horizon used |
| threshold | float | Threshold used |
| days_back | int | Number of days backtested |
| summary | dict | Summary statistics (accuracy, counts) |
| chart_path | string | Path to generated PNG chart |
| results_file | string | Path to CSV results file |

---

## 6. GET `/backtest/{symbol}`

GET version for convenience.

### Request
```bash
curl "http://localhost:8000/backtest/TQQQ?horizon=20&threshold=0.01&days_back=180"
```

Parameters same as POST (query parameters).

---

## 7. GET `/model-info`

Get information about loaded models.

### Request
```bash
curl http://localhost:8000/model-info
```

### Response
```json
{
  "version": "2.5.2",
  "models_loaded": 13,
  "available_horizons": [3, 5, 10, 15, 20, 30],
  "available_thresholds": [0.0075, 0.01, 0.015, 0.025, 0.05],
  "class_labels": ["UP", "DOWN", "UP_DOWN", "SIDEWAYS"],
  "features_count": 64,
  "best_model": "XGBoost"
}
```

---

## Understanding 4-Class Classification

### Class Definitions

| Class | Condition | Meaning |
|-------|-----------|---------|
| **UP** | max_gain > threshold AND max_loss ≤ threshold | Price goes UP more than threshold, but doesn't go down more than threshold |
| **DOWN** | max_loss > threshold AND max_gain ≥ threshold | Price goes DOWN more than threshold, but doesn't go up more than threshold |
| **UP_DOWN** | max_gain > threshold AND max_loss > threshold | Price goes both UP and DOWN more than threshold (volatile) |
| **SIDEWAYS** | max_gain ≤ threshold AND max_loss ≥ threshold | Price stays within ±threshold (no significant movement) |

### Example: 20-day horizon, 2.5% threshold

- **UP**: In the next 20 days, QQQ goes up >2.5% but never down >2.5%
- **DOWN**: In the next 20 days, QQQ goes down >2.5% but never up >2.5%
- **UP_DOWN**: In the next 20 days, QQQ goes up >2.5% AND down >2.5%
- **SIDEWAYS**: In the next 20 days, QQQ stays within ±2.5%

---

## Threshold Selection Guide

| Threshold | Use Case | Difficulty |
|-----------|----------|------------|
| 0.75% (0.0075) | Ultra-sensitive, early warning | Very hard |
| 1% (0.01) | Sensitive, short-term moves | Hard to predict |
| 1.5% (0.015) | Moderate sensitivity | Moderate |
| 2.5% (0.025) | Balanced - recommended default | Moderate |
| 5% (0.05) | Significant movements only, lower noise | Easy (mostly SIDEWAYS) |

---

## Horizon Selection Guide

| Horizon | Use Case | Typical Accuracy |
|---------|----------|----------|
| 3 days | Intraday/scalping | ~45-50% |
| 5 days | Short-term trading signals | ~60-80% |
| 10 days | Medium-term signals | ~80-85% |
| 15 days | Transition to medium-term | ~85-90% |
| 20 days | Default, balanced | ~90-95% |
| 30 days | Long-term trends | ~95-98% |

---

## Error Responses

### 400 Bad Request
```json
{"detail": "Invalid horizon. Must be one of [5, 10, 20, 30]"}
```

### 404 Not Found
```json
{"detail": "Model not found for horizon=20, threshold=0.025. Please train first."}
```

### 500 Internal Server Error
```json
{"detail": "Internal server error. Check logs for details."}
```

---

## Python Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "QQQ",
        "horizon": 20,
        "threshold": 0.025
    }
)

data = response.json()
pred = data['predictions'][0]

print(f"Prediction: {pred['prediction']}")
print(f"Confidence: {pred['confidence']:.1%}")
print(f"Probabilities: {pred['probabilities']}")

# Check if significant movement expected
if pred['probabilities']['SIDEWAYS'] < 0.5:
    print("⚠️ High chance of significant movement!")
else:
    print("✅ Likely to stay within threshold")
```

---

## Version History

- **2.5.2** (2026-02-28): Backtest API, model_used field, multi-class handling
- **2.5.1** (2026-02-28): Added response explanations, XGBoost as default
- **2.5.0** (2026-02-27): Initial 4-class classification release
