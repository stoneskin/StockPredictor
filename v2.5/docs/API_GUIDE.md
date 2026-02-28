# Stock Predictor V2.5.1 API Guide

## Overview

The V2.5.1 API provides stock price movement predictions using 4-class classification:
- **UP**: Price goes up more than threshold without going down more than threshold
- **DOWN**: Price goes down more than threshold without going up more than threshold
- **UP_DOWN**: Both up and down exceed threshold (volatile market)
- **SIDEWAYS**: Price stays within ±threshold (no significant movement)

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

---

## Endpoints

### Health Check

**GET** `/health`

Check if the API is running and get status information.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 12,
  "horizons": [5, 10, 20, 30],
  "thresholds": [0.01, 0.025, 0.05]
}
```

---

### Model Information

**GET** `/model-info`

Get information about loaded models.

**Response:**
```json
{
  "version": "2.5.1",
  "n_models": 12,
  "horizons": [5, 10, 20, 30],
  "thresholds": [0.01, 0.025, 0.05],
  "classes": ["UP", "DOWN", "UP_DOWN", "SIDEWAYS"],
  "n_features": 64,
  "best_model": "XGBoost"
}
```

---

### Single Prediction

**POST** `/predict`

Make a single prediction for a stock.

**Request Body:**
```json
{
  "symbol": "QQQ",
  "date": "2026-02-25",
  "horizon": 20,
  "threshold": 0.01
}
```

| Field | Type | Required | Description |
|-------|------|----------|--------------|
| symbol | string | Yes | Stock symbol (e.g., QQQ, AAPL, SPY) |
| date | string | No | Date in YYYY-MM-DD format. Defaults to latest |
| horizon | int | No | Prediction horizon in days (5, 10, 20, 30). Default: 20 |
| threshold | float | No | Threshold as decimal (0.01=1%, 0.025=2.5%, 0.05=5%). Default: 0.01 |

**Response:**
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

| Field | Description |
|-------|-------------|
| **prediction** | The predicted class (UP, DOWN, UP_DOWN, or SIDEWAYS) |
| **probabilities** | Probability for each class (sums to 1.0) |
| **confidence** | Highest probability (confidence in prediction) |

#### Understanding Probabilities

- **UP (0.66%)**: Probability price will go up >2.5% without dropping >2.5%
- **DOWN (8.78%)**: Probability price will drop >2.5% without rising >2.5%
- **UP_DOWN (3.77%)**: Probability of volatile movement (up AND down >2.5%)
- **SIDEWAYS (86.79%)**: Probability price stays within ±2.5%

---

### Multi-Horizon Prediction

**POST** `/predict/multi`

Make predictions for multiple horizons and thresholds in one request.

**Request Body:**
```json
{
  "symbol": "QQQ",
  "date": "2026-02-25",
  "horizons": [5, 10, 20],
  "thresholds": [0.01, 0.025]
}
```

**Response:**
```json
{
  "symbol": "QQQ",
  "date": "2026-02-25",
  "predictions": [
    {
      "horizon": 5,
      "threshold": 0.01,
      "prediction": "SIDEWAYS",
      "probabilities": {...},
      "confidence": 0.45
    },
    {
      "horizon": 5,
      "threshold": 0.025,
      "prediction": "UP",
      "probabilities": {...},
      "confidence": 0.52
    },
    ...
  ]
}
```

---

### Get Prediction by Stock

**GET** `/predict/by-stock/{symbol}`

Get prediction for a specific stock symbol.

**Parameters:**
- `symbol` (path): Stock symbol
- `date` (query, optional): Date in YYYY-MM-DD
- `horizon` (query, optional): Prediction horizon (default: 20)
- `threshold` (query, optional): Threshold as decimal (default: 0.01)

**Example:**
```
GET /predict/by-stock/QQQ?horizon=20&threshold=0.01
```

---

### Get Prediction by Date

**GET** `/predict/by-date/{date}`

Get prediction for a specific date.

**Parameters:**
- `date` (path): Date in YYYY-MM-DD format
- `symbol` (query, required): Stock symbol
- `horizon` (query, optional): Prediction horizon
- `threshold` (query, optional): Threshold

**Example:**
```
GET /predict/by-date/2026-02-25?symbol=QQQ&horizon=20
```

---

## Classification Classes (V2.5.1)

| Class | Value | Description |
|-------|-------|--------------|
| **UP** | 0 | Price goes up > threshold, down ≤ threshold |
| **DOWN** | 1 | Price goes down > threshold, up ≥ threshold |
| **UP_DOWN** | 2 | Both up AND down exceed threshold (volatile) |
| **SIDEWAYS** | 3 | Price stays within ±threshold |

### Class Definitions with Examples (20-day horizon, 2.5% threshold)

| Class | Condition | Example |
|-------|-----------|---------|
| **UP** | max_gain > 2.5% AND max_loss ≤ 2.5% | Price goes to +5% but never below -2.5% |
| **DOWN** | max_loss > 2.5% AND max_gain ≤ 2.5% | Price goes to -5% but never above +2.5% |
| **UP_DOWN** | max_gain > 2.5% AND max_loss > 2.5% | Price goes to +5% AND -5% (volatile) |
| **SIDEWAYS** | max_gain ≤ 2.5% AND max_loss ≥ -2.5% | Price stays between ±2.5% |

---

## Examples

### Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "QQQ",
        "horizon": 20,
        "threshold": 0.01
    }
)
print(response.json())

# Multi-horizon
response = requests.post(
    "http://localhost:8000/predict/multi",
    json={
        "symbol": "QQQ",
        "horizons": [5, 10, 20, 30],
        "thresholds": [0.01, 0.025, 0.05]
    }
)
print(response.json())
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ", "horizon": 20, "threshold": 0.01}'

# Multi-horizon
curl -X POST http://localhost:8000/predict/multi \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ", "horizons": [5, 10, 20], "thresholds": [0.01, 0.025]}'

# GET request
curl "http://localhost:8000/predict/by-stock/QQQ?horizon=20&threshold=0.01"
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid horizon. Must be one of [5, 10, 20, 30]"
}
```

### 404 Not Found
```json
{
  "detail": "Model not found for horizon=20, threshold=0.01. Please train first."
}
```

### 500 Internal Server Error
```json
{
  "detail": "Prediction error: <error message>"
}
```

---

## Logging

All API requests are logged to:
- `v2.5/src/logs/api/` - API request logs
- `v2.5/src/logs/prediction/` - Prediction logs

Log files are timestamped: `api_20260227_143022.log`
