# Stock Predictor V2.5 API Guide

## Overview

The V2.5 API provides stock price movement predictions using 4-class classification:
- **UP**: Price goes up more than threshold in any day within horizon
- **DOWN**: Price goes down more than threshold in any day within horizon
- **UP_DOWN**: Both up and down exceed threshold
- **SIDEWAYS**: Price stays within threshold range

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
  "version": "2.5.0",
  "n_models": 12,
  "horizons": [5, 10, 20, 30],
  "thresholds": [0.01, 0.025, 0.05],
  "classes": ["SIDEWAYS", "UP", "DOWN", "UP_DOWN"],
  "n_features": 47
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
  "date": "2026-02-25",
  "predictions": [
    {
      "horizon": 20,
      "threshold": 0.01,
      "prediction": "UP",
      "probabilities": {
        "SIDEWAYS": 0.15,
        "UP": 0.55,
        "DOWN": 0.10,
        "UP_DOWN": 0.20
      },
      "confidence": 0.55
    }
  ]
}
```

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

## Classification Classes

| Class | Value | Description |
|-------|-------|--------------|
| SIDEWAYS | 0 | Price stays within ±threshold |
| UP | 1 | Price goes up > threshold, down ≤ threshold |
| DOWN | 2 | Price goes down > threshold, up ≥ threshold |
| UP_DOWN | 3 | Both up and down exceed threshold |

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
