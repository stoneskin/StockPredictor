# 📡 API Reference

Complete documentation of all API endpoints for the Stock Predictor service.

**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

---

## Quick Reference

| Endpoint | Method | Purpose | Complexity |
|----------|--------|---------|-----------|
| `/` | GET | Get API info | ⭐ Easy |
| `/predict/simple` | POST | Quick prediction | ⭐ Easy |
| `/predict` | POST | Full prediction | ⭐⭐ Medium |
| `/health` | GET | Health check | ⭐ Easy |

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
  "title": "Stock Prediction API V2",
  "description": "Predict QQQ price direction (up/down) using classification + ensemble",
  "version": "2.0.0"
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
  "models_loaded": true,
  "horizon": 5,
  "message": "API is running and ready for predictions"
}
```

### Response (Error)
```json
{
  "status": "unhealthy",
  "models_loaded": false,
  "error": "Models not found. Run: python src/v2/train_v2.py"
}
```

---

## 3. POST `/predict/simple`

**Easiest endpoint** - Get quick predictions with default settings.

### Use Cases
- ✅ First-time testing
- ✅ Quick symbol predictions
- ✅ Simple integration
- ✅ Default time horizons

### Request

**Format**:
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "QQQ"
  }'
```

**Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/simple",
    json={"symbol": "QQQ"}
)
data = response.json()
print(data)
```

### Parameters

| Parameter | Type | Required | Default | Example |
|-----------|------|----------|---------|---------|
| `symbol` | string | Yes | - | "QQQ", "TSLA", "SPY" |

### Response

```json
{
  "symbol": "QQQ",
  "date": "2026-02-24",
  "predictions": [
    {
      "horizon": 5,
      "prediction": "UP",
      "probability_up": 0.61,
      "probability_down": 0.39,
      "confidence": 0.61
    },
    {
      "horizon": 10,
      "prediction": "DOWN",
      "probability_up": 0.46,
      "probability_down": 0.54,
      "confidence": 0.54
    },
    {
      "horizon": 20,
      "prediction": "DOWN",
      "probability_up": 0.48,
      "probability_down": 0.52,
      "confidence": 0.52
    },
    {
      "horizon": 30,
      "prediction": "DOWN",
      "probability_up": 0.49,
      "probability_down": 0.51,
      "confidence": 0.51
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Stock ticker (e.g., "QQQ") |
| `date` | string | Prediction date (YYYY-MM-DD) |
| `predictions` | array | Array of predictions for each horizon |
| `horizon` | int | Number of days ahead (5, 10, 20, 30) |
| `prediction` | string | "UP" or "DOWN" |
| `probability_up` | float | Probability of price going up (0.0-1.0) |
| `probability_down` | float | Probability of price going down (0.0-1.0) |
| `confidence` | float | Confidence in prediction (max probability) |

### Interpretation

**Example**: 
```json
{
  "horizon": 5,
  "prediction": "UP",
  "probability_up": 0.61,
  "confidence": 0.61
}
```

**Meaning**:
- 5-day outlook: Stock will go UP
- Confidence: 61% (61% up, 39% down)
- Action: Moderate confidence prediction

**With Different Confidences**:
```
Confidence 50-52%: Very weak (coin flip)
Confidence 53-55%:  Weak (barely better than random)
Confidence 56-60%:  Moderate (useful for some strategies)
Confidence 61-70%:  Good (consider trading)
Confidence 71-100%: Strong (rare, very confident)
```

### Error Responses

**Invalid Symbol**:
```json
{
  "detail": "Could not fetch data for symbol INVALID"
}
```

**Models Not Trained**:
```json
{
  "detail": "Could not load model for horizon 5"
}
```

**No Internet**:
```json
{
  "detail": "Could not fetch data from Yahoo Finance"
}
```

---

## 4. POST `/predict`

**Advanced endpoint** - Full control over predictions.

### Use Cases
- ✅ Custom horizons
- ✅ Specific history length
- ✅ Advanced integration
- ✅ Production deployments

### Request

**Format**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "TSLA",
    "horizons": [5, 10],
    "min_history": 200
  }'
```

**Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "TSLA",
        "horizons": [5, 10, 20],
        "min_history": 250
    }
)
data = response.json()
for pred in data['predictions']:
    print(f"{pred['horizon']}d: {pred['prediction']} ({pred['confidence']:.1%})")
```

### Parameters

| Parameter | Type | Required | Default | Constraints |
|-----------|------|----------|---------|-------------|
| `symbol` | string | Yes | - | Must be valid stock ticker |
| `horizons` | array | No | [5, 10, 20, 30] | Int values in [5, 10, 20, 30] |
| `min_history` | int | No | 200 | >= 100, <= 10000 |

### Parameter Details

**`symbol`**: Stock ticker symbol
```json
"symbol": "QQQ"    // Valid
"symbol": "INVALID"  // Error
```

**`horizons`**: Which time horizons to predict
```json
"horizons": [5]           // Only 5-day prediction
"horizons": [5, 10, 30]   // Multiple horizons
"horizons": [20]          // Only long-term
```

Available horizons: `[5, 10, 20, 30]` (must be trained)

**`min_history`**: Minimum historical days to use
```json
"min_history": 200   // Use 200+ days of history
"min_history": 500   // Use ~500 days (more data)
"min_history": 300   // Use ~300 days (balanced)
```

### Response

Same as `/predict/simple` but with only requested horizons:

```json
{
  "symbol": "TSLA",
  "date": "2026-02-24",
  "predictions": [
    {
      "horizon": 5,
      "prediction": "UP",
      "probability_up": 0.58,
      "probability_down": 0.42,
      "confidence": 0.58
    },
    {
      "horizon": 10,
      "prediction": "DOWN",
      "probability_up": 0.47,
      "probability_down": 0.53,
      "confidence": 0.53
    }
  ]
}
```

### Advanced Examples

**Get only short-term prediction**:
```python
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "SPY",
        "horizons": [5],
        "min_history": 150
    }
)
result = response.json()['predictions'][0]
print(f"5-day: {result['prediction']} ({result['confidence']:.0%})")
```

**Trading Strategy - Multiple Timeframes**:
```python
symbols = ["QQQ", "TSLA", "SPY"]

for symbol in symbols:
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "symbol": symbol,
            "horizons": [5, 10, 20],
            "min_history": 300
        }
    )
    
    data = response.json()
    
    # Check if all timeframes agree
    predictions = {p['horizon']: p['prediction'] for p in data['predictions']}
    
    if predictions[5] == predictions[10] == predictions[20]:
        consensus = predictions[5]
        if consensus == "UP":
            print(f"BUY {symbol} - All timeframes bullish")
        else:
            print(f"SELL {symbol} - All timeframes bearish")
    else:
        print(f"MIXED {symbol} - Timeframes disagree")
```

---

## Data Format Reference

### Request Body Format

All POST requests use JSON:
```json
{
  "symbol": "string",
  "horizons": [5, 10],
  "min_history": 200
}
```

### Response Format

All success responses (HTTP 200) have this structure:
```json
{
  "symbol": "string",
  "date": "YYYY-MM-DD",
  "predictions": [
    {
      "horizon": int,
      "prediction": "UP|DOWN",
      "probability_up": 0.0-1.0,
      "probability_down": 0.0-1.0,
      "confidence": 0.0-1.0
    }
  ]
}
```

### Error Format

All error responses (HTTP 400, 500) have this structure:
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| 200 | Success | Prediction returned successfully |
| 400 | Bad Request | Invalid parameters (e.g., wrong symbol) |
| 422 | Validation Error | Missing required fields |
| 500 | Server Error | Internal error (models not loaded, network issue) |

---

## Rate Limits & Constraints

Currently **no rate limits** - feel free to make requests!

**Practical Constraints**:
- First request for a symbol: ~2-5 seconds (downloads data)
- Subsequent requests: <100ms (cached data)
- Maximum 1 concurrent request per symbol (queued otherwise)

---

## Code Examples

### Python - Simple Prediction

```python
import requests
import json

def predict_qqq():
    response = requests.post(
        "http://localhost:8000/predict/simple",
        json={"symbol": "QQQ"}
    )
    
    if response.status_code == 200:
        data = response.json()
        for pred in data['predictions']:
            print(f"Horizon {pred['horizon']}d: {pred['prediction']} ({pred['confidence']:.1%})")
    else:
        print(f"Error: {response.text}")

predict_qqq()
```

### Python - Trading Bot Logic

```python
import requests
from datetime import datetime

def get_trading_signal(symbol):
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "symbol": symbol,
            "horizons": [5, 10, 20],
            "min_history": 300
        }
    )
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    
    # Get consensus
    signals = {p['horizon']: p['prediction'] for p in data['predictions']}
    confidence_scores = {p['horizon']: p['confidence'] for p in data['predictions']}
    
    # Calculate strength
    up_votes = sum(1 for v in signals.values() if v == "UP")
    down_votes = sum(1 for v in signals.values() if v == "DOWN")
    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "signal": "BUY" if up_votes > down_votes else "SELL",
        "strength": abs(up_votes - down_votes) / len(signals),
        "confidence": avg_confidence,
        "timeframes": signals
    }

# Example usage
signal = get_trading_signal("QQQ")
print(f"Signal: {signal['signal']} ({signal['strength']:.0%} consensus, {signal['confidence']:.0%} confidence)")
```

### Bash/curl - Batch Predictions

```bash
#!/bin/bash

symbols=("QQQ" "TSLA" "SPY" "AMD")

for symbol in "${symbols[@]}"; do
    echo "Predicting $symbol..."
    curl -X POST http://localhost:8000/predict/simple \
      -H "Content-Type: application/json" \
      -d "{\"symbol\": \"$symbol\"}" | jq '.'
    echo ""
done
```

### JavaScript/Node - API Integration

```javascript
async function predictStock(symbol) {
    const response = await fetch('http://localhost:8000/predict/simple', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({symbol: symbol})
    });
    
    const data = await response.json();
    
    data.predictions.forEach(pred => {
        console.log(`${pred.horizon}d: ${pred.prediction} (${(pred.confidence*100).toFixed(1)}%)`);
    });
}

predictStock("QQQ");
```

---

## Troubleshooting

### "Connection refused"
- Make sure API server is running: `python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000`

### "Could not fetch data"
- Check internet connection
- Verify symbol is correct (QQQ, TSLA, SPY are tested)
- Yahoo Finance might be temporarily unavailable

### "No module named src"
- Make sure you're in project root directory: `cd StockPredictor`
- Run from correct working directory

### Slow first request
- Normal - first request downloads 200+ days of historical data (~2-5 seconds)
- Subsequent requests use cache and are fast (<100ms)

### Models not found error
- Run training first: `python src/v2/train_v2.py`
- Check that models exist: `ls models/results/v2/`

---

## Next Steps

- **Try it**: Make your first API call
- **Integrate**: Use in your trading system
- **Understand**: Review [ARCHITECTURE.md](ARCHITECTURE.md) to learn how it works
- **Customize**: Modify [config_v2.py](../src/v2/config_v2.py) for different settings