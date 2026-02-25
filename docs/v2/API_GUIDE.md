# Enhanced Stock Prediction API V2 Guide

## Overview

The Stock Prediction API V2 has been enhanced to make it **much easier to use**. Instead of manually providing historical data, you can now simply provide a stock symbol and optionally a date, and the API will automatically:

1. Load data from local files
2. Automatically fetch missing data from Yahoo Finance
3. Append new data to local files for future use
4. Return predictions for multiple horizons in a single request

## Quick Start

### 1. Start the API Server

```bash
cd StockPredictor
python -m uvicorn uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 2. Make a Simple Prediction Request

The simplest possible request:

```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'
```

This will:
- Use the latest available date
- Return predictions for default horizons [5, 10, 20, 30]

### 3. API Response

```json
{
  "symbol": "QQQ",
  "date": "2026-02-24",
  "predictions": [
    {
      "horizon": 5,
      "prediction": "UP",
      "probability_up": 0.68,
      "probability_down": 0.32,
      "confidence": 0.68
    },
    {
      "horizon": 10,
      "prediction": "DOWN",
      "probability_up": 0.45,
      "probability_down": 0.55,
      "confidence": 0.55
    },
    {
      "horizon": 20,
      "prediction": "UP",
      "probability_up": 0.71,
      "probability_down": 0.29,
      "confidence": 0.71
    },
    {
      "horizon": 30,
      "prediction": "UP",
      "probability_up": 0.62,
      "probability_down": 0.38,
      "confidence": 0.62
    }
  ]
}
```

## Detailed API Reference

### Endpoint: `/predict/simple` (Recommended)

**Method:** POST

**Description:** Make predictions with automatic data loading. This is the recommended endpoint for most use cases.

#### Request Schema

```json
{
  "symbol": "QQQ",              // Required: Stock symbol (e.g., "QQQ", "AAPL", "SPY")
  "date": "2026-02-24",         // Optional: Prediction date (YYYY-MM-DD). Defaults to latest available
  "horizons": [5, 10, 20, 30]   // Optional: Prediction horizons (days). Defaults to [5, 10, 20, 30]
}
```

#### Response Schema

```json
{
  "symbol": "string",
  "date": "string (YYYY-MM-DD)",
  "predictions": [
    {
      "horizon": "int",
      "prediction": "string (UP or DOWN)",
      "probability_up": "float (0-1)",
      "probability_down": "float (0-1)",
      "confidence": "float (0-1)"
    }
  ]
}
```

#### Examples

**Example 1: Minimal request (use defaults)**
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

**Example 2: Custom date and horizons**
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "QQQ",
    "date": "2026-02-20",
    "horizons": [5, 10]
  }'
```

**Example 3: Multiple stocks (make separate requests)**
```bash
# QQQ
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'

# SPY
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY"}'

# AAPL
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

### Endpoint: `/predict` (Legacy API)

**Method:** POST

**Description:** Original API that requires manually providing all data. Use this if you want full control over the data.

#### Request Schema

```json
{
  "current": {
    "date": "2026-02-24",
    "open": 100.50,
    "high": 101.20,
    "low": 99.80,
    "close": 101.00,
    "volume": 1000000
  },
  "history": [
    {
      "date": "2026-02-20",
      "open": 99.50,
      "high": 100.20,
      "low": 98.80,
      "close": 100.00,
      "volume": 1000000
    }
    // ... at least 200 days of history required
  ],
  "horizon": 5  // Optional, defaults to 5
}
```

#### Response Schema

```json
{
  "prediction": "UP",
  "probability_up": 0.68,
  "probability_down": 0.32,
  "confidence": 0.68,
  "horizon": 5,
  "features_used": ["ma_5", "rsi_14", "bb_upper", ...]
}
```

### Endpoint: `/health` (Health Check)

**Method:** GET

**Description:** Check if the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "n_features": 65
}
```

### Endpoint: `/model-info` (Model Information)

**Method:** GET

**Description:** Get information about the loaded model.

**Response:**
```json
{
  "model_type": "Ensemble (LR + RF + GB + SVM + NB)",
  "best_horizon": "20d",
  "best_roc_auc": 0.62,
  "n_features": 65,
  "features": ["ma_5", "rsi_14", "bb_upper", ...]
}
```

## Data Management

### How Data Flows

1. **First Request:** 
   - API checks if local data file exists for the symbol
   - If not, fetches from Yahoo Finance
   - Saves to `data/raw/{symbol}.csv`

2. **Subsequent Requests:**
   - API loads from local file
   - Checks if date is within available data
   - If date is newer, automatically fetches missing data
   - Appends new data to local file

3. **Data Persistence:**
   - All fetched data is cached locally
   - No need to re-download on subsequent calls
   - Data format: CSV with columns [date, open, high, low, close, volume]

### Local Data Files

Data is stored in: `data/raw/{symbol}.csv`

Example files:
- `data/raw/qqq.csv` - QQQ (Nasdaq-100 Index ETF)
- `data/raw/aapl.csv` - Apple stock
- `data/raw/spy.csv` - SPY (S&P 500 ETF)

### Minimal Data Requirements

- **Minimum history:** 200 days of data required per prediction
- **If less than 200 days available:** API returns error with message about insufficient data

## Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Simple prediction request
def predict_stock(symbol, date=None, horizons=None):
    payload = {
        "symbol": symbol,
        "date": date,
        "horizons": horizons
    }
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}
    
    response = requests.post(
        f"{BASE_URL}/predict/simple",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    return response.json()

# Examples
print(predict_stock("QQQ"))
print(predict_stock("AAPL", horizons=[5, 10]))
print(predict_stock("SPY", date="2026-02-20", horizons=[5, 10, 20]))
```

## Error Handling

### Common Errors

**Error 1: Model not loaded**
```json
{
  "detail": "Model not loaded. Server not ready."
}
```
**Solution:** Wait for server startup or restart the API.

**Error 2: Invalid date format**
```json
{
  "detail": "Invalid date format: 2026/02/24. Use YYYY-MM-DD."
}
```
**Solution:** Use YYYY-MM-DD format for dates.

**Error 3: Insufficient data**
```json
{
  "detail": "Insufficient data for AAPL. Need 200 days, got 150"
}
```
**Solution:** API needs more historical data. Yahoo Finance might not have enough historical data for the symbol.

**Error 4: Invalid horizon**
```json
{
  "detail": "Invalid horizon: -5. Must be positive integer."
}
```
**Solution:** Horizons must be positive integers.

## Performance Tips

1. **Use the same symbol multiple times:** First request fetches data, subsequent requests are cached
2. **Batch predictions:** Make multiple `POST` requests in sequence rather than parallel (rate limiting)
3. **Default parameters:** Use defaults for horizons when applicable to save request size

## Feature Details

The model uses 65 technical features including:

- **Moving Averages:** 5, 10, 20, 50, 100, 200-day MAs
- **Relative Strength Index (RSI):** 14-period RSI
- **MACD:** Fast (12), Slow (26), Signal (9)
- **Bollinger Bands:** 20-period with 2 std dev
- **ATR:** 14-period Average True Range
- **Stochastic:** 14-period
- **Regime Features:** MA crossovers, volatility profiles
- **Market Features:** Volume indicators, daily returns

## Model Performance

- **Model Type:** Ensemble of 5 classifiers
  - Logistic Regression (20% weight)
  - Random Forest (30% weight)
  - Gradient Boosting (25% weight)
  - Support Vector Machine (15% weight)
  - Naive Bayes (10% weight)

- **Best Horizon:** 20 days
- **Best ROC-AUC:** ~0.62 (reference metrics in model-info endpoint)

## Troubleshooting

### API won't start
- Check Python version (3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Ensure port 8000 is not in use

### Module import errors
- Ensure you're running from project root: `cd StockPredictor`
- Check Python path: `python -c "import sys; print(sys.path)"`

### Data loading errors
- Check internet connection (for Yahoo Finance)
- Verify stock symbol is valid (e.g., "QQQ" not "NASDAQ")
- Check `data/raw/` directory permissions

### Slow predictions
- First prediction for a new symbol is slower (data fetching)
- Large horizon arrays take longer to compute
- Consider caching predictions for the same date

## Advanced Usage

### Custom Model

To use a different model:

```python
from src.v2.inference_v2 import predict_direction
import pandas as pd

# Your data
current_data = {...}
history_df = pd.DataFrame([...])

# Predict with custom model path
result = predict_direction(
    current_data=current_data,
    history_df=history_df,
    horizon=20,
    model_path="path/to/your/model.pkl"
)
```

### Direct Python Usage

```python
from src.v2.inference_v2 import get_stock_data, compute_features_v2, predict_direction
import pandas as pd

# Get data automatically
history_df, current_date = get_stock_data("QQQ", current_date="2026-02-24")

# Make prediction
result = predict_direction(
    current_data={
        'date': current_date,
        'open': history_df.iloc[-1]['open'],
        'high': history_df.iloc[-1]['high'],
        'low': history_df.iloc[-1]['low'],
        'close': history_df.iloc[-1]['close'],
        'volume': history_df.iloc[-1]['volume']
    },
    history_df=history_df,
    horizon=20
)
print(result)
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review example requests below
3. Check model information with `/model-info` endpoint
4. Verify data with local CSV files in `data/raw/`
