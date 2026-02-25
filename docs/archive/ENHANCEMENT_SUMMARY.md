# Stock Prediction API V2 - Enhancement Summary

## What Was Done

The Stock Prediction API V2 has been significantly enhanced to be **much easier to use**. Here are the key improvements:

### 1. **Automatic Data Management**
- **Before:** You had to manually provide 200+ days of historical data for every prediction
- **Now:** Just provide a stock symbol, and the API automatically:
  - Loads data from local files (`data/raw/{symbol}.csv`)
  - Automatically fetches missing data from Yahoo Finance
  - Caches new data locally for future use

### 2. **Simplified Request Format**
- **Before:**
  ```json
  {
    "current": { "date": "...", "open": 100.5, "high": 101.2, ... },
    "history": [{ full 200-day history }],
    "horizon": 5
  }
  ```

- **Now:**
  ```json
  {
    "symbol": "QQQ",
    "date": "2026-02-24",
    "horizons": [5, 10, 20, 30]
  }
  ```

### 3. **Multiple Horizons in One Request**
- **Before:** Had to make separate requests for each prediction horizon
- **Now:** Get predictions for multiple horizons [5, 10, 20, 30] in a single request
- **Default:** Uses [5, 10, 20, 30] if not specified

### 4. **Smart Date Handling**
- **Automatic date:** If no date provided, uses the latest available data
- **Custom date:** Automatically fetches new data if requested date is in the future
- **Validation:** Validates date format (YYYY-MM-DD)

## New API Endpoints

### Main Endpoint: `/predict/simple` (Recommended)

```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "QQQ",
    "date": "2026-02-24",
    "horizons": [5, 10, 20, 30]
  }'
```

**Response:**
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
    // ... more horizons
  ]
}
```

### Legacy Endpoint: `/predict` (Still Available)

The old endpoint that requires manual data input is still available for backward compatibility.

## Key Files Modified

1. **`src/inference_v2.py`** - Main API file with new features:
   - `get_stock_data()` - Auto-loads data from local/Yahoo Finance
   - `load_local_data()` - Loads CSV with smart format detection
   - `fetch_data_from_yahoo()` - Fetches from Yahoo Finance API
   - `append_to_local_file()` - Caches data locally
   - `predict_simple()` - New simplified endpoint
   - `compute_features_v2()` - Updated to include SPY market features

2. **`API_GUIDE.md`** - Comprehensive guide with:
   - Quick start instructions
   - Full API reference
   - Example requests
   - Python client code
   - Troubleshooting guide
   - Data management documentation

3. **`test_api.py`** - Test script demonstrating:
   - Data loading with auto-fetch
   - Model loading
   - Feature computation
   - Predictions

## How to Use

### Option 1: Start the Server and Make HTTP Requests

```bash
# Start server
python -m uvicorn uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000

# In another terminal, make predictions
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'
```

### Option 2: Use Python Directly

```python
from src.inference_v2 import get_stock_data, predict_direction, load_model

# Load model
load_model()

# Get data (auto-fetches if needed)
history_df, current_date = get_stock_data("QQQ")

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

### Option 3: Use Python Requests Library

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/simple",
    json={
        "symbol": "QQQ",
        "horizons": [5, 10, 20, 30]
    }
)

predictions = response.json()
print(predictions)
```

## Data Caching

All data fetched from Yahoo Finance is automatically cached in:
```
data/raw/{symbol}.csv
```

Example files:
- `data/raw/qqq.csv` - 1541 days of QQQ data (2020-01-03 to 2026-02-20)
- `data/raw/spy.csv` - S&P 500 ETF data (used for market features)

Next time you request the same symbol, it loads instantly from cache instead of fetching from Yahoo Finance.

## Testing

Run the test script to verify everything works:

```bash
cd StockPredictor
python test_api.py
```

Expected output shows:
- ✓ Data loading from local files
- ✓ Auto-fetching from Yahoo Finance  
- ✓ Model loading with 47 features
- ✓ Successful predictions with probabilities and confidence

## Example Requests

### 1. Minimal Request (Uses All Defaults)
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'
```

### 2. With Custom Horizons
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizons": [5, 10]}'
```

### 3. With Specific Date
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "TSLA", "date": "2026-02-20", "horizons": [20]}'
```

## Benefits Summary

✅ **Much Simpler:** Just send symbol, automatically handles data  
✅ **Faster:** Data is cached locally, no repeated fetches  
✅ **More Flexible:** Support for multiple symbols and custom dates  
✅ **Multiple Predictions:** Get forecasts for multiple time horizons at once  
✅ **Backward Compatible:** Old `/predict` endpoint still works  
✅ **Auto-Fetch:** Automatically updates data from Yahoo Finance when needed  
✅ **Extensive Documentation:** Complete API guide with examples  

## Next Steps

1. **Start the server:** `python -m uvicorn uvicorn src.v2.inference_v2:app --reload`
2. **Read the guide:** See `API_GUIDE.md` for complete documentation
3. **Run tests:** `python test_api.py` to verify setup
4. **Make predictions:** Start with simple symbol-only requests
5. **Explore:** Try different symbols, dates, and horizons
