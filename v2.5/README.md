# Stock Predictor V2.5

## Overview

Stock Predictor V2.5 is an enhanced version with **4-class classification** for predicting stock price movements.

### Key Features

- **4 Classes**: UP, DOWN, UP_DOWN, SIDEWAYS
- **Multi-Threshold**: 1%, 2.5%, 5% price movement detection
- **Multi-Horizon**: 5, 10, 20, 30-day predictions
- **7 Models**: Including XGBoost, CatBoost
- **Enhanced Logging**: Date/time-stamped log files

## Classification Logic

```
For each prediction horizon (e.g., 5 days) and threshold (e.g., 1%):

- UP:       max daily gain > threshold, max daily loss <= threshold
- DOWN:     max daily loss < threshold, max daily gain >= threshold  
- UP_DOWN:  max daily gain > threshold AND max daily loss < threshold
- SIDEWAYS: max daily gain <= threshold AND max daily loss >= threshold
```

Example: For 5-day horizon with 1% threshold:
- If in any of the next 5 days, price goes up >1% but never down >1%: **UP**
- If price goes down >1% but never up >1%: **DOWN**
- If price goes up >1% AND down >1%: **UP_DOWN**
- If price stays within ±1%: **SIDEWAYS**

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python v2.5/src/train_v2_5.py
```

### Running API

```bash
python -m uvicorn v2_5.src.inference_v2_5:app --reload --host 0.0.0.0 --port 8000
```

### Making Predictions

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

## Project Structure

```
v2.5/
├── src/
│   ├── config_v2_5.py
│   ├── data_preparation_v2_5.py
│   ├── inference_v2_5.py
│   ├── train_v2_5.py
│   ├── logging_utils.py
│   ├── models_v2/
│   └── regime_v2/
├── tests/
├── docs/
├── data/
│   └── cache/
└── models/
    └── results/
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single prediction |
| `/predict/multi` | POST | Multiple horizons/thresholds |
| `/predict/by-stock/{symbol}` | GET | Prediction by symbol |
| `/predict/by-date/{date}` | GET | Prediction by date |
| `/health` | GET | Health check |
| `/model-info` | GET | Model information |

## Logging

Logs are stored in `v2.5/src/logs/` with separate directories:
- `training/` - Training logs
- `prediction/` - Prediction logs  
- `api/` - API request logs

Log filenames include timestamps: `training_20260227_143022.log`

## Requirements

See `requirements.txt` for dependencies.

## Version History

- **2.5.0** (2026-02-27): 4-class classification, multi-threshold, XGBoost/CatBoost
- **2.0** (2025-01-01): Binary classification with ensemble (see ../v2/)
- **1.0** (2024-01-01): Initial release (see ../archive/)

## Notes

- This is the current active version
- Default threshold: 1%
- Default horizon: 20 days
- Model files saved to: `v2.5/models/results/`
