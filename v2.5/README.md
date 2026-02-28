# Stock Predictor V2.5.1

## Overview

Stock Predictor V2.5.1 is an enhanced version with **4-class classification** for predicting stock price movements.

### Key Features

- **4 Classes**: UP, DOWN, UP_DOWN, SIDEWAYS (reordered for better AUC)
- **Multi-Horizon**: 3, 5, 10, 15, 20, 30-day predictions
- **Multi-Threshold**: 0.75%, 1%, 1.5%, 2.5%, 5% price movement detection
- **7 Models**: Including XGBoost, CatBoost
- **SMOTE**: Class imbalance handling
- **64 Features**: Technical + regime + market features
- **Enhanced Logging**: Date/time-stamped log files

## Documentation

| Document | Purpose |
|----------|---------|
| **[docs/README.md](docs/README.md)** | Docs index |
| **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** | Complete API reference with response explanations |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design & data flow |
| **[docs/API_GUIDE.md](docs/API_GUIDE.md)** | API usage guide with examples |

## Quick Start

**IMPORTANT**: All commands below must be run from the `v2.5/` folder.

### Installation

```bash
# Install dependencies (from project root)
pip install -r requirements.txt
```

### Training

```bash
# Run from v2.5 folder
cd v2.5
python src/train_v2_5.py
```

### Running API

```bash
# Run from v2.5 folder
cd v2.5
python -m uvicorn src.inference_v2_5:app --reload --host 0.0.0.0 --port 8000
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

- **2.5.1** (2026-02-28): Class reordering (UP/DOWN/UP_DOWN/SIDEWAYS), SMOTE support, enhanced regime features, XGBoost/CatBoost now trained
- **2.5.0** (2026-02-27): 4-class classification, multi-threshold, XGBoost/CatBoost
- **2.0** (2025-01-01): Binary classification with ensemble (see ../v2/)
- **1.0** (2024-01-01): Initial release (see ../archive/)

## Training Results (V2.5.1)

### Performance Summary

| Horizon | Threshold | Best Model | Accuracy | AUC-ROC | Notes |
|---------|-----------|------------|----------|---------|-------|
| 5d | 1% | XGBoost | 61.19% | 0.820 | Hard to predict |
| 5d | 2.5% | XGBoost | 80.97% | 0.913 | Good |
| 5d | 5% | XGBoost | 98.51% | 0.973 | Trivial |
| 10d | 1% | XGBoost | 82.02% | 0.885 | Improved |
| 10d | 2.5% | XGBoost | 79.78% | 0.943 | Good |
| 10d | 5% | XGBoost | 98.13% | 0.982 | Trivial |
| 20d | 1% | XGBoost | 96.98% | 0.982 | Binary (3 classes) |
| 20d | 2.5% | XGBoost | 92.83% | 0.993 | Excellent |
| 20d | 5% | XGBoost | 98.49% | 0.998 | Trivial |
| 30d | 2.5% | XGBoost | 97.34% | 0.997 | Excellent |
| 30d | 5% | XGBoost | 98.10% | 0.996 | Trivial |

### Key Improvements over V2.5.0

- **XGBoost outperforms RandomForest** by 7-22% accuracy
- **10d/1% now viable** - 82% accuracy (was 60%)
- **20d/2.5% improved** - 93% accuracy (was 77-79%)
- **SMOTE handling** - balanced class distribution during training
- **64 features** - added regime detection features

### Recommendations

1. Use **XGBoost** as default model (best performer)
2. Use **2.5% threshold** for best balance of difficulty and accuracy
3. Avoid 1% threshold with short horizons (5d) - inherently hard
4. For risk-averse strategies: use 20d/5% threshold (>98% accuracy)

## Notes

- This is the current active version
- Default threshold: 1%
- Default horizon: 20 days
- Model files saved to: `v2.5/models/results/`
