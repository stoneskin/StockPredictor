"""
Stock Predictor - ML System for QQQ Stock Prediction

Project Structure:
├── src/v1/           - Legacy regression approach (reference only)
│   ├── config.py
│   ├── train.py
│   ├── inference.py
│   └── ...
│
└── src/v2/           - Current classification approach (ACTIVE)
    ├── inference_v2.py          - FastAPI server
    ├── train_v2.py              - Training pipeline
    ├── config_v2.py             - Configuration
    ├── data_preparation_v2.py   - Feature engineering
    ├── models_v2/               - 5 ensemble base models
    ├── regime_v2/               - Market regime detection
    ├── walk_forward/            - V1.5: Walk-forward validation (experimentation)
    └── train_walkforward.py      - Walk-forward trainer

Version History:
- V1: Regression on stock returns (R² < 0) - Failed, replaced
- V1.5: Walk-forward methodology - Bridge for testing strategies
- V2: Classification UP/DOWN (52-54% accuracy) - Current, production ready

Quick Start:
$ python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
$ python src/v2/train_v2.py
"""

__version__ = "2.0"
__all__ = ["v1", "v2"]