"""
Stock Predictor - ML System for QQQ Stock Prediction

Project Structure - Three Approaches:

├── src/v1/           - Regression Approach (FAILED)
│   ├── config.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── data_preparation.py
│   └── convert_model_to_onnx.py
│   └── Result: R² < 0 on test set (too noisy for regression)
│
├── src/v1_5/         - Walk-Forward Redesign (EXPERIMENTAL)
│   ├── train_walkforward.py         # Main pipeline
│   └── walk_forward/                # Validation modules
│       ├── validation.py             # Time-series validation logic
│       ├── feature_selector.py       # Feature importance analysis
│       ├── trainer.py                # Walk-forward training
│       └── evaluator.py              # Cross-fold metrics
│   └── Purpose: Feature selection + time-series validation testing
│   └── Discovered: Feature selection critical, walk-forward crucial
│
└── src/v2/           - Classification Ensemble (ACTIVE/PRODUCTION)
    ├── inference_v2.py              # FastAPI server
    ├── train_v2.py                  # Training pipeline
    ├── config_v2.py                 # Configuration
    ├── data_preparation_v2.py       # Feature engineering (25+ indicators)
    ├── models_v2/                   # 5 ensemble base models
    │   ├── logistic_model.py
    │   ├── random_forest_model.py
    │   ├── gradient_boosting_model.py
    │   ├── svm_model.py
    │   ├── naive_bayes_model.py
    │   └── ensemble.py
    ├── regime_v2/                   # Market regime detection
    │   ├── detector.py
    │   └── ma_crossover.py
    └── Result: 52-54% accuracy (binary UP/DOWN prediction)

Evolution:
- V1: Regression (predict exact return %) → Failed: R² < 0
- V1.5: Walk-forward + feature selection → Experimentation: Led to methodology insights
- V2: Binary classification (UP/DOWN) → Success: 52-54% accuracy

Quick Start (V2 Production):
$ python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
$ python src/v2/train_v2.py

Research (V1.5 Experimentation):
$ python src/v1_5/train_walkforward.py
"""

__version__ = "2.0"
__all__ = ["v1", "v1_5", "v2"]