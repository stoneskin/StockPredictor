# Stock Predictor V2
# Classification-based stock prediction using ensemble models

"""
V2 Module - Classification approach for stock prediction

Main components:
- inference_v2.py: FastAPI server for predictions
- train_v2.py: Model training pipeline
- data_preparation_v2.py: Data loading and feature engineering (25+ indicators)
- config_v2.py: Configuration and hyperparameters

Supporting modules:
- models_v2/: 5 base model implementations (LR, RF, GB, SVM, NB)
- regime_v2/: Market regime detection (bull/bear/sideways)
- walk_forward/ (V1.5): Time-series walk-forward validation for strategy testing
  * Used to validate models on unseen future data
  * Bridge between V1 and V2 for experimentation
  * train_walkforward.py: Walk-forward trainer

Version: 2.0 (Classification: UP/DOWN prediction)
Horizons: 5, 10, 20, 30 days
Accuracy: 52-54% vs 50% random baseline
"""

__version__ = "2.0.0"
__all__ = ["inference_v2", "train_v2", "config_v2"]
