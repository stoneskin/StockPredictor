# Stock Predictor V2
# Classification-based stock prediction using ensemble models

"""
V2 Module - Classification approach for stock prediction

Main components:
- inference_v2.py: FastAPI server for predictions
- train_v2.py: Model training pipeline
- data_preparation_v2.py: Data loading and feature engineering
- models_v2/: 5 base model implementations
- regime_v2/: Market regime detection
- walk_forward/: Advanced time-series validation
"""

__version__ = "2.0.0"
__all__ = ["inference_v2", "train_v2", "config_v2"]
