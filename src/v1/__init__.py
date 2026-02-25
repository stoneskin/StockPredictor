# Stock Predictor V1
# Legacy regression-based stock prediction

"""
V1 Module - Original regression approach for QQQ prediction

This is the legacy implementation predicting 15-day returns.

Files:
- config.py: Configuration for V1
- data_preparation.py: V1 data pipeline
- train.py: V1 training
- inference.py: V1 API
- evaluate.py: V1 evaluation
- convert_model_to_onnx.py: ONNX conversion

Note: For new projects, use V2 (classification approach) in src/v2/
"""

__version__ = "1.0.0"
__status__ = "legacy"
__all__ = ["config", "data_preparation", "train", "inference", "evaluate"]
