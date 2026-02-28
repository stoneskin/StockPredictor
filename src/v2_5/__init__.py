"""
Stock Predictor V2.5
Multi-class classification with 4 classes: UP, DOWN, UP_DOWN, SIDEWAYS
"""

from .config_v2_5 import (
    HORIZONS,
    THRESHOLDS,
    CLASS_LABELS,
    DEFAULT_HORIZON,
    MODEL_RESULTS_DIR,
    LOG_DIR
)

__version__ = "2.5.0"

__all__ = [
    'HORIZONS',
    'THRESHOLDS',
    'CLASS_LABELS',
    'DEFAULT_HORIZON',
    'MODEL_RESULTS_DIR',
    'LOG_DIR'
]
