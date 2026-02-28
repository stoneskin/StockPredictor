"""
CatBoost Model for Stock Predictor V2.5
"""

import numpy as np
from .base import BaseModel

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostModel(BaseModel):
    """CatBoost classifier for multi-class."""
    
    def __init__(self, params: dict = None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost not installed. Install with: pip install catboost")
        
        default_params = {
            'iterations': 100,
            'depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': 0,
            'loss_function': 'MultiClass',
            'classes_count': 4
        }
        if params:
            default_params.update(params)
        super().__init__("CatBoost", default_params)
    
    def _create_model(self):
        return CatBoostClassifier(**self.params)
