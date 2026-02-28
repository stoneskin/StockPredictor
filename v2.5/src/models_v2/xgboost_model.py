"""
XGBoost Model for Stock Predictor V2.5
"""

import numpy as np
from .base import BaseModel

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostModel(BaseModel):
    """XGBoost classifier for multi-class."""
    
    def __init__(self, params: dict = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'num_class': 4
        }
        if params:
            default_params.update(params)
        super().__init__("XGBoost", default_params)
    
    def _create_model(self):
        return XGBClassifier(**self.params)
