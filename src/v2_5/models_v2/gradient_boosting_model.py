"""
Gradient Boosting Model for Stock Predictor V2.5
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from .base import BaseModel


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier for multi-class."""
    
    def __init__(self, params: dict = None):
        default_params = {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1,
            'min_samples_split': 20,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__("GradientBoosting", default_params)
    
    def _create_model(self):
        return GradientBoostingClassifier(**self.params)
