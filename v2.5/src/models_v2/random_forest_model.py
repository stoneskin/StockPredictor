"""
Random Forest Model for Stock Predictor V2.5
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier for multi-class."""
    
    def __init__(self, params: dict = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        super().__init__("RandomForest", default_params)
    
    def _create_model(self):
        return RandomForestClassifier(**self.params)
