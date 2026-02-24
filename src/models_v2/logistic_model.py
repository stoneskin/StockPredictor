"""
Logistic Regression Model for Stock Predictor V2
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from .base import BaseModel


class LogisticModel(BaseModel):
    """
    Logistic Regression classifier.
    Good baseline for binary classification, interpretable.
    """
    
    def __init__(self, params: dict = None):
        default_params = {
            'C': 0.1,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42,
            'solver': 'lbfgs'
        }
        if params:
            default_params.update(params)
        super().__init__("LogisticRegression", default_params)
    
    def _create_model(self):
        return LogisticRegression(**self.params)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)