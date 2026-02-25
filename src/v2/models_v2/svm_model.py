"""
SVM Model for Stock Predictor V2
"""

import numpy as np
from sklearn.svm import SVC
from .base import BaseModel


class SVMModel(BaseModel):
    """
    Support Vector Machine classifier with RBF kernel.
    Good for medium-dimensional data, handles non-linear boundaries.
    """
    
    def __init__(self, params: dict = None):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__("SVM", default_params)
    
    def _create_model(self):
        return SVC(**self.params)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)