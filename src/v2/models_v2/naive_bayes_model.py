"""
Naive Bayes Model for Stock Predictor V2
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from .base import BaseModel


class NaiveBayesModel(BaseModel):
    """
    Gaussian Naive Bayes classifier.
    Fast, works well with many features, makes strong independence assumptions.
    """
    
    def __init__(self, params: dict = None):
        default_params = {
            'var_smoothing': 1e-9
        }
        if params:
            default_params.update(params)
        super().__init__("NaiveBayes", default_params)
    
    def _create_model(self):
        return GaussianNB(**self.params)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)