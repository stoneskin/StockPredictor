"""
Gradient Boosting Model for Stock Predictor V2
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from .base import BaseModel


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting classifier.
    Strong predictive power, but prone to overfitting with small data.
    """
    
    def __init__(self, params: dict = None):
        default_params = {
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.1,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__("GradientBoosting", default_params)
    
    def _create_model(self):
        return GradientBoostingClassifier(**self.params)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)