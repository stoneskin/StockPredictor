"""
Base model class for Stock Predictor V2.5
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import joblib
import os
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseModel(ABC, BaseEstimator, ClassifierMixin):
    """Abstract base class for all models."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.n_classes_ = 4
        
    @abstractmethod
    def _create_model(self):
        """Create the underlying model instance."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: list = None) -> 'BaseModel':
        """Fit the model to training data."""
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = feature_names
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        
        # Default implementation
        pred = self.predict(X)
        proba = np.zeros((len(X), self.n_classes_))
        for i, p in enumerate(pred):
            proba[i, p] = 1.0
        return proba
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).sum(axis=0)
        return None
    
    def save(self, filepath: str):
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'name': self.name,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'n_classes_': self.n_classes_
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.name = data['name']
        self.params = data['params']
        self.feature_names = data.get('feature_names')
        self.is_fitted = data.get('is_fitted', True)
        self.n_classes_ = data.get('n_classes_', 4)
    
    def __repr__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted}, classes={self.n_classes_})"
