"""
Base model class for Stock Predictor V2
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import joblib
import os
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseModel(ABC, BaseEstimator, ClassifierMixin):
    """
    Abstract base class for all models.
    Provides common interface for training, prediction, and evaluation.
    """
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize base model.
        
        Args:
            name: Model name
            params: Model parameters
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def _create_model(self):
        """Create the underlying model instance."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: list = None) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Training labels
            feature_names: List of feature names
            
        Returns:
            Self
        """
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = feature_names
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            
        Returns:
            Array of predicted labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Default implementation using predict
        pred = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[pred == 0, 0] = 1.0
        proba[pred == 1, 1] = 1.0
        return proba
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importance scores, or None if not available
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        return None
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'name': self.name,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.name = data['name']
        self.params = data['params']
        self.feature_names = data.get('feature_names')
        self.is_fitted = data.get('is_fitted', True)
    
    def __repr__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"