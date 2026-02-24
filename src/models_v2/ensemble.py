"""
Ensemble Model for Stock Predictor V2
Combines multiple models using weighted voting
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from .base import BaseModel


class EnsembleModel(BaseModel, BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier combining multiple models.
    Uses weighted voting based on validation performance.
    """
    
    def __init__(self, models: List[BaseModel], weights: Dict[str, float] = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of fitted BaseModel instances
            weights: Dictionary mapping model name to weight
        """
        super().__init__("Ensemble", {})
        self.models = models
        self.weights = weights or {m.name: 1.0 / len(models) for m in models}
        self.is_fitted = all(m.is_fitted for m in models)
        
        # Get feature names from first model
        if self.models:
            self.feature_names = self.models[0].feature_names
    
    def _create_model(self):
        """Not used for ensemble."""
        return None
    
    def fit(self, X, y, feature_names=None):
        """Fit all models in ensemble."""
        for model in self.models:
            model.fit(X, y, feature_names)
        self.is_fitted = True
        self.feature_names = feature_names
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted voting.
        
        Args:
            X: Features
            
        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        # Get weighted votes
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted average.
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        proba = np.zeros((len(X), 2))
        total_weight = 0
        
        for model in self.models:
            weight = self.weights.get(model.name, 1.0)
            try:
                model_proba = model.predict_proba(X)
                proba += weight * model_proba
                total_weight += weight
            except Exception as e:
                print(f"Warning: Model {model.name} failed to predict: {e}")
        
        # Normalize by total weight
        if total_weight > 0:
            proba /= total_weight
        
        return proba
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        """
        Predict with confidence scores.
        
        Args:
            X: Features
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        proba = self.predict_proba(X)
        predictions = (proba[:, 1] > 0.5).astype(int)
        
        # Confidence is the distance from 0.5
        confidence = np.abs(proba[:, 1] - 0.5) * 2
        
        return predictions, confidence
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get average feature importance across all models."""
        importances = []
        for model in self.models:
            imp = model.get_feature_importance()
            if imp is not None:
                importances.append(imp)
        
        if importances:
            return np.mean(importances, axis=0)
        return None
    
    def get_model_performance(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Get individual model performance.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary of model name to accuracy
        """
        from sklearn.metrics import accuracy_score
        
        performance = {}
        for model in self.models:
            pred = model.predict(X)
            acc = accuracy_score(y, pred)
            performance[model.name] = acc
        
        return performance
    
    def update_weights(self, weights: Dict[str, float]):
        """Update ensemble weights."""
        self.weights = weights
    
    def __repr__(self) -> str:
        model_names = [m.name for m in self.models]
        return f"Ensemble({model_names})"