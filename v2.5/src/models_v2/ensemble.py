"""
Ensemble Model for Stock Predictor V2.5
Combines multiple models with weighted voting for multi-class classification
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import joblib
from .base import BaseModel


class EnsembleModel(BaseModel):
    """Ensemble classifier using weighted voting for multi-class."""
    
    def __init__(self, models: List[BaseModel] = None, weights: Dict[str, float] = None):
        self.name = "Ensemble"
        self.models = models or []
        self.weights = weights or {}
        self.feature_names = None
        self.n_classes_ = 4
        self.is_fitted = False
        
        if not self.weights and self.models:
            self.weights = {m.name: 1.0 / len(self.models) for m in self.models}
    
    def _create_model(self):
        """Ensemble doesn't use a single model - this is a no-op."""
        return None
    
    def add_model(self, model: BaseModel, weight: float = None):
        """Add a model to the ensemble."""
        self.models.append(model)
        if weight:
            self.weights[model.name] = weight
        elif model.name not in self.weights:
            self.weights[model.name] = 1.0 / len(self.models)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: list = None) -> 'EnsembleModel':
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y, feature_names)
        
        self.feature_names = feature_names
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted voting."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        # Collect votes from all models
        all_proba = []
        all_weights = []
        
        for model in self.models:
            if model.name in self.weights:
                proba = model.predict_proba(X)
                all_proba.append(proba)
                all_weights.append(self.weights[model.name])
        
        if not all_proba:
            raise ValueError("No models in ensemble")
        
        # Normalize weights
        total_weight = sum(all_weights)
        normalized_weights = [w / total_weight for w in all_weights]
        
        # Weighted average of probabilities
        weighted_proba = np.zeros((len(X), self.n_classes_))
        for proba, weight in zip(all_proba, normalized_weights):
            weighted_proba += proba * weight
        
        # Return class with highest probability
        return np.argmax(weighted_proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using weighted voting."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        all_proba = []
        all_weights = []
        
        for model in self.models:
            if model.name in self.weights:
                proba = model.predict_proba(X)
                all_proba.append(proba)
                all_weights.append(self.weights[model.name])
        
        if not all_proba:
            raise ValueError("No models in ensemble")
        
        total_weight = sum(all_weights)
        normalized_weights = [w / total_weight for w in all_weights]
        
        weighted_proba = np.zeros((len(X), self.n_classes_))
        for proba, weight in zip(all_proba, normalized_weights):
            weighted_proba += proba * weight
        
        return weighted_proba
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get average feature importance from all models."""
        importances = []
        for model in self.models:
            imp = model.get_feature_importance()
            if imp is not None:
                importances.append(imp)
        
        if importances:
            return np.mean(importances, axis=0)
        return None
    
    def save(self, filepath: str):
        """Save ensemble to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'models': self.models,
            'weights': self.weights,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'n_classes_': self.n_classes_
        }, filepath)
    
    def load(self, filepath: str):
        """Load ensemble from disk."""
        data = joblib.load(filepath)
        self.models = data['models']
        self.weights = data['weights']
        self.feature_names = data.get('feature_names')
        self.is_fitted = data.get('is_fitted', True)
        self.n_classes_ = data.get('n_classes_', 4)
    
    def __repr__(self) -> str:
        model_names = [m.name for m in self.models]
        return f"Ensemble({' + '.join(model_names)})"
