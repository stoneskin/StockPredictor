"""
SVM Model for Stock Predictor V2.5
"""

import numpy as np
from sklearn.svm import SVC
from .base import BaseModel


class SVMModel(BaseModel):
    """SVM classifier for multi-class."""
    
    def __init__(self, params: dict = None):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'decision_function_shape': 'ovr'
        }
        if params:
            default_params.update(params)
        super().__init__("SVM", default_params)
    
    def _create_model(self):
        return SVC(**self.params)
