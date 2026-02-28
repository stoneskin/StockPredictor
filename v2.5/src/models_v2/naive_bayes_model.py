"""
Naive Bayes Model for Stock Predictor V2.5
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from .base import BaseModel


class NaiveBayesModel(BaseModel):
    """Naive Bayes classifier for multi-class."""
    
    def __init__(self, params: dict = None):
        default_params = {
            'var_smoothing': 1e-9
        }
        if params:
            default_params.update(params)
        super().__init__("NaiveBayes", default_params)
    
    def _create_model(self):
        return GaussianNB(**self.params)
