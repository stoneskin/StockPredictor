"""
Logging utilities for Stock Predictor V2.5
Provides separate logging for training, prediction, and API calls
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import os
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.v2_5.config_v2_5 import (
    LOG_DIR, TRAIN_LOG_DIR, PREDICTION_LOG_DIR, API_LOG_DIR,
    LOG_LEVEL, LOG_FORMAT
)


def get_log_filename(log_type: str) -> str:
    """Generate log filename with date and time."""
    now = datetime.now()
    return f"{log_type}_{now.strftime('%Y%m%d_%H%M%S')}.log"


def setup_logger(
    name: str,
    log_type: str = "general",
    log_dir: Optional[Path] = None,
    level: str = LOG_LEVEL
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_type: Type of log (training, prediction, api, general)
        log_dir: Custom log directory (overrides log_type)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    if log_dir:
        log_path = log_dir
    else:
        log_path = LOG_DIR
        
        if log_type == "training":
            log_path = TRAIN_LOG_DIR
        elif log_type == "prediction":
            log_path = PREDICTION_LOG_DIR
        elif log_type == "api":
            log_path = API_LOG_DIR
    
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / get_log_filename(log_type)
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(getattr(logging, level.upper()))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper()))
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def get_training_logger(name: str = __name__) -> logging.Logger:
    """Get logger for training operations."""
    return setup_logger(name, log_type="training")


def get_prediction_logger(name: str = __name__) -> logging.Logger:
    """Get logger for prediction operations."""
    return setup_logger(name, log_type="prediction")


def get_api_logger(name: str = __name__) -> logging.Logger:
    """Get logger for API operations."""
    return setup_logger(name, log_type="api")


class PredictionLogger:
    """Context manager for logging predictions."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.predictions = []
    
    def log_prediction(
        self,
        symbol: str,
        date: str,
        horizon: int,
        threshold: float,
        prediction: str,
        probabilities: dict,
        confidence: float
    ):
        """Log a single prediction."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'date': date,
            'horizon': horizon,
            'threshold': threshold,
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': confidence
        }
        self.predictions.append(entry)
        
        self.logger.info(
            f"Prediction: {symbol} | Date: {date} | Horizon: {horizon}d | "
            f"Threshold: {threshold*100}% | Prediction: {prediction} | "
            f"Confidence: {confidence:.2%}"
        )
    
    def save_predictions(self, filepath: Optional[Path] = None):
        """Save predictions to file."""
        import json
        
        if filepath is None:
            filepath = PREDICTION_LOG_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.predictions, f, indent=2)
        
        self.logger.info(f"Saved {len(self.predictions)} predictions to {filepath}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save_predictions()


class ModelPerformanceLogger:
    """Logger for model performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = []
    
    def log_metrics(
        self,
        model_name: str,
        horizon: int,
        threshold: float,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        auc_roc: float,
        confusion_matrix: list
    ):
        """Log model performance metrics."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'horizon': horizon,
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': confusion_matrix
        }
        self.metrics.append(entry)
        
        self.logger.info(
            f"Model: {model_name} | Horizon: {horizon}d | Threshold: {threshold*100}% | "
            f"Acc: {accuracy:.2%} | F1: {f1:.2%} | AUC: {auc_roc:.3f}"
        )
    
    def save_metrics(self, filepath: Optional[Path] = None):
        """Save metrics to file."""
        import json
        
        if filepath is None:
            filepath = TRAIN_LOG_DIR / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Saved metrics for {len(self.metrics)} models to {filepath}")
