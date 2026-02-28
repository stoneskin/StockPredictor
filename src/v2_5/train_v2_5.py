"""
Training script for Stock Predictor V2.5
Trains models with new 4-class classification targets
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.v2_5.config_v2_5 import (
    MODEL_RESULTS_DIR, HORIZONS, THRESHOLDS, ENSEMBLE_WEIGHTS,
    TRAIN_PARAMS, MODEL_PARAMS
)
from src.v2_5.data_preparation_v2_5 import prepare_data, get_target_column_name
from src.v2_5.models_v2 import (
    LogisticModel, RandomForestModel, GradientBoostingModel,
    XGBoostModel, CatBoostModel, SVMModel, NaiveBayesModel, EnsembleModel,
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE
)
from src.v2_5.logging_utils import get_training_logger, ModelPerformanceLogger, TRAIN_LOG_DIR


def create_models() -> Dict[str, object]:
    """Create all model instances."""
    models = {
        'LogisticRegression': LogisticModel(MODEL_PARAMS.get('logistic_regression')),
        'RandomForest': RandomForestModel(MODEL_PARAMS.get('random_forest')),
        'GradientBoosting': GradientBoostingModel(MODEL_PARAMS.get('gradient_boosting')),
        'SVM': SVMModel(MODEL_PARAMS.get('svm')),
        'NaiveBayes': NaiveBayesModel(MODEL_PARAMS.get('naive_bayes'))
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBoostModel(MODEL_PARAMS.get('xgboost'))
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostModel(MODEL_PARAMS.get('catboost'))
    
    return models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                   logger) -> Dict:
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    try:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except:
        metrics['auc_roc'] = 0.0
    
    return metrics


def train_single_model(model, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str, horizon: int, threshold: float,
                       logger, perf_logger) -> Tuple[object, Dict]:
    """Train and evaluate a single model."""
    logger.info(f"Training {model_name} for horizon={horizon}d, threshold={threshold*100}%")
    
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, logger)
    
    perf_logger.log_metrics(
        model_name=model_name,
        horizon=horizon,
        threshold=threshold,
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1=metrics['f1'],
        auc_roc=metrics['auc_roc'],
        confusion_matrix=metrics['confusion_matrix']
    )
    
    logger.info(f"{model_name}: Accuracy={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}")
    
    return model, metrics


def train_ensemble(models: Dict, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   horizon: int, threshold: float, logger) -> Tuple[object, Dict]:
    """Train ensemble model."""
    logger.info(f"Training Ensemble for horizon={horizon}d, threshold={threshold*100}%")
    
    ensemble = EnsembleModel()
    
    for name, model in models.items():
        weight = ENSEMBLE_WEIGHTS.get(name.lower(), 0.1)
        ensemble.add_model(model, weight)
    
    ensemble.fit(X_train, y_train)
    metrics = evaluate_model(ensemble, X_test, y_test, logger)
    
    logger.info(f"Ensemble: Accuracy={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}")
    
    return ensemble, metrics


def save_model_and_results(model, metrics: Dict, model_name: str,
                           horizon: int, threshold: float, feature_names: List[str]):
    """Save model and results to disk."""
    threshold_str = str(threshold).replace('.', '_')
    model_filename = f"{model_name.lower()}_h{horizon}_t{threshold_str}.pkl"
    
    model_path = MODEL_RESULTS_DIR / model_filename
    model.save(str(model_path))
    
    results_file = MODEL_RESULTS_DIR / f"{model_name.lower()}_results.txt"
    with open(results_file, 'a') as f:
        f.write(f"\n=== {model_name} - Horizon: {horizon}d, Threshold: {threshold*100}% ===\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1: {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
        f.write(f"Confusion Matrix: {metrics['confusion_matrix']}\n")
    
    feature_file = MODEL_RESULTS_DIR / "feature_names.txt"
    with open(feature_file, 'w') as f:
        f.write('\n'.join(feature_names))


def main():
    """Main training function."""
    logger = get_training_logger('train_v2_5')
    perf_logger = ModelPerformanceLogger(logger)
    
    logger.info("=" * 60)
    logger.info("Stock Predictor V2.5 Training")
    logger.info("=" * 60)
    
    results_summary = []
    
    for horizon in HORIZONS:
        for threshold in THRESHOLDS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training for Horizon: {horizon}d, Threshold: {threshold*100}%")
            logger.info(f"{'='*60}")
            
            try:
                X, y, feature_names, df = prepare_data(
                    horizon=horizon,
                    threshold=threshold,
                    include_spy=True
                )
                
                logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
                logger.info(f"Class distribution: {np.bincount(y)}")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=TRAIN_PARAMS['test_size'],
                    random_state=TRAIN_PARAMS['random_state']
                )
                
                models = create_models()
                best_model = None
                best_accuracy = 0
                best_metrics = None
                best_model_name = None
                
                for name, model in models.items():
                    try:
                        trained_model, metrics = train_single_model(
                            model, X_train, y_train, X_test, y_test,
                            name, horizon, threshold, logger, perf_logger
                        )
                        
                        save_model_and_results(
                            trained_model, metrics, name,
                            horizon, threshold, feature_names
                        )
                        
                        if metrics['accuracy'] > best_accuracy:
                            best_accuracy = metrics['accuracy']
                            best_model = trained_model
                            best_metrics = metrics
                            best_model_name = name
                            
                    except Exception as e:
                        logger.error(f"Error training {name}: {e}")
                
                # Train ensemble
                try:
                    models = create_models()
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                    
                    ensemble, ensemble_metrics = train_ensemble(
                        models, X_train, y_train, X_test, y_test,
                        horizon, threshold, logger
                    )
                    
                    save_model_and_results(
                        ensemble, ensemble_metrics, "Ensemble",
                        horizon, threshold, feature_names
                    )
                    
                    if ensemble_metrics['accuracy'] > best_accuracy:
                        best_accuracy = ensemble_metrics['accuracy']
                        best_model = ensemble
                        best_metrics = ensemble_metrics
                        best_model_name = "Ensemble"
                        
                except Exception as e:
                    logger.error(f"Error training ensemble: {e}")
                
                if best_model:
                    logger.info(f"\nBest Model: {best_model_name}")
                    logger.info(f"Best Accuracy: {best_accuracy:.2%}")
                    
                    results_summary.append({
                        'horizon': horizon,
                        'threshold': threshold,
                        'best_model': best_model_name,
                        'accuracy': best_accuracy,
                        'f1': best_metrics['f1'],
                        'auc_roc': best_metrics['auc_roc']
                    })
                
            except Exception as e:
                logger.error(f"Error for horizon={horizon}, threshold={threshold}: {e}")
                continue
    
    perf_logger.save_metrics()
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete - Summary")
    logger.info("=" * 60)
    for result in results_summary:
        logger.info(
            f"Horizon: {result['horizon']}d, Threshold: {result['threshold']*100}% | "
            f"Best: {result['best_model']} | Acc: {result['accuracy']:.2%}"
        )
    
    logger.info(f"\nModels saved to: {MODEL_RESULTS_DIR}")
    logger.info(f"Logs saved to: {TRAIN_LOG_DIR}")


if __name__ == "__main__":
    main()
