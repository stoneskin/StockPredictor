"""
Training script for Stock Predictor V2.5.1
Trains models with new 4-class classification targets
Includes SMOTE for class imbalance and time-series cross-validation
"""

import sys
from pathlib import Path

# Add v2.5/src to path
v25_root = Path(__file__).parent.parent
sys.path.insert(0, str(v25_root))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import joblib

# Try to import imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

from src.config_v2_5 import (
    MODEL_RESULTS_DIR, HORIZONS, THRESHOLDS, ENSEMBLE_WEIGHTS,
    TRAIN_PARAMS, MODEL_PARAMS, USE_SMOTE, USE_TIMESERIES_CV,
    THRESHOLD_PARAMS, SMOTE_K_NEIGHBORS, CLASS_LABELS
)
from src.data_preparation_v2_5 import prepare_data, get_target_column_name
from src.models_v2 import (
    LogisticModel, RandomForestModel, GradientBoostingModel,
    XGBoostModel, CatBoostModel, SVMModel, NaiveBayesModel, EnsembleModel,
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE
)
from src.logging_utils import get_training_logger, ModelPerformanceLogger, TRAIN_LOG_DIR


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, 
                logger) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE for class imbalance."""
    if not IMBLEARN_AVAILABLE:
        logger.warning("imbalanced-learn not installed, skipping SMOTE")
        return X_train, y_train
    
    if not USE_SMOTE:
        return X_train, y_train
    
    try:
        unique, counts = np.unique(y_train, return_counts=True)
        min_samples = min(counts)
        
        k_neighbors = min(SMOTE_K_NEIGHBORS, min_samples - 1)
        
        if k_neighbors < 1:
            logger.warning(f"Not enough samples for SMOTE (min={min_samples})")
            return X_train, y_train
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"SMOTE applied: {len(y_train)} -> {len(y_resampled)} samples")
        logger.info(f"New class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.warning(f"SMOTE failed: {e}, using original data")
        return X_train, y_train


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
    logger.info("Stock Predictor V2.5.1 Training")
    logger.info("=" * 60)
    logger.info(f"SMOTE enabled: {USE_SMOTE}")
    logger.info(f"Time-series CV: {USE_TIMESERIES_CV}")
    
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
                logger.info(f"Class labels: {CLASS_LABELS}")
                
                # Apply SMOTE if enabled
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=TRAIN_PARAMS['test_size'],
                    random_state=TRAIN_PARAMS['random_state'],
                    shuffle=not USE_TIMESERIES_CV
                )
                
                # Apply SMOTE to training data only
                if USE_SMOTE:
                    X_train, y_train = apply_smote(X_train, y_train, logger)
                
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
