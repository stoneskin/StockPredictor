"""
Main Training Script for Stock Predictor V2
Classification-based approach with ensemble models and regime detection
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_v2 import (
    MODEL_CHECKPOINTS_DIR, MODEL_RESULTS_DIR, HORIZONS, DEFAULT_HORIZON,
    MODEL_PARAMS, ENSEMBLE_WEIGHTS, WALK_FORWARD_PARAMS
)
from data_preparation_v2 import prepare_data
from models_v2 import (
    LogisticModel, RandomForestModel, GradientBoostingModel,
    SVMModel, NaiveBayesModel, EnsembleModel
)
from regime_v2 import MACrossoverRegime, VolatilityRegime

# Import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def create_models():
    """Create all base models for the ensemble."""
    models = [
        LogisticModel(MODEL_PARAMS['logistic_regression']),
        RandomForestModel(MODEL_PARAMS['random_forest']),
        GradientBoostingModel(MODEL_PARAMS['gradient_boosting']),
        SVMModel(MODEL_PARAMS['svm']),
        NaiveBayesModel(MODEL_PARAMS['naive_bayes'])
    ]
    return models


def evaluate_model_manual(model, X, y, cv=5):
    """
    Evaluate model using manual cross-validation.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of CV folds
        
    Returns:
        Dictionary of metrics
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    pr_auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone and fit model
        from sklearn.base import clone
        try:
            model_clone = clone(model)
        except:
            # If clone fails, create new model
            model_name = model.name if hasattr(model, 'name') else 'Unknown'
            if 'Logistic' in model_name:
                model_clone = LogisticModel(MODEL_PARAMS['logistic_regression'])
            elif 'RandomForest' in model_name:
                model_clone = RandomForestModel(MODEL_PARAMS['random_forest'])
            elif 'GradientBoosting' in model_name:
                model_clone = GradientBoostingModel(MODEL_PARAMS['gradient_boosting'])
            elif 'SVM' in model_name:
                model_clone = SVMModel(MODEL_PARAMS['svm'])
            else:
                model_clone = NaiveBayesModel(MODEL_PARAMS['naive_bayes'])
        
        model_clone.fit(X_train, y_train)
        
        # Predict
        y_pred = model_clone.predict(X_val)
        
        try:
            y_proba = model_clone.predict_proba(X_val)[:, 1]
        except:
            y_proba = y_pred.astype(float)
        
        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        
        try:
            roc_auc_scores.append(roc_auc_score(y_val, y_proba))
            pr_auc_scores.append(average_precision_score(y_val, y_proba))
        except:
            roc_auc_scores.append(0.5)
            pr_auc_scores.append(0.5)
    
    return {
        'accuracy_mean': np.mean(accuracy_scores),
        'accuracy_std': np.std(accuracy_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'roc_auc_mean': np.mean(roc_auc_scores),
        'roc_auc_std': np.std(roc_auc_scores),
        'pr_auc_mean': np.mean(pr_auc_scores),
        'pr_auc_std': np.std(pr_auc_scores)
    }


def train_and_evaluate(X, y, feature_names, horizon):
    """
    Train ensemble and evaluate performance.
    
    Args:
        X: Features
        y: Labels
        feature_names: List of feature names
        horizon: Prediction horizon
        
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"TRAINING FOR {horizon}-DAY HORIZON")
    print(f"{'='*60}")
    
    # Create models
    models = create_models()
    print(f"\nCreated {len(models)} base models")
    
    # Evaluate each model
    print("\n--- Individual Model Performance (5-fold CV) ---")
    model_results = {}
    
    for model in models:
        print(f"\nEvaluating {model.name}...")
        try:
            metrics = evaluate_model_manual(model, X, y)
            model_results[model.name] = metrics
            print(f"  Accuracy: {metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}")
            print(f"  ROC-AUC:  {metrics['roc_auc_mean']:.4f} +/- {metrics['roc_auc_std']:.4f}")
            print(f"  F1:       {metrics['f1_mean']:.4f} +/- {metrics['f1_std']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            model_results[model.name] = {'error': str(e)}
    
    # Train ensemble
    print("\n--- Training Ensemble ---")
    ensemble = EnsembleModel(models, ENSEMBLE_WEIGHTS)
    
    # Fit all models
    for model in models:
        print(f"  Training {model.name}...")
        model.fit(X, y, feature_names)
    
    # Evaluate ensemble
    print("\n--- Ensemble Performance (5-fold CV) ---")
    ensemble_metrics = evaluate_model_manual(ensemble, X, y)
    print(f"  Accuracy: {ensemble_metrics['accuracy_mean']:.4f} +/- {ensemble_metrics['accuracy_std']:.4f}")
    print(f"  ROC-AUC:  {ensemble_metrics['roc_auc_mean']:.4f} +/- {ensemble_metrics['roc_auc_std']:.4f}")
    print(f"  F1:       {ensemble_metrics['f1_mean']:.4f} +/- {ensemble_metrics['f1_std']:.4f}")
    print(f"  PR-AUC:   {ensemble_metrics['pr_auc_mean']:.4f} +/- {ensemble_metrics['pr_auc_std']:.4f}")
    
    # Get feature importance
    importance = ensemble.get_feature_importance()
    if importance is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n--- Top 10 Features ---")
        print(importance_df.head(10).to_string(index=False))
        
        importance_records = importance_df.to_dict('records')
    else:
        importance_records = []
    
    return {
        'horizon': horizon,
        'n_samples': len(X),
        'n_features': len(feature_names),
        'target_distribution': {'up': int(y.sum()), 'down': int(len(y) - y.sum())},
        'model_results': model_results,
        'ensemble_metrics': ensemble_metrics,
        'feature_importance': importance_records,
        'trained_models': models,  # Return trained models for saving
        'ensemble': ensemble,       # Return trained ensemble
        'feature_names': feature_names  # Return feature names
    }


def compare_horizons():
    """Compare performance across different prediction horizons."""
    print("\n" + "="*80)
    print("COMPARING PREDICTION HORIZONS")
    print("="*80)
    
    all_results = {}
    
    for horizon in HORIZONS:
        # Prepare data for this horizon
        X, y, feature_names, df = prepare_data(horizon=horizon)
        
        # Train and evaluate
        results = train_and_evaluate(X, y, feature_names, horizon)
        all_results[f'{horizon}d'] = results
    
    # Summary comparison
    print("\n" + "="*80)
    print("HORIZON COMPARISON SUMMARY")
    print("="*80)
    
    summary_data = []
    for horizon_key, results in all_results.items():
        metrics = results['ensemble_metrics']
        summary_data.append({
            'horizon': horizon_key,
            'accuracy': f"{metrics['accuracy_mean']:.4f}",
            'roc_auc': f"{metrics['roc_auc_mean']:.4f}",
            'f1': f"{metrics['f1_mean']:.4f}",
            'pr_auc': f"{metrics['pr_auc_mean']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    return all_results


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("STOCK PREDICTOR V2 - CLASSIFICATION APPROACH")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)
    
    # Compare horizons
    all_results = compare_horizons()
    
    # Remove model objects from results before JSON serialization
    # (they will be saved separately via joblib)
    results_for_json = {}
    for horizon_key, results in all_results.items():
        results_for_json[horizon_key] = {
            'horizon': results.get('horizon'),
            'n_samples': results.get('n_samples'),
            'n_features': results.get('n_features'),
            'target_distribution': results.get('target_distribution'),
            'model_results': results.get('model_results'),
            'ensemble_metrics': results.get('ensemble_metrics'),
            'feature_importance': results.get('feature_importance')
        }
    
    # Save results
    results_path = MODEL_RESULTS_DIR / 'horizon_comparison.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        json.dump(convert_to_serializable(results_for_json), f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Find best horizon
    best_horizon = None
    best_auc = 0
    
    for horizon_key, results in all_results.items():
        auc = results['ensemble_metrics']['roc_auc_mean']
        if auc > best_auc:
            best_auc = auc
            best_horizon = horizon_key
    
    print(f"\nBest horizon: {best_horizon} (ROC-AUC: {best_auc:.4f})")
    
    # Save trained models for the best horizon
    print(f"\n--- Saving Models for Best Horizon ({best_horizon}) ---")
    best_results = all_results[best_horizon]
    
    # Save ensemble model
    ensemble = best_results.get('ensemble')
    if ensemble is not None:
        ensemble_path = MODEL_RESULTS_DIR / 'ensemble_model.pkl'
        joblib.dump(ensemble, ensemble_path)
        print(f"  Saved ensemble model to: {ensemble_path}")
    
    # Save individual models
    trained_models = best_results.get('trained_models', [])
    for model in trained_models:
        if model is not None and hasattr(model, 'name'):
            model_filename = f"{model.name.lower().replace(' ', '_')}_model.pkl"
            model_path = MODEL_RESULTS_DIR / model_filename
            try:
                joblib.dump(model, model_path)
                print(f"  Saved {model.name} to: {model_path}")
            except Exception as e:
                print(f"  Warning: Could not save {model.name}: {e}")
    
    # Save feature names
    feature_names = best_results.get('feature_names', [])
    if feature_names:
        feature_names_path = MODEL_RESULTS_DIR / 'feature_names.txt'
        with open(feature_names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"  Saved feature names to: {feature_names_path}")
    
    # Save best horizon info
    best_horizon_path = MODEL_RESULTS_DIR / 'best_horizon.txt'
    with open(best_horizon_path, 'w') as f:
        f.write(f"Best horizon: {best_horizon}\n")
        f.write(f"ROC-AUC: {best_auc:.4f}\n")
        f.write(f"Accuracy: {best_results['ensemble_metrics']['accuracy_mean']:.4f}\n")
        f.write(f"F1: {best_results['ensemble_metrics']['f1_mean']:.4f}\n")
    print(f"  Saved best horizon info to: {best_horizon_path}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return all_results


if __name__ == '__main__':
    main()