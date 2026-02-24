"""
Walk-Forward Training module.
Trains models on each fold using different feature combinations.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json


class WalkForwardTrainer:
    """
    Trains models using walk-forward validation with different feature combinations.
    """
    
    def __init__(self, model_params: Dict, patience: int = 50):
        """
        Initialize walk-forward trainer.
        
        Args:
            model_params: LightGBM model parameters
            patience: Early stopping patience
        """
        self.model_params = model_params
        self.patience = patience
        self.fold_results = []
        self.combination_results = {}
    
    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fold_id: int,
        feature_names: List[str]
    ) -> Dict:
        """
        Train and evaluate model for a single fold.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            fold_id: Fold identifier
            feature_names: Feature names
            
        Returns:
            Dictionary with fold metrics
        """
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=[train_data, test_data],
            num_boost_round=self.model_params.get('n_estimators', 1000),
            callbacks=[
                lgb.early_stopping(self.patience, verbose=False),
                lgb.log_evaluation(period=-1)
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        
        # Win rate analysis
        errors = y_test - y_pred
        win_rate = (np.abs(errors[y_pred > 0]) < np.abs(y_pred[y_pred > 0])).mean() * 100
        
        # Monotonicity check
        df_eval = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
        if len(df_eval) > 0:
            df_eval['quantile'] = pd.qcut(df_eval['y_pred'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            quantile_means = df_eval.groupby('quantile')['y_true'].mean().values
            is_monotonic = all(quantile_means[i] <= quantile_means[i+1] for i in range(len(quantile_means)-1))
        else:
            quantile_means = []
            is_monotonic = False
        
        return {
            'fold_id': fold_id,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'win_rate': win_rate,
            'is_monotonic': is_monotonic,
            'n_samples': len(y_test),
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    def train_combination(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: List[int],
        feature_names: List[str],
        folds: List,
        combination_name: str
    ) -> Dict:
        """
        Train model across all folds for a feature combination.
        
        Args:
            X: Full feature matrix
            y: Target vector
            feature_indices: Indices of features to use
            feature_names: All feature names
            folds: List of walk-forward folds
            combination_name: Name of feature combination
            
        Returns:
            Dictionary with aggregated results across folds
        """
        fold_results = []
        
        print(f"\nTraining '{combination_name}' with {len(feature_indices)} features...")
        
        for fold in folds:
            # Get data for this fold
            X_train = X[fold.train_indices][:, feature_indices]
            y_train = y[fold.train_indices]
            X_test = X[fold.test_indices][:, feature_indices]
            y_test = y[fold.test_indices]
            
            # Train
            result = self.train_fold(
                X_train, y_train, X_test, y_test,
                fold.fold_id,
                feature_names=[feature_names[i] for i in feature_indices]
            )
            
            fold_results.append(result)
            
            print(f"  Fold {fold.fold_id}: R²={result['r2']:.4f}, "
                  f"RMSE={result['rmse']:.4f}%, Monotonic={result['is_monotonic']}")
        
        # Aggregate results
        metrics_names = ['r2', 'rmse', 'mae', 'correlation', 'win_rate']
        aggregated = {}
        
        for metric in metrics_names:
            values = [r[metric] for r in fold_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        
        aggregated['n_folds'] = len(fold_results)
        aggregated['n_features'] = len(feature_indices)
        aggregated['monotonic_count'] = sum(1 for r in fold_results if r['is_monotonic'])
        aggregated['monotonic_pct'] = (aggregated['monotonic_count'] / len(fold_results) * 100)
        aggregated['fold_results'] = fold_results
        
        return aggregated
    
    def save_results(self, results: Dict, output_dir: str, combination_name: str):
        """
        Save training results for a combination.
        
        Args:
            results: Training results dictionary
            output_dir: Output directory
            combination_name: Name of combination
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save aggregated metrics
        metrics_file = os.path.join(output_dir, f"{combination_name}_metrics.json")
        
        # Remove model objects before saving
        results_copy = results.copy()
        results_copy['fold_results'] = [
            {k: v for k, v in r.items() if k not in ['model', 'y_pred', 'y_test']}
            for r in results_copy['fold_results']
        ]
        
        with open(metrics_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
    
    def print_summary(self, combination_name: str, results: Dict):
        """
        Print summary of training results for a combination.
        
        Args:
            combination_name: Name of combination
            results: Results dictionary
        """
        print("\n" + "="*80)
        print(f"WALK-FORWARD RESULTS: {combination_name}")
        print("="*80)
        print(f"Number of Features: {results['n_features']}")
        print(f"Number of Folds: {results['n_folds']}")
        print(f"Monotonic Folds: {results['monotonic_count']}/{results['n_folds']} ({results['monotonic_pct']:.1f}%)")
        
        print("\nAggregated Metrics (Mean ± Std):")
        print("-"*80)
        
        metrics_to_show = [
            ('r2', 'R² Score'),
            ('rmse', 'RMSE (%)'),
            ('mae', 'MAE (%)'),
            ('correlation', 'Correlation'),
            ('win_rate', 'Win Rate (%)')
        ]
        
        for metric_key, metric_name in metrics_to_show:
            mean = results[f'{metric_key}_mean']
            std = results[f'{metric_key}_std']
            print(f"{metric_name:<20}: {mean:>8.4f} ± {std:.4f} "
                  f"(range: {results[f'{metric_key}_min']:.4f} to {results[f'{metric_key}_max']:.4f})")
        
        print("="*80 + "\n")
