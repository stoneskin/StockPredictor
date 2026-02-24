"""
Main Walk-Forward Training and Feature Comparison Pipeline.

This script:
1. Loads preprocessed data and trained model
2. Performs feature importance analysis
3. Generates feature combinations
4. Trains models using walk-forward validation for each combination
5. Compares results and recommends best configuration
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DATA_DIR, MODEL_CHECKPOINTS_DIR, MODEL_PARAMS,
    PATIENCE, TARGET_THRESHOLD
)
from walk_forward import (
    WalkForwardValidator, FeatureSelector, WalkForwardTrainer
)


def load_data():
    """Load preprocessed training data."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    train_df = pd.read_csv(train_path)
    
    # Separate features and targets
    target_cols = [col for col in train_df.columns if col.startswith('target_')]
    feature_cols = [col for col in train_df.columns if col not in target_cols]
    
    X = train_df[feature_cols].values
    y = train_df['target_15d'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Feature Names:\n  {feature_cols[:10]}...\n")
    
    return X, y, feature_cols, train_df


def load_model_and_extract_importance(X, y, feature_cols, df):
    """Load trained model and extract feature importance."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    model_path = os.path.join(MODEL_CHECKPOINTS_DIR, "latest_model.pkl")
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Feature selection
    selector = FeatureSelector(model, feature_cols, df[feature_cols])
    importance_df = selector.extract_importance()
    
    selector.print_summary()
    
    # Save importance
    results_dir = os.path.join(os.path.dirname(MODEL_CHECKPOINTS_DIR), "results", "walk_forward")
    os.makedirs(results_dir, exist_ok=True)
    
    importance_path = os.path.join(results_dir, "feature_importance.csv")
    selector.save_importance(importance_path)
    
    return selector, importance_df


def setup_walk_forward_validation(df):
    """Setup walk-forward validation splits."""
    print("\n" + "="*80)
    print("SETTING UP WALK-FORWARD VALIDATION")
    print("="*80)
    
    # Create date column if doesn't exist
    if 'date' not in df.columns:
        df['date'] = pd.date_range(start='2019-01-01', periods=len(df), freq='D')
    
    validator = WalkForwardValidator(
        df=df,
        date_column='date',
        train_months=24,      # 2-year training window
        test_months=3,        # 3-month test window
        step_months=3,        # Step size
        min_train_samples=500,
        min_test_samples=30
    )
    
    folds = validator.generate_folds()
    validator.print_summary()
    
    return validator, folds


def run_feature_combination_comparison(X, y, feature_cols, folds, selector):
    """Run walk-forward training for all feature combinations."""
    print("\n" + "="*80)
    print("FEATURE COMBINATION COMPARISON - WALK-FORWARD TRAINING")
    print("="*80)
    
    # Generate feature combinations
    combinations = selector.generate_combinations(
        top_k_list=[20, 15, 10, 8, 5],
        remove_redundant=True,
        correlation_threshold=0.95
    )
    
    # Save combinations
    results_dir = os.path.join(os.path.dirname(MODEL_CHECKPOINTS_DIR), "results", "walk_forward")
    combinations_path = os.path.join(results_dir, "feature_combinations.json")
    selector.save_combinations(combinations_path)
    
    print(f"\nGenerated {len(combinations)} feature combinations")
    
    # Train walk-forward models for each combination
    trainer = WalkForwardTrainer(MODEL_PARAMS, PATIENCE)
    all_results = {}
    
    for combo_name, features in combinations.items():
        # Get feature indices
        feature_indices = [np.where(np.array(feature_cols) == f)[0][0] for f in features]
        
        # Train on this combination
        results = trainer.train_combination(
            X, y,
            feature_indices,
            feature_cols,
            folds,
            combo_name
        )
        
        all_results[combo_name] = results
        
        # Save results for this combination
        trainer.save_results(results, results_dir, combo_name)
        trainer.print_summary(combo_name, results)
    
    return all_results


def compare_and_recommend(all_results):
    """Compare results and recommend best configuration."""
    print("\n" + "="*80)
    print("FEATURE COMBINATION COMPARISON SUMMARY")
    print("="*80)
    
    # Create comparison dataframe
    comparison_data = []
    
    for combo_name, results in all_results.items():
        comparison_data.append({
            'combination': combo_name,
            'n_features': results['n_features'],
            'n_folds': results['n_folds'],
            'r2_mean': results['r2_mean'],
            'r2_std': results['r2_std'],
            'rmse_mean': results['rmse_mean'],
            'rmse_std': results['rmse_std'],
            'correlation_mean': results['correlation_mean'],
            'win_rate_mean': results['win_rate_mean'],
            'monotonic_pct': results['monotonic_pct'],
            'consistency': 1 / (1 + results['r2_std'])  # Higher is better
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('r2_mean', ascending=False)
    
    print("\nRanked by R² Score (Mean):")
    print("-"*80)
    print(comparison_df[[
        'combination', 'n_features', 'r2_mean', 'r2_std',
        'rmse_mean', 'monotonic_pct'
    ]].to_string(index=False))
    
    # Find best combination by different criteria
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best_r2 = comparison_df.loc[comparison_df['r2_mean'].idxmax()]
    best_consistency = comparison_df.loc[comparison_df['consistency'].idxmax()]
    best_rmse = comparison_df.loc[comparison_df['rmse_mean'].idxmin()]
    
    print(f"\n1. Best R² Score:")
    print(f"   {best_r2['combination']}")
    print(f"   R² = {best_r2['r2_mean']:.4f} ± {best_r2['r2_std']:.4f}")
    print(f"   Features: {best_r2['n_features']}")
    
    print(f"\n2. Most Consistent (Stable across folds):")
    print(f"   {best_consistency['combination']}")
    print(f"   Consistency Score: {best_consistency['consistency']:.4f}")
    print(f"   R² = {best_consistency['r2_mean']:.4f} ± {best_consistency['r2_std']:.4f}")
    
    print(f"\n3. Best RMSE:")
    print(f"   {best_rmse['combination']}")
    print(f"   RMSE = {best_rmse['rmse_mean']:.4f} ± {best_rmse['rmse_std']:.4f}")
    
    # Save comparison results
    results_dir = os.path.join(os.path.dirname(MODEL_CHECKPOINTS_DIR), "results", "walk_forward")
    comparison_path = os.path.join(results_dir, "results_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison results saved to: {comparison_path}")
    
    # Save recommendations
    recommendations = {
        'best_r2': {
            'combination': best_r2['combination'],
            'r2_mean': float(best_r2['r2_mean']),
            'r2_std': float(best_r2['r2_std']),
            'n_features': int(best_r2['n_features'])
        },
        'best_consistency': {
            'combination': best_consistency['combination'],
            'consistency': float(best_consistency['consistency']),
            'r2_mean': float(best_consistency['r2_mean']),
            'n_features': int(best_consistency['n_features'])
        },
        'best_rmse': {
            'combination': best_rmse['combination'],
            'rmse_mean': float(best_rmse['rmse_mean']),
            'rmse_std': float(best_rmse['rmse_std']),
            'n_features': int(best_rmse['n_features'])
        }
    }
    
    rec_path = os.path.join(results_dir, "recommendations.json")
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"Recommendations saved to: {rec_path}")
    
    return comparison_df, recommendations


def main():
    """Main pipeline."""
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION WITH FEATURE COMBINATION COMPARISON")
    print("="*80)
    
    # Load data
    X, y, feature_cols, df = load_data()
    
    # Feature analysis
    selector, importance_df = load_model_and_extract_importance(X, y, feature_cols, df)
    
    # Setup walk-forward validation
    validator, folds = setup_walk_forward_validation(df)
    
    # Run feature combination comparison
    all_results = run_feature_combination_comparison(X, y, feature_cols, folds, selector)
    
    # Compare and recommend
    comparison_df, recommendations = compare_and_recommend(all_results)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nResults saved to: results/walk_forward/")
    print("  - feature_importance.csv: Feature ranking")
    print("  - feature_combinations.json: All combinations tested")
    print("  - results_comparison.csv: Performance comparison")
    print("  - recommendations.json: Top recommendations")
    print("  - *_metrics.json: Detailed results per combination")


if __name__ == "__main__":
    main()
