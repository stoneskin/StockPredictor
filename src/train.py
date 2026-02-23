"""
Training module for stock prediction model using LightGBM.
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, MODEL_CHECKPOINTS_DIR, MODEL_PARAMS,
    PATIENCE, TARGET_THRESHOLD
)

def load_data():
    """Load prepared datasets."""
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Separate features and targets
    target_cols = [col for col in train_df.columns if col.startswith('target_')]
    feature_cols = [col for col in train_df.columns if col not in target_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df['target_15d'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['target_15d'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['target_15d'].values

    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples, Test: {len(X_test)} samples")

    return (X_train, y_train, X_val, y_val, X_test, y_test), feature_cols

def train_model(X_train, y_train, X_val, y_val, params):
    """Train LightGBM model with early stopping."""
    print("\n" + "="*50)
    print("Training LightGBM Model")
    print("="*50)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=params.get('n_estimators', 1000),
        callbacks=[
            lgb.early_stopping(PATIENCE, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    print(f"\nBest iteration: {model.best_iteration}")
    return model

def evaluate_model(model, X, y, dataset_name="Test"):
    """Evaluate model performance."""
    print(f"\n{'='*50}")
    print(f"{dataset_name} Set Evaluation")
    print(f"{'='*50}")

    preds = model.predict(X)

    # Overall metrics
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}%")
    print(f"MAE: {mae:.4f}%")

    # Quantile analysis (divide predictions into quintiles)
    df = pd.DataFrame({'y_true': y, 'y_pred': preds})
    df['quantile'] = pd.qcut(df['y_pred'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    print("\nAverage actual 15d return by prediction quantile:")
    quantile_stats = df.groupby('quantile').agg({
        'y_true': ['mean', 'std', 'count']
    }).round(4)
    print(quantile_stats)

    # Check monotonicity (should increase from Q1 to Q5)
    mean_returns = df.groupby('quantile')['y_true'].mean().values
    is_monotonic = all(mean_returns[i] <= mean_returns[i+1] for i in range(len(mean_returns)-1))
    print(f"\nMonotonic increasing: {'✅ Yes' if is_monotonic else '❌ No'}")
    if not is_monotonic:
        print(f"Return sequence: {mean_returns}")

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'quantile_stats': quantile_stats,
        'is_monotonic': is_monotonic
    }

def save_model(model, feature_names):
    """Save model and feature names."""
    # Save as pickle
    model_path = os.path.join(MODEL_CHECKPOINTS_DIR, "latest_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save feature names
    feature_path = os.path.join(MODEL_CHECKPOINTS_DIR, "feature_names.txt")
    with open(feature_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"Feature names saved to: {feature_path}")

    return model_path

def convert_to_onnx(model, feature_names):
    """Convert LightGBM model to ONNX format."""
    try:
        from skl2onnx import convert_lightgbm
        from skl2onnx.common.data_types import FloatTensorType

        print("\n" + "="*50)
        print("Converting to ONNX")
        print("="*50)

        # Convert
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_type)

        onnx_path = os.path.join(MODEL_CHECKPOINTS_DIR, "..", "onnx", "model.onnx")
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"ONNX model saved to: {onnx_path}")
        return onnx_path
    except ImportError:
        print("skl2onnx not installed. Skipping ONNX conversion.")
        print("Install with: pip install skl2onnx")
        return None

def main():
    """Main training pipeline."""
    # 1. Load data
    (X_train, y_train, X_val, y_val, X_test, y_test), feature_names = load_data()

    # 2. Train model
    model = train_model(X_train, y_train, X_val, y_val, MODEL_PARAMS)

    # 3. Evaluate on all sets
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # 4. Save model
    model_path = save_model(model, feature_names)

    # 5. Convert to ONNX
    onnx_path = convert_to_onnx(model, feature_names)

    # Summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}%")
    print(f"Monotonic: {'✅' if test_metrics['is_monotonic'] else '❌'}")
    print(f"\nModel saved: {model_path}")
    if onnx_path:
        print(f"ONNX saved: {onnx_path}")

    return model, test_metrics

if __name__ == "__main__":
    model, metrics = main()