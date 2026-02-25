"""
Evaluation and backtesting utilities.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, PREDICTION_HORIZON

def load_model_and_features(model_path, feature_list_path):
    """Load trained model and feature names."""
    model = joblib.load(model_path)
    with open(feature_list_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    return model, feature_names

def generate_predictions(df, model, feature_names):
    """Generate predictions for a DataFrame."""
    X = df[feature_names].values
    preds = model.predict(X)
    return preds

def evaluate_dataset(csv_path, model, feature_names, dataset_name="Dataset"):
    """Full evaluation on a dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    feature_cols = feature_names
    target_col = f'target_{PREDICTION_HORIZON}d'

    X = df[feature_cols].values
    y_true = df[target_col].values

    # Predict
    y_pred = model.predict(X)

    # Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}%")
    print(f"MAE: {mae:.4f}%")
    print(f"Correlation: {correlation:.4f}")

    # Quantile analysis
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    results_df['quantile'] = pd.qcut(results_df['y_pred'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    quantile_summary = results_df.groupby('quantile').agg({
        'y_true': ['mean', 'std', 'count', 'min', 'max']
    }).round(4)
    print("\nActual returns by prediction quintile:")
    print(quantile_summary)

    # Monotonicity check
    mean_returns = results_df.groupby('quantile')['y_true'].mean().values
    is_monotonic = all(mean_returns[i] <= mean_returns[i+1] for i in range(len(mean_returns)-1))
    print(f"\nMonotonic increasing: {'✅' if is_monotonic else '❌'}")
    if not is_monotonic:
        print(f"Warning: Non-monotonic sequence: {mean_returns}")

    # Top vs Bottom comparison
    top20 = results_df[results_df['y_pred'] >= results_df['y_pred'].quantile(0.8)]
    bottom20 = results_df[results_df['y_pred'] <= results_df['y_pred'].quantile(0.2)]
    print(f"\nTop 20% avg actual return: {top20['y_true'].mean():.2f}%")
    print(f"Bottom 20% avg actual return: {bottom20['y_true'].mean():.2f}%")
    print(f"Spread (Top - Bottom): {top20['y_true'].mean() - bottom20['y_true'].mean():.2f}%")

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'quantile_summary': quantile_summary,
        'is_monotonic': is_monotonic,
        'spread_top_bottom': top20['y_true'].mean() - bottom20['y_true'].mean()
    }

def simulate_trading_strategy(df, model, feature_names, threshold=0.0, position_size=1.0):
    """
    Simple trading simulation based on predictions.
    Args:
        df: DataFrame with features and actual returns
        threshold: only take trades where predicted return > threshold
        position_size: fraction of capital per trade (1.0 = 100%)
    """
    X = df[feature_names].values
    df['predicted_return'] = model.predict(X)
    df['position'] = 0
    df.loc[df['predicted_return'] > threshold, 'position'] = 1  # Long only
    df['strategy_return'] = df['position'].shift(1) * df['target_15d'] / 100

    # Cumulative returns
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod() - 1
    df['cumulative_bh'] = (1 + df['target_15d'] / 100).cumprod() - 1  # Buy & Hold

    # Metrics
    n_trades = df['position'].sum()
    win_rate = (df['strategy_return'] > 0).mean()
    avg_win = df[df['strategy_return'] > 0]['strategy_return'].mean()
    avg_loss = df[df['strategy_return'] < 0]['strategy_return'].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    total_return = df['cumulative_strategy'].iloc[-1] * 100
    buy_hold_return = df['cumulative_bh'].iloc[-1] * 100

    # Sharpe ratio (assuming 0% risk-free rate, 252 trading days)
    daily_returns = df['strategy_return'].dropna()
    if len(daily_returns) > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0

    print(f"\nTrading Simulation Results (threshold={threshold:.2f}%):")
    print(f"Number of trades: {int(n_trades)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average win: {avg_win:.2%}")
    print(f"Average loss: {avg_loss:.2%}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Total strategy return: {total_return:.2f}%")
    print(f"Buy & Hold return: {buy_hold_return:.2f}%")
    print(f"Sharpe ratio: {sharpe:.2f}")

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe': sharpe
    }

def plot_results(df_ytrue_y_pred, output_dir="plots"):
    """Generate evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Scatter plot: predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(df_ytrue_y_pred['y_true'], df_ytrue_y_pred['y_pred'], alpha=0.5, s=20)
    plt.plot([df_ytrue_y_pred['y_true'].min(), df_ytrue_y_pred['y_true'].max()],
             [df_ytrue_y_pred['y_true'].min(), df_ytrue_y_pred['y_true'].max()], 'r--', lw=2)
    plt.xlabel('Actual 15d Return (%)')
    plt.ylabel('Predicted 15d Return (%)')
    plt.title('Predicted vs Actual Returns')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_pred_vs_actual.png'), dpi=150)
    plt.close()

    # Quantile bar chart
    quantile_stats = df_ytrue_y_pred.groupby(pd.qcut(df_ytrue_y_pred['y_pred'], q=5, labels=['Q1','Q2','Q3','Q4','Q5']))['y_true'].mean()
    plt.figure(figsize=(8, 5))
    quantile_stats.plot(kind='bar', color='steelblue')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('Average Actual Return (%)')
    plt.xlabel('Prediction Quintile')
    plt.title('Model Calibration: Predicted vs Actual')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantile_calibration.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to: {output_dir}")

def main():
    """Full evaluation pipeline."""
    from config import PROCESSED_DATA_DIR, MODEL_CHECKPOINTS_DIR

    # Load model
    model_path = os.path.join(MODEL_CHECKPOINTS_DIR, "latest_model.pkl")
    feature_path = os.path.join(MODEL_CHECKPOINTS_DIR, "feature_names.txt")

    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return

    print(f"Loading model from: {model_path}")
    model, feature_names = load_model_and_features(model_path, feature_path)

    # Evaluate each dataset
    datasets = {
        'Train': os.path.join(PROCESSED_DATA_DIR, "train.csv"),
        'Validation': os.path.join(PROCESSED_DATA_DIR, "val.csv"),
        'Test': os.path.join(PROCESSED_DATA_DIR, "test.csv")
    }

    metrics = {}
    for name, path in datasets.items():
        if os.path.exists(path):
            m = evaluate_dataset(path, model, feature_names, name)
            metrics[name] = m

    # Trading simulation on test set
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"))
    trade_metrics = simulate_trading_strategy(test_df, model, feature_names, threshold=1.0)

    # Plot
    test_preds = model.predict(test_df[feature_names].values)
    plot_df = pd.DataFrame({
        'y_true': test_df['target_15d'],
        'y_pred': test_preds
    })
    plot_results(plot_df)

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for name, m in metrics.items():
        print(f"{name}: R²={m['r2']:.4f}, RMSE={m['rmse']:.2f}%, Monotonic={m['is_monotonic']}")

    print(f"\nStrategy (top 20% threshold): Sharpe={trade_metrics['sharpe']:.2f}, Total Return={trade_metrics['total_return']:.1f}%")

if __name__ == "__main__":
    main()