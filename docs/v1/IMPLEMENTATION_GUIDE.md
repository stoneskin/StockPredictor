# Walk-Forward Validation Implementation Guide

## Quick Start

### Prerequisites
Ensure you have:
- Trained LightGBM model in `models/checkpoints/latest_model.pkl`
- Preprocessed data in `data/processed/train.csv`
- All required packages installed

### Run the Pipeline

```bash
# Run walk-forward analysis with feature combination comparison
C:\Users\stone\AppData\Local\Programs\Python\Python312\python.exe src/train_walkforward.py
```

**Expected Runtime**: 30-60 minutes (depends on data size and number of folds)

## Project Structure

```
src/
├── walk_forward/
│   ├── __init__.py
│   ├── validation.py         # Walk-forward fold generation
│   ├── feature_selector.py   # Feature importance & combinations
│   └── trainer.py            # Walk-forward training logic
├── train_walkforward.py      # Main orchestration script
└── [other existing files]

results/
└── walk_forward/
    ├── feature_importance.csv         # Feature ranking
    ├── feature_combinations.json       # All tested combinations
    ├── results_comparison.csv          # Performance comparison table
    ├── recommendations.json            # Top 3 recommendations
    ├── [combination_name]_metrics.json # Per-combination results
    └── ...
```

## Module Documentation

### 1. WalkForwardValidator (`walk_forward/validation.py`)

**Purpose**: Generate expanding window folds for time-series data

**Key Methods**:
- `generate_folds()` - Creates walk-forward periods
- `get_train_data(fold)` - Retrieve training data for fold
- `get_test_data(fold)` - Retrieve test data for fold
- `print_summary()` - Display fold information

**Example**:
```python
from walk_forward import WalkForwardValidator

validator = WalkForwardValidator(
    df=df,
    date_column='date',
    train_months=24,
    test_months=1,
    step_months=3
)

folds = validator.generate_folds()
validator.print_summary()

# Iterate over folds
for fold in folds:
    X_train = validator.get_train_data(fold)[features].values
    X_test = validator.get_test_data(fold)[features].values
```

**Parameters**:
- `train_months` (int): Training window size in months
- `test_months` (int): Test window size in months
- `step_months` (int): Step between folds in months
- `min_train_samples` (int): Minimum training samples required
- `min_test_samples` (int): Minimum test samples required

### 2. FeatureSelector (`walk_forward/feature_selector.py`)

**Purpose**: Analyze feature importance and generate combinations

**Key Methods**:
- `extract_importance()` - Get feature importance scores
- `calculate_correlations()` - Compute feature correlations
- `find_redundant_features()` - Identify highly correlated features
- `generate_combinations()` - Create feature sets for testing
- `save_combinations(path)` - Export to JSON
- `save_importance(path)` - Export to CSV

**Example**:
```python
from walk_forward import FeatureSelector

selector = FeatureSelector(model, feature_names, df)

# Get importance
importance_df = selector.extract_importance()
print(importance_df.head())

# Generate combinations
combinations = selector.generate_combinations(
    top_k_list=[20, 15, 10, 8, 5],
    remove_redundant=True,
    correlation_threshold=0.95
)

# Access combinations
for combo_name, features in combinations.items():
    print(f"{combo_name}: {len(features)} features")
```

**Feature Combinations Generated**:
1. `all_features` - All 31 features (baseline)
2. `top_20`, `top_15`, `top_10`, `top_8`, `top_5` - Top-K by importance
3. `non_redundant` - All features with low correlation
4. `[category]_only` - Features by category (e.g., `momentum_only`)
5. `top_per_category` - Top features from each category

### 3. WalkForwardTrainer (`walk_forward/trainer.py`)

**Purpose**: Train and evaluate models across walk-forward folds

**Key Methods**:
- `train_fold()` - Train on single fold
- `train_combination()` - Train across all folds for a feature set
- `save_results()` - Export metrics to JSON
- `print_summary()` - Display aggregated results

**Example**:
```python
from walk_forward import WalkForwardTrainer

trainer = WalkForwardTrainer(model_params, patience=50)

# Train a feature combination
results = trainer.train_combination(
    X=X,
    y=y,
    feature_indices=[0, 1, 5, 10],  # Indices of features to use
    feature_names=feature_cols,
    folds=folds,
    combination_name='top_10'
)

# Aggregated metrics
print(f"R² Mean: {results['r2_mean']:.4f}")
print(f"R² Std: {results['r2_std']:.4f}")
print(f"Folds with Monotonic Returns: {results['monotonic_count']}/{results['n_folds']}")
```

**Returned Metrics**:
```python
{
    'r2_mean': float,
    'r2_std': float,
    'r2_min': float,
    'r2_max': float,
    'rmse_mean': float,
    'rmse_std': float,
    'mae_mean': float,
    'mae_std': float,
    'correlation_mean': float,
    'correlationin_std': float,
    'win_rate_mean': float,
    'win_rate_std': float,
    'n_folds': int,
    'n_features': int,
    'monotonic_count': int,
    'monotonic_pct': float,
    'fold_results': [...]  # Per-fold metrics
}
```

### 4. Main Script (`train_walkforward.py`)

**Purpose**: Orchestrate the entire pipeline

**Pipeline Steps**:
1. Load preprocessed data and trained model
2. Extract feature importance
3. Generate feature combinations
4. Setup walk-forward validation folds
5. Train models for each combination
6. Compare results
7. Generate recommendations

**Output Files**:
- `feature_importance.csv` - Ranked features by importance
- `feature_combinations.json` - All tested combinations
- `results_comparison.csv` - Performance comparison table
- `recommendations.json` - Top 3 recommendations
- `[combination]_metrics.json` - Detailed metrics per combination

## Understanding Results

### feature_importance.csv
```
feature,importance,importance_pct,rank
momentum_12m,1024,5.32,1
volatility_30d,945,4.92,2
volume_ratio,876,4.56,3
...
```

### results_comparison.csv
```
combination,n_features,r2_mean,r2_std,rmse_mean,monotonic_pct,...
top_10,10,0.1234,0.0567,3.45,75.0,...
top_15,15,0.1456,0.0489,3.12,80.0,...
all_features,31,0.0876,0.0923,4.23,50.0,...
```

### recommendations.json
```json
{
  "best_r2": {
    "combination": "top_15",
    "r2_mean": 0.1456,
    "r2_std": 0.0489,
    "n_features": 15
  },
  "best_consistency": {
    "combination": "top_10",
    "consistency": 0.9234,
    "r2_mean": 0.1234,
    "n_features": 10
  },
  "best_rmse": {
    "combination": "top_15",
    "rmse_mean": 3.12,
    "rmse_std": 0.45,
    "n_features": 15
  }
}
```

## Interpretation Guide

### Choosing Best Configuration

**Best R² Score**:
- Highest predictive power
- May have high variance across folds
- Good if consistent across folds

**Best Consistency**:
- Most stable across different time periods
- Most reliable for production deployment
- Even if lower R², more predictable performance

**Best RMSE**:
- Lowest prediction error
- Consider with R² for balanced view

### What to Look For

**Good Sign** ✅:
- R² std < 0.05 (consistent across folds)
- Monotonic returns in 70%+ of folds
- Win rate > 45%
- Fewer features than training R²

**Warning Sign** ⚠️:
- R² std > 0.1 (unstable)
- Monotonic returns < 50%
- R² mean < previous test set
- All 31 features performs best

**Red Flag** ❌:
- Negative average R²
- 0% monotonic increases
- All metrics degrade

## Customization

### Adjust Walk-Forward Parameters

Edit in `train_walkforward.py`:
```python
validator = WalkForwardValidator(
    df=df,
    date_column='date',
    train_months=24,      # Increase for more data
    test_months=3,        # Increase for longer test period
    step_months=3,        # Decrease for more folds
    min_train_samples=500,
    min_test_samples=30
)
```

### Test Different Feature Sets

Add to `generate_combinations()` call:
```python
combinations = selector.generate_combinations(
    top_k_list=[25, 20, 15, 12, 10, 8, 5],  # Add more K values
    remove_redundant=True,
    correlation_threshold=0.90  # Adjust correlation threshold
)
```

### Change Model Parameters

Edit in `config.py`:
```python
MODEL_PARAMS = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    ...
}
```

## Troubleshooting

### Script crashes with "Not enough folds"
**Cause**: Insufficient data for walk-forward windows
**Solution**: 
- Reduce training window: `train_months=12`
- Increase test step: `step_months=6`

### Feature importance unchanged
**Cause**: Using old model file
**Solution**:
- Retrain model with `python src/train.py`
- Ensure `latest_model.pkl` is updated

### All combinations have negative R²
**Cause**: Model doesn't generalize well
**Solution**:
- Use fewer, more important features
- Try simpler models (linear regression baseline)
- Check data quality and feature engineering

## Next Steps

1. **Run the pipeline**: `python src/train_walkforward.py`
2. **Review results**: Check `results_comparison.csv`
3. **Analyze recommendations**: Read `recommendations.json`
4. **Retrain with best features**: Update `train.py` to use recommended features
5. **Deploy best model**: Use the feature combination that balances performance and stability

## Performance Benchmarks

Expected runtime (on modern CPU):
- 5 folds × 20 feature combinations = ~500 model trainings
- **~30-60 minutes** depending on data size

Memory usage:
- Dataset size: ~100MB for 2-year daily data
- Results storage: ~50-100MB for all fold metrics

## References

- [Walk-Forward Analysis for Trading](https://en.wikipedia.org/wiki/Walk_forward_optimization)
- [LightGBM Feature Importance](https://lightgbm.readthedocs.io/)
- [Time-Series Validation Best Practices](https://machinelearningmastery.com/backtest-machine-learning-models-for-time-series-forecasting/)
