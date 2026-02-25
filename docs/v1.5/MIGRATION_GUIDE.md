# Project Redesign Summary & Migration Guide

## Overview

The Stock Predictor project has been redesigned to address critical overfitting issues and improve model generalization. The new approach uses **Walk-Forward Validation** with **Feature Combination Analysis** to find the optimal feature set and model configuration.

## What Changed

### Problems Solved

| Issue | Before | After |
|-------|--------|-------|
| **Overfitting** | R² drops from 0.41 to -2.53 | Walk-forward ensures realistic performance |
| **Single Test Set** | One test period (Oct) | Multiple test periods (rolling window) |
| **Look-Ahead Bias** | Test before training data | Always test after training |
| **Feature Bloat** | 31 features (untested) | 5-15 optimal features (tested) |
| **Market Regime** | No regime testing | Tests across multiple markets |
| **Reproducibility** | One result | Average across 5-6 folds |

### Key Improvements

1. **Walk-Forward Validation**
   - Expands training window progressively
   - Tests on multiple non-overlapping periods
   - No look-ahead bias
   - Captures different market regimes

2. **Feature Selection**
   - Analyzes feature importance (LightGBM)
   - Identifies redundant features (correlation)
   - Tests multiple feature combinations
   - Automatically ranks them

3. **Better Metrics**
   - Fold consistency (std dev of metrics)
   - Monotonic return validation
   - Comparison tables
   - Automated recommendations

## New Files & Structure

### Created Files

```
📁 src/walk_forward/              # New module
├── __init__.py                   # Package initialization
├── validation.py                 # Walk-forward fold generation
├── feature_selector.py           # Feature analysis & combinations
└── trainer.py                    # Training on walk-forward folds

📄 src/train_walkforward.py       # Main orchestration script (NEW)
📄 REDESIGN.md                    # Design document (NEW)
📄 IMPLEMENTATION_GUIDE.md        # Usage guide (NEW)
📄 MIGRATION_GUIDE.md             # This file (NEW)

📁 results/walk_forward/          # Results directory (created at runtime)
├── feature_importance.csv
├── feature_combinations.json
├── results_comparison.csv
├── recommendations.json
└── [combination_name]_metrics.json
```

### Unchanged Files

- `src/train.py` - Original training (deprecated but kept)
- `src/evaluate.py` - Evaluation module (works with new approach)
- `src/config.py` - Configuration
- `data/processed/` - Data format unchanged
- `models/checkpoints/` - Model format unchanged

## How to Use

### Step 1: Ensure Prerequisites

```bash
# Verify trained model exists
ls models/checkpoints/latest_model.pkl

# Verify processed data exists
ls data/processed/train.csv
```

### Step 2: Run Walk-Forward Pipeline

```bash
# Using Python 3.12
C:\Users\stone\AppData\Local\Programs\Python\Python312\python.exe src/train_walkforward.py
```

### Step 3: Review Results

```bash
# Check feature importance
cat results/walk_forward/feature_importance.csv

# View performance comparison
cat results/walk_forward/results_comparison.csv

# Read recommendations
cat results/walk_forward/recommendations.json
```

### Step 4: Update Model with Best Features (Optional)

Once you identify the best feature combination from recommendations:

```python
# Edit src/train.py:
best_features = [list of recommended features]
X_train = train_df[best_features].values
```

## Understanding the Pipeline

### 1. Data Loading
- Loads preprocessed `train.csv`
- Extracts features and target
- Prepares for analysis

### 2. Feature Importance
```
Input:  Trained LightGBM model + features + target
Output: Ranking of features by importance
        Correlation matrix
        List of redundant features
```

### 3. Feature Combinations Generated
```
- all_features (31 total - baseline)
- top_20 (top 20 by importance)
- top_15
- top_10
- top_8
- top_5 (minimal set)
- non_redundant (without correlated features)
- [category]_only (technical, momentum, volume, etc.)
- top_per_category (best from each category)
```

### 4. Walk-Forward Validation
```
Fold 1: Train [2019-2020] → Test [2020-Q1]
Fold 2: Train [2019-2021] → Test [2021-Q1]
Fold 3: Train [2019-2022] → Test [2022-Q1]
(multiple folds with expanding windows)
```

### 5. Training Each Combination
```
For each feature combination:
  For each walk-forward fold:
    Train LightGBM on fold training data
    Test on fold test data
    Record metrics
  Average metrics across all folds
  Track consistency (std dev)
```

### 6. Comparison & Ranking
```
Input:  Results from all combinations across all folds
Output: Comparison table ranked by R²
        Top 3 recommendations
        Per-fold detailed metrics
```

## Expected Output

### Example: results_comparison.csv

```
combination,n_features,n_folds,r2_mean,r2_std,rmse_mean,rmse_std,...
top_10,10,5,-0.0234,0.0567,3.45,0.23,...
top_15,15,5,0.0125,0.0445,3.32,0.19,...
all_features,31,5,-0.1456,0.1234,4.23,0.67,...
```

### Example: recommendations.json

```json
{
  "best_r2": {
    "combination": "top_15",
    "r2_mean": 0.0125,
    "r2_std": 0.0445,
    "n_features": 15
  },
  "best_consistency": {
    "combination": "top_10",
    "consistency": 0.9567,
    "r2_mean": -0.0234,
    "n_features": 10
  },
  "best_rmse": {
    "combination": "top_15",
    "rmse_mean": 3.32,
    "rmse_std": 0.19,
    "n_features": 15
  }
}
```

## Interpreting Results

### What R² Means
- **R² > 0.2**: Good predictive power
- **R² 0 to 0.2**: Weak but usable
- **R² < 0**: Worse than predicting mean
- **R² std < 0.05**: Consistent across folds ✅

### What RMSE Means
- Average prediction error in percentage
- Lower is better
- About 3-4% is reasonable for 15-day returns

### Monotonicity
- Features should result in higher returns in top quantiles
- 70%+ folds with monotonic increases = good
- 50% or less = feature set doesn't discriminate

## Migration Checklist

- [ ] Run walk-forward pipeline successfully
- [ ] Review feature importance ranking
- [ ] Check results comparison table
- [ ] Read recommendations
- [ ] Understand trade-offs between combinations
- [ ] Decide on best configuration (R², consistency, or RMSE focus)
- [ ] Update model configuration if desired with best features
- [ ] Rerun models with recommended features
- [ ] Archive old results

## Troubleshooting

### "ImportError: No module named 'walk_forward'"
**Solution**: Ensure you're in the correct directory when running the script
```bash
cd c:\Users\stone\OneDrive\Documents\obsidian_notes\SelfProject\StockPredictor
python src/train_walkforward.py
```

### "FileNotFoundError: models/checkpoints/latest_model.pkl"
**Solution**: Train the original model first
```bash
python src/train.py
```

### Script very slow (>2 hours)
**Solution**: Reduce complexity
- Reduce number of combinations (fewer top_k values)
- Increase step_months in walk_forward (fewer folds)
- Reduce data size for testing

### All R² values negative
**Solution**: This is expected if features don't predict well
- Check if features are meaningful
- Verify target variable calculation
- Consider simpler baseline model

## Key Advantages

✅ **No Look-Ahead Bias** - Always test in the future  
✅ **Realistic Performance** - Tests across multiple market periods  
✅ **Stable Estimates** - Average across multiple folds  
✅ **Feature Ranking** - Know which features matter  
✅ **Automated Recommendations** - Clear next steps  
✅ **Detailed Reporting** - Understand trade-offs  
✅ **Production Ready** - Use recommended configuration  

## Backward Compatibility

All original files remain unchanged:
- `data/processed/` format same
- `models/checkpoints/` format same
- `src/train.py` still works
- `src/evaluate.py` still works

You can run both old and new pipelines in parallel.

## Next Steps

1. **Immediate**: Run `python src/train_walkforward.py`
2. **Review**: Check `results_comparison.csv`
3. **Decide**: Choose best configuration from recommendations
4. **Implement**: Use recommended features in training
5. **Deploy**: Use best validated model in production

## Performance Expectations

### Runtime
- 5-6 walk-forward folds
- ~20 feature combinations
- ~500 total model trainings
- **Estimated: 30-60 minutes**

### Memory
- Input: ~100-200MB (2-year data)
- Results: ~50-100MB (all metrics)
- Total: <500MB

### Improvement Potential
- Better generalization (reduce overfit)
- Identify optimal features (reduce overfitting)
- More reliable performance estimates
- Clear path to production improvement

## Questions?

Refer to:
- `REDESIGN.md` - Overall architecture
- `IMPLEMENTATION_GUIDE.md` - Detailed usage
- `src/walk_forward/*.py` - Code documentation
- `src/train_walkforward.py` - Main pipeline

---

**Updated**: February 23, 2026  
**Version**: 2.0 (Walk-Forward with Feature Selection)
