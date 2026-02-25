# Stock Predictor - Redesign with Walk-Forward Validation & Feature Selection

## Overview
This document outlines the redesign of the stock prediction project to address overfitting issues and improve generalization using:
1. **Walk-Forward Validation** - Time-series aware validation approach
2. **Feature Combination Analysis** - Test multiple feature sets to find optimal combinations

## Problem with Current Approach

### Current Issues:
- **Severe Overfitting**: Train R² = 0.41 → Test R² = -2.53
- **Static Split**: Single train/val/test split doesn't capture market regime changes
- **Feature Bloat**: 31 features may include noise and redundant signals
- **No Benchmarking**: Unknown which features actually contribute

## New Approach

### 1. Walk-Forward Validation Architecture

```
Training Periods (2-year windows) → Test Periods (1-month windows)

2019-2020 → 2020-Q1
2019-2021 → 2021-Q1
2019-2022 → 2022-Q1
2019-2023 → 2023-Q1
2019-2024 → 2024-Q1
2019-2025 → 2025-Q1
```

**Benefits:**
- No look-ahead bias
- Tests across multiple market conditions
- More robust performance estimates
- Closer to real-world deployment

### 2. Feature Selection Strategy

#### Phase 1: Feature Importance Analysis
- Identify top 20/15/10 features from correlation & LightGBM importance
- Create baseline feature sets

#### Phase 2: Feature Combination Testing
Test multiple feature sets:
- **Baseline**: All 31 features
- **Top-K**: Top 20, 15, 10, 8 features by importance
- **Category-Based**: 
  - Momentum only
  - Volatility only
  - Volume only
  - Technical indicators only
  - Combined strategic set
- **Handpicked**: Domain-expert selection

#### Phase 3: Walk-Forward Backtesting
For each feature set:
- Train on each walk-forward period
- Evaluate using Sharpe ratio, profit factor, win rate
- Compare across all feature combinations

### 3. Directory Structure

```
src/
  walk_forward/
    __init__.py
    validation.py         # Walk-forward validation logic
    feature_selector.py   # Feature importance & combinations
    trainer.py            # Walk-forward training pipeline
    evaluator.py          # Performance metrics across folds
  
  train.py              # Original training (deprecated)
  train_walkforward.py  # New walk-forward training script

results/
  walk_forward/
    feature_combinations.json      # Feature sets definition
    feature_importance.csv         # Importance scores
    results_by_combination.csv     # Performance comparison
    best_combination.txt           # Best performing set
    fold_results/
      fold_1_metrics.json
      fold_2_metrics.json
      ...
```

## Implementation Plan

### Step 1: Feature Analysis (feature_selector.py)
- Load trained model
- Extract feature importance from LightGBM
- Calculate correlation matrix
- Identify redundant features
- Generate feature combinations

### Step 2: Walk-Forward Training (trainer.py)
- Split data into expanding windows
- Train model on each window
- Track performance per fold
- Average metrics across all folds

### Step 3: Feature Combination Comparison
- Test each feature combination with walk-forward
- Compare metrics: Sharpe, profit factor, win rate, R²
- Identify best combination

### Step 4: Results Analysis & Reporting
- Generate comparison tables
- Visualize performance by feature set
- Recommend best configuration

## Expected Outcomes

### Metrics Tracked per Feature Combination:
- R² Score
- Correlation
- RMSE
- Win Rate
- Profit Factor
- Sharpe Ratio
- Trading Strategy Return
- Consistency (std dev across folds)

### Output:
- Feature importance ranking
- Performance comparison table (CSV)
- Best feature combination
- Recommendations for deployment

## Files to Create/Modify

### New Files:
1. `REDESIGN.md` (this file)
2. `src/walk_forward/validation.py`
3. `src/walk_forward/feature_selector.py`
4. `src/walk_forward/trainer.py`
5. `src/walk_forward/evaluator.py`
6. `src/walk_forward/__init__.py`
7. `src/train_walkforward.py` (main script)
8. `config_walkforward.py` (configuration)

### Modified Files:
1. `config.py` - Add walk-forward parameters
2. `README.md` - Update documentation

## Configuration Parameters

```python
# Walk-Forward Settings
TRAIN_WINDOW_MONTHS = 24    # 2-year training window
TEST_WINDOW_MONTHS = 1      # 1-month test window
MIN_TRAIN_SAMPLES = 500
STEP_MONTHS = 3             # How often to re-train

# Feature Selection
FEATURE_IMPORTANCE_THRESHOLD = 0.01
TOP_K_FEATURES = [20, 15, 10, 8, 5]
FEATURE_COMBINATIONS = {
    'all': all_31_features,
    'top_20': ...,
    'top_15': ...,
    ...
}
```

## Success Criteria

- ✅ Test metrics within ±10% of validation metrics
- ✅ Consistent win rate >40% across folds
- ✅ Positive Sharpe ratio
- ✅ Profit factor >1.0
- ✅ Feature count <15 (reduce overfitting)

## Timeline

1. Phase 1: Feature analysis & combinations (1-2 hours)
2. Phase 2: Walk-forward implementation (2-3 hours)
3. Phase 3: Run all tests & analysis (2-4 hours)
4. Phase 4: Result analysis & docs (1-2 hours)

**Total Estimated Time**: 6-11 hours
