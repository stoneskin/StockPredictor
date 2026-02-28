# Changelog - Stock Predictor V2.5.1

## 2026-02-28 - V2.5.1 Release

### New Features

1. **Class Reordering** (Breaking Change)
   - Changed from: SIDEWAYS(0), UP(1), DOWN(2), UP_DOWN(3)
   - Changed to: UP(0), DOWN(1), UP_DOWN(2), SIDEWAYS(3)
   - Reason: Better AUC-ROC calculation, minority classes learned first

2. **SMOTE Support**
   - Added `imbalanced-learn` dependency
   - Automatically applies SMOTE to training data
   - Configurable via `USE_SMOTE` in config
   - Handles class imbalance at 5% threshold

3. **Enhanced Regime Features**
   - ATR ratio (`atr_ratio`)
   - Trend strength (`trend_strength`)
   - Price vs MA ratios (`price_vs_ma50`, `price_vs_ma200`)
   - Volatility regime numeric (`volatility_regime_num`)
   - Momentum regime (`momentum_positive`, `momentum_strong`)
   - RSI regime (`rsi_overbought`, `rsi_oversold`, `rsi_neutral`)
   - Stochastic regime (`stoch_overbought`, `stoch_oversold`)
   - Bollinger Band squeeze/expansion (`bb_squeeze`, `bb_expansion`)
   - SPY correlation regime (`spy_correlation_positive/negative/neutral`)
   - SPY momentum (`spy_momentum`, `spy_momentum_positive`)

4. **Time-Series Cross-Validation (Optional)**
   - Configurable via `USE_TIMESERIES_CV`
   - Prevents data leakage from future to past

5. **Threshold-Specific Parameters**
   - Different hyperparameters per threshold
   - More regularization for 1% threshold
   - Balanced class weights for 5% threshold

6. **XGBoost and CatBoost**
   - Added to requirements.txt
   - Now trained alongside other models

### Dependencies Added

```
xgboost>=2.0.0
catboost>=1.2.0
imbalanced-learn>=0.12.0
```

### Config Changes

```python
# New config parameters (config_v2_5.py)
USE_SMOTE = True
USE_TIMESERIES_CV = False
SMOTE_K_NEIGHBORS = 5
USE_CALIBRATION = False
CALIBRATION_METHOD = 'isotonic'

THRESHOLD_PARAMS = {
    0.01: {'max_depth': 3, 'n_estimators': 150, ...},
    0.025: {'max_depth': 5, 'n_estimators': 100, ...},
    0.05: {'max_depth': 7, 'n_estimators': 100, ...}
}
```

### Breaking Changes

1. Class labels have been reordered - if you have saved models from v2.5.0, you need to retrain
2. Class 0 is now UP (was SIDEWAYS)
3. Class 1 is now DOWN
4. Class 2 is now UP_DOWN
5. Class 3 is now SIDEWAYS

### Bug Fixes

- Fixed AUC-ROC calculation for binary collapsed cases
- Fixed class imbalance at 5% threshold (was predicting mostly SIDEWAYS)

### Migration from V2.5.0

1. Install new dependencies:
   ```bash
   pip install xgboost catboost imbalanced-learn
   ```

2. Retrain all models:
   ```bash
   cd v2.5
   python src/train_v2_5.py
   ```

3. Update any code that references class indices:
   - Old: 0=SIDEWAYS, 1=UP, 2=DOWN, 3=UP_DOWN
   - New: 0=UP, 1=DOWN, 2=UP_DOWN, 3=SIDEWAYS
