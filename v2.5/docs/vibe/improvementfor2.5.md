# V2.5 Model Improvements

## Current Results Analysis

### Performance Summary

Based on the training results in `v2.5/models/results/`:

| Horizon | Threshold | Best Model | Accuracy | AUC-ROC | Issue |
|---------|-----------|------------|----------|---------|-------|
| 5d | 1% | RandomForest | 54-60% | 0.79 | Hard to predict short-term moves |
| 5d | 2.5% | RandomForest | 69-74% | 0.89 | Moderate |
| 5d | 5% | RandomForest | 97-98% | 0.99 | Trivial - mostly SIDEWAYS |
| 10d | 1% | RandomForest | 60% | 0.82 | Hard |
| 10d | 2.5% | RandomForest | 72-74% | 0.91 | Good |
| 10d | 5% | RandomForest | 94-97% | 0.98 | Trivial |
| 20d | 1% | RandomForest | 95-97% | 0.97 | Binary collapse (only 2 classes) |
| 20d | 2.5% | RandomForest | 77-79% | 0.94 | Good |
| 20d | 5% | RandomForest | 96-98% | 0.99 | Trivial |
| 30d | 1% | RandomForest | 99% | 0.00 | **Binary collapse - AUC invalid** |
| 30d | 2.5% | RandomForest | 77-79% | 0.96 | Good |
| 30d | 5% | RandomForest | 97-98% | 0.99 | Trivial |

### Key Issues Identified

1. **Class Imbalance at 5% threshold**: ~95% of samples are SIDEWAYS, making accuracy misleading
2. **Binary Class Collapse at 1% threshold, long horizons**: For 20d/30d with 1% threshold, only 2 classes appear in test set
3. **AUC-ROC = 0.0 for binary collapsed cases**: Cannot calculate multi-class AUC when only 2 classes present
4. **Low accuracy for short-term (5d) predictions**: Only 40-60% for 1% threshold

---

## Improvement Recommendations

### 1. Class Reordering (Critical)

**Current (confusing for AUC):**
```
0: SIDEWAYS, 1: UP, 2: DOWN, 3: UP_DOWN
```

**Recommended:**
```
0: UP, 1: DOWN, 2: UP_DOWN, 3: SIDEWAYS
```

**Why:**
- Current ordering puts SIDEWAYS (majority class) at 0, which skews AUC-ROC calculations
- Placing minority classes first helps the model learn them better
- AUC-ROC is calculated as average of one-vs-rest, so having minority classes first improves learning

**Background:**
The AUC-ROC metric computes the area under the ROC curve for each class treated as "positive" vs all others. When classes are imbalanced (e.g., 95% SIDEWAYS), the model focuses on predicting the majority class and gets high accuracy but poor minority class detection. Reordering helps the gradient boosting algorithms focus on harder cases early in training.

---

### 2. Address Class Imbalance with SMOTE (Important)

**Install imbalanced-learn:**
```bash
pip install imbalanced-learn
```

**Add to requirements.txt:**
```
imbalanced-learn>=0.12.0
```

**Why:**
- At 5% threshold, ~95% of samples are SIDEWAYS
- This causes the model to predict SIDEWAYS for everything
- SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic samples for minority classes

**Implementation:**
```python
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Or use SMOTE-Tomek for better results
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
```

**Background:**
SMOTE works by interpolating between existing minority class samples. For stock prediction, this is reasonable because small variations in technical indicators often lead to similar outcomes. However, we must be careful not to create unrealistic synthetic samples that don't represent true market conditions.

---

### 3. Time-Series Cross-Validation (Important)

**Current Issue:**
Using `train_test_split` with random shuffling breaks temporal ordering.

**Why This Matters:**
- Stock data is autocorrelated - today's features depend on past prices
- Random split allows "future leakage" - model sees future data in training
- This leads to overoptimistic performance estimates

**Solution:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train and evaluate
    model.fit(X_train, y_train)
```

**Background:**
Time-series cross-validation uses a sliding window approach:
- Fold 1: Train on [1-100], validate on [101-120]
- Fold 2: Train on [1-120], validate on [121-140]
- This mimics real-world deployment where you only use past data to predict the future

---

### 4. Feature Selection with RFE (Recommended)

**Current Issue:**
50+ features may cause overfitting and slower training.

**Solution:**
```python
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Use Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top 20 features
top_features = importances.head(20)['feature'].tolist()
```

**Background:**
Feature importance helps identify which indicators actually predict stock movements. In stock prediction:
- RSI, MACD, and moving averages are often most predictive
- Volume indicators help confirm trends
- Too many correlated features can dilute important signals

---

### 5. Add Class Weights Instead of SMOTE (Alternative)

If SMOTE doesn't work well, use class weights:

```python
# In config_v2_5.py
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'class_weight': 'balanced_subsample',  # Changed from 'balanced'
        'random_state': 42
    }
}
```

**Why:**
- `balanced_subsample` computes weights inversely proportional to class frequencies for each bootstrap sample
- This is more robust than global class weights

---

### 6. Threshold-Specific Model Tuning

Different thresholds need different approaches:

| Threshold | Strategy |
|----------|----------|
| 1% | Use more regularization, focus on trend indicators |
| 2.5% | Standard approach, works well currently |
| 5% | Use SMOTE, focus on volatility indicators |

```python
# Example: Different hyperparameters per threshold
THRESHOLD_PARAMS = {
    0.01: {'max_depth': 3, 'n_estimators': 150},  # More regularization
    0.025: {'max_depth': 5, 'n_estimators': 100}, # Standard
    0.05: {'max_depth': 7, 'n_estimators': 100, 'class_weight': 'balanced'} # Handle imbalance
}
```

---

### 7. Add More Regime Detection Features

Your regime detector exists but isn't fully utilized:

```python
# Add to add_regime_features() in data_preparation_v2_5.py

# Average True Range (ATR) based regime
df['atr_ratio'] = df['atr'] / close

# Trend Strength (ADX-like)
df['trend_strength'] = abs(df['ma_50'] - df['ma_200']) / df['ma_200']

# Volatility regime switching
df['volatility_regime'] = pd.cut(
    df['volatility'], 
    bins=[-np.inf, 0.01, 0.02, np.inf], 
    labels=['low', 'medium', 'high']
)

# Market correlation regime
df['spy_correlation_regime'] = pd.cut(
    df['correlation_spy_20d'],
    bins=[-1, -0.3, 0.3, 1],
    labels=['inverse', 'neutral', 'positive']
)
```

**Background:**
Market regimes (trending up, trending down, high volatility, low volatility) significantly affect which features are predictive. During high volatility, different indicators work better than during calm markets.

---

### 8. Probability Calibration (Advanced)

For better probability estimates:

```python
from sklearn.calibration import CalibratedClassifierCV

# Wrap model with calibration
base_model = RandomForestClassifier(**model_params)
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# Get calibrated probabilities
probas = calibrated_model.predict_proba(X_test)
```

**Why:**
- Raw model probabilities are often overconfident
- Calibration adjusts probabilities to match actual outcomes
- Important for your use case: "predict the possibility of the stock price will go up/down"

---

### 9. Binary Classification Option for 1% Threshold

For 1% threshold with short horizons (5d, 10d), consider binary classification:

```python
# Simplified target: UP vs NOT_UP (or DOWN vs NOT_DOWN)
def create_binary_targets(df, horizon, threshold):
    max_gain, max_loss = compute_max_daily_change(df, horizon)
    
    # Binary: either UP exceeds threshold OR DOWN exceeds threshold
    df['target_binary'] = ((max_gain > threshold) | (max_loss < -threshold)).astype(int)
    
    return df
```

**Why:**
- 4-class classification at 1% threshold is very hard
- Binary classification (movement vs no significant movement) is more tractable
- Can then decompose "movement" into direction as a second step

---

## Required Dependencies

Add to `requirements.txt`:

```
# For handling class imbalance (Recommendation #2)
imbalanced-learn>=0.12.0
```

---

## Implementation Priority

| Priority | Improvement | Expected Impact |
|----------|------------|-----------------|
| 1 | Time-Series CV | More realistic performance estimates |
| 2 | Class Weights/SMOTE | Better minority class detection |
| 3 | Feature Selection | Reduced overfitting, faster training |
| 4 | Regime Features | Better market adaptation |
| 5 | Probability Calibration | Better probability estimates |
| 6 | Class Reordering | Cleaner AUC calculation |

---

## Next Steps

1. Install `imbalanced-learn`
2. Run training with time-series CV to get realistic performance baselines
3. Apply SMOTE or class weights
4. Evaluate which features are most important
5. Consider adding the regime detection features from `regime_v2/detector.py`

---

## Questions to Consider

1. **What's your primary use case?**
   - Trading signals (buy/sell recommendations)?
   - Risk assessment (probability of large moves)?
   - Portfolio construction (regime detection)?

2. **What probability threshold do you need?**
   - For 1% threshold: Are you willing to accept more false positives for earlier signals?
   - For 5% threshold: Do you only care about "significant" moves?

3. **How do you plan to use the model?**
   - Standalone predictions?
   - Combined with other signals?
   - Risk management?

This will help me tailor the improvements to your specific needs.
