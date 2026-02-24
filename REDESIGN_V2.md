# Stock Predictor V2 - Redesign Document

## Executive Summary

This document outlines the redesign of the Stock Predictor project to improve prediction accuracy and robustness. The current regression-based approach with a single model shows limited predictive power (R² ≈ -0.04 to -0.13). We propose a multi-faceted approach combining classification, ensemble methods, and market regime detection.

---

## 1. Current Problems Analysis

### 1.1 Regression Approach Limitations
- **Negative R² scores** indicate predictions worse than simple mean
- High variance across folds (inconsistent performance)
- Single model approach lacks robustness
- 15-day prediction horizon may be too long for reliable prediction

### 1.2 Key Issues Identified
1. Regression targets are noisy (stock returns are random walks)
2. Single model overfits to historical patterns
3. No consideration of market regimes (bull/bear markets)
4. Insufficient data for complex models (583 samples)

---

## 2. Proposed Solutions

### 2.1 Switch to Classification (Up/Down Prediction)

**Rationale:**
- Classification is more robust than regression for noisy financial data
- Binary outcomes (price will go up/down) are easier to predict
- Allows for probability-based trading signals
- Aligns better with practical trading decisions

**Implementation:**
- Target: Binary classification (1 = price goes up, 0 = price goes down)
- Use probability thresholds for confidence levels
- Evaluate with AUC-ROC, Precision-Recall, F1-Score

### 2.2 Ensemble of Multiple Simple Models

**Rationale:**
- Reduces overfitting through model diversity
- Simple models (shallow trees, logistic regression) work better with limited data
- Ensemble provides more stable predictions

**Models to Ensemble:**
1. **Logistic Regression** - Linear baseline, interpretable
2. **Random Forest (shallow)** - Non-linear patterns, robust
3. **Gradient Boosting (few trees)** - Strong but prone to overfitting
4. **SVM with RBF kernel** - Good for medium-dimensional data
5. **Naive Bayes** - Fast, works well with many features

**Ensemble Strategy:**
- Voting ensemble (majority vote)
- Weighted voting based on validation performance
- Stacking with meta-learner

### 2.3 Market Regime Detection

**Rationale:**
- Stock behavior differs in bull vs bear markets
- Volatility regimes affect prediction difficulty
- Different models may work better in different regimes

**Regime Detection Methods:**
1. **VIX-based**: Use volatility index to classify regimes
2. **Moving Average Crossover**: 
   - Bull: Price > 200-day MA
   - Bear: Price < 200-day MA
3. **Rolling Volatility**: High/Low volatility regimes
4. **Hidden Markov Model (HMM)**: Probabilistic regime detection

**Regime-Specific Models:**
- Train separate models for each regime
- Use regime probability as feature/input
- Apply regime-aware ensemble weights

### 2.4 Multiple Prediction Horizons

**Rationale:**
- Different horizons suit different trading strategies
- Shorter horizons may be more predictable
- Allows for multi-timeframe analysis

**Proposed Horizons:**
| Horizon | Use Case | Expected Challenge |
|---------|----------|-------------------|
| 5-day | Swing trading | Less noise than 15-day |
| 10-day | Medium-term | Balance of signal/noise |
| 20-day | Position trading | More signal, less data |

---

## 3. Architecture Design

### 3.1 New Folder Structure

```
src/
├── train_v2.py                    # Main training script
├── config_v2.py                   # Configuration for V2
├── data_preparation_v2.py         # Data prep with classification targets
├── models/
│   ├── __init__.py
│   ├── base.py                    # Base model class
│   ├── logistic_model.py          # Logistic regression
│   ├── random_forest_model.py     # Random forest
│   ├── gradient_boosting_model.py # Gradient boosting
│   ├── svm_model.py               # SVM
│   ├── naive_bayes_model.py       # Naive Bayes
│   └── ensemble.py                # Ensemble wrapper
├── regime/
│   ├── __init__.py
│   ├── detector.py                # Regime detection base
│   ├── ma_crossover.py            # MA-based detection
│   ├── volatility_regime.py       # Volatility-based detection
│   └── hmm_regime.py              # HMM-based detection
├── features/
│   ├── __init__.py
│   ├── technical.py               # Technical indicators
│   ├── market_features.py         # Market-wide features
│   └── regime_features.py         # Regime-related features
└── evaluation/
    ├── __init__.py
    ├── metrics.py                 # Classification metrics
    └── evaluator.py               # Model evaluation
```

### 3.2 Data Flow

```
Raw Data → Feature Engineering → Regime Detection → Classification Target
                                                              ↓
                                                       Model Training
                                                              ↓
                                                    Ensemble Prediction
                                                              ↓
                                                      Trading Signals
```

---

## 4. Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] Create new folder structure
- [ ] Implement data preparation with classification targets
- [ ] Add multi-horizon target generation (5d, 10d, 20d)
- [ ] Implement basic technical indicators

### Phase 2: Regime Detection (Week 2)
- [ ] Implement MA crossover regime detector
- [ ] Implement volatility regime detector
- [ ] Add regime features to dataset
- [ ] Test regime-aware training

### Phase 3: Model Development (Week 3)
- [ ] Implement all base models
- [ ] Create ensemble wrapper
- [ ] Implement cross-validation for classification
- [ ] Add hyperparameter tuning

### Phase 4: Evaluation & Integration (Week 4)
- [ ] Comprehensive evaluation with classification metrics
- [ ] Compare with baseline (random guess, simple MA)
- [ ] Generate trading signals
- [ ] Backtest on historical data

---

## 5. Expected Improvements

| Metric | Current (Regression) | Target (Classification) |
|--------|---------------------|------------------------|
| R² / AUC-ROC | -0.04 to -0.13 | > 0.55 |
| Win Rate | ~40% | > 55% |
| Consistency | Low | High |
| Regime Adaptation | None | Full |

---

## 6. Risk Mitigation

1. **Limited Data**: Use cross-validation, simple models, regularization
2. **Overfitting**: Ensemble methods, holdout validation, early stopping
3. **Regime Changes**: Rolling window retraining, regime-aware weights
4. **Class Imbalance**: SMOTE, class weights, appropriate metrics

---

## 7. Success Criteria

- [ ] AUC-ROC > 0.55 on test set
- [ ] Consistent performance across market regimes
- [ ] Ensemble outperforms individual models
- [ ] Multiple prediction horizons available
- [ ] Clear trading signals with confidence levels

---

## Appendix: Technical Details

### A. Classification Target Generation
```python
def create_classification_target(prices, horizon=5):
    """Create binary target: 1 if price goes up after horizon days"""
    future_price = prices.shift(-horizon)
    return (future_price > prices).astype(int)
```

### B. Ensemble Prediction
```python
def ensemble_predict(models, X, weights=None):
    """Weighted voting ensemble"""
    proba = np.zeros((len(X), 2))
    for i, model in enumerate(models):
        w = weights[i] if weights else 1.0 / len(models)
        proba += w * model.predict_proba(X)
    return (proba[:, 1] > 0.5).astype(int)
```

### C. Regime Detection
```python
def detect_regime(prices, short_ma=50, long_ma=200):
    """Detect bull/bear based on MA crossover"""
    if prices[-1] > short_ma[-1] > long_ma[-1]:
        return 'bull'
    elif prices[-1] < short_ma[-1] < long_ma[-1]:
        return 'bear'
    return 'neutral'