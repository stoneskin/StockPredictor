# V3.0 Improvement Plan - Final Version
## Incorporating Grok & Gemini Professional Feedback

**Date**: 2026-03-07  
**Version**: 3.0.1  
**Previous**: 3.0 (Gemini feedback)  
**Current**: V2.5.2  
**Author**: StockPredictor Team  
**Review**: Professional feedback from Grok (Quant Trading Pro)

---

## Executive Summary

V3.0 transforms StockPredictor from a **classification system** into a **trading strategy advisor**. This version incorporates critical feedback from **Grok (Quant Trading Professional)** who identified three core issues:

1. **Lookahead bias must be eliminated first** - before any model improvements
2. **Transaction costs will kill day-trade edges** - must model 0.1% round-trip
3. **Simplify model architecture** - consolidate to 4 core models

**Key Insight from Grok**: "Profit-driven objective + strategy classification + P&L sim with stops/take-profits are excellent upgrades. BUT Phase 1 first: Strict lagged features only (kill lookahead bias)."

---

## Current State Analysis

### V2.5.2 Status
- ✅ 4-class classification (UP, DOWN, UP_DOWN, SIDEWAYS)
- ✅ 6 horizons (3,5,10,15,20,30 days), 5 thresholds (0.75%-5%)
- ✅ XGBoost achieves 90-97% accuracy on some combos
- ✅ Basic backtesting API

### Critical Problems (Grok's Diagnosis)
| Problem | Severity | V2.5 Status | V3.0 Solution |
|---------|----------|-------------|---------------|
| Lookahead bias | **CRITICAL** | Present in scaling, features | Strict lagged features only |
| Transaction costs ignored | **HIGH** | Not modeled | 0.1% round-trip simulation |
| Overfitting risk | **HIGH** | Accuracy-focused | Walk-forward + OOS validation |
| Labeling leakage | **HIGH** | May exist | Time-series correct labels |
| Model complexity | **MEDIUM** | 30 models | Consolidate to 4-6 core |

---

## Grok's Core Recommendations

### 1. KILL LOOKAHEAD BIAS FIRST (Phase 0)

**Grok**: "Phase 1 first: Strict lagged features only"

```python
# ❌ FORBIDDEN - Lookahead Bias
df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1  # LEAKAGE!
df['ma_5'] = df['close'].rolling(5).mean()  # OK - only past data
df['atr'] = compute_atr(df, lookback=14)    # OK - only past data

# ✅ REQUIRED - Strict Lagged Features
def compute_features_lagged(df):
    """ALL features must use only data available at time t"""
    features = {}
    
    # Price-based (all lagged)
    features['return_1d'] = df['close'].pct_change(1)
    features['return_5d'] = df['close'].pct_change(5)
    features['return_20d'] = df['close'].pct_change(20)
    
    # Volatility (lagged)
    features['vol_20d'] = df['close'].pct_change().rolling(20).std()
    features['atr_ratio'] = compute_atr(df, 14) / df['close']  # Already lagged
    
    # Moving averages (lagged by nature)
    features['ma_5_ratio'] = df['close'] / df['close'].rolling(5).mean()
    features['ma_20_ratio'] = df['close'] / df['close'].rolling(20).mean()
    
    # Momentum (lagged)
    features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    features['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # RSI, MACD, Bollinger - all computed from past data only
    features['rsi_14'] = compute_rsi(df['close'], 14)
    features['macd'] = compute_macd(df['close'])
    features['bb_position'] = compute_bb_position(df['close'])
    
    return pd.DataFrame(features)
```

### 2. TRANSACTION COSTS (Grok's Warning)

**Grok**: "Backtest: Add 0.1% round-trip slippage + commissions (day-trade edges vanish without)"

```python
TRANSACTION_COSTS = {
    'day_trade': 0.002,   # 0.2% (20 bps) - higher turnover
    'swing': 0.001,       # 0.1% (10 bps) - fewer trades
}

def simulate_trade_with_costs(entry_price, exit_price, strategy, direction):
    """Simulate trade with realistic costs"""
    entry_cost = entry_price * (1 + TRANSACTION_COSTS[strategy])
    exit_cost = exit_price * (1 - TRANSACTION_COSTS[strategy])
    
    if direction == 'long':
        pnl = (exit_cost - entry_cost) / entry_cost
    else:  # short
        pnl = (entry_cost - exit_cost) / entry_cost
    
    return pnl

# Grok's Rule: "If strategy can't beat 0.1% round-trip cost, it's not a strategy"
```

### 3. SAMPLE WEIGHTS (Grok's Simplification)

**Grok**: "XGBoost: Use sample_weights for profit focus; skip complex custom objective"

```python
def compute_profit_weights(df_train, threshold, lookback=20):
    """
    Weight samples by historical volatility regime.
    NO FUTURE DATA - only info available at time t.
    """
    weights = np.ones(len(df_train))
    
    returns = df_train['close'].pct_change()
    rolling_vol = returns.rolling(lookback).std()
    
    for i in range(lookback, len(df_train)):
        vol_rank = rolling_vol.iloc[i] / rolling_vol.iloc[:i].quantile(0.75)
        
        if vol_rank > 1.5:
            weights[i] = 2.0   # High volatility = opportunity
        elif vol_rank < 0.5:
            weights[i] = 0.3   # Low volatility = noise
        else:
            weights[i] = 1.0
    
    return weights

# Train with sample weights
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

### 4. HALF-KELLY SIZING (Grok's Advice)

**Grok**: "Sizing: Half-Kelly max + vol targeting"

```python
def half_kelly_sizing(win_rate: float, avg_win: float, avg_loss: float, 
                       volatility: float, portfolio_value: float) -> float:
    """
    Half-Kelly with volatility cap.
    
    Kelly: f* = p - q/b where p=win_rate, q=1-p, b=avg_win/avg_loss
    Half-Kelly: f* / 2
    """
    if win_rate <= 0 or avg_loss <= 0:
        return 0.0
    
    b = avg_win / avg_loss
    q = 1 - win_rate
    
    # Full Kelly
    kelly = win_rate - (q / b)
    
    # Half-Kelly (more conservative)
    half_kelly = kelly / 2
    
    # Volatility cap: don't risk more than 2% per trade
    vol_risk = volatility * 2  # 2x ATR equivalent
    
    # Final position: min of half-kelly and vol cap
    position = min(half_kelly, 0.02 / vol_risk, 0.25)  # Cap at 25%
    
    return max(0, position)
```

### 5. MODEL CONSOLIDATION (Grok's Simplification)

**Grok**: "Simplify: Consolidate horizons/thresholds; add direct return regressor"

| Strategy | Horizon | Threshold | Purpose |
|----------|---------|-----------|---------|
| Day Trade | 3 days | 1.0% | Quick momentum |
| Day Trade | 5 days | 1.5% | Mean reversion |
| Swing | 10 days | 1.5% | Medium trend |
| Swing | 20 days | 2.5% | Long trend |

**Plus 2 Regression Models** (NEW from Grok):
| Strategy | Horizon | Output |
|----------|---------|--------|
| Day Trade | 3 days | Expected Return % |
| Swing | 20 days | Expected Return % |

```python
class ReturnRegressor:
    """Direct return prediction (Grok's suggestion)"""
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5
        )
    
    def predict(self, X):
        """Returns expected return as percentage"""
        return self.model.predict(X) * 100  # Convert to %
```

---

## Phased Implementation (Revised)

### Phase 0: FIX LOOKAHEAD BIAS (Week 1)
**Priority: CRITICAL**

- [ ] Audit ALL features for lookahead bias
- [ ] Implement strict lagged feature computation
- [ ] Fix scaling (fit only on training window)
- [ ] Fix label computation (no future data)
- [ ] Run baseline backtest with fixed features

**Duration**: 1 week

### Phase 1: CORE MODELS + TRANSACTION COSTS (Week 2-3)
**Priority: HIGH**

- [ ] Consolidate to 4 core models
- [ ] Add transaction cost simulation (0.1% round-trip)
- [ ] Implement walk-forward backtest
- [ ] Test sensitivity: 0.05%, 0.1%, 0.2%, 0.25%

**Duration**: 2 weeks

### Phase 2: PROFIT-WEIGHTED TRAINING (Week 4)
**Priority: HIGH**

- [ ] Implement sample weight computation
- [ ] Retrain 4 core models with weights
- [ ] Add direct return regressor models
- [ ] Evaluate: precision_up, recall_up, F1, profit-weighted accuracy

**Duration**: 1 week

### Phase 3: POSITION SIZING + RISK (Week 5)
**Priority: MEDIUM**

- [ ] Implement Half-Kelly sizing
- [ ] Add volatility targeting
- [ ] Add stop loss / take profit calculations
- [ ] Add position limit reasons (volatility/liquidity/confidence)

**Duration**: 1 week

### Phase 4: STRATEGY CLASSIFIER + ALPHA (Week 6)
**Priority: MEDIUM**

- [ ] Implement StrategyClassifier
- [ ] Add liquidity filter ($50M minimum)
- [ ] Add alpha score (filter beta-chasers)
- [ ] Implement regime change detection (CUSUM)

**Duration**: 1 week

### Phase 5: VALIDATION + BENCHMARK (Week 7-8)
**Priority: MEDIUM**

- [ ] Walk-forward validation (OOS regime splits)
- [ ] Monte Carlo simulation
- [ ] Compare to SPY/QQQ buy-and-hold
- [ ] Stress test: 2020 crash simulation
- [ ] Final performance report

**Duration**: 2 weeks

---

## Success Criteria

### Primary (Must Achieve)
1. ✅ **Sharpe Ratio > 1.0** (after 0.1% costs)
2. ✅ **Max Drawdown < 15%**
3. ✅ **Beat Buy-and-Hold** (risk-adjusted)
4. ✅ **Win Rate > 55%** (day), **> 60%** (swing)

### Secondary (Nice to Have)
5. **Profit Factor > 1.5**
6. **Alpha Score > 0.5**
7. **Robust to 0.25% costs**

---

## Key Formulas

### 1. Position Sizing (Half-Kelly + Vol Targeting)
```
Position = min(Half-Kelly, 0.02/CurrentVol, 0.25)
```

### 2. Transaction Costs
```
Day Trade: 0.2% per round-trip
Swing: 0.1% per round-trip
```

### 3. Alpha Score
```
Alpha = Confidence × (1 - SPY_Correlation)
If Alpha < 0.4 → NO TRADE
```

### 4. Sample Weights
```
High Vol Regime (vol_rank > 1.5): weight = 2.0
Low Vol Regime (vol_rank < 0.5): weight = 0.3
Normal: weight = 1.0
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-------------|
| Lookahead bias | Strict lagged features, walk-forward |
| Transaction costs | 0.1% baseline, sensitivity tests |
| Overfitting | Walk-forward, OOS validation, Monte Carlo |
| Labeling leakage | Time-series correct labels |
| Regime change | CUSUM detection, monthly retrain |

---

## File Structure

```
v3.0/
├── IMPROVEMENT_PLAN_V3.md           # This file (Grok version)
├── IMPROVEMENT_PLAN.md              # Previous (Gemini version)
├── prototypes/
│   ├── lagged_features.py           # NEW: Strict lagged feature computation
│   ├── transaction_cost_sim.py      # NEW: 0.1% cost simulation
│   ├── sample_weight_train.py       # Profit-weighted training
│   ├── half_kelly_sizing.py         # Half-Kelly position sizing
│   └── return_regressor.py          # Direct return prediction
└── metrics/
    ├── profitability_metrics.py
    ├── walk_forward_backtest.py
    └── regime_detector.py
```

---

## Summary

This V3.0 plan incorporates Grok's professional feedback:

1. **Phase 0 first**: Kill lookahead bias before any model changes
2. **Transaction costs**: Model 0.1% round-trip from day 1
3. **Simplify**: 4 core models + 2 return regressors
4. **Half-Kelly**: Conservative sizing with vol cap
5. **Walk-forward**: Mandatory OOS validation

**Status**: Ready for implementation  
**Next Step**: Begin Phase 0 - Audit features for lookahead bias

---

**Document Version**: 3.0.1  
**Last Updated**: 2026-03-07  
**Review**: Grok (Quant Trading Pro) feedback integrated
