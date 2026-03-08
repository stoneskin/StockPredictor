# V3.0 Improvement Plan - Strategy-Aware Stock Prediction

**Date**: 2026-02-28  
**Version**: 3.0 (planned)  
**Current**: V2.5.2  
**Author**: StockPredictor Team  
**Review**: Incorporates feedback from Gemini (Quant Trading Professional)

---

## 🎯 Executive Summary

V3.0 transforms StockPredictor from a **classification system** into a **trading strategy advisor**. Instead of just predicting price direction, it will:

1. **Classify stocks by strategy fit**: Day Trading vs. Swing Trading
2. **Recommend actionable signals** with profitability expectations
3. **Optimize for risk-adjusted returns**, not just accuracy
4. **Provide position sizing based on volatility targeting**
5. **Beat buy-and-hold** on a risk-adjusted basis
6. **Include liquidity filters and regime change detection**

---

## 📊 Current State (V2.5.2)

### What Works:
- ✅ 4-class classification (UP, DOWN, UP_DOWN, SIDEWAYS)
- ✅ 6 horizons (3,5,10,15,20,30 days)
- ✅ 5 thresholds (0.75%,1%,1.5%,2.5%,5%)
- ✅ XGBoost as best model (90-97% accuracy on some combos)
- ✅ Backtesting API (but with lookahead bias)

### Critical Problems:
1. **Accuracy ≠ Profitability**: 95% accuracy often just predicts SIDEWAYS (majority class)
2. **No strategy guidance**: User doesn't know if stock is better for day trade or swing
3. **Missing actionable signals**: UP predictions are rare and not optimized
4. **Wrong optimization**: Training maximizes accuracy, not trading returns
5. **Lookahead bias**: Backtest artificially inflates performance (scaling, features)
6. **No risk management**: No position sizing, stop loss, or drawdown control
7. **Ignores transaction costs**: 0.1% per trade can eliminate profits
8. **No liquidity filter**: Can't trade illiquid stocks

---

## 🎯 V3.0 Vision

### Core Principle:
**Return != Profitability. Risk-Adjusted Return == Profitability.**

### New Output:
```json
{
  "symbol": "TQQQ",
  "date": "2026-02-27",
  "strategy": "swing",
  "strategy_confidence": 0.78,
  "predictions": [{
    "horizon": 20,
    "threshold": 0.015,
    "model_used": "xgboost",
    "direction": "UP",
    "probabilities": {...},
    "confidence": 0.85,
    "expected_sharpe": 1.2,           // NEW: risk-adjusted return
    "expected_return_pct": 3.2,
    "expected_volatility_pct": 1.8,
    "expected_win_rate": 0.68,
    "risk_level": "medium",
    "action": "BUY",
    "recommended_position": 0.15,     // Volatility-targeted
    "position_limit_reason": "volatility",
    "stop_loss_pct": 2.5,
    "take_profit_pct": 5.0,
    "liquidity_ok": true,
    "alpha_score": 0.65              // NOT just beta (SPY correlation check)
  }]
}
```

---

## 🔄 Key Changes (Incorporating Gemini Feedback)

### 1. Strategy Classification with Liquidity Filter

**Key Insight from Gemini**: "Add a `liquidity_filter`. If the 20-day average dollar volume is too low, the strategy should default to NEUTRAL."

```python
class StrategyClassifier:
    """
    Determine if current market is suitable for:
    - DAY_TRADING: High volatility, range-bound, quick momentum
    - SWING_TRADING: Trending, sustained moves
    - NEUTRAL: No trade (low liquidity, high correlation, unclear regime)
    """
    
    def select_strategy(self, df):
        # 1. LIQUIDITY CHECK (Gemini: market impact)
        avg_dollar_volume = (df['close'] * df['volume']).rolling(20).mean().iloc[-1]
        if avg_dollar_volume < MIN_DOLLAR_VOLUME:  # e.g., $50M
            return 'neutral', 0.9, 'low_liquidity'
        
        # 2. CORRELATION CHECK (Gemini: beta-chaser detection)
        spy_correlation = df['correlation_spy_20d'].iloc[-1]
        if spy_correlation > 0.95:
            # Stock just follows market, no alpha
            return 'neutral', 0.8, 'high_market_correlation'
        
        # 3. VOLATILITY & TREND
        volatility = df['atr_ratio'].iloc[-1] * 100
        trend_strength = df['trend_strength'].iloc[-1] * 100
        
        # 4. UP_DOWN treatment (Gemini: "high-risk for swing, opportunity for day")
        bb_squeeze = df['bb_squeeze'].iloc[-1]
        momentum_reversal = (df['momentum_5'].iloc[-1] * df['momentum_5'].iloc[-2]) < 0
        
        # Decision tree with confidence scoring
        scores = {'day_trade': 0, 'swing': 0, 'neutral': 0}
        
        if volatility > 2.5 and trend_strength < 1.0:
            scores['day_trade'] += 3
        if 1.0 < volatility < 2.0 and trend_strength > 2.0:
            scores['swing'] += 3
        
        if bb_squeeze or momentum_reversal:
            scores['day_trade'] += 1  # Mean reversion opportunity
        
        if spy_correlation > 0.85:
            scores['neutral'] += 2
        
        best = max(scores, key=scores.get)
        confidence = scores[best] / sum(scores.values())
        
        return best, confidence
```

### 2. Volatility-Targeted Position Sizing (Gemini's Key Advice)

**Gemini**: "Position sizing should be driven by Volatility Targeting. If ATR doubles, position size should halve."

```python
def calculate_position_size(
    volatility_pct: float,          # ATR ratio (e.g., 0.02 = 2%)
    confidence: float,
    strategy: str,
    portfolio_value: float,
    max_portfolio_risk_pct: float = 0.02  # Max 2% of portfolio at risk
) -> float:
    """
    Volatility-targeted position sizing (NOT Kelly).
    
    Formula: Position Size = (Portfolio Risk) / (Volatility * Stop Multiplier)
    
    If volatility doubles → position size halves.
    """
    
    # Base position as % of portfolio
    if strategy == "day_trade":
        base_risk_pct = 0.01  # 1% of portfolio per trade
        stop_multiplier = 1.5  # Stop at 1.5x ATR
    elif strategy == "swing":
        base_risk_pct = 0.015  # 1.5% of portfolio
        stop_multiplier = 2.5   # Wider stop
    else:
        return 0.0
    
    # Adjust by confidence (but cap at 1.0)
    confidence_factor = min(confidence, 1.0)
    
    # Volatility scaling: base_risk adjusted by (typical_vol / current_vol)
    typical_vol = 0.015  # 1.5% ATR is typical
    vol_scaling = typical_vol / max(volatility_pct, 0.005)
    vol_scaling = min(vol_scaling, 2.0)  # Cap at 2x
    
    position_risk = base_risk_pct * confidence_factor * vol_scaling
    position_risk = min(position_risk, max_portfolio_risk_pct)
    
    # Convert to $ amount and return as fraction of portfolio
    return position_risk
```

**Example**:
- Volatility = 3% (double normal)
- Confidence = 0.8
- Portfolio = $100k
- Normal position: $10k (10%)
- With volatility scaling: $10k * (1.5%/3%) = $5k (5%)
**Result**: Risk stays constant regardless of volatility.

### 3. Profit-Weighted Training via Sample Weights

**Gemini's Refinement**: "Instead of custom differentiable loss, use `sample_weight`."

**⚠️ CRITICAL FIX (Grok Round 2)**: Original version used future data. Must compute weights using ONLY information available at time t.

```python
def compute_sample_weights(df_train, threshold, lookback=20):
    """
    Assign higher weights to samples with high alpha potential.
    NO LOOKAHEAD: Only use information available at prediction time.
    
    Uses rolling volatility and momentum strength to estimate "interesting" days.
    """
    weights = np.ones(len(df_train))
    
    # Compute rolling metrics (no future leakage)
    returns = df_train['close'].pct_change()
    rolling_vol = returns.rolling(lookback).std()
    rolling_momentum = df_train['close'] / df_train['close'].rolling(lookback).mean() - 1
    
    for i, idx in enumerate(df_train.index):
        # Weight based on historical volatility (available at time t)
        vol_rank = rolling_vol.iloc[idx] / rolling_vol.iloc[:idx].quantile(0.75) if idx > lookback else 1.0
        
        if vol_rank > 1.5:  # High volatility regime
            weights[i] = 2.0
        elif vol_rank < 0.7:  # Low volatility (noise)
            weights[i] = 0.5
        else:
            weights[i] = 1.0
        
        # Additional: weight by price momentum strength
        momentum_abs = abs(rolling_momentum.iloc[idx])
        if momentum_abs > 0.05:  # Strong trend
            weights[i] *= 1.5
    
    return weights

# Train XGBoost with sample weights
model = xgb.XGBClassifier()
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Grok's Advice**: "Compute weights strictly inside each walk-forward training window" to absolutely prevent leakage.

### 4. Transaction Costs & Benchmark (Gemini Answers)

**Gemini's Recommendations**:

1. **Transaction Costs (Tiered)**:
   - **Day Trading**: 0.15-0.25% per trade (higher due to turnover, slippage)
   - **Swing Trading**: 0.08-0.12% per trade (lower, fewer trades)
   - **Baseline used in backtest**: 0.1% (conservative average)
   - Always run sensitivity: 0.05%, 0.15%, 0.25%

2. **Benchmark Choice**:
   - **Primary benchmark**: **SPY** (institutional standard for alpha)
   - **Secondary**: Specific symbol B&H (e.g., QQQ) for strategy comparison
   - Goal: Beat SPY on risk-adjusted basis ( Sharpe(Strategy) > Sharpe(SPY) )

3. **Minimum Liquidity Threshold (Tiered)**:
   - Day Trade: ≥ $100M 20-day avg dollar volume
   - Swing Trade: ≥ $40M 20-day avg dollar volume
   - Below $30M: Force NEUTRAL regardless of signal

**Implementation**:
```python
def get_transaction_cost(strategy: str) -> float:
    if strategy == "day_trade":
        return 0.002  # 0.2% (20 bps)
    else:  # swing
        return 0.001  # 0.1% (10 bps)

def check_liquidity(df, strategy) -> bool:
    avg_dollar_volume = (df['close'] * df['volume']).rolling(20).mean().iloc[-1]
    if strategy == "day_trade":
        return avg_dollar_volume >= 100_000_000  # $100M
    else:
        return avg_dollar_volume >= 40_000_000   # $40M
```

### 5. UP_DOWN Class Handling

**Gemini**: "Treat `UP_DOWN` as high-risk/no-trade for Swing, but prime opportunity for Mean Reversion (Day)."

```python
def interpret_up_down(confidence: float, strategy: str, volatility: float) -> Dict:
    """
    UP_DOWN means price went both up AND down > threshold.
    Interpretation differs by strategy:
    """
    if strategy == "swing":
        # Swing doesn't want choppy markets
        return {
            "action": "HOLD",
            "reason": "choppy_market",
            "risk_level": "high",
            "position_size": 0.0
        }
    elif strategy == "day_trade":
        # Day trader loves volatility (can scalp both ways)
        if confidence > 0.7 and volatility > 2.5:
            return {
                "action": "BUY",  # Mean reversion play
                "reason": "volatility_breakout",
                "position_size": 0.1,  # Small size (high risk)
                "stop_loss": 2.0,
                "take_profit": 3.0,
                "max_holding_days": 1
            }
        else:
            return {"action": "HOLD", "reason": "low_confidence"}
```

### 5. Model Proliferation Problem (Critical - Grok Round 2)

**Grok's Finding**: "Maintaining 30 separate models (6 horizons × 5 thresholds) creates severe multiple-testing and maintenance overhead."

**Recommendation**: Consolidate to **4-6 core models**:

| Strategy | Horizon | Threshold | Model |
|----------|---------|-----------|-------|
| Day Trade | 3 days | 1.0% | XGBoost |
| Day Trade | 5 days | 1.5% | XGBoost |
| Swing | 10 days | 1.5% | XGBoost |
| Swing | 20 days | 2.5% | XGBoost |

**Rationale**:
- Day trading: Short horizons (3-5d) with lower thresholds (1-1.5%) for quick moves
- Swing trading: Longer horizons (10-20d) with medium thresholds (1.5-2.5%) for sustained trends
- Eliminates noisy combinations (e.g., 3d/2.5%, 30d/0.75%)
- Reduces training time from 90 min to ~12 min
- Easier to maintain and monitor

**Implementation**:
```python
CORE_MODELS = [
    {'strategy': 'day_trade', 'horizon': 3, 'threshold': 0.01},
    {'strategy': 'day_trade', 'horizon': 5, 'threshold': 0.015},
    {'strategy': 'swing', 'horizon': 10, 'threshold': 0.015},
    {'strategy': 'swing', 'horizon': 20, 'threshold': 0.025},
]
```

**Phase 2** will retrain only these 4 core models with profit-weighted sample weights.

### 6. Backtest with Proper Walk-Forward and Scaling

**Gemini**: "Use Rolling Window Pipeline. Scale using only training window stats."

```python
class WalkForwardBacktest:
    """
    Walk-forward validation (anchored or expanding window).
    No lookahead bias in scaling.
    """
    
    def __init__(self, initial_train_days=500, test_days=30):
        self.initial_train_days = initial_train_days
        self.test_days = test_days
    
    def run(self, df):
        results = []
        
        for i in range(self.initial_train_days, len(df), self.test_days):
            # Split
            train_start = i - self.initial_train_days
            train_end = i
            test_start = i
            test_end = min(i + self.test_days, len(df))
            
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            # 1. Fit scaler on TRAIN only
            scaler = StandardScaler()
            X_train = self._compute_features(train_df)
            scaler.fit(X_train)
            
            # 2. Transform train and test with SAME scaler
            X_train_scaled = scaler.transform(X_train)
            X_test = self._compute_features(test_df)
            X_test_scaled = scaler.transform(X_test)
            
            # 3. Train on scaled train
            model = self.train_model(X_train_scaled, y_train, sample_weights)
            
            # 4. Predict on scaled test
            preds = model.predict(X_test_scaled)
            
            # 5. Simulate trading (with transaction costs)
            for j, idx in enumerate(test_df.index):
                result = self._simulate_trade(
                    preds[j], 
                    test_df.iloc[j],
                    strategy=self.determine_strategy(test_df.iloc[j])
                )
                results.append(result)
        
        return pd.DataFrame(results)
```

### 6. Forward Sharpe Ratio as Target

**Gemini**: "Predict Forward Sharpe Ratio instead of just % Return."

Instead of predicting raw return, predict:

```python
def compute_forward_sharpe(df, idx, horizon, threshold, risk_free_rate=0.02/252):
    """
    Compute realized Sharpe over horizon.
    """
    if idx + horizon >= len(df):
        return np.nan
    
    entry_price = df['close'].iloc[idx]
    future_prices = df['close'].iloc[idx+1:idx+horizon+1]
    returns = future_prices.pct_change().dropna()
    
    if len(returns) == 0:
        return 0
    
    # Annualized Sharpe
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return sharpe
```

Then train a regression model to predict Sharpe, or use Sharpe in custom evaluation.

### 7. Alpha Score: Filter Out Beta Chasers

```python
def compute_alpha_score(prediction_confidence: float, spy_correlation: float, r_squared: float) -> float:
    """
    If stock moves 95% like SPY, you're just holding beta.
    Alpha score discounts confidence by correlation.
    """
    beta_weight = 1.0 - spy_correlation  # 1 - 0.95 = 0.05
    alpha_score = prediction_confidence * beta_weight
    
    # Further discount if R² is extremely high (pure beta)
    if r_squared > 0.90:
        alpha_score *= 0.5
    
    return alpha_score
```

**Usage**: Only take trades with `alpha_score > 0.4`.

### 8. Regime Change Detection (Kill Switch)

**Gemini**: "Implement CUSUM Filter for Structural Breaks."

```python
def detect_regime_change(df, column='returns', threshold=3.0):
    """
    CUSUM filter to detect abrupt regime changes.
    Returns True if market regime changed dramatically.
    """
    returns = df[column].pct_change().dropna()
    # FIX: Use expanding mean (only past data, no lookahead)
    cumsum = (returns - returns.expanding().mean()).cumsum()
    std = returns.expanding().std()
    
    # Detect if cumsum deviates by threshold * std
    if abs(cumsum.iloc[-1]) > threshold * std:
        return True, "structural_break"
    
    # Or detect volatility regime change
    recent_vol = returns.rolling(20).std().iloc[-1]
    long_vol = returns.rolling(60).std().iloc[-1]
    if recent_vol > long_vol * 2:
        return True, "volatility_regime_change"
    
    return False, "stable"
```

If regime change detected → Move all positions to cash until StrategyClassifier re-evaluates.

---

## 📋 Revised Implementation Roadmap

### Phase 0: Foundation (Before Major Changes)
- [ ] **Document current baseline**: Run full backtest on V2.5.2, record metrics
- [ ] **Implement proper Walk-Forward backtest** (with rolling scaling)
- [ ] **Fix all lookahead bias sources** (scaling, feature medians)
- [ ] **Add transaction cost modeling** (0.1% per trade)
- [ ] **Add liquidity filter** to data loading (min dollar volume)

**Duration**: 1 week

### Phase 1: Strategy Classifier & Alpha Filter
- [ ] Implement `StrategyClassifier` with liquidity + correlation filters
- [ ] Train binary classifier: `is_day_trade_able` vs `is_swing_able`
- [ ] Add `alpha_score` computation to API
- [ ] Update API response with `strategy`, `strategy_confidence`, `liquidity_ok`, `alpha_score`
- [ ] Create strategy selection validation (test on historical regimes)

**Duration**: 1 week

### Phase 2: Profit-Weighted Training
- [ ] Compute sample weights based on forward returns/Sharpe
- [ ] Retrain all 30 models with `sample_weight` parameter (XGBoost/GradientBoosting support this)
- [ ] For models without sample_weight (SVM, etc.), use stratified sampling or SMOTE variants
- [ ] Evaluate: precision_up, recall_up, F1_macro, AND profit-weighted accuracy
- [ ] Save new models with `_pw` (profit-weighted) suffix

**Duration**: 1 week

### Phase 3: Volatility-Targeted Position Sizing & Risk
- [ ] Implement `calculate_position_size()` with inverse volatility scaling
- [ ] Add `expected_volatility_pct` prediction (regression from ATR)
- [ ] Compute `risk_level`: "low" (<1% ATR), "medium" (1-2%), "high" (>2%)
- [ ] Add stop loss & take profit calculations based on ATR
- [ ] Implement `position_limit_reason` field (volatility/liquidity/confidence)

**Duration**: 5 days

### Phase 4: Enhanced Backtest with Trading Simulation
- [ ] Rewrite backtest: walk-forward validation (anchored)
- [ ] Simulate actual trades: entry → exit (stop/target/time)
- [ ] Track: P&L, win rate, profit factor, Sharpe, max drawdown, Ulcer Index
- [ ] Compare to buy-and-hold benchmark (SPY/QQQ)
- [ ] Generate detailed performance reports (monthly, by strategy, by horizon)
- [ ] Charts: equity curve, drawdown chart, monthly returns heatmap

**Duration**: 1 week

### Phase 5: API & Documentation Updates
- [ ] Update `/predict` and `/predict/multi` with all new fields
- [ ] Update `/backtest` to return P&L simulation results
- [ ] Add `/strategy` endpoint: returns current strategy classification without full prediction
- [ ] Update `v3.0/docs/` with new response schema
- [ ] Create `STRATEGY_GUIDE.md`: when to use day vs swing
- [ ] Document risk management approach and limitations

**Duration**: 3 days

### Phase 6: Validation & Stress Testing
- [ ] Walk-forward validation: train on 2020-2022, test 2023; train on 2021-2023, test 2024
- [ ] Out-of-sample test: last 3 months completely unseen
- [ ] Monte Carlo simulation: randomize trade order, check robustness
- [ ] Sensitivity analysis: transaction costs (0.05%, 0.1%, 0.2%)
- [ ] Drawdown stress: What happens in 2020-style crash?
- [ ] Compare against simple benchmarks: buy-and-hold, 50/50, momentum

**Duration**: 1 week

---

## 📊 New Success Criteria (Risk-Adjusted)

### Primary (Must Achieve):
1. **Sharpe Ratio > 1.0** (annualized, after 0.1% per trade cost)
2. **Max Drawdown < 15%** (absolute) OR **Ulcer Index < 10%**
3. **Beat Buy-and-Hold** on risk-adjusted basis (higher Sharpe or lower MDD)
4. **Win Rate > 55%** (day), **> 60%** (swing)

### Secondary (Nice to Have):
5. **Profit Factor > 1.5**
6. **Alpha Score > 0.5** (not just beta chasing)
7. **Strategy Classification Accuracy > 75%** (on historical labeling)
8. **Transaction Cost Robust**: Still profitable at 0.2% per trade

---

## 📂 New Files/Folders (Updated)

```
v3.0/
├── IMPROVEMENT_PLAN.md              # This revised plan
├── QUICK_REFERENCE.md               # Summary (to be updated)
├── STRATEGY_DESIGN.md               # Detailed strategy logic
├── RISK_MANAGEMENT.md               # Volatility targeting, position sizing
├── FEATURE_REQUIREMENTS.md          # New features: alpha_score, liquidity
├── api_v3_spec.json                 # OpenAPI spec with new fields
├── prototypes/
│   ├── strategy_selector_proto.py
│   ├── volatility_position_sizing.py
│   ├── sample_weight_calculator.py
│   ├── walk_forward_backtest_proto.py
│   └── cusum_regime_detector.py
└── metrics/
    ├── profitability_metrics.py    # Sharpe, Ulcer, profit factor
    ├── strategy_selector.py        # StrategyClassifier
    ├── position_sizing.py          # Volatility-targeted sizing
    ├── alpha_filter.py             # Alpha score, correlation filter
    ├── regime_detector.py          # CUSUM, structural breaks
    └── transaction_costs.py        # Slippage modeling
```

Updates to existing:
- `v2.5/src/train_v2_5.py` → Add sample_weight computation, retrain with weights
- `v2.5/src/inference_v2_5.py` → Add strategy fields, alpha_score, position sizing
- `v2.5/src/backtest_v2_5.py` → Replace with walk-forward version, trading simulation
- `v2.5/docs/API_REFERENCE.md` → Document all new fields
- `v2.5/README.md` → Add strategy guide, risk warnings
- `CHANGELOG.md` → Add V3.0 planning section

---

## ⚠️ Critical Risks & Mitigations (Enhanced)

| Risk | Impact | V2.5 Status | V3.0 Mitigation |
|------|--------|-------------|-----------------|
| Overfitting | High | Accuracy may be overfitted | Walk-forward validation, out-of-sample, Monte Carlo |
| Lookahead bias | Critical | Present in scaling & features | Rolling window scaling only, time-correct features |
| Transaction costs | High | Ignored | Model 0.1% per trade, test sensitivity 0.05-0.2% |
| Liquidity assumption | Medium | Ignored | Dollar volume filter <$50M = NO TRADE |
| Regime change | High | Static model | CUSUM kill switch, monthly retraining |
| Alpha vs Beta | Medium | Not measured | Alpha score = confidence * (1 - SPY_correlation) |
| UP_DOWN mispricing | High | Single class | Strategy-specific interpretation (day=opportunity, swing=avoid) |
| Position sizing | Medium | None | Volatility targeting (inverse to ATR) |
| Model decay | High | No monitoring | Performance tracking dashboard, drift alerts |

---

## 💡 Gemini's Recommendations - Implementation Status

| Gemini Feedback | Status in V3.0 Plan |
|-----------------|---------------------|
| Elevate Max Drawdown, Ulcer Index | ✅ Added to success criteria |
| Volatility-targeted position sizing | ✅ Detailed implementation |
| UP_DOWN: day=opportunity, swing=avoid | ✅ Strategy-specific interpretation |
| Liquidity filter (dollar volume) | ✅ Added to StrategyClassifier |
| Rolling window scaling (no lookahead) | ✅ Walk-forward backtest |
| Sample weights vs custom loss | ✅ Using XGBoost sample_weight |
| Alpha decay study | ⚠️ Need to add Phase 1.5 |
| CUSUM regime change detection | ✅ Added Phase 6 |
| Forward Sharpe prediction | ✅ Optional: predict Sharpe instead of return |
| SPY correlation filter (beta-chaser) | ✅ Alpha score computation |
| Market impact modeling | ✅ Liquidity filter handles this |

---

## 🎓 Key Formulas Summary

### 1. Volatility-Targeted Position Size
```
Position Risk ($) = Portfolio × Base Risk Pct × Confidence × (Typical Vol / Current Vol)
Position Size (Shares) = Position Risk / (Current Price × Stop Distance)
```

### 2. Alpha Score
```
Alpha Score = Prediction Confidence × (1 - SPY Correlation)
If Alpha Score < 0.4 → DO NOT TRADE (just beta)
```

### 3. Sample Weight
```python
if max_future_move > threshold * 2: weight = 3.0
elif max_future_move > threshold: weight = 2.0
elif max_future_move < threshold * 0.5: weight = 0.5
else: weight = 1.0
```

### 4. Strategy Confidence
```
Strategy Score = sum(feature_matches) / total_features_checked
```

---

## ❓ Remaining Open Questions

Even with Gemini's feedback, we need your input on:

1. **Transaction Costs**: 
   - Gemini suggests 0.1% per trade for backtest. Agreed?
   - Should we model per-trade commission + spread separately?

2. **Minimum Liquidity Threshold**:
   - Gemini asks: what dollar volume cutoff?
   - Suggestion: $50M 20-day average → "NO TRADE"
   - Your preference?

3. **Model Architecture**:
   - Gemini suggests **separate models** (modular) which I'm following.
   - Train: classification (direction) + regression (Sharpe) + classification (strategy)?
   - Or single multi-output model? Gemini prefers separate.

4. **Expected Sharpe vs Expected Return**:
   - Gemini: Predict Forward Sharpe instead of % Return.
   - Pros: Risk-adjusted, more realistic.
   - Cons: Adds complexity, need volatility model.
   - Your call: Sharpe or Return?

5. **Retraining Frequency**:
   - Gemini: Monthly retraining due to regime changes.
   - Agreed? Or quarterly?

6. **Benchmark Choice**:
   - Buy-and-hold what? QQQ itself? SPY?
   - Gemini implies SPY as benchmark for alpha.
   - Clarify: compare to QQQ buy-and-hold or risk-adjusted?

---

## 🎯 Your Next Steps

1. **Review this updated plan** with Gemini's feedback integrated
2. **Decide on open questions** (see above)
3. **Prioritize phases**: Should we do all 6 phases or start with Phase 0-2 first?
4. **Resource allocation**: How much time per week can you dedicate?
5. **Start implementation**: Should I begin with **Phase 0** (baseline & walk-forward fix)?

---

## 📞 Gemini's Final Question

**Gemini asked**: *"Would you like me to draft a Python implementation for the 'Volatility-Targeted Position Sizing' logic to replace the simple 0-1 fraction?"*

**My response**: Already drafted in this plan (see Section 3). But we can refine further.

**Your decision**: 
- Option A: I implement the full plan step-by-step
- Option B: We prototype key components first (strategy selector, walk-forward backtest) before committing to full build
- Option C: Adjust plan further based on your feedback

---

**Document Version**: 2.0 (with Gemini feedback)  
**Last Updated**: 2026-02-28  
**Status**: Ready for Review & Approval to Start Phase 0
