# V3.0 Plan - Executive Summary

## 🎯 What Changed After Gemini's Review

Gemini (Quant professional) provided critical feedback. Key changes:

### Before (Original Plan):
- Focus on accuracy and raw returns
- Simple position sizing (confidence-based)
- Treat UP_DOWN same as other classes
- Backtest with potential lookahead bias
- No liquidity or alpha filtering

### After (Revised Plan):
- **Risk-adjusted returns**: Sharpe ratio, Max Drawdown, Ulcer Index
- **Volatility-targeted position sizing**: Higher volatility = smaller position
- **Strategy-specific UP_DOWN**: Day trade = opportunity, Swing = avoid
- **Walk-forward backtest**: No lookahead in scaling or features
- **Sample weights**: Weight profitable periods higher during training
- **Liquidity filter**: Low volume = NO TRADE
- **Alpha score**: Filter out beta-chasers (high SPY correlation)
- **Regime change kill switch**: CUSUM detector → go to cash
- **Transaction costs**: Model 0.1% per trade

---

## 📊 Comparison Table

| Aspect | V2.5 (Current) | V3.0 (Plan) |
|--------|----------------|-------------|
| **Target** | Accuracy | Risk-Adjusted Returns (Sharpe) |
| **Position Size** | None (assume 100%) | Volatility-targeted (0-20%) |
| **Strategy** | None | Day Trade OR Swing |
| **UP_DOWN** | Random noise | Day=Opportunity, Swing=Avoid |
| **Liquidity** | Ignored | Filter: <$50M daily volume = NO TRADE |
| **Backtest** | Single split | Walk-forward (rolling window) |
| **Scaling** | Full dataset (lookahead!) | Rolling window (no lookahead) |
| **Training Weights** | Equal | Sample weights (profitability-based) |
| **Alpha Filter** | No | Yes (SPY correlation < 0.95) |
| **Regime Change** | Static | CUSUM kill switch |
| **Transaction Costs** | 0% | 0.1% per trade |
| **Risk Metrics** | None | Sharpe, Max DD, Ulcer, Profit Factor |

---

## 🎬 Strategy Decision Flow

```
User requests prediction for TQQQ
    ↓
StrategyClassifier evaluates:
  - Liquidity: $120M daily volume → OK
  - SPY correlation: 0.87 → Has alpha
  - Volatility: 2.1% ATR → Swing territory
  - Trend strength: 3.5% → Strong uptrend
  → Strategy: SWING (confidence: 82%)
    ↓
Model predicts: UP (confidence: 78%)
    ↓
Compute risk metrics:
  - Expected volatility: 1.8%
  - Expected Sharpe: 1.3
  - Alpha score: 0.65 (78% × (1-0.87))
    ↓
Generate action:
  - Position size: 15% (volatility-targeted)
  - Stop loss: -2.5% (2.5 × 1.8% ATR)
  - Take profit: +5% (3 × 1.8%)
  - Action: BUY
    ↓
Return response with all fields
```

---

## 💰 Example: How Volatility Targeting Works

**Scenario A (Low Volatility)**:
- Stock XYZ: ATR = 1%
- Portfolio: $100k
- Day trade position: $100k × 1% risk = $1k risk per trade
- Stop at 1.5×ATR = 1.5% away
- Position size = $1k / (1.5% × price) = **$66,667** (6.7% of portfolio)

**Scenario B (High Volatility)**:
- Stock ABC: ATR = 3% (3× higher)
- Same $100k, 1% risk target
- Stop at 1.5×ATR = 4.5% away
- Position size = $1k / 4.5% = **$22,222** (2.2% of portfolio)

**Result**: Risk stays constant at $1k regardless of volatility.

---

## 🏆 Success Criteria (Quantitative)

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Sharpe Ratio > 1.0** | Annualized | Risk-adjusted return (higher = better) |
| **Max Drawdown < 15%** | Absolute | Portfolio survival (won't blow up) |
| **Win Rate > 55%** (day) / **> 60%** (swing) | Per-strategy | Signal quality |
| **Profit Factor > 1.5** | (Wins ÷ Losses) | Positive expectancy |
| **Beat Buy & Hold** | Risk-adjusted | Alpha generation |
| **Alpha Score > 0.5** | (1 - SPY corr) | Not just beta chasing |
| **Commission Robust** | Profitable @ 0.2%/trade | Real-world viability |

---

## 🕐 Timeline (Revised)

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| **Phase 0**: Baseline & Walk-Forward Fix | 1 week | None (start now) |
| **Phase 1**: Strategy Classifier | 1 week | Phase 0 complete |
| **Phase 2**: Profit-Weighted Training | 1 week | Phase 0 complete |
| **Phase 3**: Risk Management | 5 days | Phase 1, 2 |
| **Phase 4**: Trading Simulation Backtest | 1 week | Phase 1-3 |
| **Phase 5**: API & Docs | 3 days | Phase 4 |
| **Phase 6**: Validation & Stress Test | 1 week | Phase 5 |
| **Total** | **~6 weeks** | Mostly sequential |

---

## ❓ Decisions Needed (Priority Order)

### 1. Transaction Costs (High Priority)
**Question**: Model 0.1% per trade (commission + spread + slippage)? Or break it down?
- 0.05% commission + 0.05% slippage/spread?
- Template: 0.1% total

**My recommendation**: 0.1% flat per trade (round-trip)

### 2. Liquidity Threshold (High Priority)
**Question**: Minimum dollar volume for "tradeable"?
- Gemini suggests implicit cutoff
- **Recommendation**: $50M 20-day average dollar volume
- Anything below → Strategy = NEUTRAL

### 3. Benchmark (High Priority)
**Question**: Buy-and-hold what?
- Option A: QQQ itself (what we're predicting)
- Option B: SPY (broader market, less tech)
- **Recommendation**: QQQ B&H as primary benchmark (fair comparison)

### 4. Retraining Frequency (Medium)
**Question**: How often retrain V3.0 models?
- Gemini: Monthly (due to regime changes)
- Option: Weekly? Quarterly?
- **Recommendation**: Monthly (first Sunday of month)

### 5. Model Architecture (Medium)
**Question**: Separate models or multi-output?
- Gemini prefers **separate** (modular, interpretable)
  - Model 1: Direction classifier (4-class)
  - Model 2: Sharpe regressor
  - Model 3: Strategy classifier
- Option: Single multi-output model (captures interactions)
- **Recommendation**: Separate (as in plan)

### 6. Predict Target: Sharpe or Return? (Medium)
**Question**: Should model predict:
- A) Expected return % (simpler, current approach)
- B) Expected Sharpe (risk-adjusted, Gemini's suggestion)
- **Recommendation**: Start with A (return) then add B later (Phase 3)

### 7. Phase Prioritization (Low)
**Question**: Full 6-phase plan or MVP first?
- MVP = Phases 0-2 only (walk-through + strategy + weighted training)
- Then decide if Phase 3-6 worth it
- **Recommendation**: Full plan, but stop after MVP if results poor

---

## 🚀 What I Need From You

Please provide answers to the 7 questions above (can be 1-liners).

Once answered, I'll:

1. ✅ Start **Phase 0** immediately: Implement walk-forward backtest with proper scaling
2. ✅ Document baseline V2.5.2 performance (so we know V3.0 is better)
3. ✅ Build the StrategyClassifier prototype
4. ✅ Create sample_weight calculator
5. ✅ Provide weekly progress updates

**Shall I proceed with Phase 0 now?** (I'll start while you review the plan)

---

## 📚 Key References

- **Full Plan**: `v3.0/IMPROVEMENT_PLAN.md` (revised with Gemini feedback)
- **Quick Ref**: `v3.0/QUICK_REFERENCE.md`
- **Strategy Prototype**: `v3.0/prototypes/strategy_selector_proto.py` (already created)

---

**Bottom Line**: V3.0 will transform StockPredictor from an academic exercise into a **professionally-grounded trading system** with proper risk management, strategy guidance, and risk-adjusted performance.

**Risk of NOT doing this**: Model achieves 95% accuracy by always saying "SIDEWAYS" and makes no money in real trading.

**Risk of doing this**: Complexity increases, but we maintain modular design to isolate components for testing.

**Your call**: Approve plan and answer the 7 questions? Or want adjustments first?
