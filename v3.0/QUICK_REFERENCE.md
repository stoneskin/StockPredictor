# V3.0 Quick Reference

## 🎯 Core Goal
Transform from pure prediction to **strategy-aware trading advisor** that beats buy-and-hold.

## ✅ What's New (vs V2.5)

| Feature | V2.5 | V3.0 |
|---------|------|------|
| Output | Direction (UP/DOWN/SIDEWAYS) | Direction + Strategy + Expected Return |
| Optimization | Accuracy (all classes equal) | Profitability (UP weight = 4x) |
| Backtest | Classification accuracy | P&L simulation vs buy-and-hold |
| Recommendation | None | "Day trade" or "Swing" + position sizing |
| Risk metrics | None | Stop loss, take profit, max drawdown |

## 🎬 Two Strategies

### Day Trading
- **When**: High volatility (>2.5% ATR), range-bound, frequent 1% moves
- **Hold**: 1-3 days
- **Stop**: 1.5% below entry
- **Target**: 2% above entry (quick scalps)
- **Position**: Small (5-10%)

### Swing Trading
- **When**: Medium volatility, clear trend, sustained momentum
- **Hold**: 10-20 days
- **Stop**: 2.5% below entry
- **Target**: 5% above entry (let winners run)
- **Position**: Medium (10-20%)

## 📊 New API Response

```json
{
  "strategy": "swing",
  "strategy_confidence": 0.78,
  "predictions": [{
    "direction": "UP",
    "confidence": 0.85,
    "expected_return_pct": 3.2,
    "expected_win_rate": 0.68,
    "risk_level": "medium",
    "action": "BUY",
    "recommended_position": 0.15,
    "stop_loss_pct": 2.5,
    "take_profit_pct": 5.0
  }]
}
```

## 🔧 Implementation Phases

1. **Phase 1** (2 days): Fix backtest lookahead bias
2. **Phase 2** (3 days): Build strategy classifier
3. **Phase 3** (1 week): Retrain with profit-weighted objective
4. **Phase 4** (3 days): Update API & backtest simulation
5. **Phase 5** (2 days): Test & document

**Total**: ~3 weeks

## 📈 Success Criteria

- **Beat buy-and-hold** by >5% annualized (net of costs)
- **Win rate**: Day >55%, Swing >60%
- **Profit factor** >1.5
- **Max drawdown** <15%
- **Sharpe ratio** >1.0

## ⚠️ Key Risks

- Overfitting in backtest (fix: walk-forward validation)
- Ignoring transaction costs (fix: deduct 0.1% per trade)
- Strategy drift (fix: monthly retraining)

## 📁 New Files

```
v3.0/
├── IMPROVEMENT_PLAN.md      # Full detailed plan
├── QUICK_REFERENCE.md       # This file
├── STRATEGY_GUIDE.md        # When to use each strategy
├── prototypes/
│   └── strategy_selector_proto.py  # Working prototype
└── metrics/
    ├── profitability_metrics.py
    ├── strategy_selector.py
    └── position_sizing.py
```

## ❓ Your Decisions Needed

1. **Transaction costs**: Should we assume 0.1% per trade (commission + slippage)?
2. **Position sizing**: Kelly criterion vs fixed fraction?
3. **Multi-stock correlation**: Should we avoid holding too many correlated stocks?
4. **Rebalancing frequency**: Daily predictions or weekly?
5. **Minimum capital**: Assume $10,000? Or make it configurable?

---

**Next Step**: Review IMPROVEMENT_PLAN.md and provide feedback on open questions!
