**Quant Trading Pro Review:**

**Strengths:** Profit-driven objective + strategy classification + P&L sim with stops/take-profits are excellent upgrades.

**Core Advice:**
- Phase 1 first: Strict lagged features only (kill lookahead bias).
- Backtest: Add 0.1% round-trip slippage + commissions (day-trade edges vanish without).
- XGBoost: Use sample_weights for profit focus; skip complex custom objective.
- Sizing: Half-Kelly max + vol targeting.
- Testing: Walk-forward + OOS regime splits mandatory.
- Simplify: Consolidate horizons/thresholds; add direct return regressor.

**Risks:** Costs ignored, labeling leakage, overfitting. Solid plan if executed realistically.