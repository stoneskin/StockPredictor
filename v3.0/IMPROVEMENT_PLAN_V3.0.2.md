# V3.0 Improvement Plan - Execution-Focused Version (V3.0.2)

**Date**: 2026-03-08  
**Version**: 3.0.2 (proposed)  
**Supersedes**: `IMPROVEMENT_PLAN_v3.0.md`, `IMPROVEMENT_PLAN_V3.0.1.md`  
**Current Production Baseline**: V2.5.2

---

## Executive Direction

V3.0 should ship as a **bias-free, cost-aware, strategy recommendation engine** with hard release gates.

This version keeps the strongest ideas from Gemini and Grok, but fixes three planning issues:

1. Too many ideas in parallel, not enough gate criteria
2. Inconsistent design choices (Kelly vs volatility-targeting, Sharpe-as-target vs return/vol decomposition)
3. Weak linkage to current V2.5 code reality (feature duplication, stale tests, model artifact mismatch)

---

## Non-Negotiable Principles

1. **No leakage first**: no new alpha logic until leakage tests pass
2. **Cost-aware always**: all strategy metrics are net of fees/slippage
3. **Risk-adjusted over raw accuracy**: Sharpe/Sortino/Calmar + drawdown govern decisions
4. **Simplicity over model sprawl**: 4 core classifiers + 2 regressors maximum for initial release
5. **Reproducibility**: same data split + same config must reproduce metrics within tolerance

---

## Current Reality Check (from V2.5 code/doc audit)

1. `compute_features` logic is duplicated in multiple files (`train`, `inference`, `backtest`) with small differences
2. Backtest behavior differs from inference in SPY feature handling
3. Test suite is stale against current config/class mapping
4. Trained artifacts on disk do not fully match doc narrative (e.g., XGBoost dominance claims)
5. Current success framing still overweights classification metrics relative to trading outcomes

V3.0.2 addresses these as explicit early tasks, not side notes.

---

## Architecture Decisions (Finalized)

### 1) Objective Modeling
- Predict **Expected Return** and **Expected Volatility** separately
- Compute expected Sharpe downstream: `(E[r] - rf) / E[sigma]`
- Do **not** train direct Sharpe target in V3.0 GA (too noisy)

### 2) Position Sizing
- Primary sizing: **volatility-targeted risk budget**
- Kelly optional as a capped overlay only after 3 months of stable live-like paper results
- Initial formula:
  - `risk_budget_per_trade = min(base_risk * confidence * vol_scale, max_risk_cap)`
  - `shares = (equity * risk_budget_per_trade) / abs(entry - stop)`

### 3) Transaction Costs
- Default net assumptions:
  - Day-trade: **20 bps** round-trip
  - Swing: **10 bps** round-trip
- Mandatory sensitivity grid: **5 / 10 / 15 / 20 / 25 bps**

### 4) Model Set
- 4 core classifiers:
  - day: (3d, 1.0%), (5d, 1.5%)
  - swing: (10d, 1.5%), (20d, 2.5%)
- 2 regressors:
  - day expected return/volatility
  - swing expected return/volatility

### 5) Benchmarks
- Primary: symbol buy-and-hold (investor reality)
- Secondary: SPY (alpha discipline)
- V3.0 passes only if strategy is competitive on risk-adjusted basis against both

---

## Phased Plan With Release Gates

## Phase 0 - Data Integrity and Leakage Eradication (Week 1)

### Scope
- Build one shared feature pipeline module used by train/inference/backtest
- Enforce strict lagged feature construction
- Ensure scaler/normalization is fit on train folds only
- Add leakage unit tests and time-split integration tests

### Deliverables
- `v3.0/src/features/pipeline.py`
- `v3.0/tests/test_no_lookahead.py`
- Leakage audit report (markdown)

### Gate (must pass)
- All leakage tests pass
- Walk-forward and inference produce feature parity on same timestamp
- Baseline backtest rerun with fixed pipeline complete

---

## Phase 1 - Cost-Aware Backtesting Engine (Week 2)

### Scope
- Replace current backtest assumptions with event-based simulation:
  - next-bar execution
  - stop/target/time exits
  - strategy-specific costs
- Add turnover, exposure, and slippage accounting

### Deliverables
- `v3.0/src/backtest/engine.py`
- `v3.0/src/backtest/costs.py`
- `v3.0/src/backtest/reports.py`

### Gate
- Backtest outputs include: net return, Sharpe, Sortino, Calmar, MDD, Ulcer, turnover
- Sensitivity table for 5-25 bps generated automatically

---

## Phase 2 - Core Model Consolidation and Retraining (Week 3-4)

### Scope
- Retire 30-model grid for V3.0 path
- Train 4 core classifiers with sample weights (volatility regime based, no future leakage)
- Train 2 regressors for expected return/volatility
- Add probability calibration (isotonic or Platt) and threshold tuning by net utility

### Deliverables
- `v3.0/src/models/core_registry.py`
- `v3.0/src/training/train_core_models.py`
- `v3.0/src/training/calibration.py`

### Gate
- Core models outperform V2.5 baseline on net utility in walk-forward
- Calibration error (ECE/Brier) improves over uncalibrated baseline

---

## Phase 3 - Strategy Layer and Risk Controls (Week 5)

### Scope
- Implement strategy classifier (day/swing/neutral)
- Add liquidity filter and SPY-correlation alpha filter
- Add regime kill switch (CUSUM + volatility shock trigger)
- Implement volatility-targeted position sizing and risk caps

### Deliverables
- `v3.0/src/strategy/classifier.py`
- `v3.0/src/risk/position_sizing.py`
- `v3.0/src/risk/regime_guard.py`

### Gate
- Neutral mode correctly suppresses trades under low liquidity/high beta/noise regimes
- Risk caps enforced in simulation (single-trade and portfolio-level)

---

## Phase 4 - API V3 Contract and Explainability (Week 6)

### Scope
- Introduce V3 response schema with:
  - strategy recommendation + confidence
  - expected return, expected volatility, expected Sharpe
  - action, position size, stop/target, liquidity/alpha flags
- Add model/version metadata and reason codes for no-trade outcomes

### Deliverables
- `v3.0/src/api/inference_v3.py`
- `v3.0/docs/API_REFERENCE.md`
- `v3.0/docs/STRATEGY_GUIDE.md`

### Gate
- Contract tests pass for `/predict`, `/predict/multi`, `/backtest`, `/strategy`
- No-trade decisions are explainable via explicit reason fields

---

## Phase 5 - Robust Validation and Go/No-Go (Week 7-8)

### Scope
- OOS regime-split validation
- Monte Carlo trade-order resampling
- Stress periods (e.g., 2020 crash, 2022 tightening)
- Drift and decay analysis by month

### Deliverables
- Validation pack with reproducible notebook/report
- Go/No-Go memo with pass/fail by criterion

### Gate (release criteria)
- Net Sharpe > 1.0 at baseline costs
- MDD < 15% and Calmar acceptable
- Positive net edge remains at 15 bps and does not collapse at 20 bps
- Outperforms buy-and-hold on risk-adjusted basis in at least 2 major regimes

---

## Success Metrics (Ranked)

### Tier 1 (Required)
1. Net Sharpe > 1.0
2. Max Drawdown < 15%
3. Sortino > 1.2
4. Strategy remains net-positive at 10 bps and competitive at 15 bps

### Tier 2 (Strongly Preferred)
5. Profit Factor > 1.4
6. Calmar > 0.8
7. Stable monthly hit-rate without single-regime dependence

### Tier 3 (Diagnostic)
8. Strategy classifier agreement with realized regime labels > 70%
9. Alpha filter meaningfully reduces high-correlation false positives

---

## Technical Debt Cleanup (Mandatory in V3.0)

1. Single source of truth for features and label logic
2. Shared dataset contract across training, inference, backtest
3. Replace stale tests with deterministic time-series tests
4. Artifact manifest per run (model hash, feature version, config, data window)
5. Automatic report generation per experiment run

---

## Proposed V3.0 File Structure

```text
v3.0/
├── IMPROVEMENT_PLAN_V3.0.2.md
├── docs/
│   ├── API_REFERENCE.md
│   ├── STRATEGY_GUIDE.md
│   ├── VALIDATION_PROTOCOL.md
│   └── RELEASE_CHECKLIST.md
├── src/
│   ├── api/
│   │   └── inference_v3.py
│   ├── features/
│   │   └── pipeline.py
│   ├── labels/
│   │   └── targets.py
│   ├── models/
│   │   └── core_registry.py
│   ├── training/
│   │   ├── train_core_models.py
│   │   └── calibration.py
│   ├── strategy/
│   │   └── classifier.py
│   ├── risk/
│   │   ├── position_sizing.py
│   │   └── regime_guard.py
│   └── backtest/
│       ├── engine.py
│       ├── costs.py
│       └── reports.py
└── tests/
    ├── test_no_lookahead.py
    ├── test_feature_parity.py
    ├── test_backtest_costs.py
    └── test_api_contract_v3.py
```

---

## Open Decisions (Recommended Defaults)

1. **Default costs?** -> Use 20 bps day / 10 bps swing
2. **Liquidity floor?** -> Day >= $100M, Swing >= $40M, Hard no-trade < $30M
3. **Retraining cadence?** -> Monthly + trigger retrain on drift/killswitch events
4. **Benchmark reporting?** -> Always report both symbol B&H and SPY
5. **Execution assumption?** -> Next-bar open for entries/exits in backtests

---

## Immediate Next Action (Start Now)

Execute **Phase 0 only** until leakage gates pass. No model innovation work should start before that gate is green.

---

**Status**: Proposed and implementation-ready  
**Owner**: StockPredictor team  
**Decision Request**: Approve V3.0.2 as the execution baseline
