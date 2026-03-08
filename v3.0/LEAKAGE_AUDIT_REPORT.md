# V3.0 Leakage Audit Report (Phase 0)

**Date**: 2026-03-08  
**Feature Version**: `3.0.2-phase0`

## Scope

Phase 0 hard gate implementation focused on leakage prevention and deterministic feature parity:

- Shared feature module for training, inference, and backtest: `v3.0/src/features/pipeline.py`
- Shared target builder: `v3.0/src/labels/targets.py`
- Leakage and parity tests:
  - `v3.0/tests/test_no_lookahead.py`
  - `v3.0/tests/test_feature_parity.py`

## Leakage Controls Implemented

1. **Strict lagging for all engineered features**
   - Every engineered feature column is shifted by one bar (`lag_periods=1` default).
   - Feature row at timestamp `t` only uses data available at `t-1` or earlier.

2. **Train-only scaler fit contract**
   - `fit_train_scaler` accepts only the training split and never touches validation/test rows.
   - Time split helper (`split_by_time`) enforces deterministic no-shuffle partitioning.

3. **Timestamp alignment for market features**
   - SPY correlation and SPY momentum are computed on index intersection only.
   - No forward fill from future timestamps.

4. **Single source of feature truth**
   - Shared module replaces fragmented feature logic drift risk.

## Gate Test Matrix

- `test_future_mutation_does_not_change_past_features`
  - Mutating future close prices does not alter historical feature rows.
- `test_feature_row_uses_only_past_values`
  - Explicit check that `ma_5[t]` equals average of close values from `t-5` to `t-1`.
- `test_scaler_fits_train_window_only`
  - Extreme test-window outliers do not change scaler train mean.
- `test_walk_forward_matches_batch_for_same_timestamp`
  - Walk-forward generation matches full-batch value at same timestamp.

## Phase 0 Status

- [x] Shared feature pipeline created
- [x] Shared target logic created
- [x] Leakage tests added
- [x] Feature parity test added
- [ ] Baseline backtest rerun with fixed pipeline (next execution step)

## Notes

- This report documents implementation and test coverage criteria.
- Runtime gate status should be confirmed by executing `python -m pytest v3.0/tests -v`.
