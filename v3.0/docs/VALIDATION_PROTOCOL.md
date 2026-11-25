# V3.0 Validation Protocol

## Phase 0 (Data Integrity and Leakage)

1. Build all features from `src/features/pipeline.py` only.
2. Use deterministic time splits (no random shuffle) for train/test.
3. Fit normalization/scaler on train split only.
4. Verify walk-forward and batch parity at identical timestamps.
5. Run leakage tests before any model retraining.

## Required Commands

```bash
python -m pytest v3.0/tests/test_no_lookahead.py -v
python -m pytest v3.0/tests/test_feature_parity.py -v
```

## Gate

Phase 0 is considered complete only when both test files pass and leakage audit report is updated.
