"""Leakage tests for the V3.0 shared feature pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_ROOT))

from src.features.pipeline import build_feature_frame, fit_train_scaler, split_by_time


def _mock_ohlcv(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    base = np.cumsum(rng.normal(0.2, 1.0, size=rows)) + 100
    close = pd.Series(base, index=dates)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, size=rows)),
            "high": close * (1 + np.abs(rng.normal(0, 0.01, size=rows))),
            "low": close * (1 - np.abs(rng.normal(0, 0.01, size=rows))),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=rows),
        },
        index=dates,
    )


def test_future_mutation_does_not_change_past_features() -> None:
    df = _mock_ohlcv()
    baseline = build_feature_frame(df, include_spy_features=False).dataframe

    mutated = df.copy()
    mutated.iloc[-15:, mutated.columns.get_loc("close")] *= 3.0
    mutated_frame = build_feature_frame(mutated, include_spy_features=False).dataframe

    compare_cols = ["ma_5", "rsi", "macd", "volatility", "momentum_20"]
    safe_slice = slice(None, -20)

    pd.testing.assert_frame_equal(
        baseline.loc[:, compare_cols].iloc[safe_slice],
        mutated_frame.loc[:, compare_cols].iloc[safe_slice],
        check_exact=False,
        atol=1e-12,
        rtol=1e-10,
    )


def test_feature_row_uses_only_past_values() -> None:
    df = _mock_ohlcv()
    feature_df = build_feature_frame(df, include_spy_features=False).dataframe

    row_idx = 260
    ts = feature_df.index[row_idx]
    expected = df["close"].iloc[row_idx - 5 : row_idx].mean()
    actual = feature_df.loc[ts, "ma_5"]

    assert np.isfinite(actual)
    assert abs(actual - expected) < 1e-10


def test_scaler_fits_train_window_only() -> None:
    df = _mock_ohlcv()
    result = build_feature_frame(df, include_spy_features=False, dropna=True)
    feature_df = result.dataframe

    train_df, test_df = split_by_time(feature_df, train_ratio=0.8)
    test_df = test_df.copy()
    test_df.loc[:, "rsi"] = 1e6

    scaler = fit_train_scaler(train_df, ["rsi", "macd"])

    assert scaler.mean_[0] < 200
