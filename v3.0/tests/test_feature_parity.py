"""Parity tests between batch and walk-forward feature generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_ROOT))

from src.features.pipeline import build_feature_frame


def _mock_ohlcv(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    close = pd.Series(np.cumsum(rng.normal(0.1, 1.2, size=rows)) + 150, index=dates)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, size=rows)),
            "high": close * (1 + np.abs(rng.normal(0, 0.008, size=rows))),
            "low": close * (1 - np.abs(rng.normal(0, 0.008, size=rows))),
            "close": close,
            "volume": rng.integers(2_000_000, 6_000_000, size=rows),
        },
        index=dates,
    )


def test_walk_forward_matches_batch_for_same_timestamp() -> None:
    df = _mock_ohlcv()
    batch_result = build_feature_frame(df, include_spy_features=False, dropna=False)
    batch_df = batch_result.dataframe

    cols = ["ma_20", "rsi", "macd", "atr", "momentum_10"]

    for end_idx in range(260, 300):
        partial = df.iloc[: end_idx + 1].copy()
        wf_df = build_feature_frame(partial, include_spy_features=False, dropna=False).dataframe

        ts = partial.index[-1]
        batch_values = batch_df.loc[ts, cols]
        wf_values = wf_df.loc[ts, cols]

        np.testing.assert_allclose(batch_values.values, wf_values.values, atol=1e-12, rtol=1e-10, equal_nan=True)
