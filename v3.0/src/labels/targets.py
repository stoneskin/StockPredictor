"""Shared target generation logic for V3.0 models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_column_name(horizon: int, threshold: float) -> str:
    """Return deterministic target column name."""
    return f"target_{horizon}d_{threshold * 100}pct"


def build_4class_targets(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """Build V3 four-class targets using forward window max/min returns."""
    if "close" not in df.columns:
        raise ValueError("close column is required")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    close = pd.to_numeric(df["close"], errors="coerce")
    labels = pd.Series(np.nan, index=df.index, dtype=float)

    for idx in range(len(df) - horizon):
        entry = close.iloc[idx]
        future_window = close.iloc[idx + 1 : idx + horizon + 1]
        if pd.isna(entry) or future_window.isna().any():
            continue

        max_gain = (future_window.max() - entry) / entry
        max_loss = (future_window.min() - entry) / entry

        if max_gain > threshold and max_loss < -threshold:
            labels.iloc[idx] = 2
        elif max_gain > threshold:
            labels.iloc[idx] = 0
        elif max_loss < -threshold:
            labels.iloc[idx] = 1
        else:
            labels.iloc[idx] = 3

    return labels
