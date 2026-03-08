"""Shared feature engineering pipeline used by V3.0 train/inference/backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_VERSION = "3.0.2-phase0"
REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class FeatureFrameResult:
    """Container for feature matrix creation outputs."""

    dataframe: pd.DataFrame
    feature_columns: list[str]
    feature_version: str = FEATURE_VERSION


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate OHLCV dataframe."""
    work_df = df.copy()
    work_df.columns = [str(col).lower() for col in work_df.columns]

    if "date" in work_df.columns:
        work_df["date"] = pd.to_datetime(work_df["date"])
        work_df = work_df.set_index("date")

    if not isinstance(work_df.index, pd.DatetimeIndex):
        raise ValueError("Input dataframe must include a DatetimeIndex or a 'date' column")

    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in work_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = work_df.sort_index()
    for col in REQUIRED_PRICE_COLUMNS:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    return work_df


def _add_core_technical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add technical and regime features before lagging."""
    work_df = df.copy()
    close = work_df["close"]
    high = work_df["high"]
    low = work_df["low"]
    volume = work_df["volume"]

    for period in [5, 10, 20, 50, 100, 200]:
        work_df[f"ma_{period}"] = close.rolling(window=period).mean()
        work_df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    work_df["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    work_df["macd"] = ema_fast - ema_slow
    work_df["macd_signal"] = work_df["macd"].ewm(span=9, adjust=False).mean()
    work_df["macd_hist"] = work_df["macd"] - work_df["macd_signal"]

    bb_ma = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    work_df["bb_upper"] = bb_ma + (bb_std * 2)
    work_df["bb_lower"] = bb_ma - (bb_std * 2)
    work_df["bb_width"] = (work_df["bb_upper"] - work_df["bb_lower"]) / bb_ma
    work_df["bb_position"] = (close - work_df["bb_lower"]) / (work_df["bb_upper"] - work_df["bb_lower"])

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    work_df["atr"] = true_range.rolling(window=14).mean()
    work_df["atr_pct"] = work_df["atr"] / close * 100

    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    work_df["stoch_k"] = 100 * (close - lowest_low) / (highest_high - lowest_low)
    work_df["stoch_d"] = work_df["stoch_k"].rolling(window=3).mean()

    work_df["momentum_5"] = close / close.shift(5) - 1
    work_df["momentum_10"] = close / close.shift(10) - 1
    work_df["momentum_20"] = close / close.shift(20) - 1
    work_df["roc_5"] = (close - close.shift(5)) / close.shift(5) * 100
    work_df["roc_10"] = (close - close.shift(10)) / close.shift(10) * 100

    work_df["volume_ma_5"] = volume.rolling(window=5).mean()
    work_df["volume_ma_20"] = volume.rolling(window=20).mean()
    work_df["volume_ratio"] = volume / work_df["volume_ma_20"]

    work_df["price_above_ma50"] = (close > work_df["ma_50"]).astype(int)
    work_df["price_above_ma200"] = (close > work_df["ma_200"]).astype(int)

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    work_df["ma_cross_bull"] = ((close > ma50) & (close > ma200) & (ma50 > ma200)).astype(int)
    work_df["ma_cross_bear"] = ((close < ma50) & (close < ma200) & (ma50 < ma200)).astype(int)
    work_df["ma_cross_sideways"] = (~work_df["ma_cross_bull"].astype(bool) & ~work_df["ma_cross_bear"].astype(bool)).astype(int)

    daily_returns = close.pct_change()
    work_df["volatility"] = daily_returns.rolling(20).std()

    high_vol = 0.03
    low_vol = 0.005
    work_df["volatility_high"] = (work_df["volatility"] > high_vol).astype(int)
    work_df["volatility_normal"] = ((work_df["volatility"] >= low_vol) & (work_df["volatility"] <= high_vol)).astype(int)
    work_df["volatility_low"] = (work_df["volatility"] < low_vol).astype(int)

    work_df["atr_ratio"] = work_df["atr"] / close
    work_df["trend_strength"] = abs(work_df["ma_50"] - work_df["ma_200"]) / work_df["ma_200"]
    work_df["price_vs_ma50"] = (close - work_df["ma_50"]) / work_df["ma_50"]
    work_df["price_vs_ma200"] = (close - work_df["ma_200"]) / work_df["ma_200"]

    work_df["volatility_regime_num"] = 0
    work_df.loc[work_df["volatility"] > high_vol, "volatility_regime_num"] = 2
    work_df.loc[(work_df["volatility"] >= low_vol) & (work_df["volatility"] <= high_vol), "volatility_regime_num"] = 1

    work_df["momentum_positive"] = (work_df["momentum_20"] > 0).astype(int)
    work_df["momentum_strong"] = (abs(work_df["momentum_20"]) > 0.05).astype(int)
    work_df["rsi_overbought"] = (work_df["rsi"] > 70).astype(int)
    work_df["rsi_oversold"] = (work_df["rsi"] < 30).astype(int)
    work_df["rsi_neutral"] = ((work_df["rsi"] >= 30) & (work_df["rsi"] <= 70)).astype(int)
    work_df["stoch_overbought"] = (work_df["stoch_k"] > 80).astype(int)
    work_df["stoch_oversold"] = (work_df["stoch_k"] < 20).astype(int)

    bb_width_median = work_df["bb_width"].median()
    work_df["bb_squeeze"] = (work_df["bb_width"] < bb_width_median * 0.8).astype(int)
    work_df["bb_expansion"] = (work_df["bb_width"] > bb_width_median * 1.2).astype(int)

    feature_columns = [col for col in work_df.columns if col not in REQUIRED_PRICE_COLUMNS]
    return work_df, feature_columns


def _add_spy_features(base_df: pd.DataFrame, spy_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add SPY relation features with strict timestamp alignment."""
    work_df = base_df.copy()
    for col in ["correlation_spy_5d", "correlation_spy_10d", "correlation_spy_20d", "spy_momentum"]:
        work_df[col] = 0.0

    if spy_df is None or spy_df.empty:
        return work_df

    normalized_spy = _normalize_ohlcv(spy_df)
    common_idx = work_df.index.intersection(normalized_spy.index)
    if len(common_idx) <= 20:
        return work_df

    stock_ret = work_df.loc[common_idx, "close"].pct_change()
    spy_ret = normalized_spy.loc[common_idx, "close"].pct_change()
    for window in [5, 10, 20]:
        rolling_corr = stock_ret.rolling(window).corr(spy_ret)
        work_df.loc[common_idx, f"correlation_spy_{window}d"] = rolling_corr

    spy_close = normalized_spy.loc[common_idx, "close"]
    spy_ma20 = spy_close.rolling(20).mean()
    work_df.loc[common_idx, "spy_momentum"] = (spy_close - spy_ma20) / spy_ma20

    return work_df


def build_feature_frame(
    stock_df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
    include_spy_features: bool = True,
    lag_periods: int = 1,
    dropna: bool = False,
) -> FeatureFrameResult:
    """Build a deterministic, leakage-safe feature frame."""
    normalized = _normalize_ohlcv(stock_df)
    feature_df, feature_columns = _add_core_technical_features(normalized)

    for extra in ["correlation_spy_5d", "correlation_spy_10d", "correlation_spy_20d", "spy_momentum"]:
        if extra not in feature_df.columns:
            feature_df[extra] = 0.0
        if extra not in feature_columns:
            feature_columns.append(extra)

    if include_spy_features:
        feature_df = _add_spy_features(feature_df, spy_df)

    feature_df[feature_columns] = feature_df[feature_columns].shift(lag_periods)

    feature_df["spy_correlation_positive"] = (feature_df["correlation_spy_20d"] > 0.3).astype(int)
    feature_df["spy_correlation_negative"] = (feature_df["correlation_spy_20d"] < -0.3).astype(int)
    feature_df["spy_correlation_neutral"] = (
        (feature_df["correlation_spy_20d"] >= -0.3) & (feature_df["correlation_spy_20d"] <= 0.3)
    ).astype(int)
    feature_df["spy_momentum_positive"] = (feature_df["spy_momentum"] > 0).astype(int)

    for col in ["spy_correlation_positive", "spy_correlation_negative", "spy_correlation_neutral", "spy_momentum_positive"]:
        if col not in feature_columns:
            feature_columns.append(col)

    if dropna:
        feature_df = feature_df.dropna(subset=feature_columns).copy()

    return FeatureFrameResult(dataframe=feature_df, feature_columns=feature_columns)


def fit_train_scaler(train_df: pd.DataFrame, feature_columns: Iterable[str]) -> StandardScaler:
    """Fit scaler only on the training split to prevent leakage."""
    feature_list = list(feature_columns)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_list])
    return scaler


def transform_with_scaler(df: pd.DataFrame, feature_columns: Iterable[str], scaler: StandardScaler) -> pd.DataFrame:
    """Apply a pre-fit scaler to the given dataframe."""
    feature_list = list(feature_columns)
    transformed_df = df.copy()
    transformed_df[feature_list] = scaler.transform(transformed_df[feature_list])
    return transformed_df


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic time split without shuffling."""
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df
