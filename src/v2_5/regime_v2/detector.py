"""
Market Regime Detection for V2.5
Enhanced regime detection with more market states
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def detect_ma_cross_regime(df: pd.DataFrame, short_ma: int = 50, long_ma: int = 200) -> pd.DataFrame:
    """
    Detect MA crossover regime.
    
    Returns:
        DataFrame with regime columns:
        - ma_regime_bull: Price above both MAs and short > long
        - ma_regime_bear: Price below both MAs and short < long
        - ma_regime_sideways: Other conditions
    """
    df = df.copy()
    close = df['close']
    
    sma_short = close.rolling(short_ma).mean()
    sma_long = close.rolling(long_ma).mean()
    
    df['ma_regime_bull'] = ((close > sma_short) & (close > sma_long) & (sma_short > sma_long)).astype(int)
    df['ma_regime_bear'] = ((close < sma_short) & (close < sma_long) & (sma_short < sma_long)).astype(int)
    df['ma_regime_sideways'] = (~(df['ma_regime_bull'].astype(bool)) & ~(df['ma_regime_bear'].astype(bool))).astype(int)
    
    return df


def detect_volatility_regime(
    df: pd.DataFrame, 
    window: int = 20,
    high_threshold: float = 0.03,
    low_threshold: float = 0.005
) -> pd.DataFrame:
    """
    Detect volatility regime.
    
    Returns:
        DataFrame with regime columns:
        - vol_regime_high: volatility > high_threshold
        - vol_regime_normal: low_threshold <= volatility <= high_threshold
        - vol_regime_low: volatility < low_threshold
    """
    df = df.copy()
    daily_returns = df['close'].pct_change()
    
    df['volatility'] = daily_returns.rolling(window).std()
    
    df['vol_regime_high'] = (df['volatility'] > high_threshold).astype(int)
    df['vol_regime_normal'] = ((df['volatility'] >= low_threshold) & (df['volatility'] <= high_threshold)).astype(int)
    df['vol_regime_low'] = (df['volatility'] < low_threshold).astype(int)
    
    return df


def detect_momentum_regime(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.DataFrame:
    """
    Detect momentum regime.
    
    Returns:
        DataFrame with regime columns:
        - momentum_strong_up: short > long > 0
        - momentum_strong_down: short < long < 0
        - momentum_neutral: Other
    """
    df = df.copy()
    
    df['momentum_short'] = df['close'].pct_change(short_period)
    df['momentum_long'] = df['close'].pct_change(long_period)
    
    df['momentum_strong_up'] = ((df['momentum_short'] > df['momentum_long']) & (df['momentum_long'] > 0)).astype(int)
    df['momentum_strong_down'] = ((df['momentum_short'] < df['momentum_long']) & (df['momentum_long'] < 0)).astype(int)
    df['momentum_neutral'] = (~(df['momentum_strong_up'].astype(bool)) & ~(df['momentum_strong_down'].astype(bool))).astype(int)
    
    return df


def detect_volume_regime(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Detect volume regime.
    
    Returns:
        DataFrame with regime columns:
        - volume_high: volume > 1.5 * moving average
        - volume_normal: 0.5 * MA <= volume <= 1.5 * MA
        - volume_low: volume < 0.5 * MA
    """
    df = df.copy()
    
    df['volume_ma'] = df['volume'].rolling(window).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    df['volume_high'] = (df['volume_ratio'] > 1.5).astype(int)
    df['volume_normal'] = ((df['volume_ratio'] >= 0.5) & (df['volume_ratio'] <= 1.5)).astype(int)
    df['volume_low'] = (df['volume_ratio'] < 0.5).astype(int)
    
    return df


def detect_all_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all regime detection methods.
    
    Returns:
        DataFrame with all regime features added
    """
    df = detect_ma_cross_regime(df)
    df = detect_volatility_regime(df)
    df = detect_momentum_regime(df)
    df = detect_volume_regime(df)
    
    return df


def get_current_regime(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get current market regime.
    
    Returns:
        Dictionary with regime types and their states
    """
    latest = df.iloc[-1]
    
    regimes = {}
    
    # MA Cross
    if latest.get('ma_regime_bull', 0) == 1:
        regimes['ma_regime'] = 'BULL'
    elif latest.get('ma_regime_bear', 0) == 1:
        regimes['ma_regime'] = 'BEAR'
    else:
        regimes['ma_regime'] = 'SIDEWAYS'
    
    # Volatility
    if latest.get('vol_regime_high', 0) == 1:
        regimes['volatility'] = 'HIGH'
    elif latest.get('vol_regime_low', 0) == 1:
        regimes['volatility'] = 'LOW'
    else:
        regimes['volatility'] = 'NORMAL'
    
    # Momentum
    if latest.get('momentum_strong_up', 0) == 1:
        regimes['momentum'] = 'STRONG_UP'
    elif latest.get('momentum_strong_down', 0) == 1:
        regimes['momentum'] = 'STRONG_DOWN'
    else:
        regimes['momentum'] = 'NEUTRAL'
    
    # Volume
    if latest.get('volume_high', 0) == 1:
        regimes['volume'] = 'HIGH'
    elif latest.get('volume_low', 0) == 1:
        regimes['volume'] = 'LOW'
    else:
        regimes['volume'] = 'NORMAL'
    
    return regimes
