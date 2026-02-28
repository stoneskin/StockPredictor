"""
Data Preparation for Stock Predictor V2.5
Creates 4-class classification targets: UP, DOWN, UP_DOWN, SIDEWAYS
Based on thresholds: 1%, 2.5%, 5% price movement in any day within horizon
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.v2_5.config_v2_5 import (
    PROCESSED_DATA_DIR, RAW_DATA_DIR, HORIZONS, THRESHOLDS,
    THRESHOLD_LABELS, REGIME_PARAMS, FEATURE_PARAMS
)


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """Load raw stock data from CSV files."""
    data_files = {
        'qqq': RAW_DATA_DIR / 'qqq.csv',
        'spy': RAW_DATA_DIR / 'spy.csv'
    }
    
    dfs = {}
    for name, path in data_files.items():
        if path.exists():
            try:
                df = pd.read_csv(path, skiprows=3)
            except:
                df = pd.read_csv(path, parse_dates=['date'])
            
            if 'Price' in df.columns:
                df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
            elif 'Date' in df.columns:
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            for col in ['close', 'high', 'low', 'open', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            dfs[name] = df
            print(f"Loaded {name}: {len(df)} rows")
    
    if not dfs:
        raise FileNotFoundError(f"No data files found in {RAW_DATA_DIR}")
    
    return dfs


def compute_max_daily_change(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Compute the maximum daily price change within the horizon.
    
    For each day, look forward 'horizon' days and find:
    - Maximum gain (max return)
    - Maximum loss (min return)
    
    Args:
        df: DataFrame with 'close' price column
        horizon: Number of days to look forward
        
    Returns:
        DataFrame with max_gain and max_loss columns
    """
    returns = df['close'].pct_change()
    
    max_gain = pd.Series(index=df.index, dtype=float)
    max_loss = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df) - horizon):
        future_returns = returns.iloc[i+1:i+1+horizon]
        max_gain.iloc[i] = future_returns.max()
        max_loss.iloc[i] = future_returns.min()
    
    max_gain.iloc[-horizon:] = np.nan
    max_loss.iloc[-horizon:] = np.nan
    
    return max_gain, max_loss


def create_4class_targets(
    df: pd.DataFrame, 
    horizon: int, 
    threshold: float
) -> pd.DataFrame:
    """
    Create 4-class classification targets based on threshold.
    
    Classes:
    - 0: SIDEWAYS - neither max gain nor max loss exceeds threshold
    - 1: UP - max gain exceeds threshold, max loss does not
    - 2: DOWN - max loss exceeds threshold (negative), max gain does not
    - 3: UP_DOWN - both max gain and max loss exceed threshold
    
    Args:
        df: DataFrame with 'close' price column
        horizon: Prediction horizon in days
        threshold: Threshold as decimal (e.g., 0.01 for 1%)
        
    Returns:
        DataFrame with target column added
    """
    df = df.copy()
    
    max_gain, max_loss = compute_max_daily_change(df, horizon)
    
    target_col = f'target_{horizon}d_{threshold*100}pct'
    
    conditions = [
        (max_gain > threshold) & (max_loss < -threshold),  # UP_DOWN: 3
        (max_gain > threshold) & (max_loss >= -threshold),  # UP: 1
        (max_gain <= threshold) & (max_loss < -threshold),  # DOWN: 2
        (max_gain <= threshold) & (max_loss >= -threshold),  # SIDEWAYS: 0
    ]
    choices = [3, 1, 2, 0]
    
    df[target_col] = np.select(conditions, choices, default=np.nan)
    
    return df


def create_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create targets for all horizons and thresholds.
    
    Generates 16 target columns (4 horizons x 4 thresholds):
    - target_5d_1pct, target_5d_2.5pct, target_5d_5pct
    - target_10d_1pct, target_10d_2.5pct, target_10d_5pct
    - target_20d_1pct, target_20d_2.5pct, target_20d_5pct
    - target_30d_1pct, target_30d_2.5pct, target_30d_5pct
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with all target columns added
    """
    df = df.copy()
    
    for horizon in HORIZONS:
        for threshold in THRESHOLDS:
            df = create_4class_targets(df, horizon, threshold)
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features."""
    df = df.copy()
    params = FEATURE_PARAMS['technical_indicators']
    
    close = df['close']
    high = df.get('high', close)
    low = df.get('low', close)
    volume = df.get('volume', pd.Series(1, index=close.index))
    
    # Moving Averages
    for period in params['ma_periods']:
        df[f'ma_{period}'] = close.rolling(window=period).mean()
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    # RSI
    rsi_period = params['rsi_period']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    macd_fast = params['macd_fast']
    macd_slow = params['macd_slow']
    macd_signal = params['macd_signal']
    
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_period = params['bb_period']
    bb_std = params['bb_std']
    
    bb_ma = close.rolling(window=bb_period).mean()
    bb_std_val = close.rolling(window=bb_period).std()
    df['bb_upper'] = bb_ma + (bb_std_val * bb_std)
    df['bb_lower'] = bb_ma - (bb_std_val * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_ma
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    atr_period = params['atr_period']
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=atr_period).mean()
    df['atr_pct'] = df['atr'] / close * 100
    
    # Stochastic
    stoch_period = params['stoch_period']
    lowest_low = low.rolling(window=stoch_period).min()
    highest_high = high.rolling(window=stoch_period).max()
    df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Momentum
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_10'] = close / close.shift(10) - 1
    df['momentum_20'] = close / close.shift(20) - 1
    
    # Rate of Change
    df['roc_5'] = (close - close.shift(5)) / close.shift(5) * 100
    df['roc_10'] = (close - close.shift(10)) / close.shift(10) * 100
    
    # Volume indicators
    df['volume_ma_5'] = volume.rolling(window=5).mean()
    df['volume_ma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_ma_20']
    
    # Price position relative to MAs
    df['price_above_ma50'] = (close > df['ma_50']).astype(int)
    df['price_above_ma200'] = (close > df['ma_200']).astype(int)
    
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime-related features."""
    df = df.copy()
    params = REGIME_PARAMS
    
    close = df['close']
    
    # MA Crossover Regime
    short_ma = close.rolling(window=params['short_ma']).mean()
    long_ma = close.rolling(window=params['long_ma']).mean()
    
    # Bull: price above both MAs and short > long
    # Bear: price below both MAs and short < long
    # Sideways: other conditions
    df['ma_cross_bull'] = ((close > short_ma) & (close > long_ma) & (short_ma > long_ma)).astype(int)
    df['ma_cross_bear'] = ((close < short_ma) & (close < long_ma) & (short_ma < long_ma)).astype(int)
    df['ma_cross_sideways'] = (~(df['ma_cross_bull'].astype(bool)) & ~(df['ma_cross_bear'].astype(bool))).astype(int)
    
    # Volatility Regime
    volatility_window = params['volatility_window']
    daily_returns = close.pct_change()
    df['volatility'] = daily_returns.rolling(window=volatility_window).std()
    
    high_vol = params.get('additional_regimes', {}).get('high_volatility', 0.03)
    low_vol = params.get('additional_regimes', {}).get('low_volatility', 0.005)
    
    df['volatility_high'] = (df['volatility'] > high_vol).astype(int)
    df['volatility_normal'] = ((df['volatility'] >= low_vol) & (df['volatility'] <= high_vol)).astype(int)
    df['volatility_low'] = (df['volatility'] < low_vol).astype(int)
    
    return df


def add_market_features(df: pd.DataFrame, spy_data: pd.DataFrame) -> pd.DataFrame:
    """Add market correlation features with SPY."""
    df = df.copy()
    
    if spy_data is None or len(spy_data) == 0:
        df['correlation_spy_5d'] = 0.0
        df['correlation_spy_10d'] = 0.0
        df['correlation_spy_20d'] = 0.0
        return df
    
    spy_close = spy_data['close']
    stock_close = df['close']
    
    # Align by index
    common_idx = df.index.intersection(spy_data.index)
    if len(common_idx) > 20:
        stock_ret = stock_close.loc[common_idx].pct_change().dropna()
        spy_ret = spy_close.loc[common_idx].pct_change().dropna()
        
        aligned_stock = stock_ret.loc[spy_ret.index]
        
        df.loc[common_idx, 'correlation_spy_5d'] = aligned_stock.rolling(5).corr(spy_ret.rolling(5))
        df.loc[common_idx, 'correlation_spy_10d'] = aligned_stock.rolling(10).corr(spy_ret.rolling(10))
        df.loc[common_idx, 'correlation_spy_20d'] = aligned_stock.rolling(20).corr(spy_ret.rolling(20))
    
    df['correlation_spy_5d'] = df['correlation_spy_5d'].fillna(0)
    df['correlation_spy_10d'] = df['correlation_spy_10d'].fillna(0)
    df['correlation_spy_20d'] = df['correlation_spy_20d'].fillna(0)
    
    return df


def prepare_data(
    horizon: int = 20,
    threshold: float = 0.01,
    include_spy: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Prepare data for training.
    
    Args:
        horizon: Prediction horizon (5, 10, 20, or 30)
        threshold: Threshold for classification (0.01, 0.025, 0.05)
        include_spy: Whether to include SPY features
        
    Returns:
        X: Feature array
        y: Target array
        feature_names: List of feature names
        df: Full DataFrame
    """
    # Load data
    dfs = load_raw_data()
    df = dfs['qqq'].copy()
    
    # Add features
    df = add_technical_indicators(df)
    df = add_regime_features(df)
    
    if include_spy and 'spy' in dfs:
        df = add_market_features(df, dfs['spy'])
    
    # Create targets
    df = create_4class_targets(df, horizon, threshold)
    
    # Get target column
    target_col = f'target_{horizon}d_{threshold*100}pct'
    
    # Remove rows with NaN targets
    df = df.dropna(subset=[target_col])
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['date', 'open', 'high', 'low', 'volume']
    exclude_cols += [c for c in df.columns if c.startswith('target_')]
    exclude_cols += [c for c in df.columns if c.startswith('return_')]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values.astype(int)
    
    return X, y, feature_cols, df


def get_target_column_name(horizon: int, threshold: float) -> str:
    """Get the target column name for given horizon and threshold."""
    return f'target_{horizon}d_{threshold*100}pct'


def get_all_target_columns() -> List[str]:
    """Get all target column names."""
    targets = []
    for horizon in HORIZONS:
        for threshold in THRESHOLDS:
            targets.append(f'target_{horizon}d_{threshold*100}pct')
    return targets
