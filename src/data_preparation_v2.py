"""
Data Preparation for Stock Predictor V2
Creates classification targets and multi-horizon features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_v2 import (
    PROCESSED_DATA_DIR, RAW_DATA_DIR, HORIZONS, DEFAULT_HORIZON,
    REGIME_PARAMS, FEATURE_PARAMS
)


def load_raw_data():
    """Load raw stock data from CSV files."""
    data_files = {
        'qqq': RAW_DATA_DIR / 'qqq.csv',
        'spy': RAW_DATA_DIR / 'spy.csv'
    }
    
    dfs = {}
    for name, path in data_files.items():
        if path.exists():
            # Read CSV, skip the first 3 rows (header rows with ticker info)
            df = pd.read_csv(path, skiprows=3)
            
            # Rename columns - first column is Date
            df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Convert numeric columns
            for col in ['close', 'high', 'low', 'open', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with missing data
            df = df.dropna()
            
            dfs[name] = df
            print(f"Loaded {name}: {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
    
    if not dfs:
        raise FileNotFoundError(f"No data files found in {RAW_DATA_DIR}")
    
    return dfs


def create_classification_targets(df: pd.DataFrame, horizons: list = None) -> pd.DataFrame:
    """
    Create binary classification targets for multiple horizons.
    
    Target: 1 if price goes up after horizon days, 0 otherwise
    
    Args:
        df: DataFrame with 'close' price column
        horizons: List of prediction horizons in days
        
    Returns:
        DataFrame with added target columns
    """
    if horizons is None:
        horizons = HORIZONS
    
    df = df.copy()
    
    for horizon in horizons:
        # Future price
        future_price = df['close'].shift(-horizon)
        
        # Binary target: 1 if price goes up, 0 otherwise
        target_col = f'target_{horizon}d'
        df[target_col] = (future_price > df['close']).astype(int)
        
        # Also create return target for reference
        return_col = f'return_{horizon}d'
        df[return_col] = ((future_price - df['close']) / df['close'] * 100)
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    params = FEATURE_PARAMS['technical_indicators']
    
    # Price data
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
    
    # ATR (Average True Range)
    atr_period = params['atr_period']
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=atr_period).mean()
    df['atr_pct'] = df['atr'] / close * 100
    
    # Stochastic Oscillator
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
    """
    Add market regime-related features.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with added regime features
    """
    df = df.copy()
    params = REGIME_PARAMS
    
    close = df['close']
    
    # Moving Average Crossover Regime
    short_ma = close.rolling(window=params['short_ma']).mean()
    long_ma = close.rolling(window=params['long_ma']).mean()
    
    # Bull: price above both MAs and short > long
    # Bear: price below both MAs and short < long
    # Neutral: otherwise
    df['ma_regime'] = 0  # Neutral
    df.loc[(close > short_ma) & (close > long_ma) & (short_ma > long_ma), 'ma_regime'] = 1  # Bull
    df.loc[(close < short_ma) & (close < long_ma) & (short_ma < long_ma), 'ma_regime'] = -1  # Bear
    
    # Volatility regime
    returns = close.pct_change()
    rolling_vol = returns.rolling(window=params['volatility_window']).std()
    df['volatility'] = rolling_vol
    df['volatility_regime'] = 0  # Normal
    df.loc[rolling_vol > params['volatility_threshold'], 'volatility_regime'] = 1  # High
    df.loc[rolling_vol < params['volatility_threshold'] * 0.5, 'volatility_regime'] = -1  # Low
    
    # Trend strength
    df['trend_strength'] = (short_ma - long_ma) / long_ma * 100
    
    # Distance from moving averages
    df['distance_ma50'] = (close - short_ma) / short_ma * 100
    df['distance_ma200'] = (close - long_ma) / long_ma * 100
    
    return df


def add_market_features(df: pd.DataFrame, market_data: dict = None) -> pd.DataFrame:
    """
    Add market-wide features (e.g., SPY returns for QQQ).
    
    Args:
        df: DataFrame with stock data
        market_data: Dictionary of market DataFrames (e.g., {'spy': spy_df})
        
    Returns:
        DataFrame with added market features
    """
    df = df.copy()
    
    if market_data is not None and 'spy' in market_data:
        spy = market_data['spy']['close']
        
        # SPY returns at different horizons
        for horizon in [5, 10, 15, 20]:
            df[f'spy_return_{horizon}d'] = spy.pct_change(horizon) * 100
        
        # SPY position relative to MAs
        spy_ma50 = spy.rolling(50).mean()
        spy_ma200 = spy.rolling(200).mean()
        df['spy_above_ma200'] = (spy > spy_ma200).astype(int)
        
        # Correlation with SPY (rolling)
        stock_returns = df['close'].pct_change()
        spy_returns = spy.pct_change()
        df['correlation_spy_20d'] = stock_returns.rolling(20).corr(spy_returns)
    
    return df


def prepare_data(horizon: int = None, include_spy_features: bool = True) -> tuple:
    """
    Prepare complete dataset for training.
    
    Args:
        horizon: Prediction horizon (5, 10, or 20). If None, uses DEFAULT_HORIZON
        include_spy_features: Whether to include SPY-related features
        
    Returns:
        Tuple of (X, y, feature_names, df)
    """
    if horizon is None:
        horizon = DEFAULT_HORIZON
    
    print(f"\n{'='*60}")
    print(f"DATA PREPARATION - V2")
    print(f"{'='*60}")
    print(f"Prediction horizon: {horizon} days")
    
    # Load raw data
    dfs = load_raw_data()
    main_df = dfs.get('qqq', list(dfs.values())[0])
    market_data = dfs if include_spy_features else None
    
    # Add features
    print("\nAdding technical indicators...")
    main_df = add_technical_indicators(main_df)
    
    print("Adding regime features...")
    main_df = add_regime_features(main_df)
    
    if include_spy_features:
        print("Adding market features...")
        main_df = add_market_features(main_df, market_data)
    
    # Create classification targets
    print("Creating classification targets...")
    main_df = create_classification_targets(main_df)
    
    # Remove rows with NaN (from feature calculations)
    target_col = f'target_{horizon}d'
    initial_len = len(main_df)
    main_df = main_df.dropna(subset=[target_col])
    main_df = main_df.dropna()  # Drop all NaN rows
    final_len = len(main_df)
    
    print(f"\nData shape: {main_df.shape}")
    print(f"Removed {initial_len - final_len} rows with NaN")
    
    # Separate features and target
    exclude_cols = [f'target_{h}d' for h in HORIZONS] + [f'return_{h}d' for h in HORIZONS]
    exclude_cols += ['open', 'high', 'low', 'volume'] if 'open' in main_df.columns else []
    
    feature_cols = [col for col in main_df.columns if col not in exclude_cols]
    
    X = main_df[feature_cols].values
    y = main_df[target_col].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target distribution: Up={y.sum()} ({y.mean()*100:.1f}%), Down={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    return X, y, feature_cols, main_df


def get_feature_importance_names() -> dict:
    """Get human-readable feature names mapping."""
    return {
        'ma_5': '5-day Moving Average',
        'ma_10': '10-day Moving Average',
        'ma_20': '20-day Moving Average',
        'ma_50': '50-day Moving Average',
        'ma_100': '100-day Moving Average',
        'ma_200': '200-day Moving Average',
        'rsi': 'Relative Strength Index',
        'macd': 'MACD Line',
        'macd_signal': 'MACD Signal',
        'macd_hist': 'MACD Histogram',
        'bb_upper': 'Bollinger Upper Band',
        'bb_lower': 'Bollinger Lower Band',
        'bb_width': 'Bollinger Band Width',
        'bb_position': 'Bollinger Band Position',
        'atr': 'Average True Range',
        'atr_pct': 'ATR Percentage',
        'stoch_k': 'Stochastic %K',
        'stoch_d': 'Stochastic %D',
        'momentum_5': '5-day Momentum',
        'momentum_10': '10-day Momentum',
        'momentum_20': '20-day Momentum',
        'volume_ratio': 'Volume Ratio',
        'ma_regime': 'MA Regime (Bull/Bear/Neutral)',
        'volatility_regime': 'Volatility Regime',
        'trend_strength': 'Trend Strength',
        'spy_return_5d': 'SPY 5-day Return',
        'correlation_spy_20d': '20-day Correlation with SPY'
    }


if __name__ == '__main__':
    # Test data preparation
    X, y, feature_names, df = prepare_data(horizon=5)
    print(f"\nFeature names: {feature_names[:10]}...")
    print(f"\nData preparation complete!")