"""
Data preparation module: download, feature engineering, label creation.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import (
    TICKER, MARKET_BENCHMARK,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    VEGAS_MA1_PERIOD, VEGAS_MA2_PERIOD, VEGAS_MA12_PERIOD, VEGAS_MA22_PERIOD,
    HULL_LENGTH, HULL_MODE, PREDICTION_HORIZON, CALCULATE_MAX_DRAWDOWN, MAX_DRAWDOWN_WINDOW,
    TARGET_THRESHOLD
)

def download_data(ticker, start_date, end_date, save_path=None):
    """Download OHLCV data from yfinance."""
    print(f"Downloading {ticker} from {start_date} to {end_date}...")
    # Use period='max' and then filter by date range
    data = yf.download(ticker, period="max", progress=False)
    # Filter by date range
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    if save_path:
        data.to_csv(save_path)
        print(f"Saved to {save_path}")
    return data

def calculate_vegas_channels(df, ma1_period=144, ma2_period=169, ma12_period=576, ma22_period=676, ma_type='ema'):
    """Calculate Vegas Channel indicators."""
    # yfinance returns multi-level columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # Keep only ticker level
    close = df['Close'].squeeze()  # Ensure 1D

    if ma_type == 'sma':
        ma_func = ta.trend.sma_indicator
    elif ma_type == 'ema':
        ma_func = ta.trend.ema_indicator
    elif ma_type == 'wma':
        ma_func = ta.trend.wma_indicator
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")

    df['MA1'] = ma_func(close, ma1_period)
    df['MA2'] = ma_func(close, ma2_period)
    df['MA12'] = ma_func(close, ma12_period)
    df['MA22'] = ma_func(close, ma22_period)

    # Vegas signals
    df['veg_fast_band_width'] = (df['MA1'] - df['MA2']) / df['MA2']
    df['veg_slow_band_width'] = (df['MA12'] - df['MA22']) / df['MA22']
    df['veg_price_position'] = (close - df['MA1']) / (df['MA22'] - df['MA1'])
    df['veg_filter'] = ta.trend.ema_indicator(close, 12)
    df['veg_filter_slope'] = (df['veg_filter'] - df['veg_filter'].shift(1)) / df['veg_filter'].shift(1)

    # Channel signals
    df['Channel_Allow_Long'] = (df['MA1'] > df['MA12']) & (df['MA1'] > df['MA22']) & (close > df['MA1']) & (close > df['MA12']) & (df['veg_filter_slope'] > 0)
    df['Channel_Allow_Short'] = (df['MA1'] < df['MA12']) & (df['MA1'] < df['MA22']) & (close < df['MA1']) & (close < df['MA12']) & (df['veg_filter_slope'] < 0)
    df['veg_signal'] = df['Channel_Allow_Long'].astype(int) - df['Channel_Allow_Short'].astype(int)

    return df

def calculate_hull_ma(df, length=55, mode=1):
    """Calculate Hull Moving Average (HMA/EHMA/THMA)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    def wma(src, period):
        return ta.trend.wma_indicator(src, period)

    def ema(src, period):
        return ta.trend.ema_indicator(src, period)

    if mode == 1:  # HMA
        hull = wma(2 * wma(close, length // 2) - wma(close, length), int(np.sqrt(length)))
    elif mode == 2:  # EHMA
        hull = ema(2 * ema(close, length // 2) - ema(close, length), int(np.sqrt(length)))
    elif mode == 3:  # THMA
        hull = wma(
            wma(close, length // 3) * 3 - wma(close, length // 2) - wma(close, length),
            length
        )
    else:
        raise ValueError(f"Invalid Hull mode: {mode}")

    df['HULL'] = hull
    df['hull_direction'] = (hull > hull.shift(1)).astype(int)
    df['hull_slope'] = (hull - hull.shift(1)) / hull.shift(1)
    df['Hull_Allow_Long'] = (hull > hull.shift(2)) & (low > hull) & (low > hull.shift(2))
    df['Hull_Allow_Short'] = (hull <= hull.shift(2)) & (high < hull) & (high < hull.shift(2))
    df['hull_signal'] = df['Hull_Allow_Long'].astype(int) - df['Hull_Allow_Short'].astype(int)

    return df

def calculate_macd_rsi_stoch(df, fast=12, slow=26, signal=9, rsi_len=14, stoch_len=14, stoch_k=3, stoch_d=3):
    """Calculate MACD, RSI, Stochastic."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    # MACD
    macd_line = ta.trend.ema_indicator(close, fast) - ta.trend.ema_indicator(close, slow)
    signal_line = ta.trend.ema_indicator(macd_line, signal)
    hist = macd_line - signal_line

    df['macd_line'] = macd_line
    df['signal_line'] = signal_line
    df['macd_hist'] = hist
    df['macd_bullish'] = ((macd_line > 0) & (signal_line > 0)).astype(int)
    df['macd_crossover'] = (macd_line > signal_line).astype(int)
    df['macd_signal'] = (macd_line > 0).astype(int)

    # RSI
    df['rsi'] = ta.momentum.rsi(close, rsi_len)
    df['rsi_overbought'] = 0
    df.loc[df['rsi'] > 70, 'rsi_overbought'] = 1
    df.loc[df['rsi'] < 30, 'rsi_overbought'] = -1

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=stoch_len, smooth_window=stoch_d)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_bullish'] = (df['stoch_k'] > df['stoch_d']).astype(int)

    return df

def calculate_volume_volatility(df, volume_ma=20, atr_period=14, bb_period=20, bb_std=2):
    """Calculate volume and volatility indicators."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    # Volume
    df['volume_ma'] = ta.trend.sma_indicator(volume, volume_ma)
    df['volume_ratio'] = volume / df['volume_ma']
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan)

    # Volatility
    df['atr'] = ta.volatility.AverageTrueRange(high, low, close, atr_period).average_true_range()
    df['atr_pct'] = df['atr'] / close

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, bb_period, bb_std)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / close

    return df

def add_time_features(df):
    """Add time-based features."""
    df['day_of_week'] = df.index.dayofweek  # 0=Monday
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    return df

def create_labels(df, horizon=15):
    """Create target labels."""
    # Future return (%)
    future_close = df['Close'].shift(-horizon)
    df[f'target_{horizon}d'] = (future_close / df['Close'] - 1) * 100

    # Max drawdown within horizon (risk metric)
    if CALCULATE_MAX_DRAWDOWN:
        max_drawdown = []
        for i in range(len(df) - horizon):
            future_prices = df['Close'].iloc[i:i+horizon+1]
            running_max = future_prices.cummax()
            drawdown = (future_prices - running_max) / running_max
            max_dd = drawdown.min() * 100  # as percentage
            max_drawdown.append(max_dd)
        # Pad with NaN
        max_drawdown.extend([np.nan] * horizon)
        df['target_max_drawdown'] = max_drawdown

    # Classification label (success/failure)
    df['target_success'] = (df[f'target_{horizon}d'] > TARGET_THRESHOLD).astype(int)

    return df

def prepare_features(df):
    """Combine all features into a feature matrix."""
    features = pd.DataFrame(index=df.index)

    # Vegas Channel
    features['veg_fast_band_width'] = df['veg_fast_band_width']
    features['veg_slow_band_width'] = df['veg_slow_band_width']
    features['veg_price_position'] = df['veg_price_position']
    features['veg_filter_slope'] = df['veg_filter_slope']
    features['veg_signal'] = df['veg_signal']

    # Hull
    features['hull_value'] = df['HULL']
    features['hull_slope'] = df['hull_slope']
    features['hull_signal'] = df['hull_signal']

    # MACD+RSI+Stoch
    features['macd_line'] = df['macd_line']
    features['signal_line'] = df['signal_line']
    features['macd_hist'] = df['macd_hist']
    features['macd_bullish'] = df['macd_bullish']
    features['macd_crossover'] = df['macd_crossover']
    features['rsi'] = df['rsi']
    features['rsi_overbought'] = df['rsi_overbought']
    features['stoch_k'] = df['stoch_k']
    features['stoch_d'] = df['stoch_d']
    features['stoch_bullish'] = df['stoch_bullish']

    # Volume & Volatility
    features['volume_ratio'] = df['volume_ratio']
    features['atr_pct'] = df['atr_pct']
    features['bb_width'] = df['bb_width']

    # Market benchmark (requires SPY data)
    if 'spy_return_15d' in df.columns:
        features['spy_return_15d'] = df['spy_return_15d']
        features['relative_strength'] = df['Close'] / df['spy_close']  # QQQ/SPY ratio

    # Time features
    features['day_of_week'] = df['day_of_week']
    features['month'] = df['month']
    features['quarter'] = df['quarter']

    # Lag features (past returns)
    for lag in [1, 3, 5, 10, 20]:
        features[f'return_{lag}d'] = df['Close'].pct_change(lag) * 100

    return features

def main():
    """Main data preparation pipeline."""
    print("=" * 50)
    print("Stock Prediction Data Preparation")
    print("=" * 50)

    # 1. Download QQQ data
    qqq_data = download_data(
        TICKER,
        TRAIN_START,
        TEST_END,
        save_path=os.path.join(RAW_DATA_DIR, f"{TICKER.lower()}.csv")
    )

    # 2. Download SPY (benchmark)
    spy_data = download_data(
        MARKET_BENCHMARK,
        TRAIN_START,
        TEST_END,
        save_path=os.path.join(RAW_DATA_DIR, f"{MARKET_BENCHMARK.lower()}.csv")
    )

    # 3. Feature engineering on QQQ
    print("\nCalculating features...")
    df = qqq_data.copy()
    df = calculate_vegas_channels(df)
    df = calculate_hull_ma(df, length=HULL_LENGTH, mode=HULL_MODE)
    df = calculate_macd_rsi_stoch(df)
    df = calculate_volume_volatility(df)
    df = add_time_features(df)

    # 4. Merge SPY features
    df['spy_close'] = spy_data['Close']
    df['spy_return_15d'] = df['spy_close'].pct_change(15) * 100
    df['relative_strength'] = df['Close'] / df['spy_close']

    # 5. Create labels
    print("Creating target labels...")
    df = create_labels(df, horizon=PREDICTION_HORIZON)

    # 6. Prepare feature matrix
    print("Assembling feature matrix...")
    features = prepare_features(df)
    targets = df[[f'target_{PREDICTION_HORIZON}d', 'target_max_drawdown', 'target_success']]

    # Combine
    full_df = pd.concat([features, targets], axis=1)
    full_df = full_df.dropna()  # Remove rows with NaN (from indicators or future targets)

    # 7. Split data by date
    print("\nSplitting data...")
    full_df['date'] = full_df.index

    train_mask = (full_df['date'] >= TRAIN_START) & (full_df['date'] <= TRAIN_END)
    val_mask = (full_df['date'] >= VAL_START) & (full_df['date'] <= VAL_END)
    test_mask = (full_df['date'] >= TEST_START) & (full_df['date'] <= TEST_END)

    train_df = full_df[train_mask].drop(columns=['date'])
    val_df = full_df[val_mask].drop(columns=['date'])
    test_df = full_df[test_mask].drop(columns=['date'])

    print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")

    # 8. Save
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Also save feature matrix for inspection
    feature_list = train_df.drop(columns=[f'target_{PREDICTION_HORIZON}d', 'target_max_drawdown', 'target_success']).columns.tolist()
    print(f"\nFeatures ({len(feature_list)}):")
    for f in feature_list:
        print(f"  - {f}")

    print(f"\nSaved datasets to {PROCESSED_DATA_DIR}")
    print("\nData preparation complete!")
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'feature_names': feature_list,
        'n_features': len(feature_list)
    }

if __name__ == "__main__":
    result = main()