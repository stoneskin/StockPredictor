"""
Backtest module for Stock Predictor V2.5
Run historical predictions and compare with actual results
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

v25_root = Path(__file__).parent.parent
sys.path.insert(0, str(v25_root))

import logging
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple

from src.config_v2_5 import (
    MODEL_RESULTS_DIR, CACHE_DATA_DIR, RAW_DATA_DIR,
    HORIZONS, THRESHOLDS, CLASS_LABELS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('backtest_v2_5')

BACKTEST_DIR = v25_root / "data" / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

CHART_DIR = BACKTEST_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_data_file(symbol: str) -> Path:
    """Get cache file path for symbol."""
    return CACHE_DATA_DIR / f"{symbol.lower()}.csv"


def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load stock data for backtesting."""
    cache_file = get_cache_data_file(symbol)
    training_file = RAW_DATA_DIR / f"{symbol.lower()}.csv"
    
    df = None
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=['date'])
            df = df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Cache corrupted for {symbol}: {e}")
    
    if df is None and training_file.exists():
        try:
            df = pd.read_csv(training_file, parse_dates=['date'])
            df = df.sort_values('date').reset_index(drop=True)
        except:
            pass
    
    if df is None:
        logger.info(f"Fetching {symbol} data from Yahoo Finance")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        data = data.reset_index()
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        data['date'] = pd.to_datetime(data['date'])
        df = data.sort_values('date').reset_index(drop=True)
    
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators as features."""
    df = df.copy()
    close = df['close']
    high = df.get('high', close)
    low = df.get('low', close)
    volume = df.get('volume', pd.Series(1, index=close.index))
    
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'ma_{period}'] = close.rolling(window=period).mean()
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    bb_ma = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['bb_upper'] = bb_ma + (bb_std * 2)
    df['bb_lower'] = bb_ma - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_ma
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / close * 100
    
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_10'] = close / close.shift(10) - 1
    df['momentum_20'] = close / close.shift(20) - 1
    
    df['roc_5'] = (close - close.shift(5)) / close.shift(5) * 100
    df['roc_10'] = (close - close.shift(10)) / close.shift(10) * 100
    
    df['volume_ma_5'] = volume.rolling(window=5).mean()
    df['volume_ma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_ma_20']
    
    df['price_above_ma50'] = (close > df['ma_50']).astype(int)
    df['price_above_ma200'] = (close > df['ma_200']).astype(int)
    
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df['ma_cross_bull'] = ((close > ma50) & (close > ma200) & (ma50 > ma200)).astype(int)
    df['ma_cross_bear'] = ((close < ma50) & (close < ma200) & (ma50 < ma200)).astype(int)
    df['ma_cross_sideways'] = (~(df['ma_cross_bull'].astype(bool)) & ~(df['ma_cross_bear'].astype(bool))).astype(int)
    
    daily_returns = close.pct_change()
    df['volatility'] = daily_returns.rolling(20).std()
    
    high_vol = 0.03
    low_vol = 0.005
    df['volatility_high'] = (df['volatility'] > high_vol).astype(int)
    df['volatility_normal'] = ((df['volatility'] >= low_vol) & (df['volatility'] <= high_vol)).astype(int)
    df['volatility_low'] = (df['volatility'] < low_vol).astype(int)
    
    df['atr_ratio'] = df['atr'] / close
    df['trend_strength'] = abs(df['ma_50'] - df['ma_200']) / df['ma_200']
    df['price_vs_ma50'] = (close - df['ma_50']) / df['ma_50']
    df['price_vs_ma200'] = (close - df['ma_200']) / df['ma_200']
    
    df['volatility_regime_num'] = 0
    df.loc[df['volatility'] > high_vol, 'volatility_regime_num'] = 2
    df.loc[(df['volatility'] >= low_vol) & (df['volatility'] <= high_vol), 'volatility_regime_num'] = 1
    
    df['momentum_positive'] = (df['momentum_20'] > 0).astype(int)
    df['momentum_strong'] = (abs(df['momentum_20']) > 0.05).astype(int)
    
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)
    
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    
    bb_width_median = df['bb_width'].median()
    df['bb_squeeze'] = (df['bb_width'] < bb_width_median * 0.8).astype(int)
    df['bb_expansion'] = (df['bb_width'] > bb_width_median * 1.2).astype(int)
    
    for col in ['correlation_spy_5d', 'correlation_spy_10d', 'correlation_spy_20d', 'spy_momentum']:
        df[col] = 0.0
    
    df['spy_correlation_positive'] = (df['correlation_spy_20d'] > 0.3).astype(int)
    df['spy_correlation_negative'] = (df['correlation_spy_20d'] < -0.3).astype(int)
    df['spy_correlation_neutral'] = ((df['correlation_spy_20d'] >= -0.3) & (df['correlation_spy_20d'] <= 0.3)).astype(int)
    df['spy_momentum_positive'] = (df['spy_momentum'] > 0).astype(int)
    
    df = df.fillna(0)
    
    return df


def load_model(horizon: int, threshold: float) -> Tuple[Optional[object], Optional[str]]:
    """Load model for specific horizon and threshold."""
    threshold_str = str(threshold).replace('.', '_')
    
    model_priority = [
        (f"xgboost_h{horizon}_t{threshold_str}.pkl", "xgboost"),
        (f"gradientboosting_h{horizon}_t{threshold_str}.pkl", "gradientboosting"),
        (f"randomforest_h{horizon}_t{threshold_str}.pkl", "randomforest"),
        (f"ensemble_h{horizon}_t{threshold_str}.pkl", "ensemble"),
    ]
    
    for model_filename, model_name in model_priority:
        model_path = MODEL_RESULTS_DIR / model_filename
        
        if model_path.exists():
            try:
                data = joblib.load(model_path)
                model = data.get('model')
                if hasattr(model, 'classes_'):
                    logger.info(f"Model {model_filename} has {len(model.classes_)} classes")
                return model, model_name
            except Exception as e:
                logger.warning(f"Error loading {model_filename}: {e}")
                continue
    
    return None, None


def calculate_actual_class(df: pd.DataFrame, idx: int, horizon: int, threshold: float) -> int:
    """Calculate actual class based on future price movement."""
    if idx + horizon >= len(df):
        return -1
    
    current_price = df['close'].iloc[idx]
    future_prices = df['close'].iloc[idx+1:idx+horizon+1]
    
    max_gain = (future_prices.max() - current_price) / current_price
    max_loss = (future_prices.min() - current_price) / current_price
    
    gain_threshold = threshold
    loss_threshold = -threshold
    
    if max_gain > gain_threshold and max_loss < loss_threshold:
        return 2  # UP_DOWN
    elif max_gain > gain_threshold:
        return 0  # UP
    elif max_loss < loss_threshold:
        return 1  # DOWN
    else:
        return 3  # SIDEWAYS


def run_backtest(
    symbol: str = "TQQQ",
    horizon: int = 20,
    threshold: float = 0.01,
    days_back: int = 180,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Run backtest for a symbol.
    
    Args:
        symbol: Stock symbol to backtest
        horizon: Prediction horizon in days
        threshold: Threshold as decimal (0.01 = 1%)
        days_back: Number of days to backtest
        end_date: End date (default: today)
    
    Returns:
        DataFrame with backtest results
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    start_date = (datetime.now() - timedelta(days=days_back + 100)).strftime('%Y-%m-%d')
    
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    df = load_data(symbol, start_date, end_date)
    
    if df is None or len(df) == 0:
        raise ValueError(f"No data available for {symbol}")
    
    logger.info(f"Loaded {len(df)} rows of data")
    
    df_features = compute_features(df)
    
    feature_cols = [c for c in df_features.columns 
                    if c not in ['date', 'open', 'high', 'low', 'volume']]
    
    model, model_name = load_model(horizon, threshold)
    
    if model is None:
        raise ValueError(f"No model found for horizon={horizon}, threshold={threshold}")
    
    logger.info(f"Using {model_name} model for backtest")
    
    df_features = df_features.dropna(subset=feature_cols).copy()
    
    cutoff_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days_back)
    df_backtest = df_features[df_features['date'] >= pd.to_datetime(cutoff_date)].copy()
    
    trading_dates = df_backtest.index.tolist()
    
    results = []
    
    logger.info(f"Running backtest for {len(trading_dates)} trading days...")
    
    for i, idx in enumerate(trading_dates):
        if idx + horizon >= len(df_features):
            break
        
        X = df_features[feature_cols].iloc[idx:idx+1].values
        
        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
        except Exception as e:
            logger.warning(f"Error predicting at index {idx}: {e}")
            continue
        
        actual_class = calculate_actual_class(df_features, idx, horizon, threshold)
        
        if actual_class == -1:
            continue
        
        prediction_str = CLASS_LABELS[pred]
        actual_str = CLASS_LABELS[actual_class]
        
        current_price = df_features['close'].iloc[idx]
        future_prices = df_features['close'].iloc[idx+1:idx+horizon+1]
        max_gain = (future_prices.max() - current_price) / current_price * 100
        max_loss = (future_prices.min() - current_price) / current_price * 100
        
        correct = 1 if pred == actual_class else 0
        
        # Build full probability dict for all 4 classes
        prob_full = {cls: 0.0 for cls in CLASS_LABELS}
        if hasattr(model, 'classes_'):
            for i, class_idx in enumerate(model.classes_):
                if class_idx < len(CLASS_LABELS):
                    prob_full[CLASS_LABELS[class_idx]] = float(proba[i])
        else:
            # Fallback: assume order is correct
            for i, cls in enumerate(CLASS_LABELS):
                if i < len(proba):
                    prob_full[cls] = float(proba[i])
        
        confidence = max(prob_full.values())
        
        results.append({
            'date': df_features['date'].iloc[idx].strftime('%Y-%m-%d'),
            'symbol': symbol,
            'horizon': horizon,
            'threshold': threshold,
            'model_used': model_name,
            'prediction': prediction_str,
            'actual': actual_str,
            'correct': correct,
            'confidence': round(confidence, 3),
            'max_gain_pct': round(max_gain, 2),
            'max_loss_pct': round(max_loss, 2),
            'pred_UP': round(prob_full['UP'], 3),
            'pred_DOWN': round(prob_full['DOWN'], 3),
            'pred_UP_DOWN': round(prob_full['UP_DOWN'], 3),
            'pred_SIDEWAYS': round(prob_full['SIDEWAYS'], 3)
        })
        
        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i+1}/{len(trading_dates)} days")
    
    results_df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"backtest_{symbol}_h{horizon}_t{int(threshold*1000)}d_{timestamp}.csv"
    filepath = BACKTEST_DIR / filename
    results_df.to_csv(filepath, index=False)
    
    logger.info(f"Backtest results saved to {filepath}")
    
    return results_df


def generate_charts(results_df: pd.DataFrame, symbol: str) -> Dict[str, str]:
    """Generate comparison charts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    charts = {}
    
    if len(results_df) == 0:
        logger.warning("No results to generate charts")
        return charts
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    confusion = pd.crosstab(results_df['actual'], results_df['prediction'], 
                            rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'Confusion Matrix - {symbol}')
    
    accuracy_by_class = results_df.groupby('actual')['correct'].mean() * 100
    accuracy_by_class.plot(kind='bar', ax=axes[0, 1], color='steelblue')
    axes[0, 1].set_title('Accuracy by Actual Class')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
    
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df_sorted = results_df.sort_values('date')
    
    axes[1, 0].plot(results_df_sorted['date'], results_df_sorted['correct'].cumsum() / 
                    np.arange(1, len(results_df_sorted) + 1) * 100, 
                    label='Cumulative Accuracy', color='green')
    axes[1, 0].set_title('Cumulative Accuracy Over Time')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    prediction_counts = results_df['prediction'].value_counts()
    axes[1, 1].pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6'])
    axes[1, 1].set_title('Prediction Distribution')
    
    plt.tight_layout()
    
    chart_file = CHART_DIR / f"backtest_{symbol}_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    charts['summary'] = str(chart_file)
    logger.info(f"Chart saved to {chart_file}")
    
    return charts


def get_backtest_summary(results_df: pd.DataFrame) -> Dict:
    """Get backtest summary statistics."""
    if len(results_df) == 0:
        return {}
    
    total = len(results_df)
    correct = results_df['correct'].sum()
    accuracy = correct / total * 100
    
    summary = {
        'total_predictions': total,
        'correct_predictions': int(correct),
        'accuracy_pct': round(accuracy, 2),
        'by_class': {},
        'by_prediction': {}
    }
    
    for cls in CLASS_LABELS:
        cls_data = results_df[results_df['actual'] == cls]
        if len(cls_data) > 0:
            summary['by_class'][cls] = {
                'count': len(cls_data),
                'correct': int(cls_data['correct'].sum()),
                'accuracy_pct': round(cls_data['correct'].mean() * 100, 2)
            }
    
    for pred in CLASS_LABELS:
        pred_data = results_df[results_df['prediction'] == pred]
        if len(pred_data) > 0:
            summary['by_prediction'][pred] = {
                'count': len(pred_data),
                'actual_correct': int(pred_data['correct'].sum()),
                'precision_pct': round(pred_data['correct'].mean() * 100, 2)
            }
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtest for Stock Predictor V2.5')
    parser.add_argument('--symbol', type=str, default='TQQQ', help='Stock symbol')
    parser.add_argument('--horizon', type=int, default=20, help='Prediction horizon in days')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold as decimal')
    parser.add_argument('--days', type=int, default=180, help='Number of days to backtest')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    results = run_backtest(
        symbol=args.symbol,
        horizon=args.horizon,
        threshold=args.threshold,
        days_back=args.days,
        end_date=args.end_date
    )
    
    summary = get_backtest_summary(results)
    print(f"\nBacktest Summary for {args.symbol}:")
    print(f"Total Predictions: {summary['total_predictions']}")
    print(f"Accuracy: {summary['accuracy_pct']}%")
    print(f"\nBy Class:")
    for cls, stats in summary['by_class'].items():
        print(f"  {cls}: {stats['accuracy_pct']}% ({stats['correct']}/{stats['count']})")
    
    charts = generate_charts(results, args.symbol)
    print(f"\nCharts saved to: {charts.get('summary', 'N/A')}")
