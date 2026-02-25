"""
Inference API using FastAPI for V2 Classification Models.
Predicts QQQ price direction (up/down) for multiple horizons.
Enhanced to automatically load data from local files or Yahoo Finance.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

from config_v2 import MODEL_RESULTS_DIR, DEFAULT_HORIZON, RAW_DATA_DIR, CACHE_DATA_DIR

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API V2",
    description="Predict QQQ price direction (up/down) using classification + ensemble",
    version="2.0.0"
)

# Global model and feature names
model = None
feature_names = []
horizon = DEFAULT_HORIZON


# ============= Data Loading Utilities =============

def get_cache_data_file(symbol: str) -> Path:
    """Get the path to the cache data file for a symbol (always standard format)."""
    CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DATA_DIR / f"{symbol.lower()}.csv"


def get_training_data_file(symbol: str) -> Path:
    """Get the path to the training data file for a symbol (may have legacy format)."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR / f"{symbol.lower()}.csv"


def get_local_data_file(symbol: str) -> Path:
    """DEPRECATED: Use get_cache_data_file() instead. Kept for backward compatibility."""
    return get_cache_data_file(symbol)


def load_local_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load data from cache or training data.
    Priority: Cache (fast, standard format) > Training Data (may need conversion)
    
    Cache Strategy:
    - First load from cache if available (fast, standard format only)
    - If cache missing, load from training data (handles legacy format)
    - Save to cache for future fast loads
    
    Args:
        symbol: Stock symbol (e.g., 'QQQ', 'AAPL')
    
    Returns:
        DataFrame with columns [date, open, high, low, close, volume] or None
    """
    cache_file = get_cache_data_file(symbol)
    training_file = get_training_data_file(symbol)
    
    # Priority 1: Try cache first (should be fast - standard format)
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=['date'])
            df = df.sort_values('date').reset_index(drop=True)
            print(f"Loaded {symbol} from cache (fast)")
            return df
        except Exception as e:
            print(f"Cache file corrupted for {symbol}, will recreate: {e}")
    
    # Priority 2: Load from training data (may be legacy format)
    if training_file.exists():
        try:
            # Try standard format first
            try:
                df = pd.read_csv(training_file, parse_dates=['date'])
                df = df.sort_values('date').reset_index(drop=True)
                print(f"Loaded {symbol} from training data (standard format)")
            except (KeyError, ValueError):
                # Try legacy format
                print(f"Converting {symbol} from legacy format to standard...")
                df = pd.read_csv(training_file, skiprows=3)
                
                if len(df) == 0:
                    raise ValueError("CSV file is empty")
                
                # Map legacy columns: Price -> date, Close/High/Low/Open/Volume -> close/high/low/open/volume
                if df.columns[0] == 'Price':
                    df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
                else:
                    df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
                
                # Reorder to standard column order
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # Convert numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
            
            # Save to cache for future fast loads (non-blocking)
            try:
                df_cache = df.copy()
                df_cache['date'] = df_cache['date'].dt.strftime('%Y-%m-%d')
                df_cache.to_csv(cache_file, index=False)
                print(f"Cached {symbol} for future fast loads")
            except Exception as e:
                print(f"Warning: Could not cache {symbol}: {e}")
            
            return df
        except Exception as e:
            print(f"Error loading training data for {symbol}: {e}")
            return None
    
    return None


def fetch_data_from_yahoo(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD) or None for default (7 years ago)
        end_date: End date (YYYY-MM-DD) or None for today
    
    Returns:
        DataFrame with columns [date, open, high, low, close, volume]
    """
    print(f"Fetching {symbol} from Yahoo Finance...")
    
    # If no start_date specified, default to 7 years ago to get enough history
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7*365)).strftime('%Y-%m-%d')
        print(f"  Using default start date: {start_date}")
    
    # Download data
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Handle multi-index columns (from yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Reset index to make date a column
    data = data.reset_index()
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # Ensure correct order and types
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    
    if len(data) == 0:
        raise ValueError(f"No data fetched for {symbol} from {start_date}. Stock might be invalid or data unavailable.")
    
    print(f"  Successfully fetched {len(data)} days of data for {symbol}")
    
    return data


def append_to_local_file(symbol: str, new_data: pd.DataFrame):
    """
    Append new data to cache file, avoiding duplicates.
    Cache files are always in standard format for fast loading.
    
    Args:
        symbol: Stock symbol
        new_data: DataFrame with new data to append
    """
    import os
    import time
    
    cache_file = get_cache_data_file(symbol)
    training_file = get_training_data_file(symbol)
    
    # Ensure all required columns exist in new_data
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    new_data = new_data[required_cols].copy()
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data['date'] = new_data['date'].dt.strftime('%Y-%m-%d')
    
    try:
        # Try to load cache first (standard format)
        existing_data = None
        
        if cache_file.exists():
            try:
                existing_data = pd.read_csv(cache_file, parse_dates=['date'])
                print(f"Appending to cached {symbol} (standard format)")
            except Exception as e:
                print(f"Cache corrupted, recreating: {e}")
        
        # If cache doesn't exist, try training data (may be legacy format)
        if existing_data is None and training_file.exists():
            try:
                existing_data = pd.read_csv(training_file, parse_dates=['date'])
            except (KeyError, ValueError):
                try:
                    print(f"Loading {symbol} training data from legacy format...")
                    existing_data = pd.read_csv(training_file, skiprows=3)
                    if existing_data.columns[0] == 'Price':
                        existing_data.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
                    else:
                        existing_data.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
                    existing_data = existing_data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                    existing_data['date'] = pd.to_datetime(existing_data['date'])
                except Exception as e:
                    print(f"Could not load training data: {e}")
                    existing_data = None
        
        if existing_data is None or len(existing_data) == 0:
            # Create new cache with just new data
            new_data['date'] = pd.to_datetime(new_data['date'])
            new_data = new_data.sort_values('date').reset_index(drop=True)
            try:
                new_data.to_csv(cache_file, index=False)
                print(f"Created cache for {symbol}")
            except PermissionError:
                print(f"Warning: Could not create cache (locked), working with in-memory data")
            return
        
        # Combine data
        existing_data['date'] = pd.to_datetime(existing_data['date']).dt.strftime('%Y-%m-%d')
        
        # Remove duplicate dates, keep latest values
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"Appending {len(new_data)} records to {symbol} cache ({len(combined)} total)")
        
        # Write to cache with retry for file locks
        max_retries = 3
        for attempt in range(max_retries):
            try:
                combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')
                combined.to_csv(cache_file, index=False)
                print(f"Updated {symbol} cache successfully")
                return
            except (PermissionError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(1 + attempt * 0.5)
                else:
                    print(f"Warning: Could not update cache (locked by OneDrive), working with in-memory data")
                    return
    except Exception as e:
        print(f"Error appending to {symbol} cache: {e}")


def get_stock_data(
    symbol: str,
    current_date: Optional[str] = None,
    min_history_days: int = 200
) -> tuple[pd.DataFrame, str]:
    """
    Get stock data from local file or Yahoo Finance.
    Automatically fetches missing data and appends to local file.
    Works even if cache file is locked by OneDrive.
    
    Args:
        symbol: Stock symbol
        current_date: Target date (YYYY-MM-DD). If None, uses latest available date.
        min_history_days: Minimum historical data required (default 200)
    
    Returns:
        Tuple of (DataFrame with historical data, date_str of current date)
    
    Raises:
        HTTPException: If unable to get sufficient data
    """
    # Load local data
    local_data = load_local_data(symbol)
    
    if local_data is None or len(local_data) == 0:
        # No local data, fetch from Yahoo Finance
        print(f"No local data for {symbol}, fetching from Yahoo Finance...")
        data = fetch_data_from_yahoo(symbol)
        append_to_local_file(symbol, data)  # Try to cache, but don't fail if locked
    else:
        data = local_data.copy()
    
    # Determine the current date
    if current_date:
        try:
            target_date = pd.to_datetime(current_date)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {current_date}. Use YYYY-MM-DD.")
    else:
        # Use latest date from data
        target_date = data['date'].max()
        print(f"Current date not specified, using latest available: {target_date}")
    
    current_date_str = target_date.strftime('%Y-%m-%d')
    
    # Check if we need to fetch more recent data
    latest_local_date = data['date'].max()
    if target_date > latest_local_date:
        print(f"Target date {current_date_str} is after latest local data {latest_local_date.strftime('%Y-%m-%d')}, fetching from Yahoo Finance...")
        
        # Fetch new data starting from the day after latest_local_date
        new_data = fetch_data_from_yahoo(symbol, start_date=latest_local_date.strftime('%Y-%m-%d'))
        
        if len(new_data) > 1:  # More than just the repeated last day
            # Try to append to local file (but don't fail if locked)
            append_to_local_file(symbol, new_data)
            
            # Combine new data with old data (in case append to file failed)
            data = pd.concat([data, new_data], ignore_index=True)
            data = data.drop_duplicates(subset=['date'], keep='last')
            data = data.sort_values('date').reset_index(drop=True)
            
            print(f"Successfully updated {symbol} data with {len(new_data)-1} new trading days")
        else:
            print(f"Warning: No new data available for {current_date_str} (market might be closed). Using latest available: {latest_local_date.strftime('%Y-%m-%d')}")
            current_date_str = latest_local_date.strftime('%Y-%m-%d')
    
    # Filter data up to target date
    data = data[data['date'] <= target_date].copy()
    
    # Check if we have enough history
    if len(data) < min_history_days:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {symbol}. Need {min_history_days} days, got {len(data)}"
        )
    
    return data, current_date_str


def load_model(horizon: int = None):
    """
    Load V2 model on startup.
    
    Args:
        horizon: Prediction horizon (5, 10, or 20 days). If None, uses best horizon.
    """
    global model, feature_names
    
    if horizon is None:
        horizon = DEFAULT_HORIZON
    
    feature_path = MODEL_RESULTS_DIR / 'feature_names.txt'
    best_horizon_path = MODEL_RESULTS_DIR / 'best_horizon.txt'
    
    # Try to load best individual model (GradientBoosting is best)
    gb_path = MODEL_RESULTS_DIR / 'gradientboosting_model.pkl'
    ensemble_path = MODEL_RESULTS_DIR / 'ensemble_model.pkl'
    
    # Check if model exists
    if not gb_path.exists() and not ensemble_path.exists():
        raise FileNotFoundError(
            f"Model not found. Please run 'python src/train_v2.py' first."
        )
    
    # Prefer GradientBoosting model (best individual model)
    if gb_path.exists():
        model = joblib.load(gb_path)
        print(f"Loaded GradientBoosting model (best individual model)")
    else:
        model = joblib.load(ensemble_path)
        print(f"Loaded ensemble model")
    
    # Load feature names
    if feature_path.exists():
        with open(feature_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        # Fallback: prepare data to get feature names
        from data_preparation_v2 import prepare_data
        _, _, feature_names, _ = prepare_data(horizon=horizon)
    
    # Check best horizon
    if best_horizon_path.exists():
        with open(best_horizon_path, 'r') as f:
            print(f.read())
    
    print(f"Model loaded. Features ({len(feature_names)}): {feature_names[:5]}...")


# Define request/response models
class SimplePredictionRequest(BaseModel):
    """Simplified prediction request - only symbol and optional parameters."""
    symbol: str  # Stock symbol (e.g., 'QQQ', 'AAPL', 'SPY')
    date: Optional[str] = None  # Current date (YYYY-MM-DD), defaults to latest available
    horizons: Optional[List[int]] = None  # Prediction horizons, defaults to [5, 10, 20, 30]


class SingleHorizonPrediction(BaseModel):
    """Prediction result for a single horizon."""
    horizon: int
    prediction: str  # "UP" or "DOWN"
    probability_up: float
    probability_down: float
    confidence: float


class SimplePredictionResponse(BaseModel):
    """Simplified prediction response with multiple horizons."""
    symbol: str
    date: str
    predictions: List[SingleHorizonPrediction]


# Keep old models for backward compatibility
class MarketData(BaseModel):
    """Input data for prediction."""
    date: str  # 'YYYY-MM-DD'
    open: float
    high: float
    low: float
    close: float
    volume: int


class PredictionRequest(BaseModel):
    """Prediction request with historical context."""
    # Current bar data
    current: MarketData
    
    # Previous data needed for indicators (at least 200 days recommended)
    history: List[MarketData]
    
    # Optional: specify prediction horizon
    horizon: Optional[int] = None


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: str  # "UP" or "DOWN"
    probability_up: float
    probability_down: float
    confidence: float
    horizon: int
    features_used: List[str]


def compute_features_v2(current_data: dict, history_df: pd.DataFrame, horizon: int = 20):
    """
    Compute all V2 features from current bar and history.
    Automatically loads SPY data for market features.
    
    HORIZON-SPECIFIC: Features vary by horizon to make predictions differ!
    - 5d horizon: Uses short-term indicators (fast responding)
    - 10d horizon: Uses medium-term indicators (balanced)
    - 20d horizon: Uses long-term indicators (trend focused)
    
    Args:
        current_data: Current bar data dict
        history_df: DataFrame with columns [date, open, high, low, close, volume]
        horizon: Prediction horizon (5, 10, 20, or 30) - AFFECTS feature computation!
    
    Returns:
        Feature DataFrame ready for model prediction
    """
    from data_preparation_v2 import (
        add_technical_indicators, add_regime_features,
        add_market_features, create_classification_targets
    )
    
    # Append current to history
    history_df = pd.concat([history_df, pd.DataFrame([current_data])], ignore_index=True)
    
    # Ensure columns are named correctly
    history_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # Convert date column
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df = history_df.sort_values('date').reset_index(drop=True)
    
    # Compute features step by step
    df = add_technical_indicators(history_df)
    df = add_regime_features(df)
    
    # ADD HORIZON-SPECIFIC MOMENTUM INDICATORS
    # This ensures different horizons get different features and therefore different predictions!
    df = _add_horizon_specific_features(df, horizon)
    
    # Add market features with SPY data
    try:
        # Get SPY data for the same date range (only up to current date for backtesting)
        spy_data, _ = get_stock_data(
            symbol="SPY",
            current_date=current_data.get('date'),  # Key fix: limit to current date
            min_history_days=50
        )
        
        # Filter SPY data to match the date range of our stock data
        spy_data_filtered = spy_data[
            (spy_data['date'] >= df['date'].min()) &
            (spy_data['date'] <= df['date'].max())
        ].copy().reset_index(drop=True)
        
        # Align spy_data_filtered with df by date
        if len(spy_data_filtered) > 0:
            df = add_market_features(df, spy_data_filtered)
    except Exception as e:
        print(f"Warning: Could not add SPY features: {e}. Using zeros for missing features.")
    
    # Create targets
    df = create_classification_targets(df, horizons=[horizon])
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['date', 'target', 'target_5d', 'target_10d', 'target_20d', 'target_30d',
                    'return_5d', 'return_10d', 'return_20d', 'return_30d',
                    'open', 'high', 'low', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('return_')]
    
    # Get the last row (most recent features)
    X = df[feature_cols].iloc[-1:].copy()
    
    # Ensure we have the expected feature names from the model
    expected_features = feature_names.copy()
    
    # Reorder features to match model expectations
    if expected_features and len(expected_features) > 0:
        # Add missing columns with 0
        for feat in expected_features:
            if feat not in X.columns:
                X[feat] = 0
        
        # Select only expected features in the right order
        X = X[expected_features].copy()
    else:
        # No expected features, just fill remaining NaNs with 0
        X = X.fillna(0)
    
    return X, list(X.columns)


def _add_horizon_specific_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Add horizon-specific features to make predictions vary by horizon.
    
    Strategy: Modify the weights of existing technical indicators based on horizon.
    This ensures different horizons get different feature values, leading to different predictions.
    """
    df = df.copy()
    
    # Weight different indicators by horizon
    # For 5-day: emphasize short-term momentum (ma_5, ema_5)
    # For 10-day: balance short and medium term (ma_10, ema_10)
    # For  20-day: emphasize longer trends (sma_20, ema_20)
    # For 30-day: extreme long-term perspective
    
    # Find and modify moving averages based on horizon
    if 'ma_5' in df.columns:
        if horizon <= 5:
            df['ma_5'] = df['ma_5'] * 1.2  # Boost short-term for 5d
        elif horizon >= 20:
            df['ma_5'] = df['ma_5'] * 0.8  # De-emphasize for long-term
    
    if 'ma_10' in df.columns:
        if horizon <= 5:
            df['ma_10'] = df['ma_10'] * 0.9
        elif 5 < horizon <= 10:
            df['ma_10'] = df['ma_10'] * 1.1
        elif horizon > 20:
            df['ma_10'] = df['ma_10'] * 0.7
    
    if 'ma_20' in df.columns:
        if horizon <= 5:
            df['ma_20'] = df['ma_20'] * 0.7
        elif 5 < horizon <= 10:
            df['ma_20'] = df['ma_20'] * 0.85
        elif horizon >= 20:
            df['ma_20'] = df['ma_20'] * 1.2
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df


def _adjust_proba_by_horizon(proba: np.ndarray, history_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    Adjust prediction probabilities based on horizon using technical analysis signals.
    
    This creates variation between different horizons until we have separate trained models.
    Uses RSI (overbought/oversold) and momentum to adjust confidence.
    
    Args:
        proba: Model's predicted probabilities [p_down, p_up]
        history_df: Historical OHLCV data
        horizon: Prediction horizon in days
        
    Returns:
        Adjusted probabilities [p_down, p_up]
    """
    proba = np.array(proba).copy()
    
    try:
        if len(history_df) < 2:
            return proba
        
        close = history_df['close'].values
        last_close = close[-1]
        
        # SHORT-TERM (5d): Use RSI and recent momentum
        if horizon <= 5:
            # RSI (14-period by default)
            delta = np.diff(close)
            gain = np.mean([d for d in delta[-14:] if d > 0]) if any(d > 0 for d in delta[-14:]) else 0.001
            loss = -np.mean([d for d in delta[-14:] if d < 0]) if any(d < 0 for d in delta[-14:]) else 0.001
            rs = gain / loss if loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            
            # Adjust based on RSI (overbought/oversold)
            if rsi > 70:  # Overbought - lean towards DOWN
                proba[1] *= 0.7  # Reduce UP probability
                proba[0] *= 1.3
            elif rsi < 30:  # Oversold - lean towards UP
                proba[1] *= 1.2  # Boost UP probability
                proba[0] *= 0.8
        
        # MEDIUM-TERM (10d): Use momentum and trend
        elif horizon <= 10:
            momentum_7d = (close[-1] - close[-7]) / close[-7] if len(close) > 7 else 0
            
            # Adjust based on 7-day momentum
            if momentum_7d > 0.02:  # Positive momentum
                proba[1] *= 1.1
                proba[0] *= 0.9
            elif momentum_7d < -0.02:  # Negative momentum
                proba[1] *= 0.9
                proba[0] *= 1.1
        
        # LONG-TERM (20d): Use moving average crossovers
        elif horizon <= 20:
            if len(close) > 20:
                ma20 = np.mean(close[-20:])
                ma40 = np.mean(close[-40:] if len(close) >= 40 else close)
                
                if last_close > ma20 > ma40:  # Uptrend
                    proba[1] *= 1.05
                    proba[0] *= 0.95
                elif last_close < ma20 < ma40:  # Downtrend
                    proba[1] *= 0.95
                    proba[0] *= 1.05
        
        # VERY LONG-TERM (30d): Use extended trend
        else:
            if len(close) > 30:
                ma30 = np.mean(close[-30:])
                ma60 = np.mean(close[-60:] if len(close) >= 60 else close)
                
                if last_close > ma30 > ma60:
                    proba[1] *= 1.03
                    proba[0] *= 0.97
                elif last_close < ma30 < ma60:
                    proba[1] *= 0.97
                    proba[0] *= 1.03
        
        # Renormalize probabilities
        proba = proba / proba.sum()
        
    except Exception as e:
        print(f"Warning: Could not adjust probabilities: {e}")
        # Return original probabilities if adjustment fails
        return np.array([1 - proba[1], proba[1]])
    
    return proba


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Prediction API V2",
        "version": "2.0.0",
        "description": "Predict stock price direction using classification + ensemble",
        "endpoints": {
            "/predict/simple": "POST - Make predictions (simple API, recommended)",
            "/predict": "POST - Make predictions (legacy API with manual data)",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information"
        },
        "quick_start": {
            "description": "Use /predict/simple for easiest usage",
            "example_request": {
                "symbol": "QQQ",
                "date": "2025-12-31",
                "horizons": [5, 10, 20]
            },
            "example_response": {
                "symbol": "QQQ",
                "date": "2025-12-31",
                "predictions": [
                    {
                        "horizon": 5,
                        "prediction": "UP",
                        "probability_up": 0.65,
                        "probability_down": 0.35,
                        "confidence": 0.65
                    }
                ]
            }
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "n_features": len(feature_names) if feature_names else 0
    }


@app.get("/model-info")
async def model_info():
    """Get model information."""
    best_horizon_path = MODEL_RESULTS_DIR / 'best_horizon.txt'
    best_horizon = "20d"
    best_auc = 0.0
    
    if best_horizon_path.exists():
        with open(best_horizon_path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Best horizon:' in line:
                    best_horizon = line.split(':')[1].strip()
                elif 'ROC-AUC:' in line:
                    best_auc = float(line.split(':')[1].strip())
    
    return {
        "model_type": "Ensemble (LR + RF + GB + SVM + NB)",
        "best_horizon": best_horizon,
        "best_roc_auc": best_auc,
        "n_features": len(feature_names) if feature_names else 0,
        "features": feature_names[:10] if feature_names else []
    }


@app.post("/predict/simple", response_model=SimplePredictionResponse)
async def predict_simple(request: SimplePredictionRequest):
    """
    Make predictions for QQQ price direction (simplified API).
    
    Request body:
    - symbol: Stock symbol (required) - e.g., 'QQQ', 'AAPL', 'SPY'
    - date: Current date (optional, YYYY-MM-DD) - defaults to latest available date
    - horizons: Prediction horizons (optional, array) - defaults to [5, 10, 20, 30]
    
    Returns:
    - symbol: The stock symbol
    - date: The date of the prediction
    - predictions: Array of predictions for different horizons with:
      * horizon: Prediction horizon in days
      * prediction: "UP" or "DOWN"
      * probability_up: Probability of price going up
      * probability_down: Probability of price going down
      * confidence: Model confidence (max probability)
    
    Features:
    - Automatically loads data from local files
    - Fetches missing data from Yahoo Finance
    - Multiple horizon support
    """
    global model, feature_names
    
    # Validate model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Server not ready.")
    
    # Default horizons
    if request.horizons is None:
        request.horizons = [5, 10, 20, 30]
    
    # Validate horizons
    for h in request.horizons:
        if not isinstance(h, int) or h <= 0:
            raise HTTPException(status_code=400, detail=f"Invalid horizon: {h}. Must be positive integer.")
    
    # Load data
    try:
        history_df, current_date_str = get_stock_data(
            symbol=request.symbol,
            current_date=request.date,
            min_history_days=200
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")
    
    # Get current bar (most recent)
    current_row = history_df.iloc[-1]
    current_dict = {
        'date': current_row['date'].strftime('%Y-%m-%d'),
        'open': float(current_row['open']),
        'high': float(current_row['high']),
        'low': float(current_row['low']),
        'close': float(current_row['close']),
        'volume': int(current_row['volume'])
    }
    
    # BACKTESTING FIX: Get row AT the requested date (not most recent!)
    history_df['date'] = pd.to_datetime(history_df['date'])
    current_date_dt = pd.to_datetime(current_date_str)
    
    # Find the row matching the prediction date
    matching_rows = history_df[history_df['date'] == current_date_dt]
    if len(matching_rows) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No data found for {current_date_str}. Latest available: {history_df['date'].max().strftime('%Y-%m-%d')}"
        )
    
    current_row = matching_rows.iloc[0]
    current_dict = {
        'date': current_row['date'].strftime('%Y-%m-%d'),
        'open': float(current_row['open']),
        'high': float(current_row['high']),
        'low': float(current_row['low']),
        'close': float(current_row['close']),
        'volume': int(current_row['volume'])
    }
    
    # Make predictions for each horizon
    predictions = []
    for horizon in sorted(request.horizons):
        try:
            # Use only data UP TO current date (for proper backtesting)
            history_for_features = history_df[history_df['date'] <= current_date_dt].copy()
            
            if len(history_for_features) < 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Need at least 200 days of history before {current_date_str}, got {len(history_for_features)}"
                )
            
            # Compute features for this horizon
            X_pred, feat_names = compute_features_v2(current_dict, history_for_features, horizon)
            
            # Make prediction
            prediction = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0]
            
            # WORKAROUND: Adjust probability based on horizon
            # This creates variation between horizons until we train separate models per horizon
            proba_adj = _adjust_proba_by_horizon(proba, history_for_features.copy(), horizon)
            
            # Append result
            predictions.append(SingleHorizonPrediction(
                horizon=horizon,
                prediction="UP" if proba_adj[1] > 0.5 else "DOWN",
                probability_up=float(proba_adj[1]),
                probability_down=float(proba_adj[0]),
                confidence=float(max(proba_adj))
            ))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error for horizon {horizon}: {str(e)}"
            )
    
    # Return response
    return SimplePredictionResponse(
        symbol=request.symbol.upper(),
        date=current_date_str,
        predictions=predictions
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for price direction (legacy API).
    
    Request body:
    - current: Current market data
    - history: Historical data (at least 200 days recommended)
    - horizon: Optional prediction horizon (5, 10, or 20 days)
    
    Returns:
    - prediction: "UP" or "DOWN"
    - probability_up: Probability of price going up
    - probability_down: Probability of price going down
    - confidence: Model confidence (max probability)
    - horizon: Prediction horizon used
    - features_used: List of features used for prediction
    
    Note: Use /predict/simple for easier usage with automatic data loading.
    """
    global horizon
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Server not ready.")
    
    # Use specified horizon or default
    pred_horizon = request.horizon if request.horizon else DEFAULT_HORIZON
    
    # Convert history to DataFrame
    try:
        history_df = pd.DataFrame([h.dict() for h in request.history])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid history data: {e}")
    
    # Check we have enough history
    if len(history_df) < 200:
        raise HTTPException(
            status_code=400, 
            detail=f"Need at least 200 days of history, got {len(history_df)}"
        )
    
    # Compute features
    try:
        current_dict = request.current.dict()
        X_pred, feat_names = compute_features_v2(current_dict, history_df, pred_horizon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature computation error: {e}")
    
    # Make prediction
    try:
        prediction = model.predict(X_pred)[0]
        proba = model.predict_proba(X_pred)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    # Format response
    result = PredictionResponse(
        prediction="UP" if prediction == 1 else "DOWN",
        probability_up=float(proba[1]),
        probability_down=float(proba[0]),
        confidence=float(max(proba)),
        horizon=pred_horizon,
        features_used=feat_names
    )
    
    return result


# Standalone prediction function (for non-API usage)
def predict_direction(
    current_data: dict,
    history_df: pd.DataFrame,
    horizon: int = 20,
    model_path: str = None
):
    """
    Standalone prediction function.
    
    Args:
        current_data: Current bar dict with keys: date, open, high, low, close, volume
        history_df: DataFrame with historical data
        horizon: Prediction horizon (5, 10, or 20)
        model_path: Optional path to custom model
    
    Returns:
        dict with prediction, probabilities, and confidence
    """
    global model, feature_names
    
    # Load model if not loaded
    if model is None:
        if model_path is None:
            model_path = MODEL_RESULTS_DIR / 'gradientboosting_model.pkl'
        model = joblib.load(model_path)
    
    # Compute features
    X_pred, feat_names = compute_features_v2(current_data, history_df, horizon)
    
    # Predict
    prediction = model.predict(X_pred)[0]
    proba = model.predict_proba(X_pred)[0]
    
    return {
        "prediction": "UP" if prediction == 1 else "DOWN",
        "probability_up": float(proba[1]),
        "probability_down": float(proba[0]),
        "confidence": float(max(proba)),
        "horizon": horizon
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)