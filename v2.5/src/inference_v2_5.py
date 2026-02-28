"""
Inference API for Stock Predictor V2.5
FastAPI server with new 4-class classification
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config_v2_5 import (
    MODEL_RESULTS_DIR, DEFAULT_HORIZON, HORIZONS, THRESHOLDS,
    CACHE_DATA_DIR, RAW_DATA_DIR, CLASS_LABELS
)
from .logging_utils import get_api_logger, get_prediction_logger, PredictionLogger

logging.basicConfig(level=logging.INFO)
logger = get_api_logger('inference_v2_5')

app = FastAPI(
    title="Stock Prediction API V2.5",
    description="Predict stock price movements with 4-class classification: UP, DOWN, UP_DOWN, SIDEWAYS",
    version="2.5.0"
)

models = {}
feature_names = []


def get_cache_data_file(symbol: str) -> Path:
    """Get cache file path for symbol."""
    CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DATA_DIR / f"{symbol.lower()}.csv"


def load_local_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load data from cache."""
    cache_file = get_cache_data_file(symbol)
    training_file = RAW_DATA_DIR / f"{symbol.lower()}.csv"
    
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        except Exception as e:
            logger.warning(f"Cache corrupted for {symbol}: {e}")
    
    if training_file.exists():
        try:
            df = pd.read_csv(training_file, parse_dates=['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        except:
            pass
    
    return None


def fetch_data_from_yahoo(symbol: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch data from Yahoo Finance."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7*365)).strftime('%Y-%m-%d')
    
    data = yf.download(symbol, start=start_date, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    data = data.reset_index()
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    
    return data


def get_stock_data(symbol: str, current_date: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    """Get stock data, fetching from Yahoo if needed."""
    local_data = load_local_data(symbol)
    
    if local_data is None or len(local_data) == 0:
        data = fetch_data_from_yahoo(symbol)
    else:
        data = local_data.copy()
    
    if current_date:
        try:
            target_date = pd.to_datetime(current_date)
        except:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {current_date}")
    else:
        target_date = data['date'].max()
    
    if target_date > data['date'].max():
        new_data = fetch_data_from_yahoo(symbol)
        data = pd.concat([data, new_data], ignore_index=True)
        data = data.drop_duplicates(subset=['date'], keep='last')
        data = data.sort_values('date').reset_index(drop=True)
    
    current_date_str = target_date.strftime('%Y-%m-%d')
    data = data[data['date'] <= target_date].copy()
    
    if len(data) < 200:
        raise HTTPException(status_code=400, detail=f"Insufficient data for {symbol}")
    
    return data, current_date_str


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators as features."""
    df = df.copy()
    close = df['close']
    high = df.get('high', close)
    low = df.get('low', close)
    volume = df.get('volume', pd.Series(1, index=close.index))
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'ma_{period}'] = close.rolling(window=period).mean()
        df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_ma = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['bb_upper'] = bb_ma + (bb_std * 2)
    df['bb_lower'] = bb_ma - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_ma
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / close * 100
    
    # Stochastic
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Momentum
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_10'] = close / close.shift(10) - 1
    df['momentum_20'] = close / close.shift(20) - 1
    
    # Volume
    df['volume_ma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_ma_20']
    
    # MA Crossover Regime
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df['ma_cross_bull'] = ((close > ma50) & (close > ma200) & (ma50 > ma200)).astype(int)
    df['ma_cross_bear'] = ((close < ma50) & (close < ma200) & (ma50 < ma200)).astype(int)
    
    # Volatility
    daily_returns = close.pct_change()
    df['volatility'] = daily_returns.rolling(20).std()
    
    # Fill NaN
    df = df.fillna(0)
    
    return df


def load_models(horizon: int, threshold: float):
    """Load models for specific horizon and threshold."""
    threshold_str = str(threshold).replace('.', '_')
    model_filename = f"gradientboosting_h{horizon}_t{threshold_str}.pkl"
    model_path = MODEL_RESULTS_DIR / model_filename
    
    if not model_path.exists():
        return None
    
    try:
        data = joblib.load(model_path)
        return data.get('model')
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    date: Optional[str] = None
    horizon: Optional[int] = DEFAULT_HORIZON
    threshold: Optional[float] = 0.01


class MultiHorizonRequest(BaseModel):
    symbol: str
    date: Optional[str] = None
    horizons: Optional[List[int]] = None
    thresholds: Optional[List[float]] = None


class PredictionResult(BaseModel):
    horizon: int
    threshold: float
    prediction: str
    probabilities: Dict[str, float]
    confidence: float


class PredictionResponse(BaseModel):
    symbol: str
    date: str
    predictions: List[PredictionResult]


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global models, feature_names
    
    logger.info("Loading models for all horizons and thresholds...")
    
    for horizon in HORIZONS:
        for threshold in THRESHOLDS:
            model = load_models(horizon, threshold)
            if model:
                key = f"{horizon}_{threshold}"
                models[key] = model
                logger.info(f"Loaded model for horizon={horizon}d, threshold={threshold*100}%")
    
    feature_file = MODEL_RESULTS_DIR / "feature_names.txt"
    if feature_file.exists():
        with open(feature_file) as f:
            feature_names = [line.strip() for line in f]
    
    logger.info(f"API initialized with {len(models)} models")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Prediction API V2.5",
        "version": "2.5.0",
        "description": "4-class classification: UP, DOWN, UP_DOWN, SIDEWAYS",
        "endpoints": [
            "/predict",
            "/predict/multi",
            "/health",
            "/model-info"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "horizons": HORIZONS,
        "thresholds": THRESHOLDS
    }


@app.get("/model-info")
async def model_info():
    """Model information."""
    return {
        "version": "2.5.0",
        "n_models": len(models),
        "horizons": HORIZONS,
        "thresholds": THRESHOLDS,
        "classes": CLASS_LABELS,
        "n_features": len(feature_names) if feature_names else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction for a single horizon and threshold."""
    pred_logger = get_prediction_logger('prediction')
    
    logger.info(f"Prediction request: {request.symbol}, horizon={request.horizon}, threshold={request.threshold}")
    
    horizon = request.horizon or DEFAULT_HORIZON
    threshold = request.threshold or 0.01
    
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail=f"Invalid horizon. Must be one of {HORIZONS}")
    
    if threshold not in THRESHOLDS:
        raise HTTPException(status_code=400, detail=f"Invalid threshold. Must be one of {THRESHOLDS}")
    
    data, date_str = get_stock_data(request.symbol, request.date)
    df_features = compute_features(data)
    
    feature_cols = [c for c in df_features.columns 
                   if c not in ['date', 'open', 'high', 'low', 'volume']]
    X = df_features[feature_cols].iloc[-1:].values
    
    key = f"{horizon}_{threshold}"
    model = models.get(key)
    
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for horizon={horizon}, threshold={threshold}. Please train first."
        )
    
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    prediction_str = CLASS_LABELS[pred]
    confidence = float(max(proba))
    
    prob_dict = {CLASS_LABELS[i]: float(p) for i, p in enumerate(proba)}
    
    pred_logger.log_prediction(
        symbol=request.symbol,
        date=date_str,
        horizon=horizon,
        threshold=threshold,
        prediction=prediction_str,
        probabilities=prob_dict,
        confidence=confidence
    )
    
    return PredictionResponse(
        symbol=request.symbol.upper(),
        date=date_str,
        predictions=[
            PredictionResult(
                horizon=horizon,
                threshold=threshold,
                prediction=prediction_str,
                probabilities=prob_dict,
                confidence=confidence
            )
        ]
    )


@app.post("/predict/multi", response_model=PredictionResponse)
async def predict_multi(request: MultiHorizonRequest):
    """Make predictions for multiple horizons and thresholds."""
    pred_logger = get_prediction_logger('prediction_multi')
    
    horizons = request.horizons or HORIZONS
    thresholds = request.thresholds or THRESHOLDS
    
    data, date_str = get_stock_data(request.symbol, request.date)
    df_features = compute_features(data)
    
    feature_cols = [c for c in df_features.columns 
                   if c not in ['date', 'open', 'high', 'low', 'volume']]
    X = df_features[feature_cols].iloc[-1:].values
    
    all_predictions = []
    
    for horizon in horizons:
        for threshold in thresholds:
            key = f"{horizon}_{threshold}"
            model = models.get(key)
            
            if model is None:
                continue
            
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            
            prediction_str = CLASS_LABELS[pred]
            confidence = float(max(proba))
            prob_dict = {CLASS_LABELS[i]: float(p) for i, p in enumerate(proba)}
            
            pred_logger.log_prediction(
                symbol=request.symbol,
                date=date_str,
                horizon=horizon,
                threshold=threshold,
                prediction=prediction_str,
                probabilities=prob_dict,
                confidence=confidence
            )
            
            all_predictions.append(PredictionResult(
                horizon=horizon,
                threshold=threshold,
                prediction=prediction_str,
                probabilities=prob_dict,
                confidence=confidence
            ))
    
    if not all_predictions:
        raise HTTPException(status_code=404, detail="No models available for requested horizons/thresholds")
    
    return PredictionResponse(
        symbol=request.symbol.upper(),
        date=date_str,
        predictions=all_predictions
    )


@app.get("/predict/by-stock/{symbol}")
async def predict_by_stock(
    symbol: str,
    date: Optional[str] = None,
    horizon: int = Query(DEFAULT_HORIZON, description="Prediction horizon in days"),
    threshold: float = Query(0.01, description="Threshold as decimal (0.01=1%)")
):
    """Get prediction by stock symbol."""
    request = PredictionRequest(
        symbol=symbol,
        date=date,
        horizon=horizon,
        threshold=threshold
    )
    return await predict(request)


@app.get("/predict/by-date/{date}")
async def predict_by_date(
    date: str,
    symbol: str = Query(..., description="Stock symbol"),
    horizon: int = Query(DEFAULT_HORIZON),
    threshold: float = Query(0.01)
):
    """Get prediction by date."""
    request = PredictionRequest(
        symbol=symbol,
        date=date,
        horizon=horizon,
        threshold=threshold
    )
    return await predict(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
