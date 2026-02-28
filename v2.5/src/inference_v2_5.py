"""
Inference API for Stock Predictor V2.5
FastAPI server with new 4-class classification
"""

import sys
from pathlib import Path

# Add v2.5 to path
v25_root = Path(__file__).parent.parent
sys.path.insert(0, str(v25_root))

import os
import logging
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.config_v2_5 import (
    MODEL_RESULTS_DIR, DEFAULT_HORIZON, HORIZONS, THRESHOLDS,
    CACHE_DATA_DIR, RAW_DATA_DIR, CLASS_LABELS
)
from src.logging_utils import get_api_logger, get_prediction_logger, PredictionLogger

logging.basicConfig(level=logging.INFO)
logger = get_api_logger('inference_v2_5')

app = FastAPI(
    title="Stock Prediction API V2.5.1",
    description="Predict stock price movements with 4-class classification: UP, DOWN, UP_DOWN, SIDEWAYS",
    version="2.5.1"
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


def compute_features(df: pd.DataFrame, include_spy_features: bool = True) -> pd.DataFrame:
    """Compute technical indicators as features.
    
    Must match features used in training (V2.5.1).
    """
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
    
    # Rate of Change
    df['roc_5'] = (close - close.shift(5)) / close.shift(5) * 100
    df['roc_10'] = (close - close.shift(10)) / close.shift(10) * 100
    
    # Volume
    df['volume_ma_5'] = volume.rolling(window=5).mean()
    df['volume_ma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_ma_20']
    
    # Price position relative to MAs
    df['price_above_ma50'] = (close > df['ma_50']).astype(int)
    df['price_above_ma200'] = (close > df['ma_200']).astype(int)
    
    # MA Crossover Regime
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df['ma_cross_bull'] = ((close > ma50) & (close > ma200) & (ma50 > ma200)).astype(int)
    df['ma_cross_bear'] = ((close < ma50) & (close < ma200) & (ma50 < ma200)).astype(int)
    df['ma_cross_sideways'] = (~(df['ma_cross_bull'].astype(bool)) & ~(df['ma_cross_bear'].astype(bool))).astype(int)
    
    # Volatility
    daily_returns = close.pct_change()
    df['volatility'] = daily_returns.rolling(20).std()
    
    high_vol = 0.03
    low_vol = 0.005
    df['volatility_high'] = (df['volatility'] > high_vol).astype(int)
    df['volatility_normal'] = ((df['volatility'] >= low_vol) & (df['volatility'] <= high_vol)).astype(int)
    df['volatility_low'] = (df['volatility'] < low_vol).astype(int)
    
    # V2.5.1 Additional regime features
    
    # ATR Ratio
    df['atr_ratio'] = df['atr'] / close
    
    # Trend Strength
    df['trend_strength'] = abs(df['ma_50'] - df['ma_200']) / df['ma_200']
    
    # Price vs MA
    df['price_vs_ma50'] = (close - df['ma_50']) / df['ma_50']
    df['price_vs_ma200'] = (close - df['ma_200']) / df['ma_200']
    
    # Volatility regime numeric
    df['volatility_regime_num'] = 0
    df.loc[df['volatility'] > high_vol, 'volatility_regime_num'] = 2
    df.loc[(df['volatility'] >= low_vol) & (df['volatility'] <= high_vol), 'volatility_regime_num'] = 1
    
    # Momentum regime
    df['momentum_positive'] = (df['momentum_20'] > 0).astype(int)
    df['momentum_strong'] = (abs(df['momentum_20']) > 0.05).astype(int)
    
    # RSI regime
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)
    
    # Stochastic regime
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    
    # Bollinger Band squeeze/expansion
    bb_width_median = df['bb_width'].median()
    df['bb_squeeze'] = (df['bb_width'] < bb_width_median * 0.8).astype(int)
    df['bb_expansion'] = (df['bb_width'] > bb_width_median * 1.2).astype(int)
    
    # SPY correlation features (if SPY data available)
    if include_spy_features:
        try:
            spy_data = get_spy_data(df.index)
            if spy_data is not None and len(spy_data) > 0:
                spy_close = spy_data['close']
                stock_close = close
                
                common_idx = df.index.intersection(spy_data.index)
                if len(common_idx) > 20:
                    stock_ret = stock_close.loc[common_idx].pct_change()
                    spy_ret = spy_close.loc[common_idx].pct_change()
                    
                    for window in [5, 10, 20]:
                        corr = stock_ret.rolling(window).corr(spy_ret)
                        df.loc[common_idx, f'correlation_spy_{window}d'] = corr.fillna(0)
                    
                    # SPY momentum
                    spy_ma20 = spy_close.rolling(20).mean()
                    spy_momentum = (spy_close - spy_ma20) / spy_ma20
                    df.loc[common_idx, 'spy_momentum'] = spy_momentum.loc[common_idx]
        except:
            pass
    
    # Initialize SPY correlation columns if not set
    for col in ['correlation_spy_5d', 'correlation_spy_10d', 'correlation_spy_20d', 'spy_momentum']:
        if col not in df.columns:
            df[col] = 0.0
    df['correlation_spy_5d'] = df['correlation_spy_5d'].fillna(0)
    df['correlation_spy_10d'] = df['correlation_spy_10d'].fillna(0)
    df['correlation_spy_20d'] = df['correlation_spy_20d'].fillna(0)
    df['spy_momentum'] = df['spy_momentum'].fillna(0)
    
    # SPY correlation regime
    df['spy_correlation_positive'] = (df['correlation_spy_20d'] > 0.3).astype(int)
    df['spy_correlation_negative'] = (df['correlation_spy_20d'] < -0.3).astype(int)
    df['spy_correlation_neutral'] = ((df['correlation_spy_20d'] >= -0.3) & (df['correlation_spy_20d'] <= 0.3)).astype(int)
    df['spy_momentum_positive'] = (df['spy_momentum'] > 0).astype(int)
    
    # Fill NaN
    df = df.fillna(0)
    
    return df


def get_spy_data(index) -> Optional[pd.DataFrame]:
    """Get SPY data aligned with given index."""
    try:
        cache_file = CACHE_DATA_DIR / "spy.csv"
        if cache_file.exists():
            spy_df = pd.read_csv(cache_file, parse_dates=['date'])
            spy_df = spy_df.set_index('date')
            return spy_df
    except:
        pass
    return None


def load_models(horizon: int, threshold: float):
    """Load models for specific horizon and threshold.
    
    Priority: XGBoost > GradientBoosting > RandomForest > Ensemble
    XGBoost is the best performer in V2.5.1
    """
    threshold_str = str(threshold).replace('.', '_')
    
    # Model priority order (best first)
    model_priority = [
        f"xgboost_h{horizon}_t{threshold_str}.pkl",
        f"gradientboosting_h{horizon}_t{threshold_str}.pkl",
        f"randomforest_h{horizon}_t{threshold_str}.pkl",
        f"ensemble_h{horizon}_t{threshold_str}.pkl",
    ]
    
    for model_filename in model_priority:
        model_path = MODEL_RESULTS_DIR / model_filename
        
        if model_path.exists():
            try:
                data = joblib.load(model_path)
                logger.info(f"Loaded {model_filename} for horizon={horizon}d, threshold={threshold*100}%")
                return data.get('model')
            except Exception as e:
                logger.warning(f"Error loading {model_filename}: {e}")
                continue
    
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
    base_logger = get_prediction_logger('prediction')
    pred_logger = PredictionLogger(base_logger)
    
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
    base_logger = get_prediction_logger('prediction_multi')
    pred_logger = PredictionLogger(base_logger)
    
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
