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

from config_v2 import MODEL_RESULTS_DIR, DEFAULT_HORIZON, RAW_DATA_DIR

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

def get_local_data_file(symbol: str) -> Path:
    """Get the path to the local data file for a symbol."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR / f"{symbol.lower()}.csv"


def load_local_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load data from local CSV file.
    
    Args:
        symbol: Stock symbol (e.g., 'QQQ', 'AAPL')
    
    Returns:
        DataFrame with columns [date, open, high, low, close, volume] or None if file doesn't exist
    """
    file_path = get_local_data_file(symbol)
    
    if not file_path.exists():
        return None
    
    try:
        # Try to load as standard CSV first
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        except:
            # Try to load with header skipping (for older format)
            df = pd.read_csv(file_path, skiprows=3)
            
            # Rename columns based on first column
            if df.columns[0] == 'Price':
                # Old format: Price, Close, High, Low, Open, Volume
                df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
            else:
                # Assume standard format already
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # Ensure columns are in the right order
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove NaN rows
            df = df.dropna()
            
            return df
    except Exception as e:
        print(f"Error loading local data for {symbol}: {e}")
        return None


def fetch_data_from_yahoo(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD) or None for start of data
        end_date: End date (YYYY-MM-DD) or None for today
    
    Returns:
        DataFrame with columns [date, open, high, low, close, volume]
    """
    print(f"Fetching {symbol} from Yahoo Finance...")
    
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
    
    return data


def append_to_local_file(symbol: str, new_data: pd.DataFrame):
    """
    Append new data to local CSV file, avoiding duplicates.
    
    Args:
        symbol: Stock symbol
        new_data: DataFrame with new data to append
    """
    file_path = get_local_data_file(symbol)
    
    if not file_path.exists():
        # Create new file
        new_data.to_csv(file_path, index=False)
        print(f"Created new data file: {file_path}")
        return
    
    # Load existing data
    existing_data = pd.read_csv(file_path)
    existing_data['date'] = pd.to_datetime(existing_data['date'])
    new_data = new_data.copy()
    new_data['date'] = pd.to_datetime(new_data['date'])
    
    # Combine and remove duplicates (keep latest)
    combined = pd.concat([existing_data, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date'], keep='last')
    combined = combined.sort_values('date').reset_index(drop=True)
    
    # Save back to file
    combined.to_csv(file_path, index=False)
    print(f"Updated data file: {file_path} with {len(new_data)} new records")


def get_stock_data(
    symbol: str,
    current_date: Optional[str] = None,
    min_history_days: int = 200
) -> tuple[pd.DataFrame, str]:
    """
    Get stock data from local file or Yahoo Finance.
    Automatically fetches missing data and appends to local file.
    
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
        append_to_local_file(symbol, data)
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
        new_data = fetch_data_from_yahoo(symbol, start_date=latest_local_date.strftime('%Y-%m-%d'))
        
        if len(new_data) > 0:
            append_to_local_file(symbol, new_data)
            data = pd.concat([data, new_data], ignore_index=True)
            data = data.drop_duplicates(subset=['date'], keep='last')
            data = data.sort_values('date').reset_index(drop=True)
        else:
            print(f"Warning: Could not fetch data for {current_date_str}, using latest available")
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
    
    Args:
        current_data: Current bar data dict
        history_df: DataFrame with columns [date, open, high, low, close, volume]
        horizon: Prediction horizon (5, 10, or 20)
    
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
    
    # Add market features with SPY data
    try:
        # Get SPY data for the same date range
        spy_data, _ = get_stock_data(
            symbol="SPY",
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
    
    # Make predictions for each horizon
    predictions = []
    for horizon in sorted(request.horizons):
        try:
            # Compute features
            X_pred, feat_names = compute_features_v2(current_dict, history_df.copy(), horizon)
            
            # Make prediction
            prediction = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0]
            
            # Append result
            predictions.append(SingleHorizonPrediction(
                horizon=horizon,
                prediction="UP" if prediction == 1 else "DOWN",
                probability_up=float(proba[1]),
                probability_down=float(proba[0]),
                confidence=float(max(proba))
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