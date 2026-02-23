"""
Inference API using FastAPI for local deployment.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CHECKPOINTS_DIR, MODEL_PARAMS, API_HOST, API_PORT

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    description="Predict QQQ 15-day forward return using technical indicators",
    version="1.0.0"
)

# Global model and feature names
model = None
feature_names = []

def load_model():
    """Load model on startup."""
    global model, feature_names
    model_path = os.path.join(MODEL_CHECKPOINTS_DIR, "latest_model.pkl")
    feature_path = os.path.join(MODEL_CHECKPOINTS_DIR, "feature_names.txt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")

    model = joblib.load(model_path)
    with open(feature_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    print(f"✅ Model loaded. Features ({len(feature_names)}): {feature_names[:5]}...")

# Define request/response models
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

    # Previous data needed for indicators (at least 60 days)
    history: List[MarketData]

class PredictionResponse(BaseModel):
    """Prediction response."""
    expected_15d_return_pct: float
    risk_score: Optional[float]  # max drawdown prediction (if available)
    signal_strength: float  # normalized 0-1
    confidence: Optional[float]  # model confidence (if available)
    features_used: List[str]

from src.data_preparation import (
    calculate_vegas_channels,
    calculate_hull_ma,
    calculate_macd_rsi_stoch,
    calculate_volume_volatility,
    add_time_features
)
import ta

def compute_features(current_data: dict, history_df: pd.DataFrame):
    """
    Compute all features from current bar and history.
    history_df: DataFrame with columns [Open, High, Low, Close, Volume]
    Must include current bar at the end.
    """
    # Append current to history
    history_df = pd.concat([history_df, pd.DataFrame([current_data])], ignore_index=True)

    # Ensure columns are named correctly
    history_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Calculate all features (will produce multiple columns)
    df = history_df.copy()
    df = calculate_vegas_channels(df)
    df = calculate_hull_ma(df)
    df = calculate_macd_rsi_stoch(df)
    df = calculate_volume_volatility(df)
    df = add_time_features(df)

    # Add lag returns
    for lag in [1, 3, 5, 10, 20]:
        df[f'return_{lag}d'] = df['Close'].pct_change(lag) * 100

    # Get the last row (current bar) feature values
    features = df.iloc[-1]

    # Build feature vector in the order the model expects
    feature_vector = []
    for name in feature_names:
        if name in features.index:
            val = features[name]
            feature_vector.append(val if not np.isnan(val) and not np.isinf(val) else 0.0)
        else:
            # Feature not found, default to 0
            feature_vector.append(0.0)

    return np.array(feature_vector).reshape(1, -1)

@app.on_event("startup")
def startup_event():
    """Load model when API starts."""
    try:
        load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Optionally exit if model is required

@app.get("/")
def root():
    return {
        "message": "Stock Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "features": len(feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict 15-day return for QQQ."""
    global model, feature_names

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert history to DataFrame
    history_data = []
    for h in request.history:
        history_data.append([h.open, h.high, h.low, h.close, h.volume])
    history_df = pd.DataFrame(history_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    # Compute features
    try:
        X = compute_features(request.current.dict(), history_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature computation error: {str(e)}")

    # Predict
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Normalize signal strength to 0-1 (based on empirical range)
    # Typical 15d returns range from -20% to +20%
    signal_strength = max(0, min(1, (pred + 20) / 40))

    return PredictionResponse(
        expected_15d_return_pct=round(float(pred), 2),
        risk_score=None,  # Could add if model predicts risk
        signal_strength=round(float(signal_strength), 2),
        confidence=None,  # Could add probability calibration
        features_used=feature_names[:10]  # Only first 10 in response (shorten)
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host=API_HOST, port=API_PORT)