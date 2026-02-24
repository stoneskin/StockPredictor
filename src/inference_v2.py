"""
Inference API using FastAPI for V2 Classification Models.
Predicts QQQ price direction (up/down) for multiple horizons.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_v2 import MODEL_RESULTS_DIR, DEFAULT_HORIZON

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


def load_model(horizon: int = None):
    """
    Load V2 model on startup.
    
    Args:
        horizon: Prediction horizon (5, 10, or 20 days). If None, uses best horizon.
    """
    global model, feature_names, horizon
    
    if horizon is None:
        horizon = DEFAULT_HORIZON
    
    # Try to load ensemble model first
    ensemble_path = MODEL_RESULTS_DIR / 'ensemble_model.pkl'
    feature_path = MODEL_RESULTS_DIR / 'feature_names.txt'
    best_horizon_path = MODEL_RESULTS_DIR / 'best_horizon.txt'
    
    # Check if model exists
    if not ensemble_path.exists():
        # Try loading best individual model (GradientBoosting is best)
        gb_path = MODEL_RESULTS_DIR / 'gradientboosting_model.pkl'
        if not gb_path.exists():
            raise FileNotFoundError(
                f"Model not found at {ensemble_path}. Please run 'python src/train_v2.py' first."
            )
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
    # Note: For inference, we skip SPY features as they require additional data
    # df = add_market_features(df, market_data)
    df = create_classification_targets(df, horizons=[horizon])
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['date', 'target', 'target_5d', 'target_10d', 'target_20d',
                    'open', 'high', 'low', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Get the last row (most recent features)
    X = df[feature_cols].iloc[-1:].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    return X, feature_cols


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
        "description": "Predict QQQ price direction using classification + ensemble",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information"
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for QQQ price direction.
    
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
    """
    global horizon
    
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