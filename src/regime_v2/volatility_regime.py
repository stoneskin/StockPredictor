"""
Volatility Regime Detector for Stock Predictor V2
Detects high/low volatility market regimes
"""

import pandas as pd
import numpy as np
from .detector import RegimeDetector


class VolatilityRegime(RegimeDetector):
    """
    Detects market regime based on volatility levels.
    
    - High Volatility: Rolling volatility above threshold
    - Low Volatility: Rolling volatility below threshold
    - Normal Volatility: Otherwise
    """
    
    def __init__(self, window: int = 20, high_threshold: float = 0.02, 
                 low_threshold: float = 0.01):
        """
        Initialize Volatility regime detector.
        
        Args:
            window: Rolling window for volatility calculation
            high_threshold: Threshold for high volatility (e.g., 0.02 = 2%)
            low_threshold: Threshold for low volatility (e.g., 0.01 = 1%)
        """
        super().__init__("Volatility")
        self.window = window
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        # Regime codes
        self.HIGH = 1
        self.LOW = -1
        self.NORMAL = 0
    
    def detect(self, prices: pd.Series) -> int:
        """
        Detect current regime from price series.
        
        Args:
            prices: Price series
            
        Returns:
            1 (High Volatility), -1 (Low Volatility), or 0 (Normal)
        """
        if len(prices) < self.window + 1:
            return self.NORMAL
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if len(returns) < self.window:
            return self.NORMAL
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.window).std().iloc[-1]
        
        if pd.isna(volatility):
            return self.NORMAL
        
        # Classify regime
        if volatility > self.high_threshold:
            return self.HIGH
        elif volatility < self.low_threshold:
            return self.LOW
        else:
            return self.NORMAL
    
    def get_regime_name(self, regime_code: int) -> str:
        """Get human-readable name for regime code."""
        names = {
            self.HIGH: "High Volatility",
            self.LOW: "Low Volatility",
            self.NORMAL: "Normal Volatility"
        }
        return names.get(regime_code, "Unknown")
    
    def get_volatility_features(self, prices: pd.Series) -> dict:
        """
        Get volatility-related features for a price series.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary of volatility features
        """
        if len(prices) < self.window + 1:
            return {
                'volatility_regime': 0,
                'volatility': 0,
                'volatility_percentile': 0
            }
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Current volatility
        volatility = returns.rolling(window=self.window).std().iloc[-1]
        
        # Historical volatility (for percentile calculation)
        rolling_vol = returns.rolling(window=self.window).std().dropna()
        
        if len(rolling_vol) > 0 and not pd.isna(volatility):
            # Calculate percentile
            percentile = (rolling_vol < volatility).sum() / len(rolling_vol) * 100
        else:
            percentile = 50
        
        return {
            'volatility_regime': self.detect(prices),
            'volatility': volatility if not pd.isna(volatility) else 0,
            'volatility_percentile': percentile
        }