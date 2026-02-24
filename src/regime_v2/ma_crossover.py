"""
MA Crossover Regime Detector for Stock Predictor V2
Detects bull/bear markets based on moving average crossovers
"""

import pandas as pd
import numpy as np
from .detector import RegimeDetector


class MACrossoverRegime(RegimeDetector):
    """
    Detects market regime using moving average crossover.
    
    - Bull: Price above both short and long MA, short MA above long MA
    - Bear: Price below both short and long MA, short MA below long MA
    - Neutral: Otherwise
    """
    
    def __init__(self, short_period: int = 50, long_period: int = 200):
        """
        Initialize MA Crossover regime detector.
        
        Args:
            short_period: Short MA period (e.g., 50 days)
            long_period: Long MA period (e.g., 200 days)
        """
        super().__init__("MACrossover")
        self.short_period = short_period
        self.long_period = long_period
        
        # Regime codes
        self.BULL = 1
        self.BEAR = -1
        self.NEUTRAL = 0
    
    def detect(self, prices: pd.Series) -> int:
        """
        Detect current regime from price series.
        
        Args:
            prices: Price series
            
        Returns:
            1 (Bull), -1 (Bear), or 0 (Neutral)
        """
        if len(prices) < self.long_period:
            return self.NEUTRAL
        
        # Calculate moving averages
        short_ma = prices.rolling(window=self.short_period).mean()
        long_ma = prices.rolling(window=self.long_period).mean()
        
        current_price = prices.iloc[-1]
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        
        # Check for NaN
        if pd.isna(current_short_ma) or pd.isna(current_long_ma):
            return self.NEUTRAL
        
        # Bull: Price above both MAs and short MA above long MA
        if (current_price > current_short_ma and 
            current_price > current_long_ma and 
            current_short_ma > current_long_ma):
            return self.BULL
        
        # Bear: Price below both MAs and short MA below long MA
        elif (current_price < current_short_ma and 
              current_price < current_long_ma and 
              current_short_ma < current_long_ma):
            return self.BEAR
        
        # Neutral: Otherwise
        return self.NEUTRAL
    
    def get_regime_name(self, regime_code: int) -> str:
        """Get human-readable name for regime code."""
        names = {
            self.BULL: "Bull",
            self.BEAR: "Bear",
            self.NEUTRAL: "Neutral"
        }
        return names.get(regime_code, "Unknown")
    
    def get_regime_features(self, prices: pd.Series) -> dict:
        """
        Get regime-related features for a price series.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary of regime features
        """
        if len(prices) < self.long_period:
            return {
                'ma_regime': 0,
                'trend_strength': 0,
                'distance_to_ma50': 0,
                'distance_to_ma200': 0
            }
        
        short_ma = prices.rolling(window=self.short_period).mean()
        long_ma = prices.rolling(window=self.long_period).mean()
        
        current_price = prices.iloc[-1]
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        
        # Trend strength: (short_ma - long_ma) / long_ma
        trend_strength = ((current_short_ma - current_long_ma) / current_long_ma * 100 
                         if current_long_ma != 0 else 0)
        
        # Distance to MAs
        distance_ma50 = ((current_price - current_short_ma) / current_short_ma * 100 
                        if current_short_ma != 0 else 0)
        distance_ma200 = ((current_price - current_long_ma) / current_long_ma * 100 
                         if current_long_ma != 0 else 0)
        
        return {
            'ma_regime': self.detect(prices),
            'trend_strength': trend_strength,
            'distance_to_ma50': distance_ma50,
            'distance_to_ma200': distance_ma200
        }