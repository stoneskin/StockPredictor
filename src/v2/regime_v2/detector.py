"""
Base Regime Detector for Stock Predictor V2
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, List


class RegimeDetector(ABC):
    """
    Abstract base class for market regime detection.
    Detects different market states (bull, bear, neutral, high/low volatility).
    """
    
    def __init__(self, name: str = "RegimeDetector"):
        self.name = name
        self.current_regime = None
        self.regime_history = []
    
    @abstractmethod
    def detect(self, prices: pd.Series) -> int:
        """
        Detect current regime from price series.
        
        Args:
            prices: Price series
            
        Returns:
            Regime code (e.g., 1 for bull, -1 for bear, 0 for neutral)
        """
        pass
    
    @abstractmethod
    def get_regime_name(self, regime_code: int) -> str:
        """
        Get human-readable name for regime code.
        
        Args:
            regime_code: Numeric regime code
            
        Returns:
            String name of regime
        """
        pass
    
    def detect_series(self, prices: pd.Series) -> pd.Series:
        """
        Detect regime for entire price series.
        
        Args:
            prices: Price series
            
        Returns:
            Series of regime codes
        """
        regimes = pd.Series(index=prices.index, dtype=int)
        
        for i in range(len(prices)):
            # Use past data only (no look-ahead bias)
            past_prices = prices.iloc[:i+1]
            if len(past_prices) > 20:  # Need some history
                regimes.iloc[i] = self.detect(past_prices)
            else:
                regimes.iloc[i] = 0  # Neutral during warmup
        
        self.regime_history = regimes.tolist()
        self.current_regime = regimes.iloc[-1] if len(regimes) > 0 else None
        
        return regimes
    
    def get_regime_distribution(self) -> dict:
        """Get distribution of regimes in history."""
        if not self.regime_history:
            return {}
        
        unique, counts = np.unique(self.regime_history, return_counts=True)
        return {self.get_regime_name(r): c / len(self.regime_history) 
                for r, c in zip(unique, counts)}
    
    def __repr__(self) -> str:
        current = self.get_regime_name(self.current_regime) if self.current_regime is not None else 'N/A'
        return f"{self.name}(current={current})"