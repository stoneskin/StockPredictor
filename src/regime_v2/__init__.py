"""
Regime Detection package for Stock Predictor V2
"""

from .detector import RegimeDetector
from .ma_crossover import MACrossoverRegime
from .volatility_regime import VolatilityRegime

__all__ = [
    'RegimeDetector',
    'MACrossoverRegime',
    'VolatilityRegime'
]