"""
Market Regime Detection V2.5
"""

from .detector import (
    detect_all_regimes,
    detect_ma_cross_regime,
    detect_volatility_regime,
    detect_momentum_regime,
    detect_volume_regime,
    get_current_regime
)

__all__ = [
    'detect_all_regimes',
    'detect_ma_cross_regime',
    'detect_volatility_regime',
    'detect_momentum_regime',
    'detect_volume_regime',
    'get_current_regime'
]
