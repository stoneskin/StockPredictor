"""
Strategy Selector Prototype for V3.0
Determines if a stock is suitable for day trading or swing trading
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class StrategySelector:
    """
    Classify market regime into trading strategies.
    """
    
    def __init__(self):
        # Thresholds based on historical analysis (tunable)
        self.day_trade_thresholds = {
            'volatility_ratio': 2.5,      # ATR/price > 2.5%
            'trend_strength': 1.0,        # |(MA50-MA200)/MA200| < 1%
            'bb_squeeze_freq': 0.3,       # BB squeeze occurs >30% of time
            'momentum_reversal': True,    # 5d momentum changes sign frequently
            'volume_spike_ratio': 1.5     # Volume spikes > 50% avg
        }
        
        self.swing_thresholds = {
            'volatility_ratio': (1.0, 2.0),  # ATR/price 1-2%
            'trend_strength': 2.0,           # |(MA50-MA200)/MA200| > 2%
            'ma_alignment': True,            # MA50 and MA200 aligned with trend
            'momentum_consistency': 0.6,     # 20d momentum same direction >60% of time
            'volume_growth': True            # Volume increases on up days
        }
    
    def compute_features(self, df: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
        """
        Compute strategy-relevant features from recent data.
        """
        recent = df.tail(lookback).copy()
        close = recent['close']
        high = recent['high']
        low = recent['low']
        
        features = {}
        
        # 1. Volatility (ATR ratio)
        atr = self._compute_atr(recent)
        features['volatility_ratio'] = (atr / close.iloc[-1]) * 100
        
        # 2. Trend strength
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        features['trend_strength'] = abs(ma50 - ma200) / ma200 * 100
        
        # 3. Bollinger Band characteristics
        bb_ma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_ma + (bb_std * 2)
        bb_lower = bb_ma - (bb_std * 2)
        bb_width = (bb_upper - bb_lower) / bb_ma
        features['bb_squeeze_freq'] = (bb_width < bb_width.median() * 0.8).mean()
        features['avg_bb_width'] = bb_width.mean()
        
        # 4. Momentum characteristics
        momentum_5 = close / close.shift(5) - 1
        momentum_20 = close / close.shift(20) - 1
        
        # Momentum reversal: how often does 5d momentum change sign?
        momentum_sign_changes = (momentum_5 * momentum_5.shift(1) < 0).sum() / len(momentum_5)
        features['momentum_reversal'] = momentum_sign_changes
        
        # Momentum consistency
        features['momentum_consistency'] = (momentum_20 > 0).mean() if momentum_20.iloc[-1] > 0 else (momentum_20 < 0).mean()
        features['current_momentum_20'] = momentum_20.iloc[-1]
        
        # 5. Volume characteristics
        if 'volume' in recent.columns:
            volume_ma20 = recent['volume'].rolling(20).mean()
            volume_spikes = (recent['volume'] > volume_ma20 * 1.5).mean()
            features['volume_spike_ratio'] = volume_spikes
            
            # Volume trend: up days vs down days volume
            up_days = recent['close'] > recent['close'].shift(1)
            down_days = ~up_days
            avg_volume_up = recent.loc[up_days, 'volume'].mean() if up_days.any() else 0
            avg_volume_down = recent.loc[down_days, 'volume'].mean() if down_days.any() else 0
            features['volume_growth'] = avg_volume_up > avg_volume_down * 1.2
        else:
            features['volume_spike_ratio'] = 0.3  # default
            features['volume_growth'] = False
        
        return features
    
    def select_strategy(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Determine best trading strategy based on features.
        
        Returns:
            (strategy_type, confidence)
            strategy_type: "day_trade", "swing", or "neutral"
            confidence: 0-1 score
        """
        scores = {
            'day_trade': 0,
            'swing': 0,
            'neutral': 0
        }
        
        # Day trade scoring
        if features['volatility_ratio'] > self.day_trade_thresholds['volatility_ratio']:
            scores['day_trade'] += 2
        if features['trend_strength'] < self.day_trade_thresholds['trend_strength']:
            scores['day_trade'] += 2
        if features['bb_squeeze_freq'] > self.day_trade_thresholds['bb_squeeze_freq']:
            scores['day_trade'] += 1
        if features.get('momentum_reversal', 0) > 0.4:
            scores['day_trade'] += 1
        if features['volume_spike_ratio'] > 0.3:
            scores['day_trade'] += 1
        
        # Swing trading scoring
        vol = features['volatility_ratio']
        if self.swing_thresholds['volatility_ratio'][0] < vol < self.swing_thresholds['volatility_ratio'][1]:
            scores['swing'] += 2
        if features['trend_strength'] > self.swing_thresholds['trend_strength']:
            scores['swing'] += 2
        if features.get('volume_growth', False):
            scores['swing'] += 1
        
        # Neither strategy (too low volatility or unclear)
        if features['volatility_ratio'] < 0.5:
            scores['neutral'] += 2
        if features['trend_strength'] < 0.2 and features['volatility_ratio'] < 1.0:
            scores['neutral'] += 1
        
        # Find best strategy
        best_strategy = max(scores, key=scores.get)
        total_score = sum(scores.values())
        
        if total_score == 0:
            confidence = 0.5
        else:
            confidence = scores[best_strategy] / total_score
        
        # Minimum score requirement
        if scores[best_strategy] < 2:
            best_strategy = 'neutral'
            confidence = 0.7
        
        return best_strategy, round(confidence, 3)
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = (high - close.shift(1)).abs()
        tr['l-pc'] = (low - close.shift(1)).abs()
        true_range = tr.max(axis=1)
        
        return true_range.rolling(period).mean().iloc[-1]
    
    def get_strategy_parameters(self, strategy: str, threshold: float) -> Dict:
        """
        Get recommended trading parameters for the strategy.
        """
        base = {
            'position_size_limit': 0.1,  # Max 10% of capital per trade
            'max_positions': 5,
        }
        
        if strategy == "day_trade":
            base.update({
                'min_confidence': 0.65,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.0,
                'max_holding_days': 3,
                'max_daily_loss_pct': 2.0,
                'expected_return_pct': threshold * 2.5,
                'position_size_formula': 'confidence * 0.8'
            })
        elif strategy == "swing":
            base.update({
                'min_confidence': 0.55,
                'stop_loss_multiplier': 2.5,
                'take_profit_multiplier': 4.0,
                'max_holding_days': 20,
                'max_drawdown_pct': 8.0,
                'expected_return_pct': threshold * 5,
                'position_size_formula': 'confidence * 0.6'
            })
        else:  # neutral
            base.update({
                'min_confidence': 0.80,  # Very high confidence needed
                'action': 'HOLD',
                'position_size': 0
            })
        
        return base


# Quick test
if __name__ == "__main__":
    # Create mock data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=200, freq='D')
    close = 100 + np.random.randn(200).cumsum() * 0.5
    high = close * (1 + np.random.uniform(0, 0.02, 200))
    low = close * (1 - np.random.uniform(0, 0.02, 200))
    volume = np.random.randint(1000000, 5000000, 200)
    
    df = pd.DataFrame({
        'date': dates,
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    selector = StrategySelector()
    features = selector.compute_features(df)
    strategy, confidence = selector.select_strategy(features)
    params = selector.get_strategy_parameters(strategy, threshold=0.02)
    
    print(f"Features: {features}")
    print(f"\nRecommended Strategy: {strategy.upper()} (confidence: {confidence:.1%})")
    print(f"Parameters: {params}")
