#!/usr/bin/env python
"""Test QQQ with the fixed data append logic"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

# Test the data handling
from src.v2.inference_v2 import (
    get_stock_data,
    load_model
)

print("=" * 70)
print("Testing QQQ Data Append Fix")
print("=" * 70)

# Test: Get QQQ data and request a date after latest
print("\n[Test] Fetching QQQ data for 2026-02-23 (after cached 2026-02-20)...")
try:
    history_df, current_date_str = get_stock_data(
        symbol="QQQ",
        current_date="2026-02-23",
        min_history_days=200
    )
    print(f"[SUCCESS] Retrieved {len(history_df)} days of QQQ data")
    print(f"          Requested date: 2026-02-23")
    print(f"          Using date: {current_date_str}")
    print(f"          Latest in cache: {history_df['date'].max().strftime('%Y-%m-%d')}")
    
    # Check the date column format
    print(f"\n[Info] First row date type: {type(history_df['date'].iloc[0])}")
    print(f"       Last row date: {history_df['date'].iloc[-1]}")
    print(f"       Data range: {history_df['date'].min().strftime('%Y-%m-%d')} to {history_df['date'].max().strftime('%Y-%m-%d')}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test: Load model and make prediction
print("\n[Test] Loading model and making prediction...")
try:
    load_model()
    print("[OK] Model loaded")
    
    # Make sure we can use compute_features_v2 on the loaded data
    from src.v2.inference_v2 import compute_features_v2
    
    current_row = history_df.iloc[-1]
    current_data = {
        'date': current_row['date'].strftime('%Y-%m-%d') if hasattr(current_row['date'], 'strftime') else str(current_row['date']),
        'open': float(current_row['open']),
        'high': float(current_row['high']),
        'low': float(current_row['low']),
        'close': float(current_row['close']),
        'volume': int(current_row['volume'])
    }
    
    X_pred, feat_names = compute_features_v2(current_data, history_df.copy(), 20)
    print(f"[SUCCESS] Features computed: {len(feat_names)} features")
    print(f"          Shape: {X_pred.shape}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("All tests passed! QQQ data append is working correctly.")
print("=" * 70)
