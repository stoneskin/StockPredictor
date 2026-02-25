#!/usr/bin/env python
"""Test TSLA prediction through the API"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

# Remove old TSLA data to test fresh fetch
import os
tsla_file = Path("data/raw/tsla.csv")
if tsla_file.exists():
    print("Removing old TSLA cache to test fresh fetch...")
    os.remove(tsla_file)
    print("Old cache removed.\n")

from src.v2.inference_v2 import (
    get_stock_data,
    load_model,
    predict_direction
)

print("=" * 60)
print("TSLA Data Fetching & Prediction Test")
print("=" * 60)

# Test 1: Get TSLA data
print("\n[Test 1] Fetching TSLA data...")
try:
    history_df, current_date_str = get_stock_data(
        symbol="TSLA",
        min_history_days=200
    )
    print(f"[OK] Retrieved {len(history_df)} days of TSLA data")
    print(f"     Latest date: {current_date_str}")
    print(f"     Date range: {history_df['date'].min().strftime('%Y-%m-%d')} to {history_df['date'].max().strftime('%Y-%m-%d')}")
except Exception as e:
    print(f"[ERROR] Failed to get TSLA data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Load model
print("\n[Test 2] Loading prediction model...")
try:
    load_model()
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# Test 3: Make predictions
print("\n[Test 3] Making predictions for multiple horizons...")
try:
    current_row = history_df.iloc[-1]
    current_data = {
        'date': current_row['date'].strftime('%Y-%m-%d'),
        'open': float(current_row['open']),
        'high': float(current_row['high']),
        'low': float(current_row['low']),
        'close': float(current_row['close']),
        'volume': int(current_row['volume'])
    }
    
    print(f"     Using current data: {current_data['date']}, Close: ${current_data['close']:.2f}")
    print()
    
    results = {}
    for horizon in [5, 10, 20, 30]:
        result = predict_direction(
            current_data=current_data,
            history_df=history_df.copy(),
            horizon=horizon
        )
        results[horizon] = result
        print(f"     {horizon}d: {result['prediction']:4s} ({result['confidence']*100:5.1f}% - {result['probability_up']*100:.1f}% UP, {result['probability_down']*100:.1f}% DOWN)")
    
    print("\n[OK] Predictions successful!")
except Exception as e:
    print(f"[ERROR] Failed to make predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! TSLA API is working correctly.")
print("=" * 60)

# Verify file was cached
tsla_file = Path("data/raw/tsla.csv")
if tsla_file.exists():
    print(f"\n[OK] TSLA data cached to: {tsla_file}")
    print(f"     File size: {tsla_file.stat().st_size:,} bytes")
else:
    print("\n[WARNING] TSLA data was not cached")
