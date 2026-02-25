#!/usr/bin/env python
"""Quick test for TSLA data fetching"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.inference_v2 import fetch_data_from_yahoo

try:
    print("Fetching TSLA data...")
    data = fetch_data_from_yahoo('TSLA')
    print(f"[SUCCESS] Fetched {len(data)} days of TSLA data")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    if len(data) >= 200:
        print("[OK] Sufficient data for predictions (>= 200 days)")
    else:
        print(f"[WARNING] Only {len(data)} days available (need 200)")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
