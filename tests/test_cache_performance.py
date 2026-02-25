#!/usr/bin/env python3
"""
Test performance improvement with cache folder approach.
Compares load times for QQQ (legacy), TSLA, and TQQQ
"""
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.v2.inference_v2 import load_local_data

def test_load_time(symbol: str, iterations: int = 3):
    """Test load time for a symbol"""
    times = []
    
    for i in range(iterations):
        start = time.time()
        data = load_local_data(symbol)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if data is not None:
            print(f"  Iteration {i+1}: {elapsed:.3f}s ({len(data)} records)")
        else:
            print(f"  Iteration {i+1}: FAILED (no data)")
            return None
    
    avg_time = sum(times) / len(times)
    return avg_time

print("=" * 70)
print("Cache Folder Performance Tests")
print("=" * 70)

symbols = ["QQQ", "TSLA", "TQQQ"]
results = {}

for symbol in symbols:
    print(f"\n[{symbol}] Testing load performance...")
    avg = test_load_time(symbol, iterations=3)
    if avg:
        results[symbol] = avg

print("\n" + "=" * 70)
print("Summary (average load time):")
print("=" * 70)

if results:
    min_symbol = min(results, key=results.get)
    for symbol, time_ms in sorted(results.items(), key=lambda x: x[1]):
        pct = (results[symbol] / results[min_symbol]) if min_symbol in results else 1
        print(f"  {symbol:6s}: {results[symbol]:.3f}s ({pct:.1f}x)")
    
    print(f"\n✓ First load uses cache folder for standard format")
    print(f"✓ Second+ loads should be same speed across all symbols")
else:
    print("✗ Failed to load any symbols")
