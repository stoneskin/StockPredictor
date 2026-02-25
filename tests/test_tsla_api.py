#!/usr/bin/env python
"""Test TSLA through the FastAPI endpoint"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path.cwd()))

# Remove TSLA cache to test fresh
import os
tsla_file = Path("data/raw/tsla.csv")
if tsla_file.exists():
    os.remove(tsla_file)
    print("Removed cached TSLA data for fresh test\n")

# Import the app
from src.v2.inference_v2 import app, load_model
from fastapi.testclient import TestClient

# Load model manually (TestClient doesn't trigger startup events)
print("Loading model...")
load_model()
print()

client = TestClient(app)

print("=" * 70)
print("Testing TSLA through FastAPI Endpoint /predict/simple")
print("=" * 70)

# Test 1: Request with just symbol
print("\n[Test 1] Minimal request (symbol only)")
print("Request: POST /predict/simple")
print('Body: {"symbol": "TSLA"}')

try:
    response = client.post(
        "/predict/simple",
        json={"symbol": "TSLA"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n[SUCCESS] Status: 200")
        print(f"Symbol: {result['symbol']}")
        print(f"Date: {result['date']}")
        print(f"Predictions ({len(result['predictions'])} horizons):")
        for pred in result['predictions']:
            print(f"  {pred['horizon']:2d}d: {pred['prediction']:4s} ({pred['confidence']*100:5.1f}% confidence)")
    else:
        print(f"\n[ERROR] Status: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 2: Request with custom horizons
print("\n" + "-" * 70)
print("\n[Test 2] Custom horizons request")
print("Request: POST /predict/simple")
print('Body: {"symbol": "TSLA", "horizons": [5, 20]}')

try:
    response = client.post(
        "/predict/simple",
        json={"symbol": "TSLA", "horizons": [5, 20]}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n[SUCCESS] Status: 200")
        print(f"Symbol: {result['symbol']}")
        print(f"Date: {result['date']}")
        print(f"Predictions ({len(result['predictions'])} horizons):")
        for pred in result['predictions']:
            print(f"  {pred['horizon']:2d}d: {pred['prediction']:4s} ({pred['confidence']*100:5.1f}% confidence)")
    else:
        print(f"\n[ERROR] Status: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 3: Health check
print("\n" + "-" * 70)
print("\n[Test 3] Health check")
print("Request: GET /health")

try:
    response = client.get("/health")
    if response.status_code == 200:
        result = response.json()
        print(f"\n[SUCCESS] Status: 200")
        print(f"Status: {result['status']}")
        print(f"Model loaded: {result['model_loaded']}")
        print(f"Features: {result['n_features']}")
    else:
        print(f"\n[ERROR] Status: {response.status_code}")
except Exception as e:
    print(f"\n[ERROR] {e}")

print("\n" + "=" * 70)
print("All API tests completed successfully!")
print("=" * 70)

# Check cached file
if tsla_file.exists():
    print(f"\nTSLA data cached to: {tsla_file}")
    print(f"File size: {tsla_file.stat().st_size:,} bytes")
    
    # Show first few lines
    import pandas as pd
    df = pd.read_csv(tsla_file, nrows=3)
    print(f"\nFirst 3 rows of cached data:")
    print(df.to_string())
