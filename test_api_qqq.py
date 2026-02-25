#!/usr/bin/env python3
"""
Test the /predict/simple endpoint with QQQ for date after cached data
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_qqq_prediction():
    print("=" * 70)
    print("Testing /predict/simple endpoint with QQQ")
    print("=" * 70)
    
    # Give server time to start
    time.sleep(2)
    
    # Test 1: Basic QQQ prediction for date after cached data (2026-02-23)
    print("\n[Test 1] Predicting QQQ for 2026-02-23 (after cached 2026-02-20)...")
    
    payload = {
        "symbol": "QQQ",
        "date": "2026-02-23",
        "horizons": [5, 10, 20, 30]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/simple", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Status: 200 OK")
            print(f"  Response keys: {list(result.keys())}")
            print(f"  Symbol: {result.get('symbol', 'N/A')}")
            print(f"  Date: {result.get('date', 'N/A')}")
            print(f"  Predictions count: {len(result.get('predictions', []))}")
            
            if 'predictions' in result:
                for pred in result['predictions'][:2]:  # Show first 2
                    print(f"    - Horizon {pred['horizon']}d: {pred['prediction']} (confidence: {pred['confidence']:.2%})")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Details: {response.json()}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qqq_latestdate():
    """Test QQQ with latest available date"""
    print("\n[Test 2] Predicting QQQ for latest available date...")
    
    payload = {
        "symbol": "QQQ"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/simple", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Status: 200 OK")
            print(f"  Using date: {result.get('date', 'N/A')}")
            print(f"  Horizon predictions: {[p['horizon'] for p in result.get('predictions', [])]}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Details: {response.json()}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_tsla_prediction():
    """Test TSLA prediction to verify other symbols still work"""
    print("\n[Test 3] Predicting TSLA (to ensure other symbols work)...")
    
    payload = {
        "symbol": "TSLA",
        "date": "2026-02-23",
        "horizons": [5, 10]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/simple", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Status: 200 OK")
            print(f"  Symbol: {result['symbol']}")
            print(f"  Date: {result['date']}")
            print(f"  Horizons retrieved: {len(result['predictions'])}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Details: {response.json()}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    results = []
    
    try:
        results.append(test_qqq_prediction())
        results.append(test_qqq_latestdate())
        results.append(test_tsla_prediction())
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "=" * 70)
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("  - QQQ predictions with date after cached data: OK")
        print("  - QQQ predictions with latest date: OK")
        print("  - TSLA predictions: OK")
        print("=" * 70)
    else:
        print(f"✗ Some tests failed: {sum(results)}/{len(results)} passed")
        print("=" * 70)
        exit(1)

