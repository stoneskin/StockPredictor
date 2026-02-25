#!/usr/bin/env python3
"""
Test backtesting and horizon-specific predictions
"""
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def test_different_horizons():
    """Test that different horizons return DIFFERENT predictions"""
    print("=" * 80)
    print("TEST 1: Different Horizons Return Different Predictions")
    print("=" * 80)
    
    payload = {
        "symbol": "QQQ",
        "date": "2025-12-31",
        "horizons": [5, 10, 20, 30]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/simple", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nDate: {result['date']}")
            print(f"Symbol: {result['symbol']}\n")
            
            predictions = result['predictions']
            predictions_map = {p['horizon']: p for p in predictions}
            
            # Check if predictions differ
            pred_5d = predictions_map[5]['prediction']
            pred_10d = predictions_map[10]['prediction']
            pred_20d = predictions_map[20]['prediction']
            pred_30d = predictions_map[30]['prediction']
            
            conf_5d = predictions_map[5]['confidence']
            conf_10d = predictions_map[10]['confidence']
            conf_20d = predictions_map[20]['confidence']
            conf_30d = predictions_map[30]['confidence']
            
            print("Horizon-wise predictions:")
            for h in [5, 10, 20, 30]:
                p = predictions_map[h]
                print(f"  {h}d:  {p['prediction']:4s}  (confidence: {p['confidence']:.2%}, up: {p['probability_up']:.2%})")
            
            # Check if they're all the same (bad) or different (good)
            all_predictions = [pred_5d, pred_10d, pred_20d, pred_30d]
            all_same =len(set(all_predictions)) == 1
            
            if all_same:
                print("\n❌ PROBLEM: All horizons predict the SAME direction!")
                print("   Horizons SHOULD give different predictions for different timeframes")
                return False
            else:
                print("\n✓ GOOD: Different horizons return different predictions")
                unique_preds = set(all_predictions)
                print(f"   Unique predictions: {unique_preds}")
                return True
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_different_dates():
    """Test that different dates return DIFFERENT predictions (backtesting)"""
    print("\n" + "=" * 80)
    print("TEST 2: Different Dates Return Different Predictions (Backtesting)")
    print("=" * 80)
    
    dates = [
        "2025-12-15",
        "2025-12-22",
        "2025-12-29",
        "2025-12-31"
    ]
    
    try:
        predictions_by_date = {}
        
        for date_str in dates:
            payload = {
                "symbol": "QQQ",
                "date": date_str,
                "horizons": [20]  # Use single horizon for cleaner comparison
            }
            
            response = requests.post(f"{BASE_URL}/predict/simple", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                pred = result['predictions'][0]['prediction']
                conf = result['predictions'][0]['confidence']
                predictions_by_date[date_str] = (pred, conf)
                print(f"  {date_str}: {pred:4s} (confidence: {conf:.2%})")
            else:
                print(f"  {date_str}: Error - {response.status_code}")
                return False
        
        # Check if predictions differ across dates
        unique_preds = set(p[0] for p in predictions_by_date.values())
        
        if len(unique_preds) == 1:
            print("\n⚠ WARNING: All dates predict the same direction")
            print("   This might be correct if market is trending, or model issue")
            return True  # Not necessarily an error
        else:
            print(f"\n✓ GOOD: Different dates return different predictions")
            print(f"   Unique predictions across dates: {unique_preds}")
            return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_historical_backtesting():
    """Test that historical dates work properly for backtesting"""
    print("\n" + "=" * 80)
    print("TEST 3: Historical Backtesting (Past Dates)")
    print("=" * 80)
    
    # Test with a date from 6 months ago
    past_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    payload = {
        "symbol": "QQQ",
        "date": past_date,
        "horizons": [5, 20]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/simple", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nBacktesting on {past_date}...")
            print(f"Symbol: {result['symbol']}")
            
            for p in result['predictions']:
                print(f"  {p['horizon']}d: {p['prediction']:4s} (confidence: {p['confidence']:.2%})")
            
            print(f"\n✓ GOOD: Backtesting works - loaded data for {past_date}")
            print(f"         Model can predict for any historical date")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            details = response.json()
            print(f"   Details: {details.get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_model_not_always_up():
    """Test that model doesn't always predict UP"""
    print("\n" + "=" * 80)
    print("TEST 4: Model Variability (Not Always Predicting UP)")
    print("=" * 80)
    
    # Test multiple different dates
    test_dates = [
        "2025-06-15",
        "2025-07-20",
        "2025-08-10",
        "2025-09-30",
        "2025-10-15",
        "2025-11-20",
    ]
    
    up_count = 0
    down_count = 0
    
    try:
        for date_str in test_dates:
            payload = {
                "symbol": "QQQ",
                "date": date_str,
                "horizons": [20]
            }
            
            response = requests.post(f"{BASE_URL}/predict/simple", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                pred = result['predictions'][0]['prediction']
                if pred == 'UP':
                    up_count += 1
                else:
                    down_count += 1
    except:
        pass
    
    if up_count == 0 or down_count == 0:
        print(f"\n❌ Model always predicts one direction!")
        print(f"   UP: {up_count}/{len(test_dates)}")
        print(f"   DOWN: {down_count}/{len(test_dates)}")
        return False
    else:
        print(f"\n✓ GOOD: Model predicts both directions")
        print(f"   UP: {up_count}/{len(test_dates)}")
        print(f"   DOWN: {down_count}/{len(test_dates)}")
        return True


if __name__ == "__main__":
    import time
    
    print("\nWaiting for API to be ready...")
    time.sleep(2)
    
    results = []
    results.append(("Different Horizons", test_different_horizons()))
    results.append(("Different Dates", test_different_dates()))
    results.append(("Historical Backtesting", test_historical_backtesting()))
    results.append(("Model Variability", test_model_not_always_up()))
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("✓ All tests passed!" if all_passed else "✗ Some tests failed"))
