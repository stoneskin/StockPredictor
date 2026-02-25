#!/usr/bin/env python
"""
Test script for the enhanced Stock Prediction API V2
Demonstrates the new simplified API usage
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path (parent of tests folder)
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_api_locally():
    """
    Test the API functions locally without starting a server
    """
    print("=" * 60)
    print("Stock Prediction API V2 - Local Test")
    print("=" * 60)
    
    # Import after path is set
    from src.v2.inference_v2 import (
        get_stock_data, 
        load_model, 
        predict_direction,
        load_local_data,
        fetch_data_from_yahoo
    )
    
    # Test 1: Load local data
    print("\n[Test 1] Loading local data for QQQ...")
    try:
        qqq_data = load_local_data("QQQ")
        if qqq_data is not None:
            print(f"[OK] Loaded {len(qqq_data)} days of QQQ data")
            print(f"     Date range: {qqq_data['date'].min()} to {qqq_data['date'].max()}")
        else:
            print("[OK] No local data found (expected for first run)")
    except Exception as e:
        print(f"[ERROR] Error loading local data: {e}")
    
    # Test 2: Get stock data with auto-fetch
    print("\n[Test 2] Getting stock data (auto-fetch if needed)...")
    try:
        history_df, current_date_str = get_stock_data(
            symbol="SPY",
            min_history_days=200
        )
        print(f"[OK] Retrieved {len(history_df)} days of SPY data")
        print(f"     Latest date: {current_date_str}")
        print(f"     Available data range: {history_df['date'].min().strftime('%Y-%m-%d')} to {history_df['date'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"[ERROR] Error getting stock data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Load model
    print("\n[Test 3] Loading prediction model...")
    try:
        load_model()
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return
    
    # Test 4: Make a prediction
    print("\n[Test 4] Making a prediction...")
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
        
        result = predict_direction(
            current_data=current_data,
            history_df=history_df,
            horizon=20
        )
        
        print(f"[OK] Prediction successful")
        print(f"     Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
        print(f"     Probability UP: {result['probability_up']:.2%}")
        print(f"     Probability DOWN: {result['probability_down']:.2%}")
    except Exception as e:
        print(f"[ERROR] Error making prediction: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Local tests completed!")
    print("=" * 60)
    print("\nTo start the API server, run:")
    print("  python -m uvicorn uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000")
    print("\nThen test with:")
    print('  curl -X POST http://localhost:8000/predict/simple \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"symbol": "QQQ"}\'')
    print("=" * 60)


def test_api_request_format():
    """
    Print example API requests and responses
    """
    print("\n" + "=" * 60)
    print("Example API Requests & Responses")
    print("=" * 60)
    
    print("\n[Example 1] Simple request with defaults")
    print("Request:")
    print(json.dumps({
        "symbol": "QQQ"
    }, indent=2))
    print("\nExpected Response:")
    print(json.dumps({
        "symbol": "QQQ",
        "date": "2026-02-24",
        "predictions": [
            {
                "horizon": 5,
                "prediction": "UP",
                "probability_up": 0.65,
                "probability_down": 0.35,
                "confidence": 0.65
            },
            {
                "horizon": 10,
                "prediction": "DOWN",
                "probability_up": 0.45,
                "probability_down": 0.55,
                "confidence": 0.55
            }
        ]
    }, indent=2))
    
    print("\n" + "-" * 60)
    print("\n[Example 2] Request with custom date and horizons")
    print("Request:")
    print(json.dumps({
        "symbol": "AAPL",
        "date": "2026-02-20",
        "horizons": [5, 10, 20]
    }, indent=2))
    
    print("\n" + "-" * 60)
    print("\n[Example 3] Curl command")
    print("""
curl -X POST http://localhost:8000/predict/simple \\
  -H "Content-Type: application/json" \\
  -d '{
    "symbol": "QQQ",
    "date": "2026-02-24",
    "horizons": [5, 10, 20, 30]
  }'
    """.strip())
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\nStock Prediction API V2 - Enhanced with Auto Data Loading")
    print("=" * 60)
    
    # Show examples
    test_api_request_format()
    
    # Run local tests
    try:
        test_api_locally()
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
