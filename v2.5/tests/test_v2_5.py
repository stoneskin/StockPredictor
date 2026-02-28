"""
Test script for Stock Predictor V2.5
Tests the 4-class classification API
"""

import sys
import json
from pathlib import Path

# Add v2.5/src to path
V25_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(V25_ROOT))
sys.path.insert(0, str(V25_ROOT / "src"))


def test_4class_classification():
    """Test that 4-class classification works correctly."""
    print("=" * 60)
    print("Testing V2.5 4-Class Classification")
    print("=" * 60)
    
    from src.data_preparation_v2_5 import create_4class_targets
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=50)
    close = 100 + np.cumsum(np.random.randn(50) * 2)
    
    df = pd.DataFrame({'close': close}, index=dates)
    
    df = create_4class_targets(df, horizon=5, threshold=0.01)
    
    target_col = 'target_5d_1.0pct'
    
    if target_col in df.columns:
        unique_classes = df[target_col].dropna().unique()
        print(f"[OK] Target column created: {target_col}")
        print(f"     Unique classes: {sorted(unique_classes)}")
        
        class_names = {0: 'SIDEWAYS', 1: 'UP', 2: 'DOWN', 3: 'UP_DOWN'}
        for cls in sorted(unique_classes):
            count = (df[target_col] == cls).sum()
            print(f"     {class_names.get(cls, cls)}: {count} samples")
    else:
        print(f"[ERROR] Target column not found: {target_col}")
        return False
    
    print("\n[OK] 4-class classification test passed!")
    return True


def test_config():
    """Test config values."""
    print("\n[Test] Configuration")
    
    from src.config_v2_5 import HORIZONS, THRESHOLDS, CLASS_LABELS
    
    print(f"     Horizons: {HORIZONS}")
    print(f"     Thresholds: {THRESHOLDS}")
    print(f"     Classes: {CLASS_LABELS}")
    
    assert len(HORIZONS) == 4
    assert len(THRESHOLDS) == 3
    assert len(CLASS_LABELS) == 4
    
    print("[OK] Config test passed!")
    return True


def test_models():
    """Test model imports."""
    print("\n[Test] Model Imports")
    
    from src.models_v2 import (
        LogisticModel, RandomForestModel, GradientBoostingModel,
        XGBoostModel, CatBoostModel, SVMModel, NaiveBayesModel
    )
    
    print("[OK] All models imported successfully!")
    return True


def test_api_request_format():
    """Print example API requests."""
    print("\n" + "=" * 60)
    print("Example API Requests")
    print("=" * 60)
    
    print("\n[Example 1] Single prediction")
    print(json.dumps({
        "symbol": "QQQ",
        "horizon": 20,
        "threshold": 0.01
    }, indent=2))
    
    print("\n[Example 2] Multi-horizon prediction")
    print(json.dumps({
        "symbol": "QQQ",
        "horizons": [5, 10, 20],
        "thresholds": [0.01, 0.025]
    }, indent=2))
    
    print("\n[Example 3] GET request")
    print("/predict/by-stock/QQQ?horizon=20&threshold=0.01")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\nStock Predictor V2.5 Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    if test_config():
        tests_passed += 1
    
    if test_models():
        tests_passed += 1
    
    if test_4class_classification():
        tests_passed += 1
    
    test_api_request_format()
    
    print("\n" + "=" * 60)
    print(f"Tests: {tests_passed}/{total_tests} passed")
    print("=" * 60)
