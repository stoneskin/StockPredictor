# ✅ Tests Guide

How to run tests and what they verify.

---

## Quick Start

```bash
# Run all tests
python -m pytest tests/ -v

# Or run individual test
python tests/test_api.py
python tests/test_qqq_fix.py
python tests/test_cache_performance.py
```

---

## Test Files

### `test_api.py`
**What it tests**: Main API functionality

**Requires**:
- API server running: `python -m uvicorn src.v2.inference_v2:app --reload`
- Models trained: `python src/v2/train_v2.py`

**Tests**:
- Prediction endpoints work
- Response format is correct
- Different symbols work
- Error handling works

**How to run**:
```bash
# Terminal 1: Start API
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Run test
python tests/test_api.py
```

---

### `test_qqq_fix.py`
**What it tests**: QQQ-specific prediction

**What it verifies**:
- Data fetching works
- Feature computation is correct
- Prediction values are reasonable
- No NaN or infinite values

**How to run**:
```bash
python tests/test_qqq_fix.py
```

---

### `test_cache_performance.py`
**What it tests**: Caching and performance

**What it measures**:
- First request time (with data download)
- Cached request time (should be much faster)
- Cache hit rate
- Memory usage

**Expected results**:
```
First prediction: 2-5 seconds (downloads data)
Cached prediction: <100ms
Performance difference: 20-50x faster
```

**How to run**:
```bash
python tests/test_cache_performance.py
```

---

### `test_performance_comparison.py`
**What it tests**: Model comparison

**Compares**:
- Each model individually
- Ensemble vs individual models
- Time taken per model
- Accuracy metrics

**How to run**:
```bash
python tests/test_performance_comparison.py
```

---

## Running with pytest

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage report
python -m pytest tests/ --cov=src

# Stop on first failure
python -m pytest tests/ -x

# Show print statements
python -m pytest tests/ -s
```

---

## Writing Your Own Tests

### Test Template

```python
import pytest
import requests
from src.v2.inference_v2 import load_model, compute_features_v2

def test_something():
    """Test description"""
    
    # Setup
    model = load_model(horizon=5)
    
    # Execute
    result = model.predict([[1, 2, 3, ...]])
    
    # Assert
    assert result is not None
    assert result.shape == (1,)
    assert 0 <= result[0] <= 1


def test_api_endpoint():
    """Test API endpoint"""
    
    response = requests.post(
        "http://localhost:8000/predict/simple",
        json={"symbol": "QQQ"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0
```

### Common Asserts

```python
# Value checks
assert value == 5
assert value > 0
assert 0 <= value <= 1

# Type checks
assert isinstance(result, dict)
assert isinstance(proba, np.ndarray)

# Collection checks
assert len(predictions) == 4  # 4 horizons
assert "up" in response.lower()

# Exception checks
with pytest.raises(ValueError):
    some_function_that_fails()
```

---

## Debugging Tests

### See All Output

```bash
python -m pytest tests/ -s -v
```

### Stop on First Failure

```bash
python -m pytest tests/ -x
```

### Run Specific Test

```bash
# Run test function
python -m pytest tests/test_api.py::test_predict_simple -v

# Run test class
python -m pytest tests/test_api.py::TestPredictions -v
```

### Add Debug Prints

```python
def test_something():
    result = some_function()
    print(f"DEBUG: result = {result}")  # Will show with -s flag
    assert result == expected
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run tests
        run: |
          python -m pytest tests/ -v
```

---

## Troubleshooting Tests

### "API connection error"

**Problem**: API not running

**Solution**:
```bash
# Terminal 1 (keep running)
python -m uvicorn src.v2.inference_v2:app --reload

# Terminal 2
python tests/test_api.py
```

### "Models not found"

**Problem**: Models not trained

**Solution**:
```bash
python src/v2/train_v2.py
python tests/test_api.py
```

### "No module named 'tests'"

**Problem**: Running from wrong directory

**Solution**:
```bash
cd StockPredictor
python -m pytest tests/ -v
```

---

## Performance Benchmarks

Expected performance on decent hardware:

```
Operation            Time        Status
─────────────────────────────────────
First prediction     2-5s        Normal (data fetch)
Cached prediction    <100ms      ✅
Model training       30-120s     ✅
Single prediction    5-10ms      ✅
```

If significantly slower, check:
1. CPU/disk usage - other processes consuming resources
2. Network - slow internet for data fetch
3. Disk - slow SSD/HDD

---

## Test Coverage

Current test coverage:

```
Module                  Coverage  Status
──────────────────────────────────
inference_v2.py        70%       Good
models_v2/             80%       Excellent
regime_v2/             60%       Fair
train_v2.py            40%       Fair (slow to test)
```

Increase coverage by:
1. Adding unit tests for each function
2. Testing edge cases (NaN, empty data, etc.)
3. Testing error conditions

---

## Continuous Testing

Monitor predictions over time:

```python
# Log predictions daily
import json
from datetime import datetime

def log_prediction(symbol, prediction):
    log_entry = {
        "date": datetime.now().isoformat(),
        "symbol": symbol,
        "prediction": prediction
    }
    
    with open("prediction_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Analyze after 30 days
import pandas as pd
df = pd.read_json("prediction_log.jsonl", lines=True)
df['date'] = pd.to_datetime(df['date'])

accuracy = (df['prediction'] == df['actual']).mean()
print(f"Real-world accuracy: {accuracy:.1%}")
```

---

## Next Steps

1. **Run tests**: `python -m pytest tests/`
2. **Fix failures**: Address any errors
3. **Add more tests**: Expand test coverage
4. **Monitor**: Track real predictions
5. **Improve**: Use test insights to enhance models

---

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start
- [API_REFERENCE.md](API_REFERENCE.md) - API endpoints
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues