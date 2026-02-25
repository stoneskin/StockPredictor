# 🛠️ Troubleshooting Guide

Solutions for common problems and how to debug them.

---

## Installation Issues

### "pip: command not found"

**Problem**: pip is not installed or not in PATH

**Solutions**:
```bash
# Use Python directly
python -m pip install -r requirements.txt

# Or use python3
python3 -m pip install -r requirements.txt
```

### "No module named 'pandas'" (or other import errors)

**Problem**: Dependencies not installed

**Solution 1 - Reinstall all**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Solution 2 - Install specific package**:
```bash
pip install pandas numpy scikit-learn fastapi uvicorn
```

### "ModuleNotFoundError: No module named 'ta'"

**Problem**: Technical analysis library not installed

**Solution**:
```bash
pip install ta
```

---

## API Server Issues

### "Connection refused" when starting API

**Problem**: Port 8000 already in use

**Solution**:
```bash
# Use a different port
python -m uvicorn src.v2.inference_v2:app --host 0.0.0.0 --port 8001

# Or kill existing process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>
```

### "Address already in use"

**Problem**: Another process using port 8000

**Solution**: Use different port or kill the process (see above)

### "No module named 'src'"

**Problem**: Running from wrong directory

**Solution**:
```bash
# Navigate to project root
cd StockPredictor

# Then run
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

### "Module not found: config_v2"

**Problem**: Path issues when loading config

**Solution**:
```bash
# Make sure you're in project root
pwd  # or cd on Windows
# Should show: .../StockPredictor

# Then run API
python -m uvicorn src.v2.inference_v2:app --reload
```

### API starts but "/health" returns "models_loaded: false"

**Problem**: Models not trained yet

**Solution**:
```bash
python src/v2/train_v2.py

# Wait for training to complete, then restart API
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

---

## Prediction Errors

### "Could not fetch data for symbol QQQ"

**Causes**:
1. No internet connection
2. Yahoo Finance is down
3. Invalid stock symbol

**Solutions**:
```bash
# Test internet
ping google.com

# Try another symbol
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY"}'

# Check if Yahoo Finance is working
python -c "import yfinance; print(yfinance.download('QQQ', start='2025-01-01', end='2025-02-24'))"
```

### "Insufficient data available"

**Problem**: Not enough historical data for the symbol

**Solution**:
```bash
# Use more history
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "TSLA",
    "min_history": 150
  }'
```

### Prediction returns all zeros/same value

**Problem**: Bad data or model issue

**Solutions**:
1. Check if data was fetched correctly
2. Retrain models: `python src/v2/train_v2.py`
3. Check model files exist in `models/results/v2/`

---

## Training Issues

### "ModuleNotFoundError" during training

**Problem**: Import paths not set correctly

**Solution**:
```bash
# Make sure you're in project root
cd StockPredictor

# Then run training
python src/v2/train_v2.py
```

### Training is very slow

**Possible causes & solutions**:

```bash
# Check if data file exists
ls data/processed/

# If not, prepare data first
python src/v2/data_preparation_v2.py

# Then train (should be faster with cached data)
python src/v2/train_v2.py
```

### Training fails with memory error

**Problem**: Too much data or too little RAM

**Solutions**:
1. Use less data - edit `config_v2.py`:
```python
TRAIN_YEARS = 2  # Instead of 5
```

2. Reduce features - edit `config_v2.py`:
```python
FEATURE_WINDOWS = [5, 20, 50]  # Instead of [5, 10, 20, 50, 200]
```

3. Then retrain:
```bash
python src/v2/train_v2.py
```

### "Models not found after training"

**Check**: Did training actually complete?

```bash
# Look for results
ls models/results/v2/

# If empty, training failed. Check output for errors
```

---

## Data Issues

### "Cannot download data" or "Connection error"

**Causes**: Network issues, Yahoo Finance down

**Solutions**:
```bash
# Test internet connection
ping -c 4 google.com  # Linux/Mac
ping google.com       # Windows

# Try downloading manually
python -c "
import yfinance
data = yfinance.download('QQQ', start='2020-01-01')
print(data.head())
"

# If that works, clear cache and try again
rm data/raw/*.csv
rm data/processed/*.csv

# Then run training
python src/v2/train_v2.py
```

### Historical data is incomplete or has gaps

**Solution**: 
```bash
# Clean and redownload
rm data/raw/qqq.csv data/raw/spy.csv
python src/v2/data_preparation_v2.py
```

---

## Testing Issues

### "Test scripts won't run"

**Problem**: Import or path issues

**Solution**:
```bash
# Make sure you're in project root
cd StockPredictor

# Run test
python tests/test_api.py

# Or use pytest
python -m pytest tests/ -v
```

### "pytest: command not found"

**Solution**:
```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Performance Issues

### Predictions are slow (>1 second response)

**Causes**:

1. **First prediction** - downloads data (normal, takes 2-5 seconds)
2. **Network latency** - API communication
3. **Server overload** - other processes

**Solutions**:
```bash
# Check if it's the first time
python -c "
from src.v2.inference_v2 import fetch_data_from_yahoo
data = fetch_data_from_yahoo('QQQ', 200)
print(f'Data shape: {data.shape}')
"

# If data is cached, subsequent requests should be <100ms
```

### API sometimes returns 500 error

**Causes**:
1. Data fetch failed temporarily
2. Insufficient memory
3. Model loading failed

**Solutions**:
```bash
# Restart API
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000

# Check logs for details about what failed
# Keep the terminal open to see error messages
```

---

## Code Issues

### "SyntaxError" when running code

**Problem**: Python syntax error in code

**Solution**:
```bash
# Check Python version (needs 3.8+)
python --version

# If older, use python3
python3 src/v2/train_v2.py
```

### Import errors in custom scripts

**Ensure imports are correct**:
```python
# ✅ Correct (from project root)
from src.v2.inference_v2 import load_model

# ❌ Wrong
from inference_v2 import load_model
from v2.inference_v2 import load_model
```

---

## Windows-Specific Issues

### "python: command not found" (PowerShell)

**Solution**:
```powershell
# Use python directly or python3
python -m uvicorn src.v2.inference_v2:app --reload

# Or use full path
C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python312\python.exe src/v2/train_v2.py
```

### Paths with backslashes cause errors

**Problem**: String escaping issues

**Solution**:
```python
from pathlib import Path

# Use Path (works on Windows, Linux, Mac)
config_file = Path("src/v2/config_v2.py")

# Instead of:
# config_file = "src\v2\config_v2.py"  # ❌ Problems on some systems
```

### "Long path" errors

**Problem**: Windows has path length limit

**Solution**:
```bash
# Use shorter directory names
# Move project closer to root
# C:\StockPredictor  (good)
# C:\Users\long_username\folder\StockPredictor  (risky)
```

---

## Mac-Specific Issues

### "command not found: python"

**Solution**:
```bash
# Use python3 instead
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/v2/train_v2.py
```

### Port permission denied

**Problem**: Can't use port <1024 without sudo

**Solution**:
```bash
# Use port >=1024
python3 -m uvicorn src.v2.inference_v2:app --port 8000
```

---

## Linux-Specific Issues

### "pip: permission denied"

**Solution**:
```bash
# Use venv (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use --user flag
pip install --user -r requirements.txt
```

### Library not found errors

**Solution**:
```bash
# Install missing system libraries
# For Ubuntu/Debian:
sudo apt-get install python3-dev python3-pip

# Then reinstall packages
pip install --upgrade -r requirements.txt
```

---

## Debugging Tips

### Enable verbose logging

```python
# In your script
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run code
python -u your_script.py 2>&1 | tee output.log
```

### Check configuration

```bash
# View current config
python -c "from src.v2.config_v2 import *; print(f'Symbol: {SYMBOL}'); print(f'Horizons: {HORIZONS}')"
```

### Test individual components

```python
# Test data loading
from src.v2.inference_v2 import fetch_data_from_yahoo
data = fetch_data_from_yahoo('QQQ', 200)
print(f"Data shape: {data.shape}")
print(data.head())

# Test feature computation
from src.v2.inference_v2 import compute_features_v2
features = compute_features_v2(data)
print(f"Features shape: {features.shape}")
print(features.head())

# Test model loading
from src.v2.inference_v2 import load_model
model = load_model(horizon=5)
print(f"Model loaded: {model is not None}")
```

---

## Getting Help

If you're stuck:

1. **Check the error message carefully** - it usually tells you what's wrong
2. **Try the solutions in this guide** - most common issues covered
3. **Read [ARCHITECTURE.md](ARCHITECTURE.md)** - understand the system
4. **Check [GETTING_STARTED.md](GETTING_STARTED.md)** - review basic usage
5. **Look at code comments** in `src/v2/` - well-documented
6. **Search for your error** online - likely someone has encountered it

---

## Still Stuck?

**Checklist**:
- ✅ Python 3.8+ installed? (`python --version`)
- ✅ In project root directory? (`pwd` or `cd StockPredictor`)
- ✅ Dependencies installed? (`pip list | grep scikit-learn`)
- ✅ Models trained? (`ls models/results/v2/horizon_5/`)
- ✅ API server running? (check terminal output)
- ✅ Correct port? (default 8000, check with `netstat -an`)

If all pass and you're still stuck, the error message should give direction on what to fix next!