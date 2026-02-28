# StockPredictor - AGENTS.md

This guide contains information for agentic coding assistants working in this repository.

## Build/Test Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_api.py -v

# Run a specific test function
python -m pytest tests/test_api.py::test_something -v

# Run a specific test class
python -m pytest tests/test_api.py::TestClass -v

# Run tests with coverage
python -m pytest tests/ --cov=src

# Run with verbose output and print statements
python -m pytest tests/ -v -s

# Stop on first failure
python -m pytest tests/ -x

# Run single file directly (no pytest)
python tests/test_api.py
```

### Starting the API Server
```bash
# Development with hot reload
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000

# Production
python -m uvicorn src.v2.inference_v2:app --host 0.0.0.0 --port 8000
```

### Training Models
```bash
# Train V2 models (current version)
python src/v2/train_v2.py

# Train V1 models (legacy)
python src/v1/train.py
```

## Code Style Guidelines

### Imports
- Order: stdlib → third-party → local/relative imports
- Use `sys.path.insert()` to add project root when needed
- Use absolute imports from project root: `from src.v2.config_v2 import ...`
- Use relative imports for same package: `from .base import BaseModel`
- Group related imports with blank lines between groups

```python
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from sklearn.linear_model import LogisticRegression

from src.v2.config_v2 import DEFAULT_HORIZON
from .base import BaseModel
```

### Formatting
- 4-space indentation (no tabs)
- 120-160 character line length (no strict 80-char limit)
- Blank line between functions and classes
- Docstrings for all public functions and classes
- Use explicit `.copy()` when creating DataFrame copies

### Types
- Use type hints extensively for function signatures
- Parameter types: `horizon: int = None`, `symbol: str`
- Return types: `-> pd.DataFrame`, `-> Dict[str, Any]`
- Use `Optional[T]` for nullable types
- Modern Python 3.9+ type unions: `dict[str, int]` preferred over `Dict[str, int]`
- Use `list[T]` instead of `List[T]` for Python 3.9+
- Use tuple[A, B] for fixed-length tuples

```python
def get_stock_data(
    symbol: str,
    current_date: Optional[str] = None,
    min_history_days: int = 200
) -> tuple[pd.DataFrame, str]:
    """Get stock data with type hints."""
    pass
```

### Naming Conventions
- Classes: `PascalCase` (`BaseModel`, `LogisticModel`, `FastAPI`)
- Functions: `snake_case` (`load_model`, `predict_simple`)
- Variables: `snake_case` (`horizon`, `feature_names`)
- Constants: `UPPER_SNAKE_CASE` (`DEFAULT_HORIZON`, `LOG_LEVEL`)
- Private functions: `_snake_case` with leading underscore (`_helper_function`)
- Pydantic models: `PascalCase` ending with `Request` or `Response`

### Error Handling
- Use specific exception types
- Wrap external calls in try/except blocks
- Log errors with appropriate level: `logger.error()`, `logger.warning()`
- For API endpoints, raise `HTTPException` with status codes
- Provide helpful error messages

```python
try:
    data = fetch_data(symbol)
except FileNotFoundError as e:
    logger.error(f"Data not found for {symbol}: {e}")
    raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol}")
```

### Docstrings
- Use triple quotes for docstrings
- First line is a brief summary
- Args section lists parameters with types and descriptions
- Returns section describes return value
- Raises section (optional) lists exceptions

```python
def load_model(horizon: int = None):
    """
    Load V2 model on startup.

    Args:
        horizon: Prediction horizon (5, 10, or 20 days). If None, uses best horizon.

    Returns:
        Loaded model instance
    """
    pass
```

### Project Structure
- `src/` - Source code organized by version
  - `src/v2/` - Active version (USE THIS)
  - `src/v1/` - Legacy regression models (deprecated)
  - `src/v1_5/` - Experimental walk-forward validation
- `tests/` - Test files in root directory
- `docs/` - Documentation organized by version
- `data/` - Data files (raw/, cache/, processed/)
- `models/` - Trained models (checkpoints/, results/v2/)

### Version Management
- V2 is the current active version
- Always work in `src/v2/` for new features
- Prefer `GradientBoosting` model (best performance at 88.7% accuracy)
- Default prediction horizon is 20 days

### Data Files
- CSV format with lowercase column names: `date, open, high, low, close, volume`
- Cache files in `data/cache/` with format: `{symbol}.csv`
- Standard date format: `YYYY-MM-DD` as strings in CSV
- Convert to datetime in pandas: `pd.to_datetime(df['date'])`

### FastAPI Endpoints
- Keep endpoints async
- Use Pydantic models for request/response validation
- Add `response_model` parameter to endpoint decorators
- Use `Query()` for GET parameters
- Return Pydantic instances, not dicts

```python
@app.post("/predict/simple", response_model=SimplePredictionResponse)
async def predict_simple(request: SimplePredictionRequest):
    """Prediction endpoint with Pydantic models."""
    return SimplePredictionResponse(...)
```

### Configuration
- Store config in `src/v2/config_v2.py`
- Use `Path` from pathlib for file paths
- Resolve paths relative to `Path(__file__).parent` in config files
- Use uppercase constants for config values

### Logging
- Use Python's built-in logging module
- Configure logging at module level with: `logging.basicConfig(level=logging.INFO)`
- Use logger: `logger = logging.getLogger(__name__)`
- Log at appropriate levels: DEBUG, INFO, WARNING, ERROR
- Log important actions: data loading, model loading, predictions

### File I/O
- Use `joblib` for model saving/loading
- Use `Path` objects for file operations
- Create directories with `mkdir(parents=True, exist_ok=True)`
- Handle file locks gracefully (especially for sync services like OneDrive)
- Use `try/except` blocks for file operations

```python
import joblib
from pathlib import Path

model_path = MODEL_DIR / "model.pkl"
joblib.dump(model, model_path)
```

### Testing
- Test files in `tests/` directory (not src/)
- Name test files: `test_*.py`
- Use pytest for running tests
- Write tests for API endpoints with `requests` library
- Test data loading, feature computation, and predictions separately
- Mock external dependencies (yfinance, file I/O) when needed

### Working with Pandas
- Always import pandas as `pd` and numpy as `np`
- Chain methods: `df.sort_values('date').reset_index(drop=True)`
- Use `.copy()` when modifying DataFrames to avoid SettingWithCopyWarning
- Handle NaN values: `df.fillna(0)` or `df.dropna()`
- Convert to numeric: `pd.to_numeric(df[col], errors='coerce')`
- Use `.astype(int)` for boolean columns used as targets

### Performance Optimization
- Cache data in `data/cache/` for faster subsequent loads
- Use pandas operations instead of loops where possible
- Sort DataFrames by date: `df.sort_values('date')`
- Use vectorized numpy operations
- Avoid re-computing features in loops

### Git Workflow
- Don't commit to `.gitignore` patterns: `__pycache__/`, `.pyc`, models/results
- Don't commit cached data files unless for testing
- Commit trained models to share with team
- Always update tests when adding new features
- Run tests before commit

### Common Patterns

Loading local data:
```python
from pathlib import Path
import pandas as pd

data_file = DATA_DIR / "symbol.csv"
df = pd.read_csv(data_file, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
```

Adding project root to path:
```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
```

Defining Pydantic models:
```python
from pydantic import BaseModel
from typing import Optional, List

class PredictionRequest(BaseModel):
    symbol: str
    horizon: Optional[int] = 20
    feature_names: Optional[List[str]] = None
```

Computing features with DataFrame chains:
```python
df = (df
    .assign(ma_5=df['close'].rolling(5).mean())
    .assign(ma_10=df['close'].rolling(10).mean())
    .dropna()
    .reset_index(drop=True)
)
```

### Legacy Code Notes
- V1 uses regression (deprecated, don't use)
- V1_5 uses walk-forward validation (experimental only)
- V2 uses classification with ensemble (current, use this)
- Legacy CSV format has 3 header rows to skip
- New format is standard OHLCV with lowercase column names

### Dependencies
- Core: scikit-learn, pandas, numpy
- ML: lightgbm
- Technical Analysis: ta, yfinance
- API: fastapi, uvicorn, pydantic
- ONNX: onnx, skl2onnx, onnxruntime
- Visualization: matplotlib, seaborn

### Platform Notes
- Project works on Windows, Linux, Mac
- Note: Windows with OneDrive may lock cache files - handle gracefully
- Use `time.sleep()` retries for file lock issues on Windows

When making changes:
1. Update tests for new functionality
2. Run tests: `python -m pytest tests/ -v`
3. Ensure backward compatibility with API
4. Update relevant documentation in `docs/v2/`
5. Test with both cached and fresh data
6. Check that models can still be loaded from disk
