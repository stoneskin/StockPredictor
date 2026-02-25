# 📚 V1 Documentation - Legacy Regression Approach

Historical documentation for the original Stock Predictor V1, which used regression to predict 15-day return percentage.

---

## Overview

**V1 Status**: ✅ Reference / Learning only  
**V2 Status**: 🚀 Current version (recommended)

V1 attempted to predict exact QQQ 15-day returns using:
- Single LightGBM regression model
- Technical indicators (Vegas Channel, Hull, MACD, RSI, etc.)
- 583 samples of data
- Result: R² ≈ -0.04 (negative, worse than predicting mean)

---

## Why V1 Was Replaced

**Problems with regression approach**:
- Stock returns are extremely noisy
- R² scores were negative (model worse than baseline)
- Difficult to predict exact percentage returns
- High variance across validation folds
- Limited data (583 samples) led to overfitting

**V2 Solution**: Classification (UP/DOWN prediction)
- More robust binary target
- Better generalization with limited data
- 52-54% accuracy (vs 50% random baseline)
- More actionable probability outputs

---

## V1 Files

### Code
- **[../../src/v1/](../../src/v1/)** - Original V1 source code
  - `config.py` - V1 configuration
  - `data_preparation.py` - Data pipeline
  - `train.py` - Training script
  - `inference.py` - Prediction API
  - `evaluate.py` - Evaluation metrics
  - `convert_model_to_onnx.py` - ONNX conversion

### Documentation
- **[API_GUIDE.md](API_GUIDE.md)** - V1 API documentation
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Implementation details
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - How to upgrade to V2
- **[Stock Prediction with SageMaker.md](Stock%20Prediction%20with%20SageMaker.md)** - AWS deployment

### Scripts
- **[train_deploy_sagemaker.py](train_deploy_sagemaker.py)** - SageMaker deployment

---

## Running V1 (For Learning)

If you want to understand the V1 approach:

```python
# First, move back to src root (files are in src/v1 now)
from src.v1.config import *
from src.v1.data_preparation import prepare_data
from src.v1.train import train_model

# Load and train
X_train, y_train, X_test, y_test = prepare_data()
model = train_model(X_train, y_train)
predictions = model.predict(X_test)
```

**Training**:
```bash
# Note: Scripts are in src/v1/ now, not root
python src/v1/train.py
```

**Inference**:
```bash
# Note: API is in src/v1/ now
python -m uvicorn src.v1.inference:app --port 9000
```

---

## Lessons Learned from V1

✅ **What Worked**:
- Technical indicator feature engineering
- Data downloading and caching
- API structure with FastAPI
- Model evaluation framework

❌ **What Didn't Work**:
- Regression on noisy financial data
- Single model approach
- 15-day prediction horizon (too long)
- Insufficient data handling

🎯 **Applied to V2**:
- Switched to classification
- Ensemble of 5 models
- Multiple shorter horizons (5/10/20/30 days)
- Market regime detection
- Better validation strategy

---

## For Historical Context

These documents explain the V1 approach and why it was changed:

1. **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - How V1 became V2
2. **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - V1 implementation details
3. **[API_GUIDE.md](API_GUIDE.md)** - V1 API specifics

---

## Migration to V2

**If you're using V1 and want to upgrade to V2:**

1. Read: [GETTING_STARTED.md](GETTING_STARTED.md)
2. Compare approaches: [ARCHITECTURE.md](ARCHITECTURE.md)
3. Run V2: `python -m uvicorn src.v2.inference_v2:app --reload`

**Key differences**:
- Endpoint path: `/predict` (same)
- Response: Probability of UP/DOWN (instead of return %)
- Prediction horizons: 5/10/20/30 days (instead of 15 days)
- Model: Ensemble of 5 (instead of single LightGBM)

---

## See Also

- **Current Project**: [../README.md](../README.md)
- **V2 Docs**: [v2/README.md](v2/README.md)
- **Getting Started**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Full Archive**: [../archive/](../archive/)

---

## Archive Note

V1 code is kept for:
- ✅ Learning regression vs classification
- ✅ Understanding why ensemble is better
- ✅ Reference on what didn't work
- ✅ Historical context

**For new projects, use V2!** 🚀