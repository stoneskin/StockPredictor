# 📈 V2 Documentation - Classification Approach

Complete documentation for Stock Predictor V2, the current classification-based prediction system.

---

## Overview

**V2 Status**: 🚀 Current / Production Ready  
**Approach**: Classification (UP/DOWN prediction)  
**Models**: 5 ensemble models  
**Horizons**: 5, 10, 20, 30 days  
**Accuracy**: 52-54% (vs 50% random baseline)

---

## V2 Files

### Core Documentation (Main repo)
See [../../](../../) for:
- **README.md** - Main project overview
- **GETTING_STARTED.md** - Quick start guide
- **ARCHITECTURE.md** - System design
- **API_REFERENCE.md** - API endpoints
- **V2_CLASSIFICATION.md** - Deep dive into classification approach
- **TROUBLESHOOTING.md** - Problem solving

### V2-Specific Docs (This folder)
- **[README_V2.md](README_V2.md)** - Original V2 README (reference)
- **[REDESIGN_V2.md](REDESIGN_V2.md)** - V2 redesign document
- **[train_deploy_sagemaker_v2.py](train_deploy_sagemaker_v2.py)** - AWS SageMaker deployment

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python -m uvicorn uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000

# Make prediction (in another terminal)
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'
```

**Result**: Predictions for 5, 10, 20, 30 day horizons

---

## V2 Components

### Source Code
```
src/v2/
├── config_v2.py              # Configuration
├── data_preparation_v2.py    # Data pipeline
├── train_v2.py               # Training
├── inference_v2.py           # API server
├── models_v2/                # Base models
│   ├── base.py
│   ├── logistic_model.py
│   ├── random_forest_model.py
│   ├── gradient_boosting_model.py
│   ├── svm_model.py
│   ├── naive_bayes_model.py
│   └── ensemble.py
└── regime_v2/                # Market detection
    ├── detector.py
    └── ma_crossover.py
```

**Note:** Walk-forward validation methodology is in [V1.5](../../src/v1_5/) for research and experimentation.

### Models

**5 Base Models** (each has strengths):
1. **Logistic Regression** - Fast, interpretable linear baseline
2. **Random Forest** - Parallel trees, handles non-linearity
3. **Gradient Boosting** - Sequential trees, powerful predictions
4. **SVM (RBF)** - Kernel methods for complex boundaries
5. **Naive Bayes** - Probabilistic, fast inference

**Ensemble Strategy**:
- Weighted voting based on validation performance
- Default weights: RF (30%), GB (25%), LR (20%), SVM (15%), NB (10%)

---

## Key Features

✅ **Classification**: Binary UP/DOWN prediction  
✅ **Ensemble**: 5 models for robustness  
✅ **Multiple Horizons**: 5/10/20/30 day predictions  
✅ **Technical Indicators**: 25+ calculated features  
✅ **Market Regime**: Detects bull/bear/sideways  
✅ **API Server**: FastAPI with async support  
✅ **Caching**: Fast cached predictions  
✅ **Walk-Forward**: Realistic time-series validation  

---

## Performance

| Horizon | Accuracy | AUC-ROC | Best Model | Verdict |
|---------|----------|---------|-----------|---------|
| 5-day   | 54%      | 0.56    | Random Forest | ✅ Useful |
| 10-day  | 52%      | 0.54    | Ensemble | ⚠️ Marginal |
| 20-day  | 51%      | 0.52    | Ensemble | ⚠️ Weak |
| 30-day  | 50%      | 0.51    | Chance | ❌ Same as guessing |

**Interpretation**:
- 5-day predictions have 4% edge over random
- Longer horizons approach randomness
- Better results with ensemble than individual models

---

## Configuration

Edit `src/v2/config_v2.py`:

```python
# Data
SYMBOL = "QQQ"                  # Stock to predict
TRAIN_YEARS = 5                 # Years of history
MIN_HISTORY = 200               # Min days for prediction

# Models
HORIZONS = [5, 10, 20, 30]      # Prediction timeframes
MODEL_WEIGHTS = {               # Ensemble weights
    "random_forest": 0.30,
    "gradient_boosting": 0.25,
    "logistic": 0.20,
    "svm": 0.15,
    "naive_bayes": 0.10
}

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
```

---

## Training

```bash
# Full training pipeline
python src/v2/train_v2.py

# Output:
# ├─ models/results/v2/horizon_5/  - 5-day models + weights
# ├─ models/results/v2/horizon_10/ - 10-day models + weights
# ├─ models/results/v2/horizon_20/ - 20-day models + weights
# ├─ models/results/v2/horizon_30/ - 30-day models + weights
# └─ horizon_comparison.json        - Performance summary
```

---

## API Usage

**Endpoint**: `/predict/simple`

```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ"}'
```

**Response**:
```json
{
  "symbol": "QQQ",
  "date": "2026-02-24",
  "predictions": [
    {
      "horizon": 5,
      "prediction": "UP",
      "probability_up": 0.54,
      "probability_down": 0.46,
      "confidence": 0.54
    }
  ]
}
```

See [../../API_REFERENCE.md](../../API_REFERENCE.md) for full endpoint documentation.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Or individual tests  
python tests/test_api.py
python tests/test_qqq_fix.py
python tests/test_cache_performance.py
```

---

## AWS SageMaker Deployment

Deploy to AWS:

```bash
python docs/v2/train_deploy_sagemaker_v2.py
```

See [train_deploy_sagemaker_v2.py](train_deploy_sagemaker_v2.py) for configuration.

---

## Differences from V1

| Aspect | V1 | V2 |
|--------|----|----|
| **Target** | Return % (regression) | UP/DOWN (classification) |
| **Horizon** | 15 days | 5/10/20/30 days |
| **Models** | 1 LightGBM | 5 ensemble |
| **Accuracy** | R² < 0 | 52-54% |
| **Market Regimes** | None | Bull/Bear/Sideways |
| **API** | Single endpoint | Multiple endpoints |

---

## Next Steps

### Getting Started ⏱️5 mins
1. Read [../../GETTING_STARTED.md](../../GETTING_STARTED.md)
2. Run quick prediction
3. Check API documentation

### Understanding 📖 1 hour
1. Read [../../ARCHITECTURE.md](../../ARCHITECTURE.md)
2. Review [../../V2_CLASSIFICATION.md](../../V2_CLASSIFICATION.md)
3. Examine source code comments

### Improving 🔧 1+ weeks
1. Modify features in `src/data_preparation_v2.py`
2. Try different model hyperparameters
3. Test with different stocks
4. Analyze real prediction accuracy

### Deploying 🚀 1-2 weeks
1. Configure in `src/config_v2.py`
2. Train final models
3. Deploy API to production
4. Set up monitoring

---

## Support

- **Quick Start**: [../../GETTING_STARTED.md](../../GETTING_STARTED.md)
- **Architecture**: [../../ARCHITECTURE.md](../../ARCHITECTURE.md)
- **API Help**: [../../API_REFERENCE.md](../../API_REFERENCE.md)
- **Issues**: [../../TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
- **Deep Dive**: [../../V2_CLASSIFICATION.md](../../V2_CLASSIFICATION.md)

---

## See Also

- **Main README**: [../../README.md](../../README.md)
- **V1 Docs**: [../v1/README.md](../v1/README.md)
- **Archived Docs**: [../archive/](../archive/)

---

**Ready to use V2? Start with [../../GETTING_STARTED.md](../../GETTING_STARTED.md)** 🚀