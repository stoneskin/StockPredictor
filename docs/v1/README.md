# 📚 V1 Documentation - Regression Approach (Historical)

Previous version documentation for Stock Predictor V1. This approach used regression to predict exact stock returns and **failed** with R² < 0 on test set.

---

## Status

🔴 **V1 Status**: FAILED / DEPRECATED  
❌ **Why Replaced**: Regression on noisy stock returns (exactly predicting return % is too hard)  
📚 **Purpose**: Learning reference - understand what didn't work and why  
→ **Use Instead**: [V2 Classification](../v2/README.md)

---

## What Was V1?

### Approach
- **Type**: Regression (LightGBM)
- **Target**: Predict exact daily return percentage
- **Horizon**: Fixed 15-day prediction
- **Models**: Single LightGBM model
- **Features**: 31 technical indicators

### Results
```
Training:  R² score =  0.41  (overfitting!)
Testing:   R² score = -2.53  (predicting worse than baseline)
```

**Why Failed**: Predicting exact returns on stock prices is extremely difficult - daily returns are too noisy and random, even with good features.

---

## Code (Now in src/v1/)

| File | Purpose |
|------|---------|
| `config.py` | Configuration: symbols, horizons, thresholds |
| `train.py` | Training pipeline for LightGBM |
| `inference.py` | Prediction API server |
| `data_preparation.py` | Feature engineering (31 indicators) |
| `evaluate.py` | Evaluation metrics |
| `convert_model_to_onnx.py` | ONNX model conversion |

### Run V1 (for learning/reference)
```bash
cd src/v1
python train.py           # Train the model
python inference.py        # Start API server
python evaluate.py         # Check performance
```

---

## Key Files in This Folder

### Original Documentation
- **Stock Prediction with SageMaker.md** - AWS SageMaker deployment guide (V1 version)
- **codeReview-mm25.md** - Code review notes on V1 implementation

### How They Relate
- Both documents are historical references showing how V1 was developed and deployed

---

## Lessons Learned (Why V1 Failed)

### Problem 1: Regression is Hard for Stock Returns
- Stock returns are dominated by random noise
- Many factors are unpredictable (news, sentiment, macroeconomics)
- Even perfect features can't overcome inherent randomness
- Exact predictions (regression) are fundamentally harder than direction (classification)

### Problem 2: Overfitting
- Train R² = 0.41 → Test R² = -2.53 (catastrophic drop)
- Model learned training noise instead of signal
- Fixed train/val/test split couldn't capture market regime changes

### Problem 3: Wrong Problem
- Asking "what % will QQQ return?" is too hard
- Better question: "will QQQ go UP or DOWN?" (binary classification)
- Classification targets are more stable and achievable

---

## The Solution: V1.5 & V2

### V1.5 (Experimentation)
After V1 failed, an experimental phase tested:
- Walk-forward validation (prevents look-ahead bias)
- Feature selection (find what actually matters)
- Feature combinations (optimize feature set)
- Key discovery: **Walk-forward + feature selection is critical**

See [src/v1_5/](../../src/v1_5/) for methodology

### V2 (Current Success)
Learned from V1 failure, V2 uses:
- **Classification** instead of regression (UP/DOWN, not exact %)
- **Ensemble of 5 models** for robustness
- **Walk-forward validation** for realistic evaluation
- **25+ carefully selected features**
- **Multiple horizons** (5/10/20/30 days)
- **Result**: 52-54% accuracy (vs 50% random baseline)

See [../v2/README.md](../v2/README.md) for current system

---

## Version Evolution

```
V1 (Regression)         → R² < 0, Failed
  ↓ Led to research phase
V1.5 (Walk-Forward)     → Experimental methodology
  ↓ Led to design insights  
V2 (Classification)     → 52-54% accuracy, Production ✅
```

---

## Resources

### Understanding Regression vs Classification
- Regression: Predict continuous values (e.g., stock return %)
- Classification: Predict categories (e.g., UP or DOWN)
- Why binary classification works better: Predicting direction is more stable than exact values

### Understanding Why V1 Failed
1. **Randomness**: Stock returns contain too much random noise
2. **Overfitting**: Fixed splits don't adapt to market changes
3. **Problem Design**: Exact prediction is harder than direction

### What Changed in V2
1. **Problem**: Changed from regression to classification
2. **Validation**: Changed from fixed split to walk-forward
3. **Ensembles**: Changed from single model to 5 models
4. **Strategy**: Changed from 1 horizon to 4 horizons (5/10/20/30 days)

---

## For Researchers

If you're interested in understanding:
- **Why regression failed** → See code in `src/v1/train.py`
- **What features were tested** → See `src/v1/data_preparation.py`
- **How overfitting happened** → See `src/v1/evaluate.py` results
- **What V1.5 tried** → See `src/v1_5/train_walkforward.py`
- **How V2 succeeded** → See `src/v2/train_v2.py` (classification approach)

---

## Recommendations

### Don't Use V1 For
- ❌ Making actual predictions
- ❌ Trading decisions
- ❌ New research (outdated approach)

### Use V1 For
- ✅ Understanding failed approaches
- ✅ Learning what not to do
- ✅ Studying regression pitfalls
- ✅ Understanding project evolution
- ✅ Code examples (general structure)

### Use V2 Instead For
- ✅ Actual predictions
- ✅ Current research
- ✅ Production deployment
- ✅ Best results

---

## Next Steps

1. **If you want to understand the project history**:
   - Read this page (you're here!)
   - Check `src/v1/` code
   - Review `Stock Prediction with SageMaker.md`

2. **If you want to use the current system**:
   - Go to [V2 Documentation](../v2/README.md)
   - Follow [GETTING_STARTED.md](../GETTING_STARTED.md)

3. **If you want to understand research methodology**:
   - Check [V1.5 approach](../../src/v1_5/)
   - Review [REDESIGN.md](../archive/REDESIGN.md)

---

## Quick Links

- **Main README**: [README.md](../../README.md)
- **V2 Current System**: [docs/v2/README.md](../v2/README.md)
- **V1.5 Research**: [src/v1_5/](../../src/v1_5/)
- **Getting Started V2**: [GETTING_STARTED.md](../GETTING_STARTED.md)
- **Architecture**: [ARCHITECTURE.md](../ARCHITECTURE.md)

---

**Why are you reading this?** If you want to use the system, [go to V2](../v2/README.md). If you're curious about history, keep reading! 📚

