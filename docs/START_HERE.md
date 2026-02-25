# 🎯 START HERE

Welcome to the Stock Predictor project! This page will guide you to the right documentation.

---

## Choose Your Path

### 👶 Complete Beginner
*"I know nothing about machine learning or this project"*

**Read in this order:**
1. **[Main README.md](../README.md)** - 5 min overview
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - 10 min hands-on
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - 20 min understanding

**Result**: You'll have made your first prediction and understand the basics!

**Time**: ~35 minutes

---

### 🧑‍💻 ML Student / Developer
*"I know Python and want to understand ML systems"*

**Read in this order:**
1. **[Main README.md](../README.md)** - project overview
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - system design (detailed)
3. **[V2_CLASSIFICATION.md](V2_CLASSIFICATION.md)** - in-depth approach
4. **Code**: Review `src/` with focus on:
   - `src/train_v2.py` - training pipeline
   - `src/models_v2/` - model implementations
   - `src/inference_v2.py` - API server

**Result**: Deep understanding of classification, ensemble methods, and technical trading

**Time**: 2-3 hours

---

### 🏢 Production / DevOps
*"I need to deploy this to production"*

**Read in this order:**
1. **[Main README.md](../README.md)** - quick overview
2. **[API_REFERENCE.md](API_REFERENCE.md)** - endpoints & integration
3. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - common issues
4. **[tests/README.md](../tests/README.md)** - testing procedures

**Setup:**
- Configure in `src/config_v2.py`
- Train models: `python src/train_v2.py`
- Start server: `python -m uvicorn src.inference_v2:app --host 0.0.0.0 --port 8000`
- Monitor with: `python tests/test_api.py`

**Result**: Production-ready prediction system

**Time**: 1-2 hours

---

### 🔧 Something's Broken
*"The code doesn't work"*

**Go to:**
1. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - find your error
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - verify setup
3. **Check**: Is server running? Are models trained? Right directory?

**Still stuck?** Check the checklist at bottom of TROUBLESHOOTING.md

---

### 📚 Want Deep Learning Knowledge
*"I want to understand machine learning concepts"*

**Topics in documentation:**
1. **Classification vs Regression** → [V2_CLASSIFICATION.md](V2_CLASSIFICATION.md) (Why Binary?)
2. **Ensemble Methods** → [V2_CLASSIFICATION.md](V2_CLASSIFICATION.md) (Why 5 models?)
3. **Technical Indicators** → [ARCHITECTURE.md](ARCHITECTURE.md) (Features)
4. **Time Series Validation** → [V2_CLASSIFICATION.md](V2_CLASSIFICATION.md) (Data splits)
5. **Model Training** → [V2_CLASSIFICATION.md](V2_CLASSIFICATION.md) (Detailed walkthrough)

**External resources:**
- Andrew Ng's ML Course (Coursera): Fundamentals
- Fast.ai: Practical deep learning
- Scikit-learn docs: Model implementations
- Investopedia: Technical analysis

---

## Quick Links

### Documentation Map

```
📖 Documentation Structure
├─ README.md                      ← Main entry point
├─ docs/
│  ├─ START_HERE.md              ← You are here
│  ├─ GETTING_STARTED.md         ← Quick start (beginner)
│  ├─ ARCHITECTURE.md           ← System design
│  ├─ API_REFERENCE.md          ← API endpoints
│  ├─ V2_CLASSIFICATION.md       ← Deep dive
│  ├─ TROUBLESHOOTING.md        ← Problem solving
│  └─ archive/                  ← Old docs
└─ tests/
   └─ README.md                  ← Testing guide
```

### Common Questions

**Q: How do I make my first prediction?**  
→ [GETTING_STARTED.md](GETTING_STARTED.md) - 5 minute quick start

**Q: How does the system work?**  
→ [ARCHITECTURE.md](ARCHITECTURE.md) - Complete explanation

**Q: What are all the API endpoints?**  
→ [API_REFERENCE.md](API_REFERENCE.md) - Full reference

**Q: Something's broken, help!**  
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

**Q: How do I run tests?**  
→ [tests/README.md](../tests/README.md) - Testing guide

**Q: How do I deploy to production?**  
→ [Main README.md](../README.md) - See "Quick Start"

**Q: Can I use this with other stocks?**  
→ Change `SYMBOL` in `src/config_v2.py`, then `python src/train_v2.py`

---

## What This Project Does

```
INPUT: Stock symbol (e.g., "QQQ")
  ↓
PROCESS: Download 200+ days of data
  ↓
CALCULATE: 25+ technical indicators
  ↓
PREDICT: Will price go UP or DOWN in 5/10/20/30 days?
  ↓
OUTPUT: Probability + Confidence level
```

**Example prediction:**
```
Symbol: QQQ
Date: 2026-02-24

5-day forecast:  UP    (54% confidence)
10-day forecast: DOWN  (52% confidence)
20-day forecast: DOWN  (51% confidence)
30-day forecast: DOWN  (50% confidence)
```

---

## Next Steps

### Right Now
1. Choose your path above ⬆️
2. Start reading the recommended documentation
3. If beginner: Follow [GETTING_STARTED.md](GETTING_STARTED.md)

### In 1 Hour
- Made your first prediction ✓
- Understand basic architecture ✓
- Know how to use the API ✓

### In 1 Week
- Understand ensemble methods ✓
- Reviewed code structure ✓
- Trained models with different settings ✓
- Experimented with features ✓

### In 1 Month
- Deep understanding of ML system ✓
- Deployed to production ✓
- Monitoring real predictions ✓
- Contributing improvements ✓

---

## Pro Tips

💡 **For Beginners**:
- Don't worry about understanding everything at first
- Run code and experiment
- Read comments in source files
- Ask questions (TROUBLESHOOTING.md has answers)

💡 **For ML Folks**:
- Review the ensemble strategy (5 models)
- Check feature engineering (data_preparation_v2.py)
- Analyze performance by horizon
- Experiment with new models in models_v2/

💡 **For Traders**:
- Start with 5-day predictions (most reliable)
- Monitor 10-20 day for medium-term
- Don't rely on 30+ day predictions
- Combine with your own analysis

💡 **For Everyone**:
- Retrain models monthly with new data
- Monitor predictions vs actual results
- Tweak parameters in config_v2.py
- Share improvements back!

---

## Estimated Time to Understand

| Level | Time | Prerequisites |
|-------|------|---------------|
| Beginner | 30 min | None |
| Intermediate | 1-2 hours | Python basics |
| Advanced | 2-4 hours | ML concepts |
| Production | 3-5 hours | DevOps experience |

---

## Success Criteria

**You've succeeded when you can:**
- ✅ Start the API server
- ✅ Make predictions via API
- ✅ Understand how it works
- ✅ Train models with new data
- ✅ Troubleshoot when things break

**If you've done all 5, you're ready!** 🎉

---

## Questions?

1. **Getting started?** → [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Understanding system?** → [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Using API?** → [API_REFERENCE.md](API_REFERENCE.md)
4. **Something broken?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
5. **Want deep knowledge?** → [V2_CLASSIFICATION.md](V2_CLASSIFICATION.md)

---

## Let's Get Started! 🚀

**Choose your path above and start reading. Good luck!**

For most people: → [GETTING_STARTED.md](GETTING_STARTED.md)
