# 📚 V2.5.1 Documentation

Documentation for Stock Predictor V2.5.1.

---

## Contents

| Document | Description |
|----------|-------------|
| **[API_REFERENCE.md](API_REFERENCE.md)** | Complete API endpoint reference with response examples |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, data flow, and component details |
| **[API_GUIDE.md](API_GUIDE.md)** | API usage guide with examples |

---

## Quick Links

- **[API_REFERENCE.md](API_REFERENCE.md)** - Start here for integration
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand the system design
- **[../README.md](../README.md)** - Main project README
- **[../CHANGELOG.md](../CHANGELOG.md)** - Version history

---

## Key Features (V2.5.1)

- **4-Class Classification**: UP, DOWN, UP_DOWN, SIDEWAYS
- **XGBoost**: Best performing model
- **SMOTE**: Handles class imbalance
- **64 Features**: Technical + regime + market features

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
cd v2.5
python src/train_v2_5.py

# Start API
python -m uvicorn src.inference_v2_5:app --reload --port 8000
```

---

## Making Predictions

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "QQQ",
        "horizon": 20,
        "threshold": 0.025
    }
)

print(response.json())
# {
#   "prediction": "SIDEWAYS",
#   "probabilities": {
#     "UP": 0.01, "DOWN": 0.09, "UP_DOWN": 0.04, "SIDEWAYS": 0.87
#   },
#   "confidence": 0.87
# }
```

---

## Response Explanation

The API returns probabilities for each class:

- **UP**: Price will go up > threshold without going down > threshold
- **DOWN**: Price will go down > threshold without going up > threshold  
- **UP_DOWN**: Price will go both up AND down > threshold (volatile)
- **SIDEWAYS**: Price stays within ±threshold (no significant movement)

See [API_REFERENCE.md](API_REFERENCE.md) for detailed response format.

---

## Troubleshooting

See main [troubleshooting docs](../docs/TROUBLESHOOTING.md) for common issues.

### Common Issues

| Issue | Solution |
|-------|----------|
| "Model not found" | Run training: `python src/train_v2_5.py` |
| "Feature shape mismatch" | Ensure API and training versions match |
| Low accuracy | Try different threshold (2.5% recommended) |

---

## Version

**Current Version**: 2.5.1  
**Last Updated**: 2026-02-28
