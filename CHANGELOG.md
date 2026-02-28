# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.5.2] - 2026-02-28

### Added
- **Backtesting API**: New endpoints `/backtest` (POST) and `/backtest/{symbol}` (GET) for historical prediction analysis
  - Configurable backtest period (default 6 months)
  - Daily predictions with actual outcome comparison
  - PNG charts automatically generated and saved to `v2.5/data/backtest/charts/`
  - Results saved as CSV with datetime in filename
- **Model Information in Responses**: `model_used` field now shows which model (xgboost, gradientboosting, etc.) produced the prediction
- **Multi-class Probability Handling**: API now gracefully handles models with fewer than 4 classes by filling missing classes with 0 probability
- **Backtest Module**: `v2.5/src/backtest_v2_5.py` with comprehensive backtesting capabilities

### Changed
- Version bumped to 2.5.2
- API response now includes probabilities for all 4 classes (UP, DOWN, UP_DOWN, SIDEWAYS), with 0 for missing classes
- Improved error handling for models with incomplete class coverage
- Backtest results include datetime stamps for easy tracking

### Fixed
- Fixed class ordering for proper AUC calculation
- Fixed SMOTE handling for extreme class imbalances

## [2.5.1] - 2026-02-28

### Added
- **XGBoost Support**: XGBoost added as primary model (best performer)
- **CatBoost Support**: Added CatBoost to model ensemble
- **SMOTE Integration**: Automatic class balancing for training
- **Enhanced Regime Features**: ATR ratio, trend strength, RSI/stoch regimes, SPY correlation
- **New Horizons**: 3d and 15d added
- **New Thresholds**: 0.75% and 1.5% added
- **Model Priority System**: Loads best available model (XGBoost > GradientBoosting > RandomForest > Ensemble)

### Changed
- Expanded from 12 to 30 model combinations (6 horizons × 5 thresholds)
- Improved feature set (64 features total)

## [2.5.0] - 2026-02-27

### Added
- **4-Class Classification**: New classification targets (UP, DOWN, UP_DOWN, SIDEWAYS)
- **Multi-Threshold Support**: 1%, 2.5%, 5% price movement thresholds
- **Multi-Horizon Support**: 5, 10, 20, 30-day prediction horizons
- **New Models**: XGBoost and CatBoost model support
- **Enhanced Logging**: Separate log directories for training, prediction, and API
- **Date/Time Logging**: Log filenames include timestamp for performance tracking
- **New API Endpoints**: 
  - `/predict/multi` - Multiple horizons/thresholds in one request
  - `/predict/by-stock/{symbol}` - Get prediction by stock symbol
  - `/predict/by-date/{date}` - Get prediction by date
- **Enhanced Regime Detection**: MA crossover, volatility, momentum, volume regimes

### Changed
- Classification logic: Now checks max daily gain/loss within horizon
  - UP: max gain > threshold, max loss <= threshold
  - DOWN: max loss < threshold, max gain >= threshold
  - UP_DOWN: both max gain > threshold AND max loss < threshold
  - SIDEWAYS: neither exceeds threshold
- Moved v1 and v1.5 to archive/

### Removed
- Legacy v1 code (moved to archive/)
- Legacy v1.5 code (moved to archive/)

## [2.0] - 2025-01-01

### Added
- Binary classification (UP/DOWN)
- 5 ensemble models (Logistic Regression, Random Forest, Gradient Boosting, SVM, Naive Bayes)
- 47 technical indicators
- FastAPI server
- Market regime detection (bull/bear/sideways)

### Features
- Multiple prediction horizons (5, 10, 20 days)
- Weighted ensemble voting
- Automated data fetching from Yahoo Finance

## [1.5] - 2024-06-01

### Added
- Walk-forward validation
- Feature selection

### Notes
- Experimental version

## [1.0] - 2024-01-01

### Added
- Initial release
- Regression-based prediction
- Basic technical indicators

### Notes
- Legacy version, deprecated

---

## Versioning History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2024-01-01 | Initial release with regression |
| 1.5 | 2024-06-01 | Walk-forward validation (experimental) |
| 2.0 | 2025-01-01 | Binary classification with ensemble |
| 2.5.0 | 2026-02-27 | 4-class classification with thresholds |
| 2.5.1 | 2026-02-28 | XGBoost, SMOTE, new horizons/thresholds, regime features |
| 2.5.2 | 2026-02-28 | Backtesting API, model info in responses, multi-class handling |

## Adding New Versions

When adding a new version:

1. Add a new section header: `## [X.Y.Z] - YYYY-MM-DD`
2. Include sections:
   - `Added` - New features
   - `Changed` - Changes to existing functionality
   - `Deprecated` - Soon-to-be removed features
   - `Removed` - Removed features
   - `Fixed` - Bug fixes
   - `Security` - Security-related changes
3. Update the version table at the bottom
4. Update AGENTS.md with new commands if needed
5. Update README.md with new features

## Folder Structure by Version

```
StockPredictor/
├── v2.5/                 # Current version (2.5.2)
│   ├── src/
│   ├── tests/
│   ├── docs/
│   ├── data/
│   └── models/
├── v2/                   # Legacy version (2.0)
├── archive/              # Old versions (v1, v1.5)
├── CHANGELOG.md         # This file
└── README.md            # Main documentation
```
