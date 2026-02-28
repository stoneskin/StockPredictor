"""
V1.5 Module - Walk-Forward Validation & Feature Selection Approach

After V1 (regression) failed with R² < 0, this intermediate approach was developed
to test different strategies using time-series-aware validation.

V1.5 Purpose:
- Validate models with walk-forward methodology (prevents look-ahead bias)
- Analyze feature combinations to reduce overfitting
- Test different approaches before committing to V2
- Provide rigorous backtesting framework for feature selection

Components:
- train_walkforward.py: Main walk-forward training and analysis pipeline
  * Performs feature importance analysis
  * Tests multiple feature combinations
  * Runs walk-forward validation across time periods
  * Compares results and recommends best configuration

- walk_forward/: Walk-forward validation modules
  * validation.py: Time-series validation logic
  * feature_selector.py: Feature importance & combination generation
  * trainer.py: Walk-forward training on multiple periods
  * evaluator.py: Cross-fold performance metrics

Key Insight:
V1.5 discovered that feature selection + walk-forward validation was more important
than the prediction approach. This led to V2's classification methodology with
proper time-series validation.

Status: Experimental/Reference - Use for understanding feature analysis
        and walk-forward validation methodology, not for production predictions.

Version History:
- V1: Regression (R² < 0) - Failed
- V1.5: Walk-forward redesign - Intermediate experimentation
- V2: Classification with ensembles - Current production system
"""

__version__ = "1.5"
__all__ = ["train_walkforward", "walk_forward"]
