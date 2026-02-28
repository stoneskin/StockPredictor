**Professional Review of Updated V3.0 Improvement Plan (Version 2.0)**

Thank you for sharing the revised plan. I have conducted a thorough review as a quantitative trading specialist with experience in systematic equity strategies, risk management, and production backtesting systems. The updates demonstrate a clear understanding and successful incorporation of the prior feedback, particularly in the areas of walk-forward validation, volatility-targeted sizing, liquidity filtering, alpha scoring, and regime-change detection. The plan has evolved from a classification-focused system into a more institutionally viable trading-advisory framework.

### Overall Assessment
This version is substantially stronger and more executable than the original. The emphasis on risk-adjusted returns, transaction-cost realism, and causal feature engineering addresses the most common failure modes of retail prediction systems. The roadmap is now phased logically, with appropriate success criteria centered on Sharpe ratio, maximum drawdown, and Ulcer Index. I would grade the revised plan **A-** (previously B-). Implementation can proceed with high confidence once the few remaining critical issues below are resolved.

### Key Strengths
- Walk-forward backtest with train-only scaling eliminates lookahead bias in feature transformation — a critical and correctly prioritized fix.
- Volatility-targeted position sizing (inverse-volatility scaling) is correctly formulated and aligns with professional risk-parity practices.
- Liquidity filter combined with SPY-correlation alpha score prevents beta-chasing and illiquid executions.
- CUSUM regime detector as a “kill switch” provides practical drawdown protection.
- Shift to sample weights (instead of a complex custom loss) is pragmatic and directly supported by XGBoost.
- Strategy-specific handling of the UP_DOWN class and tiered success criteria reflect real trading considerations.

### Critical Issues Requiring Immediate Attention
1. **Lookahead Bias in Sample-Weight Calculation**  
   The current `compute_sample_weights` function inspects future price moves (`df_train['close'].iloc[idx+1:idx+horizon+1]`). This reintroduces leakage even within the walk-forward framework.  
   **Recommended fix**: Compute weights using only information available at time t (e.g., rolling 20-day volatility rank, recent momentum strength, or historical move frequency observed up to t). Perform weight calculation strictly inside each walk-forward training window.

2. **Model Proliferation**  
   Maintaining 30 separate models (6 horizons × 5 thresholds) creates severe multiple-testing and maintenance overhead.  
   **Recommendation**: Consolidate to 4–6 core models:  
   - Day-trading: 3-day and 5-day horizons at 1.0 % and 1.5 % thresholds  
   - Swing-trading: 10-day and 20-day horizons at 1.5 % and 2.5 % thresholds  
   This reduction improves statistical robustness without material loss of coverage.

3. **Transaction-Cost Realism**  
   0.1 % round-trip is a reasonable baseline for swing trading but optimistic for day trading (especially leveraged names such as TQQQ).  
   **Better approach**: Model spread + slippage as a function of volatility and dollar volume; apply 0.15–0.25 % for day trades and 0.08–0.12 % for swing trades in sensitivity tables.

4. **Forward-Sharpe Prediction**  
   Direct prediction of forward Sharpe over short horizons is statistically noisy.  
   **Preferred alternative**: Train separate regression heads (or models) for expected return and expected volatility (ATR forecast), then derive implied Sharpe. This yields more stable outputs and integrates naturally with position sizing.

### Responses to the Open Questions in the Plan
1. **Transaction Costs**  
   Agree with 0.1 % round-trip as the primary baseline. Implement sensitivity testing at 0.05 %, 0.15 %, and 0.25 %. Model commissions and effective spread separately for transparency.

2. **Minimum Liquidity Threshold**  
   $50 M 20-day average dollar volume is acceptable as a starting point. I recommend tiered thresholds:  
   - Day trading: ≥ $100 M  
   - Swing trading: ≥ $40 M  
   - Below $30 M → force neutral (with high confidence).

3. **Model Architecture**  
   Separate modular models (direction classifier + return/volatility regressors + strategy classifier) remain the optimal choice for interpretability, debugging, and incremental improvement.

4. **Expected Sharpe vs. Expected Return**  
   Predict expected return and volatility independently, then compute Sharpe. This approach is more robust and directly supports the volatility-targeting logic already present.

5. **Retraining Frequency**  
   Monthly retraining is appropriate. Augment with event-driven triggers based on the CUSUM regime detector to avoid unnecessary retrains during stable periods.

6. **Benchmark Choice**  
   Primary benchmark: buy-and-hold of the specific symbol (e.g., TQQQ itself). Secondary benchmark: SPY or QQQ (risk-adjusted) to quantify true alpha generation.

### Additional Professional Recommendations
- Enforce strict next-bar execution in the backtest (predict on close t, enter at open t+1) to simulate realistic slippage and avoid any implicit look-ahead.
- Implement portfolio-level risk limits: maximum 4–5 % risk per trade and 15–20 % aggregate exposure.
- Add turnover tracking and a simple ensemble (top 2–3 models per strategy) for smoother equity curves.
- For the day-trading component, explicitly acknowledge that daily data limits true intraday execution; consider adding optional intraday features in a future iteration if higher-frequency data becomes available.
- Include Calmar ratio and Sortino ratio in the metrics module to emphasize downside protection.

### Suggested Immediate Next Steps
1. Fix the sample-weight lookahead bias (complete before any retraining).
2. Implement and validate the clean walk-forward backtest (Phase 0).
3. Build the StrategyClassifier with liquidity and alpha filters.
4. Reduce the model set and retrain using corrected weights.
5. Integrate volatility-targeted sizing and run the enhanced simulation.

If you would like, I can immediately provide:
- A corrected, lookahead-free `compute_sample_weights` function,
- An enhanced volatility-position-sizing module (with full ATR integration and portfolio-level checks), or
- A simplified model configuration template (the recommended 4–6 models).

Please indicate your preferred starting point or any specific section you wish to refine further. I am ready to assist with code prototypes or detailed design documents as the project advances.