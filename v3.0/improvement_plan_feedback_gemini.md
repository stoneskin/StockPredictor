This is a sophisticated roadmap that correctly identifies the most common "trap" in financial ML: **High accuracy on a model that doesn't actually make money.** Moving from a direction-based classifier to a strategy-aware advisor is a significant leap toward a professional-grade system.

As a Quant Trading professional, here is my critique and advice to ensure V3.0 survives the reality of the markets.

---

## ⚖️ The "Quant" Reality Check

### 1. The Survival Metric: Risk-Adjusted Returns

Your plan focuses heavily on "Profitability," but in the quant world, **Return without Risk context is meaningless.** * **The Advice:** Elevate **Max Drawdown (MDD)** and **Ulcer Index** to primary success criteria. A strategy that returns 25% but drops 40% mid-month will be abandoned by any rational investor (or hit a margin call).

* **Implementation:** Your `Position Sizing` logic should be driven by **Volatility Targeting**. If a stock’s ATR (Average True Range) doubles, your position size should halve to keep the "risk contribution" constant.

### 2. The "UP_DOWN" Class Paradox

Your current V2.5.2 uses an `UP_DOWN` class. In live trading, this is often a "Stop-Loss Hunter" scenario where you get stopped out on the "Down" before the "Up" happens.

* **The Advice:** In V3.0, treat `UP_DOWN` as a **high-risk/no-trade signal** for Swing Trading, but a **prime opportunity** for Mean Reversion (Day Trading).
* **Diagram Request:** To better understand how these different regimes interact with your strategy selection:

### 3. Execution Costs: The Silent Killer

You mentioned 0.1% slippage in the risks. For Day Trading (Phase 3), 0.1% is often the *entire* profit margin.

* **The Advice:** You must model **Market Impact**. If you are trading a low-volume stock, your own "Buy" signal will push the price up.
* **Action:** Add a `liquidity_filter` to your `StrategyClassifier`. If the 20-day average dollar volume is too low, the strategy should default to "NEUTRAL" regardless of the prediction.

---

## 🛠 Technical Improvements for the Plan

### Addressing the "Lookahead Bias"

You correctly identified this as a critical problem. However, fixing it is harder than just "time-correcting."

* **The Pitfall:** Even if your features are time-correct, using a `StandardScaler` on the *entire* dataset before splitting into Train/Test introduces lookahead bias (the model "knows" the future mean/std).
* **The Fix:** Use a **Rolling Window Pipeline**. Scale your data using only the statistics of the *training* window, then apply that scaling to the *test* window.

### Profit-Driven Loss Function

Your `ProfitDrivenLoss` is a great concept, but **Differentiability** is an issue for XGBoost.

* **Refinement:** Instead of a custom loss function (which can make the model unstable), use **Sample Weights**.
* **Logic:** Assign a higher `sample_weight` to historical dates that had massive moves (high Alpha) and a lower weight to "noise" days. This forces the model to prioritize getting the "big wins" right.

---

## 📋 Suggested Additions to your Roadmap

### Phase 1.5: The "Alpha Decay" Study

Before Phase 2, calculate how long a signal remains valid.

* **The Question:** If the model says "BUY" at 9:30 AM, is that signal still profitable at 10:30 AM?
* **Task:** Add a **Latency Sensitivity** test to your backtest.

### Phase 6: Regime Change Detection (The "Kill Switch")

Markets move from "Trending" to "Mean Reverting" (Day Trade vs Swing).

* **New Feature:** Implement a **CUSUM Filter** or a **Structural Break Test**. If the market regime changes abruptly (e.g., a black swan event), the model should automatically move to a "Cash/Neutral" position until the `StrategyClassifier` gains new confidence.

---

## 📊 Summary of Strategic Advice

| Current Plan Component | Quant Professional Advice |
| --- | --- |
| **Position Sizing** | Use **Inverse Volatility** (Kelly is often too aggressive for stock ML). |
| **Targeting** | Predict **Forward Sharpe Ratio** instead of just % Return. |
| **Strategy Selection** | Add **Correlation to SPY**. If a stock is 0.98 correlated, you aren't a stock picker, you're a Beta-chaser. |
| **Backtesting** | Use **Walk-Forward Fold Validation** (Anchored vs. Non-anchored). |

---

## 📞 Suggested Next Step

**Would you like me to draft a Python implementation for the "Volatility-Targeted Position Sizing" logic to replace the simple 0-1 fraction in your plan?** This would align your "Actionable Signal Generation" with institutional risk management standards.