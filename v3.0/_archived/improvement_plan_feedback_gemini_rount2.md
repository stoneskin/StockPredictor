This is a phenomenal translation of high-level quant concepts into a concrete engineering roadmap. You have successfully mapped out the transition from a naive predictive model to a robust, risk-aware trading system. The inclusion of walk-forward validation and volatility-targeted sizing alone will put this project ahead of 90% of retail algorithmic trading attempts.

As your AI quant counterpart, I have reviewed the V3.0 plan. I am genuinely impressed by the architecture, but I spotted a few mathematical and structural traps you need to avoid before writing the code.

Here is my review, including direct answers to your Open Questions.

---

## 🔬 Critical Code & Math Corrections

### 1. The CUSUM Lookahead Bug

In your `detect_regime_change` prototype, you wrote:
`cumsum = (returns - returns.mean()).cumsum()`
**The Trap:** If `df` represents your entire historical dataset, using `returns.mean()` calculates the mean across the *entire* timeline, injecting future knowledge into past regime detection.
**The Fix:** You must use an expanding or rolling mean:
`cumsum = (returns - returns.expanding().mean()).cumsum()`

### 2. Position Sizing Formula Clarification

Your conceptual formula is correct, but translating it to actual shares requires strict definitions. Predicting risk requires isolating the trade risk per share from the total account equity.
For formal implementation, calculate your share size using this equation:

$$Shares = \frac{E \times R}{|P_{entry} - P_{stop}|}$$

Where:

* $E$ is your total Portfolio Equity.
* $R$ is your targeted Risk Percentage (e.g., 0.01 for 1%).
* $P_{entry}$ is your entry price.
* $P_{stop}$ is your volatility-adjusted stop-loss price (e.g., $P_{entry} - (1.5 \times ATR)$).

### 3. Forward Sharpe is a Noisy Target

Predicting the Sharpe Ratio directly is notoriously unstable because you are predicting a ratio of two distributions. A slight miscalculation in the denominator (volatility) causes the target variable to explode.
**The Fix:** Predict **Expected Return** and **Expected Volatility (ATR)** using *separate* models, then calculate the Expected Sharpe dynamically in your `StrategyClassifier`.

---

## 🎯 Answers to Your Open Questions

**1. Transaction Costs:** * **Verdict:** Stick to 0.1% (10 bps) as a blended proxy for now. It covers moderate slippage and minimal commissions. Only build separate spread/commission models if you plan to trade micro-caps or move into high-frequency day trading where the bid-ask spread dominates the cost.

**2. Minimum Liquidity Threshold:**

* **Verdict:** Your suggestion of a **$50M 20-day average** is excellent. This keeps you in the S&P 500 / Russell 1000 territory. At this level, a retail-sized trade will have absolute zero market impact.

**3. Model Architecture:**

* **Verdict:** **Separate models.** Multi-output models (predicting direction, volatility, and strategy all at once) often suffer from "competing gradients" during training. Train one XGBoost model for direction, a separate regression model for volatility, and use a rules-based overlay (like your `StrategyClassifier`) to combine them.

**4. Expected Sharpe vs Expected Return:**

* **Verdict:** Predict **Return** and **Volatility** separately (as noted in the corrections above). Let your logic layer calculate the Sharpe.

**5. Retraining Frequency:**

* **Verdict:** **Monthly.** Weekly introduces too much noise and overfitting to short-term chop. Quarterly is too slow to catch macro regime shifts (e.g., a sudden shift from a bull market to a bear market). Monthly strikes the right balance.

**6. Benchmark Choice:**

* **Verdict:** **SPY.** If you want to prove your model generates actual "Alpha," you must beat the broader market. Comparing against QQQ is only necessary if your stock universe is 100% tech. SPY is the institutional gold standard for benchmarking equity strategies.

---

## 🚀 Priority Recommendation

Your plan is solid, but **Phase 0 is non-negotiable**. Do not build new features until the walk-forward backtesting framework is perfectly airtight. If your testing environment has lookahead bias, all the advanced modeling in Phases 1-5 will be built on sand.

**Would you like me to write a rigorous, bug-free implementation of the `WalkForwardBacktest` class for Phase 0, complete with the rolling StandardScaler logic to ensure absolute zero lookahead bias?**