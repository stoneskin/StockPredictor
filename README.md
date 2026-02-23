# 📈 QQQ Stock Prediction with LightGBM

基于 Vegas Channel + Hull 和 MACD+RSI 技术指标，使用 LightGBM 预测 QQQ 未来15日收益率。

## 🎯 项目目标

- **标的**: QQQ (Invesco QQQ Trust)
- **预测**: 未来15日收益率（%）
- **数据范围**: 2020-01-01 至今
- **模型**: LightGBM 回归
- **部署**: 本地 ONNX + FastAPI，可选 SageMaker

## 📂 项目结构

```
StockPredictor/
├── data/
│   ├── raw/           # 原始数据 (yfinance 下载)
│   ├── processed/     # 特征工程后的 CSV
│   └── splits/        # 自动生成的训练/验证/测试集
├── src/
│   ├──数据准备.py    # 下载数据、特征工程、标签生成
│   ├── train.py        # 模型训练 + ONNX 导出
│   ├── evaluate.py     # 评估 + 回测模拟
│   ├── inference.py    # FastAPI 推理服务
│   └── config.py       # 配置参数
├── models/
│   ├── checkpoints/    # 训练好的模型 (.pkl)
│   └── onnx/           # ONNX 格式模型
├── train_deploy_sagemaker.py  # SageMaker 训练/部署脚本
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
python src/data_preparation.py
```

这将：
- 从 yfinance 下载 QQQ 和 SPY 数据
- 计算所有技术指标（Vegas Channel, Hull, MACD, RSI, Stochastic 等）
- 创建 15日收益率标签
- 保存训练/验证/测试集到 `data/processed/`

### 3. 训练模型

```bash
python src/train.py
```

输出：
- 模型保存到 `models/checkpoints/latest_model.pkl`
- ONNX 模型到 `models/onnx/model.onnx`（需要 skl2onnx）
- 特征列表到 `models/checkpoints/feature_names.txt`

### 4. 评估模型

```bash
python src/evaluate.py
```

功能：
- 计算 R², RMSE, Correlation
- 按预测分位数分析实际收益（应单调递增）
- 回测模拟（只交易预测值最高的20%样本）
- 生成可视化图表（`plots/`）

### 5. 启动推理 API

```bash
python src/inference.py
```

API 运行在 `http://localhost:8000`

**端点**:
- `GET /` - 基本信息
- `POST /predict` - 预测 15日收益率
  ```json
  {
    "current": {"date": "2025-02-20", "open": 450.0, "high": 455.0, "low": 449.0, "close": 454.0, "volume": 50000000},
    "history": [...previous 60+ bars...]
  }
  ```
- `GET /health` - 健康检查

文档：`http://localhost:8000/docs` (Swagger UI)

### 6. 查看文档
- API 文档: http://localhost:8000/docs
- 项目方案: `完整实施方案.md`
- 策略分析: `策略分析与ML应用建议.md`

---

## 🧠 特征说明

| 类别 | 特征 | 描述 |
|------|------|------|
| Vegas Channel | `veg_fast_band_width`, `veg_price_position`, `veg_signal` | 双层 EMA 通道宽度、价格位置、信号 |
| Hull | `hull_value`, `hull_slope`, `hull_signal` | Hull MA 值、斜率、交易信号 |
| MACD | `macd_hist`, `macd_bullish`, `macd_crossover` | 柱状图、看涨状态、交叉信号 |
| RSI | `rsi`, `rsi_overbought` | RSI 值、超买超卖状态 |
| Stochastic | `stoch_k`, `stoch_d`, `stoch_bullish` | K/D线、金叉死叉 |
| Volume | `volume_ratio` | 成交量相对强弱 |
| Volatility | `atr_pct`, `bb_width` | ATR 百分比、布林带宽度 |
| Time | `day_of_week`, `month`, `quarter` | 星期、月份、季度 |
| Market | `spy_return_15d`, `relative_strength` | SPY 基准收益率、相对强度 |
| Lag Returns | `return_1d`, `return_3d`, `return_5d` | 过去N日收益率 |

---

## 📊 评估指标

### 核心指标
- **R² (R-squared)**: 模型解释方差比例，目标 > 0.05（有预测能力即可）
- **RMSE**: 预测误差的均方根，单位%
- **Correlation**: 预测与实际的相关系数

### 决策指标
- **分位数单调性**: 将预测分为5组（Q1-Q5），每组实际平均收益应单调递增（Q5 > Q4 > ... > Q1）
- **Top 20% vs Bottom 20% 收益差**: 应该是正数且显著
- **交易模拟**: 只交易预测值最高的样本，计算胜率、盈亏比、夏普比率

### 预期结果（底线）
- R² > 0.02（2%的解释方差）
- 分位数单调递增
- Top 20% 相比 Bottom 20% 平均收益差 > 0.5%

---

## 💰 SageMaker 部署（可选）

### 训练并部署到 SageMaker

```bash
python train_deploy_sagemaker.py --mode train
```

这将：
1. 上传训练/验证数据到 S3
2. 提交 SKLearn 训练作业（使用 LightGBM 容器）
3. 自动部署 Endpoint（ml.m5.large）

### 成本估算
- 训练（ml.m5.xlarge, 2h）: ~$0.4
- Endpoint 24/7: ~$90/月（ml.m5.large）
- 建议：非交易时间关闭端点（API Gateway + Lambda 按需启动）

---

## 🔮 本地推理示例

```python
import joblib
import numpy as np

# Load model
model = joblib.load("models/checkpoints/latest_model.pkl")
with open("models/checkpoints/feature_names.txt") as f:
    features = [line.strip() for line in f]

# Prepare feature vector (shape: [1, n_features])
X = np.array([[...]])  # your data
prediction = model.predict(X)[0]
print(f"Predicted 15d return: {prediction:.2f}%")
```

---

## ⚠️ 注意事项

1. **未来信息泄露**: 特征计算必须严格使用当时已知数据
2. **时间序列分割**: 严禁随机 shuffle，必须按时间切分
3. **交易成本**: 预测收益需 > 佣金+滑点才有意义
4. **模型更新**: 市场变化，建议每月重训
5. **回测陷阱**: 避免过拟合，样本外测试才是真实表现

---

## 🔗 相关笔记

- [[策略分析与ML应用建议]] - 策略解读与ML方案设计
- [[完整实施方案]] - 本项目的详细计划
- [[Bedrock 集成指南]] - 可调用 Bedrock 生成分析报告
- [[SageMaker 完整学习与应用指南]] - SageMaker 详细教程

---

## 📝 TODO

- [ ] 实现 ONNX 推理验证
- [ ] 添加特征重要性分析
- [ ] 实现多任务学习（同时预测收益和风险）
- [ ] 集成 TradingView Webhook（实时信号）
- [ ] 添加 CI/CD 自动化重训（GitHub Actions + SageMaker）

---

祝交易顺利！🚀
