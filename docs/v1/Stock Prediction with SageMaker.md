---
date: 2026-02-22
project: stock-prediction-sagemaker
status: planning
tags: [ml, sagemaker, trading, pine-script, onnx]
---

# 📈 Stock Prediction with SageMaker

> 使用 Pine Script 策略逻辑训练轻量级模型（<15GB），本地部署进行股票走势预测

---

## 🎯 项目目标

- **输入**：历史股票数据（价格、成交量）+ Pine Script 技术指标
- **输出**：次日涨跌信号（1=涨，0=跌）或多分类（涨/跌/震荡）
- **模型**：LightGBM / XGBoost（轻量，<100MB）
- **训练**：Amazon SageMaker（SkLearn Estimator）
- **部署**：本地 ONNX 运行时，通过 FastAPI 暴露 REST 接口
- **集成**：可接入 TradingView Alert Webhook 或独立 CLI

---

## 🗺️ 10 步路线图

### 步骤 1：Pine Script → Python 特征映射
- 将你的 Pine Script 策略中的技术指标改写为 Python 函数
- 例如：RSI、MACD、Bollinger Bands、Volume Profile 等
- 确保计算逻辑一致

### 步骤 2：数据收集
- 使用 `yfinance` 下载历史数据（AAPL、TSLA 等）
- 列：`date, open, high, low, close, volume`
- 保存为 CSV

```python
import yfinance as yf
data = yf.download("AAPL", start="2015-01-01", end="2025-12-31")
data.to_csv("AAPL_history.csv")
```

### 步骤 3：特征工程
- 使用 `ta` 库或自定义函数计算指标
- 定义标签（如：次日收盘价 > 今日 → 1，否则 0）
- 生成特征矩阵 X 和标签 y

```python
import ta
df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
df['macd'] = ta.trend.MACD(close=df['close']).macd()
# ... 更多指标
df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
```

### 步骤 4：本地基线模型
```python
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = lgb.LGBMClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)
joblib.dump(model, "model.joblib")
```

### 步骤 5：SageMaker 训练脚本
- 创建 `train.py`，支持 `--data-dir` 和 `--model-dir` 参数
- 从 SM_CHANNEL_TRAINING 读取数据，输出到 SM_MODEL_DIR

### 步骤 6：SageMaker 训练作业
```python
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# 上传训练数据到 S3
train_path = sagemaker_session.upload_data("train.csv", key_prefix="stock-prediction/train")

# 定义 Estimator
estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="1.0-1",
    py_version="py3"
)

# 提交训练
estimator.fit({"train": train_path})
```

### 步骤 7：模型导出为 ONNX
- 下载 SageMaker 输出的 `model.joblib`
- 转换为 ONNX 格式

```python
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load("model.joblib")
initial_type = [('float_input', FloatTensorType([1, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### 步骤 8：本地推理测试
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
sample = X_test.iloc[0:1].values.astype(np.float32)
pred = session.run(None, {input_name: sample})
print(pred)
```

### 步骤 9：FastAPI 服务封装
```python
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()
session = ort.InferenceSession("model.onnx")

@app.post("/predict")
def predict(features: list):
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    result = session.run(None, {session.get_inputs()[0].name: arr})
    return {"signal": int(result[0][0])}
```

### 步骤 10：集成到工作流
- **Option A**：收盘后自动运行特征计算 → 预测 → 保存结果
- **Option B**：TradingView Alert Webhook → 调用本地 API → 获取信号
- **Option C**：预测结果写入 Google Sheets / 数据库

---

## 💰 成本估算（SageMaker）

| 项目 | 费用 |
|------|------|
| 训练实例 (ml.m5.xlarge) | ~$0.19/小时 × 2h ≈ **$0.38** |
| S3 存储 (10MB) | <$0.001/月 |
| **总计** | **< $1** |

---

## ⚠️ 关键注意事项

1. **过拟合**：股票预测极难，目标应是发现微弱信号，而非高准确率
2. **避免未来信息泄露**：特征计算只能用当时已知数据（严格使用 `shift(1)`）
3. **模型大小**：LightGBM + 少量特征（<50）通常 <10MB，远低于 15GB
4. **本地部署**：ONNX Runtime 支持 Win/macOS/Linux，无依赖问题
5. **延迟要求**：ONNX 预测 <10ms，满足实时需求

---

## 📋 周计划清单（第一周）

- [ ] Day 1: 选 Pine Script 策略，转换为 Python 特征函数
- [ ] Day 2: 下载历史数据，构建特征/标签矩阵
- [ ] Day 3: 本地训练 LightGBM 基线，评估准确率
- [ ] Day 4: 编写 SageMaker `train.py`，上传数据到 S3
- [ ] Day 5: 在 SageMaker 上运行训练，下载模型
- [ ] Day 6: 转换为 ONNX，本地测试推理
- [ ] Day 7: FastAPI 封装服务，测试完整流程

---

## 🔗 相关链接

- [[Bedrock 集成指南]]（对比：Bedrock 用于 NLP，本项目用 SageMaker 做表格预测）
- [[SageMaker 完整学习与应用指南]]
- [[SelfProject/QuoteAiAgent]]（另一个 AI 项目参考）

---

### 🎯 下一步

准备好 Pine Script 代码后，我来帮你：
1. 分析指标逻辑 → Python 转换
2. 设计特征矩阵结构
3. 选择合适的标签定义（分类/回归）

---