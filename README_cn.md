# Stock Predictor V2 - Classification Approach

---
**🌐 语言**: [English Version](README.md)

基于分类方法的股票预测系统，使用集成学习 + 市场状态识别来预测 QQQ 未来价格走势（上涨/下跌）。

## 项目目标

- **标的**: QQQ (Invesco QQQ Trust)
- **预测**: 未来N日价格走势（分类：上涨=1，下跌=0）
- **数据范围**: 2020-01-01 至今
- **方法**: 分类 + 集成学习 + 市场状态识别
- **预测周期**: 5日、10日、20日（可配置）

## 核心改进 (vs V1)

| 特性 | V1 (回归) | V2 (分类) |
|------|----------|----------|
| 目标 | 预测收益率 | 预测涨跌 |
| 模型 | 单模型 | 5模型集成 |
| 市场状态 | 无 | 牛/熊/震荡识别 |
| 预测周期 | 15日 | 5/10/20日 |
| 评估指标 | R², RMSE | AUC-ROC, F1, 准确率 |

## 项目结构

```
StockPredictor/
├── data/
│   ├── raw/           # 原始数据 (QQQ, SPY)
│   └── processed/     # 特征工程后的数据
├── src/
│   ├── config_v2.py           # V2配置参数
│   ├── data_preparation_v2.py # 数据准备（分类标签）
│   ├── train_v2.py            # V2训练脚本
│   ├── models_v2/             # 模型定义
│   │   ├── base.py           # 基类
│   │   ├── logistic_model.py # 逻辑回归
│   │   ├── random_forest_model.py
│   │   ├── gradient_boosting_model.py
│   │   ├── svm_model.py
│   │   ├── naive_bayes_model.py
│   │   └── ensemble.py       # 集成模型
│   └── regime_v2/            # 市场状态识别
│       ├── detector.py       # 基类
│       ├── ma_crossover.py   # MA交叉状态
│       └── volatility_regime.py # 波动率状态
├── models/
│   ├── checkpoints/    # V1模型
│   ├── onnx/           # V1 ONNX模型
│   └── results/v2/     # V2结果
├── REDESIGN_V2.md      # 详细设计文档
└── README2.md          # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行训练

```bash
python src/train_v2.py
```

这将：
- 加载 QQQ 和 SPY 数据
- 计算技术指标和分类标签
- 训练5个基础模型 + 集成模型
- 评估不同预测周期（5/10/20日）
- 保存结果到 `models/results/v2/`

### 3. 查看结果

结果保存在 `models/results/v2/horizon_comparison.json`

## 模型说明

### 5个基础模型

1. **Logistic Regression** - 线性基线，解释性强
2. **Random Forest** - 随机森林，处理非线性
3. **Gradient Boosting** - 梯度提升，预测能力强
4. **SVM** - 支持向量机，RBF核
5. **Naive Bayes** - 朴素贝叶斯，快速

### 集成策略

- **加权投票**: 根据验证集性能分配权重
- **默认权重**: RF 30%, GB 25%, LR 20%, SVM 15%, NB 10%

### 市场状态识别

1. **MA交叉状态** (50日/200日MA)
   - 牛市场: 价格 > MA50 > MA200
   - 熊市场: 价格 < MA50 < MA200
   - 震荡市场: 其他

2. **波动率状态**
   - 高波动: 日波动率 > 2%
   - 低波动: 日波动率 < 1%
   - 正常: 其他

## 性能结果

### 20日预测周期（最佳）

| 模型 | 准确率 | ROC-AUC | F1 |
|------|--------|---------|-----|
| Gradient Boosting | 88.7% | 0.954 | 0.918 |
| Random Forest | 88.5% | 0.945 | 0.911 |
| Ensemble | 64.8% | 0.578 | 0.775 |

### 预测周期对比

| 周期 | 准确率 | ROC-AUC | F1 | PR-AUC |
|------|--------|---------|-----|--------|
| 5日 | 58.2% | 0.536 | 0.726 | 0.627 |
| 10日 | 61.7% | 0.541 | 0.750 | 0.652 |
| 20日 | 64.8% | 0.578 | 0.775 | 0.716 |

### 重要特征

1. `trend_strength` - 趋势强度 (MA50-MA200)/MA200
2. `distance_ma200` - 距离200日MA百分比
3. `volatility` - 波动率
4. `correlation_spy_20d` - 与SPY 20日相关性

## 如何使用模型进行预测

### 基本用法

```python
import sys
sys.path.insert(0, 'src')

import numpy as np
from data_preparation_v2 import prepare_data
from models_v2 import (
    GradientBoostingModel, RandomForestModel,
    GradientBoostingModel, LogisticModel, SVMModel,
    NaiveBayesModel, EnsembleModel
)
from config_v2 import MODEL_PARAMS, ENSEMBLE_WEIGHTS

# 1. 准备数据
X, y, feature_names, df = prepare_data(horizon=20)

# 2. 创建并训练模型（使用最佳配置）
models = [
    GradientBoostingModel(MODEL_PARAMS['gradient_boosting']),
    RandomForestModel(MODEL_PARAMS['random_forest']),
]

# 3. 训练
for model in models:
    model.fit(X, y, feature_names)

# 4. 预测
ensemble = EnsembleModel(models, ENSEMBLE_WEIGHTS)
prediction = ensemble.predict(X[:10])
probability = ensemble.predict_proba(X[:10])

print(f"预测: {prediction}")
print(f"上涨概率: {probability[:, 1]}")
```

### 使用训练好的模型

训练完成后，模型保存在 `models/results/v2/` 目录：

```bash
# 查看保存的模型
ls -la models/results/v2/
```

**模型文件说明：**

| 文件 | 说明 |
|------|------|
| `ensemble_model.pkl` | 集成模型（5个模型加权投票） |
| `gradientboosting_model.pkl` | 梯度提升模型（最佳单模型，88.7%准确率） |
| `randomforest_model.pkl` | 随机森林模型 |
| `logisticregression_model.pkl` | 逻辑回归模型 |
| `svm_model.pkl` | 支持向量机模型 |
| `naivebayes_model.pkl` | 朴素贝叶斯模型 |
| `feature_names.txt` | 特征名称列表（47个特征） |
| `best_horizon.txt` | 最佳预测周期信息 |

**加载模型进行预测：**

```python
import joblib
import numpy as np

# 加载最佳模型（GradientBoosting）
model = joblib.load('models/results/v2/gradientboosting_model.pkl')

# 加载特征名称
with open('models/results/v2/feature_names.txt') as f:
    feature_names = [line.strip() for line in f]

# 准备特征数据（需要47个特征）
# X_new = ... (你的特征数据)

# 预测
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)

print(f"预测: {'上涨' if prediction[0] == 1 else '下跌'}")
print(f"上涨概率: {probability[0][1]:.2%}")
```

## 启动推理 API

使用 FastAPI 启动本地推理服务：

### 1. 安装依赖

```bash
pip install fastapi uvicorn
```

### 2. 启动 API 服务

```bash
python -m uvicorn uvicorn src.v2.inference_v2:app --reload --port 8000
```

### 3. API 端点

服务启动后，访问 http://localhost:8000 查看 API 文档。

**主要端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | API 信息 |
| `/health` | GET | 健康检查 |
| `/model-info` | GET | 模型信息 |
| `/predict/simple` | GET/POST | 简单预测接口（推荐 GET） |
| `/predict` | POST | 预测接口 |

### 4. 调用预测接口

**使用 GET（推荐用于快速测试）：**
```bash
curl "http://localhost:8000/predict/simple?symbol=QQQ&date=2025-04-28&horizons=5,10"
```

**使用 POST：**
```bash
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ", "date": "2025-04-28", "horizons": [5, 10]}'
```

**使用 Python（GET）：**
```python
import requests
response = requests.get(
    "http://localhost:8000/predict/simple",
    params={"symbol": "QQQ", "date": "2025-04-28", "horizons": "5,10"}
)
print(response.json())
```

**使用 Python（POST）：**
```python
import requests
response = requests.post(
    "http://localhost:8000/predict/simple",
    json={"symbol": "QQQ", "date": "2025-04-28", "horizons": [5, 10]}
)
print(response.json())
```



**使用 Python：**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "current": {
            "date": "2026-02-20",
            "open": 520.0,
            "high": 525.0,
            "low": 518.0,
            "close": 522.0,
            "volume": 50000000
        },
        "history": [
            # 需要至少200天的历史数据
            {"date": "2026-01-01", "open": 500, "high": 510, "low": 495, "close": 505, "volume": 45000000},
            # ... 更多历史数据
        ],
        "horizon": 20
    }
)

result = response.json()
print(f"预测: {result['prediction']}")
print(f"上涨概率: {result['probability_up']:.2%}")
print(f"置信度: {result['confidence']:.2%}")
```

**响应格式：**

```json
{
  "prediction": "UP",
  "probability_up": 0.75,
  "probability_down": 0.25,
  "confidence": 0.75,
  "horizon": 20,
  "features_used": ["trend_strength", "distance_ma200", ...]
}
```

## 训练并部署到 SageMaker

### 1. 准备 AWS 环境

```bash
# 配置 AWS 凭证
aws configure

# 安装 SageMaker SDK
pip install sagemaker
```

### 2. 部署到 SageMaker

使用提供的脚本进行训练和部署：

```bash
# 方式一：直接部署已训练好的模型
python train_deploy_sagemaker_v2.py --mode deploy

# 方式二：训练并部署（需要先准备数据）
python train_deploy_sagemaker_v2.py --mode train
```

**脚本参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 操作模式：train/deploy/predict/cleanup | train |
| `--model-path` | 模型文件路径 | models/results/v2/gradientboosting_model.pkl |
| `--endpoint` | 端点名称 | stock-predictor-v2 |
| `--horizon` | 预测周期 | 20 |

### 3. SageMaker 推理

部署完成后，可以使用 SDK 进行推理：

```python
import boto3
import json

# 创建 SageMaker 运行时
runtime = boto3.client('sagemaker-runtime')

# 准备特征数据
payload = {
    "features": [0.1, 0.05, -0.02, ...]  # 47个特征
}

# 调用端点
response = runtime.invoke_endpoint(
    EndpointName='stock-predictor-v2',
    ContentType='application/json',
    Body=json.dumps(payload)
)

# 解析结果
result = json.loads(response['Body'].read().decode())
print(f"预测: {result['prediction']}")
```

### 4. 成本估算

| 项目 | 成本 |
|------|------|
| ml.m5.large 实例 | $0.23/小时 |
| 训练（1小时） | ~$0.23 |
| 推理（24/7） | ~$165/月 |
| S3 存储 | ~$0.023/GB/月 |

**成本优化建议：**
- 使用 Serverless Inference（按调用计费）
- 使用 Batch Transform 进行批量预测
- 不用时删除端点

### 5. 清理资源

```bash
# 删除 SageMaker 端点
python train_deploy_sagemaker_v2.py --mode cleanup --endpoint stock-predictor-v2
```

## 配置文件

主要配置在 `src/config_v2.py`:

```python
# 预测周期
HORIZONS = [5, 10, 20]
DEFAULT_HORIZON = 5

# 模型参数
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 20,
    },
    # ...
}

# 集成权重
ENSEMBLE_WEIGHTS = {
    'LogisticRegression': 0.2,
    'RandomForest': 0.3,
    'GradientBoosting': 0.25,
    'SVM': 0.15,
    'NaiveBayes': 0.1
}
```

## 进一步优化

1. ~~**模型持久化**~~ - 已完成 ✓
2. ~~**推理API**~~ - 已完成（FastAPI）✓
3. ~~**SageMaker部署**~~ - 已完成 ✓
4. **超参数调优** - 使用Optuna进行贝叶斯优化
5. **特征选择** - 基于重要性筛选最佳特征
6. **时间序列CV** - 使用TimeSeriesSplit替代StratifiedKFold
7. **模型更新** - 添加增量学习支持
8. **实时数据** - 集成实时市场数据API

## 注意事项

1. **过拟合风险**: 树模型在训练集上表现很好，需要更多验证
2. **市场变化**: 模型基于历史数据，需定期重训练
3. **交易成本**: 预测准确率不代表实际盈利能力
4. **风险提示**: 本系统仅供研究参考，不构成投资建议