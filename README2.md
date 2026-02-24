# Stock Predictor V2 - Classification Approach

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

模型目前保存在内存中（训练时）。如需持久化，可以扩展 `train_v2.py` 添加模型保存功能：

```python
# 保存模型
import joblib
joblib.dump(ensemble, 'models/results/v2/ensemble_model.pkl')

# 加载模型
ensemble = joblib.load('models/results/v2/ensemble_model.pkl')
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

1. **模型持久化** - 添加模型保存/加载功能
2. **超参数调优** - 使用Optuna进行贝叶斯优化
3. **特征选择** - 基于重要性筛选最佳特征
4. **时间序列CV** - 使用TimeSeriesSplit替代StratifiedKFold
5. **模型更新** - 添加增量学习支持

## 注意事项

1. **过拟合风险**: 树模型在训练集上表现很好，需要更多验证
2. **市场变化**: 模型基于历史数据，需定期重训练
3. **交易成本**: 预测准确率不代表实际盈利能力
4. **风险提示**: 本系统仅供研究参考，不构成投资建议