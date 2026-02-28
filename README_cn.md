# 📈 Stock Predictor - 股票预测交易系统

---

**🌐 语言**: [English Version](README.md)

基于分类 + 集成学习的QQQ股票预测系统。使用多个机器学习模型和市场状态识别来预测QQQ未来价格走势。

**状态**: ✅ 完全可用 | **版本**: 2.5 | **平台**: Windows/Linux/Mac | **框架**: scikit-learn + FastAPI

> **注意**: 最新版本（V2.5）在 `v2.5/` 文件夹中。V2版本在 `src/v2/`。请参阅 [CHANGELOG.md](CHANGELOG.md) 查看版本历史。

---

## 🎯 快速概览

| 项目 | 详情 |
|------|------|
| **预测标的** | QQQ (纳斯达克100科技股ETF) |
| **预测目标** | 4分类：上涨(UP)、下跌(DOWN)、上涨下跌(UP_DOWN)、横盘(SIDEWAYS) |
| **预测周期** | 5天、10天、20天、30天（可配置） |
| **阈值** | 1%、2.5%、5%价格变动 |
| **模型** | 7个集成模型（逻辑回归、随机森林、梯度提升、XGBoost、CatBoost、SVM、朴素贝叶斯） |
| **特征** | 47个技术指标 + 市场状态识别 |
| **响应速度** | 每次预测 <100ms |

### ✨ 核心特性

- **🤖 7个集成模型**: 新增XGBoost和CatBoost，基于验证集性能加权投票
- **📊 47个技术特征**: MA、RSI、MACD、ATR、布林带、趋势、市场状态
- **⚡ 实时API服务**: FastAPI服务器，自动从雅虎财经获取数据
- **🔮 多周期多阈值预测**: 同时预测5天、10天、20天、30天，支持1%、2.5%、5%阈值
- **🎓 完整文档**: 系统设计、API指南、故障排除
- **🧪 模型持久化**: 预训练模型已保存，包含特征名称
- **📈 市场状态识别**: 自动识别牛市/熊市/震荡市及波动率状态
- **☁️ 云部署就绪**: 支持AWS SageMaker部署
- **📝 详细日志**: 按日期时间记录训练、预测、API日志

---

## 📚 文档

| 文档 | 用途 |
|------|------|
| **[v2.5/README.md](v2.5/README.md)** | 最新版本（V2.5）- 4分类预测 |
| **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** | 快速开始（V2版本，先读这个！） |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | 系统架构及数据流 |
| **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** | 所有API端点及示例 |
| **[docs/V2_CLASSIFICATION.md](docs/V2_CLASSIFICATION.md)** | ML方法及特征工程 |
| **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | 常见问题及解决方案 |
| **[CHANGELOG.md](CHANGELOG.md)** | 版本历史 |

---

## 🚀 快速开始（3个步骤）

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

### 2️⃣ 启动API服务器（V2.5）
```bash
cd v2.5
python -m uvicorn src.inference_v2_5:app --reload --host 0.0.0.0 --port 8000
```

### 3️⃣ 进行第一次预测

**使用Python：**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "QQQ",
        "horizon": 20,
        "threshold": 0.01
    }
)
print(response.json())
```

**使用curl：**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "QQQ", "horizon": 20, "threshold": 0.01}'
```

---

## 📁 项目结构

```
StockPredictor/
├── v2.5/                      # 最新版本 (2.5.0)
│   ├── src/                   # 源代码
│   ├── tests/                 # 测试文件
│   ├── docs/                 # 文档
│   ├── data/                  # 数据文件
│   └── models/                # 模型文件
├── src/v2/                    # 旧版本 (2.0)
├── archive/                   # 归档版本 (v1, v1.5)
├── CHANGELOG.md              # 版本变更日志
├── README.md                 # 英文文档
└── README_cn.md              # 中文文档
```

---

## 🧠 模型架构

### 7个基础模型

| 模型 | 特点 | 使用场景 |
|------|------|----------|
| **逻辑回归** | 可解释，简单基线 | 简单模式，需要解释 |
| **随机森林** | 处理非线性，稳健，快速 | 通用场景 |
| **梯度提升** | 强大，高性能 | 主要预测 |
| **XGBoost** | 高性能，梯度提升 | 高精度需求 |
| **CatBoost** | 处理类别特征 | 分类特征数据 |
| **SVM (RBF)** | 复杂决策边界 | 非线性可分模式 |
| **朴素贝叶斯** | 非常快，概率输出 | 实时性要求 |

### 集成策略

模型通过基于验证集性能的**加权投票**进行组合：
- **随机森林**: 20% 权重
- **梯度提升**: 20% 权重
- **XGBoost**: 20% 权重
- **逻辑回归**: 15% 权重
- **CatBoost**: 15% 权重
- **SVM**: 5% 权重
- **朴素贝叶斯**: 5% 权重

---

## 📊 4分类系统

V2.5引入了4分类系统，根据阈值判断价格变动：

| 分类 | 条件 |
|------|------|
| **UP (上涨)** | 任意一天最大涨幅超过阈值，且最大跌幅不超过阈值 |
| **DOWN (下跌)** | 任意一天最大跌幅超过阈值，且最大涨幅不超过阈值 |
| **UP_DOWN (上涨下跌)** | 任意一天最大涨幅超过阈值 AND 最大跌幅超过阈值 |
| **SIDEWAYS (横盘)** | 最大涨幅不超过阈值 AND 最大跌幅不超过阈值 |

**示例**：对于5天周期，1%阈值：
- 如果5天内价格上涨超过1%但从未下跌超过1% → **UP**
- 如果5天内价格下跌超过1%但从未上涨超过1% → **DOWN**
- 如果5天内价格上涨超过1% AND 下跌超过1% → **UP_DOWN**
- 如果5天内价格波动在±1%以内 → **SIDEWAYS**

---

## 💻 使用示例

### API使用（推荐）

**1. 启动服务器：**
```bash
cd v2.5
python -m uvicorn src.inference_v2_5:app --reload --port 8000
```

**2. 简单预测：**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "symbol": "QQQ",
        "date": "2026-02-25",
        "horizons": [5, 10, 20]
    }
)
print(response.json())
```

**3. 多周期多阈值预测：**
```python
response = requests.post(
    "http://localhost:8000/predict/multi",
    json={
        "symbol": "QQQ",
        "horizons": [5, 10, 20, 30],
        "thresholds": [0.01, 0.025, 0.05]
    }
)
```

---

## ⚙️ 配置与调优

### 预测周期

编辑 `v2.5/src/config_v2_5.py`：
```python
HORIZONS = [5, 10, 20, 30]  # 预测天数
DEFAULT_HORIZON = 20          # 默认周期
THRESHOLDS = [0.01, 0.025, 0.05]  # 阈值
```

### 模型超参数

```python
MODEL_PARAMS = {
    'gradient_boosting': {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.1,
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
    },
}
```

---

## 📞 故障排除

### API无法启动

**错误**: 端口已被占用
```bash
# Windows
netstat -ano | findstr :8000

# 使用其他端口
python -m uvicorn src.inference_v2_5:app --port 8001
```

### 模块未找到

```bash
# 确保在v2.5目录
cd v2.5

# 从项目根目录运行
python -m uvicorn v2_5.src.inference_v2_5:app --reload
```

### 模型未找到

```bash
# 先训练模型
python v2.5/src/train_v2_5.py
```

---

## 📝 项目详情

| 项目 | 详情 |
|------|------|
| **语言** | Python 3.8+ |
| **核心库** | scikit-learn, pandas, numpy, yfinance |
| **API框架** | FastAPI + uvicorn |
| **ML模型** | 7个集成模型（4分类） |
| **预测目标** | QQQ价格方向（UP/DOWN/UP_DOWN/SIDEWAYS） |
| **数据源** | 雅虎财经（免费） |
| **训练数据** | 5年每日OHLCV数据 |
| **特征** | 47个技术指标 |
| **状态** | 生产就绪 ✅ |

---

## 📚 版本历史

| 版本 | 方法 | 状态 | 备注 |
|------|------|------|------|
| **2.5** | 4分类 + 多阈值 | ✅ 当前 | 使用此版本！ |
| **2.0** | 2分类 + 集成 | 📚 旧版 | 见 src/v2/ |
| **1.5** | Walk-Forward验证 | ⚠️ 实验 | 仅研究用 |
| **1.0** | 回归 | 📚 旧版 | 学习参考 |

---

## 🎯 建议的下一步

1. **阅读** [v2.5/README.md](v2.5/README.md) - V2.5详细文档
2. **了解** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - 系统设计
3. **学习** [docs/V2_CLASSIFICATION.md](docs/V2_CLASSIFICATION.md) - ML方法详情
4. **尝试** [v2.5/docs/API_GUIDE.md](v2.5/docs/API_GUIDE.md) - API端点
5. **排查** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - 解决问题

---

## 📄 许可证与免责声明

本项目仅供学习和研究目的。免费使用。

**免责声明**: 本模型仅供教育目的，不构成投资建议，不保证盈利。交易和投资涉及重大风险，可能造成损失。在做出投资决策前，请务必进行自己的研究并咨询合格的财务顾问。

---

**准备好了吗？** 🚀

1. **安装**: `pip install -r requirements.txt`
2. **训练**: `cd v2.5 && python src/train_v2_5.py`（约5分钟）
3. **预测**: `cd v2.5 && python -m uvicorn src.inference_v2_5:app --reload --port 8000`
4. **访问**: http://localhost:8000/docs

有问题？请查看 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

祝您预测愉快！ 📊
