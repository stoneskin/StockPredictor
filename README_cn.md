# 📈 Stock Predictor - 股票预测交易系统

---

**🌐 语言**: [English Version](README.md)

基于分类 + 集成学习的QQQ股票预测系统。使用多个机器学习模型和市场状态识别来预测QQQ未来价格走势（上涨/下跌）。

**状态**: ✅ 完全可用 | **版本**: 2.0 | **平台**: Windows/Linux/Mac | **框架**: scikit-learn + FastAPI

---

## 🎯 快速概览

| 项目 | 详情 |
|------|------|
| **预测标的** | QQQ (纳斯达克100科技股ETF) |
| **预测目标** | 价格方向：上涨 (↑) 或 下跌 (↓) |
| **预测周期** | 5天、10天、20天（可配置） |
| **模型** | 5个集成模型（逻辑回归、随机森林、梯度提升、SVM、朴素贝叶斯） |
| **特征** | 47个技术指标 + 市场状态识别 |
| **准确率** | 52-65%（取决于预测周期，随机基准为50%） |
| **响应速度** | 每次预测 <100ms |

### ✨ 核心特性

- **🤖 5个集成模型**: 基于验证集性能加权投票
- **📊 47个技术特征**: MA、RSI、MACD、ATR、布林带、趋势、市场状态、与SPY相关性
- **⚡ 实时API服务**: FastAPI服务器，自动从雅虎财经获取数据
- **🔮 多周期预测**: 同时预测5天、10天、20天等多个时间段
- **🎓 完整文档**: 系统设计、API指南、故障排除
- **🧪 模型持久化**: 预训练模型已保存，包含特征名称
- **📈 市场状态识别**: 自动识别牛市/熊市/震荡市
- **☁️ 云部署就绪**: 支持AWS SageMaker部署

---

## 📚 文档

| 文档 | 用途 |
|------|------|
| **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** | 快速开始（先读这个！） |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | 系统架构及数据流 |
| **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** | 所有API端点及示例 |
| **[docs/V2_CLASSIFICATION.md](docs/V2_CLASSIFICATION.md)** | ML方法及特征工程 |
| **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | 常见问题及解决方案 |
| **[docs/v2/CACHE_OPTIMIZATION.md](docs/v2/CACHE_OPTIMIZATION.md)** | 性能优化指南 |

---

## 🚀 快速开始（3个步骤）

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

### 2️⃣ 启动API服务器
```bash
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

### 3️⃣ 进行第一次预测

**使用GET请求（推荐用于快速测试）：**
```bash
curl "http://localhost:8000/predict/simple?symbol=QQQ&date=2026-02-25&horizons=5,10,20"
```

**使用Python：**
```python
import requests

response = requests.get(
    "http://localhost:8000/predict/simple",
    params={"symbol": "QQQ", "date": "2026-02-25", "horizons": "5,10,20"}
)
print(response.json())
```

**查看实时API文档**: http://localhost:8000/docs

---

## 📁 项目结构

```
StockPredictor/
├── 📚 docs/                           # 文档（按版本分类）
│   ├── GETTING_STARTED.md            # ← 从这里开始!
│   ├── ARCHITECTURE.md               # 系统设计
│   ├── API_REFERENCE.md              # API端点
│   ├── V2_CLASSIFICATION.md          # ML细节
│   ├── TROUBLESHOOTING.md            # 常见问题
│   ├── v1/                           # V1文档（已弃用）
│   ├── v1.5/                         # V1.5文档（实验性）
│   ├── v2/                           # V2文档（当前）
│   │   ├── README.md
│   │   ├── API_GUIDE.md
│   │   └── CACHE_OPTIMIZATION.md
│   └── archive/                      # 旧文档
│
├── 🧠 src/                            # 源代码
│   ├── v1/                           # V1: 回归 [已弃用]
│   │   ├── config.py
│   │   ├── train.py
│   │   ├── data_preparation.py
│   │   ├── inference.py
│   │   └── ...
│   ├── v1_5/                         # V1.5: 前向验证 [实验性]
│   │   ├── train_walkforward.py
│   │   └── walk_forward/
│   ├── v2/                           # V2: 分类 [当前活跃 ✅]
│   │   ├── inference_v2.py           # API服务器 (FastAPI)
│   │   ├── train_v2.py               # 训练管道
│   │   ├── config_v2.py              # 配置 & 参数
│   │   ├── data_preparation_v2.py    # 数据加载 & 特征工程
│   │   ├── models_v2/                # 5个ML模型
│   │   │   ├── base.py              # 基类
│   │   │   ├── logistic_model.py    # 逻辑回归
│   │   │   ├── random_forest_model.py
│   │   │   ├── gradient_boosting_model.py
│   │   │   ├── svm_model.py
│   │   │   ├── naive_bayes_model.py
│   │   │   └── ensemble.py          # 集成投票
│   │   └── regime_v2/                # 市场状态检测
│   │       ├── detector.py
│   │       ├── ma_crossover.py
│   │       └── volatility_regime.py
│   └── common/                       # 共享工具
│
├── 📊 data/                           # 数据目录
│   ├── raw/                          # 雅虎财经原始数据
│   │   ├── qqq.csv
│   │   └── spy.csv
│   ├── cache/                        # 缓存用于快速加载
│   │   ├── qqq.csv
│   │   ├── spy.csv
│   │   └── ...
│   ├── processed/                    # 工程特征（训练/测试分割）
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── splits/                       # 交叉验证分割
│
├── 🤖 models/                         # 训练好的模型
│   ├── checkpoints/                  # 中间检查点
│   ├── onnx/                         # ONNX格式（用于部署）
│   │   └── model.onnx
│   └── results/v2/                   # V2结果（使用这些）
│       ├── gradientboosting_model.pkl ← 最佳单模型（88.7%准确率）
│       ├── randomforest_model.pkl
│       ├── ensemble_model.pkl
│       ├── logisticregression_model.pkl
│       ├── svm_model.pkl
│       ├── naivebayes_model.pkl
│       ├── feature_names.txt         # 47个特征名称
│       ├── best_horizon.txt          # 最佳周期配置
│       ├── horizon_comparison.json   # 各周期性能对比
│       └── results.txt               # 详细指标
│
├── ✅ tests/                          # 测试脚本
│   ├── test_api.py
│   ├── test_performance_comparison.py
│   ├── test_qqq_fix.py
│   ├── test_cache_performance.py
│   └── test_output.txt
│
├── 📋 requirements.txt                # Python依赖
├── README.md                          # 英文版本
└── README_cn.md                       # 中文版本（本文件）
```

---

## 🧠 模型架构

### 5个基础模型

| 模型 | 优势 | 用途 |
|------|------|------|
| **逻辑回归** | 可解释性强、简单基线 | 简单模式、需要解释 |
| **随机森林** | 处理非线性、鲁棒、快速 | 通用、良好基线 |
| **梯度提升** | 功能强大、最佳表现（88.7%准确率） | 主要预测器、高精度 |
| **SVM (RBF)** | 复杂决策边界 | 非线性可分模式 |
| **朴素贝叶斯** | 非常快速、概率模型 | 实时预测（需要速度） |

### 集成策略

通过**加权投票**组合模型，权重基于验证集性能：
- **梯度提升**: 25% 权重（最准确）
- **随机森林**: 30% 权重（最可靠）  
- **逻辑回归**: 20% 权重（基线）
- **SVM**: 15% 权重（非线性模式）
- **朴素贝叶斯**: 10% 权重（快速推理）

### 市场状态识别

**MA交叉状态**（50/200日移动平均线）：
- 🟢 **牛市**: 价格 > MA50 > MA200
- 🔴 **熊市**: 价格 < MA50 < MA200
- 🟠 **震荡**: 其他情况

**波动率状态**：
- **高**: 日波动率 > 2%
- **正常**: 1% - 2%
- **低**: < 1%

---

## 📊 性能结果

### 最佳周期：20日预测

| 模型 | 准确率 | ROC-AUC | F1分数 | 精准度 | 召回率 |
|------|--------|---------|---------|---------|---------|
| **梯度提升** | 88.7% | 0.954 | 0.918 | 0.92 | 0.92 |
| **随机森林** | 88.5% | 0.945 | 0.911 | 0.90 | 0.92 |
| **逻辑回归** | 62.5% | 0.512 | 0.760 | 0.75 | 0.77 |
| **SVM** | 61.9% | 0.508 | 0.755 | 0.74 | 0.77 |
| **朴素贝叶斯** | 55.3% | 0.485 | 0.715 | 0.72 | 0.71 |
| **集成** | 64.8% | 0.578 | 0.775 | 0.79 | 0.76 |

### 周期对比（集成模型）

| 周期 | 准确率 | ROC-AUC | F1分数 | 备注 |
|------|--------|---------|---------|------|
| 5天 | 58.2% | 0.536 | 0.726 | 太短，噪音多 |
| 10天 | 61.7% | 0.541 | 0.750 | 中等 |
| **20天** | 64.8% | 0.578 | 0.775 | **最佳 - 推荐** |
| 30天 | 52.1% | 0.485 | 0.695 | 太长，信号弱 |

**基准**: 50% (随机猜测)  
**注意**: 梯度提升单模型表现优于集成 - 考虑调整投票权重

### 重要特征排名

1. `trend_strength` - (MA50 - MA200) / MA200
2. `distance_ma200` - 距200日MA百分比
3. `volatility` - 历史波动率（20天）
4. `correlation_spy_20d` - 与SPY指数相关性
5. `rsi` - 相对强弱指数
6. `macd_signal` - MACD信号线

---

## 💻 使用示例

### API使用（推荐）

**1. 启动服务器：**
```bash
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

**2. 简单预测（GET）：**
```python
import requests

response = requests.get(
    "http://localhost:8000/predict/simple",
    params={
        "symbol": "QQQ",
        "date": "2026-02-25",
        "horizons": "5,10,20"
    }
)
print(response.json())
```

**3. 高级预测（POST）：**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "current": {
            "date": "2026-02-25",
            "open": 520.0,
            "high": 525.0,
            "low": 518.0,
            "close": 522.0,
            "volume": 50000000
        },
        "horizon": 20
    }
)

result = response.json()
print(f"预测: {result['prediction']}")  # 'UP' 或 'DOWN'
print(f"上涨概率: {result['probability_up']:.1%}")
```

**4. 查看API文档：**
在浏览器中打开 `http://localhost:8000/docs` 查看交互式文档

### Python脚本使用

```python
import sys
sys.path.insert(0, 'src/v2')

from data_preparation_v2 import prepare_data
import joblib

# 1. 准备数据
X, y, feature_names, df = prepare_data(horizon=20)

# 2. 加载预训练模型（梯度提升 - 最佳）
model = joblib.load('models/results/v2/gradientboosting_model.pkl')

# 3. 预测
prediction = model.predict(X[-10:])  # 最后10个样本
probability = model.predict_proba(X[-10:])

print(f"预测: {prediction}")  # [0, 1, 0, ...]
print(f"上涨概率: {probability[:, 1]}")  # [0.45, 0.92, ...]
```

### 加载预训练模型

```python
import joblib

# 加载最佳模型（梯度提升 - 88.7%准确率）
model = joblib.load('models/results/v2/gradientboosting_model.pkl')

# 加载特征名称
with open('models/results/v2/feature_names.txt') as f:
    feature_names = [line.strip() for line in f]

# 特征总数为47
print(f"特征数: {len(feature_names)}")
print(f"前5个特征: {feature_names[:5]}")

# 进行预测
import numpy as np
X_new = np.random.randn(10, 47)  # 10个样本，47个特征
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

---

## 🎓 训练自己的模型

### 完整训练管道

```bash
# 1. 导航到项目根目录
cd StockPredictor

# 2. 运行训练（下载数据、工程特征、训练5个模型+集成）
python src/v2/train_v2.py

# 3. 检查结果
cat models/results/v2/results.txt
```

**训练做什么：**
- 从雅虎财经下载5年QQQ和SPY每日数据
- 工程化47个技术指标（MA、RSI、MACD、ATR、布林带等）
- 创建分类标签（UP=1如果N天内价格上涨，DOWN=0）
- 训练5个基础模型（逻辑回归、RF、GB、SVM、NB）
- 训练集成模型（加权投票）
- 在测试集上用多个指标评估
- 保存结果和预训练模型

### 配置

编辑 `src/v2/config_v2.py`：

```python
# 股票 & 数据
SYMBOL = "QQQ"              # 预测的股票代码
TRAIN_YEARS = 5             # 历史数据年数

# 预测周期
HORIZONS = [5, 10, 20]      # 这些天数的预测
DEFAULT_HORIZON = 20        # 未指定时默认值

# 模型参数
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 20,
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
    },
    # ...
}

# 集成权重（总和应=1.0）
ENSEMBLE_WEIGHTS = {
    'GradientBoosting': 0.25,
    'RandomForest': 0.30,
    'LogisticRegression': 0.20,
    'SVM': 0.15,
    'NaiveBayes': 0.10,
}
```

然后重新训练：
```bash
python src/v2/train_v2.py
```

---

## ☁️ 部署

### 选项1：本地FastAPI服务器（推荐开发）

```bash
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

访问地址: http://localhost:8000  
API文档: http://localhost:8000/docs

### 选项2：生产FastAPI（使用Gunicorn）

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 src.v2.inference_v2:app
```

### 选项3：AWS SageMaker（生产规模）

**准备AWS环境：**
```bash
pip install sagemaker boto3
aws configure  # 设置AWS凭证
```

**部署模型：**
```bash
python docs/v2/train_deploy_sagemaker_v2.py --mode deploy --endpoint stock-predictor-v2
```

**使用SageMaker预测：**
```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='stock-predictor-v2',
    ContentType='application/json',
    Body=json.dumps({"features": [0.1, 0.05, -0.02, ...]})
)
result = json.loads(response['Body'].read().decode())
print(f"预测: {result['prediction']}")
```

### 选项4：Docker容器

**创建Dockerfile：**
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ src/
COPY models/ models/
CMD ["python", "-m", "uvicorn", "src.v2.inference_v2:app", "--host", "0.0.0.0", "--port", "8000"]
```

**构建并运行：**
```bash
docker build -t stock-predictor .
docker run -p 8000:8000 stock-predictor
```

---

## ⚙️ 配置 & 调优

### 预测周期

在 `src/v2/config_v2.py` 中调整：
```python
HORIZONS = [5, 10, 20]     # 预测多少天
DEFAULT_HORIZON = 20        # API默认使用
```

**建议**：
- **5天**: 快速交易，噪音多（58%准确率）- 风险大
- **10天**: 中期（62%准确率）- 中等
- **20天**: 最佳平衡（65%准确率）- ← **推荐**
- **30+天**: 信号弱（50%准确率）- 避免

### 特征工程

`src/v2/data_preparation_v2.py` 中的核心特征：
- **趋势**: 移动平均线（10、20、50、200天）
- **动量**: RSI、MACD、随机振荡器
- **波动率**: ATR、布林带
- **相关性**: 与SPY的相关性
- **市场状态**: MA交叉、波动率状态
- **价格模式**: 跳空、反转、支撑/阻力

修改以添加/删除特征，然后重新训练。

### 模型超参数

在 `src/v2/config_v2.py` MODEL_PARAMS 中调整：
```python
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,    # 更多树=更好但更慢
        'max_depth': 5,         # 限制深度防止过拟合
        'min_samples_split': 20,
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,   # 更小=更准确但更慢
        'max_depth': 3,
    },
    # ...
}
```

然后重新训练：`python src/v2/train_v2.py`

---

## 📞 故障排除

### API无法启动

**错误**: `Connection refused` 或端口已被使用
```bash
# 检查什么在使用8000端口
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Mac/Linux

# 使用不同端口
python -m uvicorn src.v2.inference_v2:app --host 0.0.0.0 --port 8001
```

### ModuleNotFoundError

**错误**: `No module named 'src'`
```bash
# 确保在项目根目录
cd StockPredictor

# 从项目根目录运行
python -m uvicorn src.v2.inference_v2:app --reload --host 0.0.0.0 --port 8000
```

### 找不到模型

**错误**: `FileNotFoundError: models/results/v2/...`
```bash
# 先训练模型
python src/v2/train_v2.py

# 验证文件存在
ls models/results/v2/
```

### 数据下载问题

**错误**: `无法从雅虎财经下载数据`
```bash
# 检查网络连接
# 验证缓存存在
ls data/cache/

# 手动下载
python -c "import yfinance as yf; yf.download('QQQ', start='2020-01-01', end='2025-12-31').to_csv('data/cache/qqq.csv')"
```

### 依赖未安装

```bash
# 重新安装所有依赖
pip install --upgrade -r requirements.txt

# 或使用虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

详见: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 📝 项目详情

| 项目 | 详情 |
|------|------|
| **语言** | Python 3.8+ |
| **核心库** | scikit-learn、pandas、numpy、yfinance |
| **API框架** | FastAPI + uvicorn |
| **ML模型** | 5个集成（分类） |
| **预测目标** | QQQ价格方向（上涨/下跌） |
| **数据源** | 雅虎财经（免费） |
| **训练数据** | 5年每日OHLCV |
| **特征数** | 47个技术指标 |
| **状态** | 生产就绪 ✅ |

---

## 📚 版本历史

| 版本 | 方法 | 状态 | 备注 |
|------|------|------|------|
| **V2** | 分类 + 集成 | ✅ 活跃 | 当前 - **使用这个！** 最佳结果 |
| **V1.5** | 前向验证 | ⚠️ 实验性 | 研究 & 优化仅用 |
| **V1** | 回归（连续） | 📚 已弃用 | 学习参考 - 交易不建议使用 |

---

## ❓ 常见问题

**Q: 能预测除QQQ外的其他股票吗？**  
A: 可以！在 `src/v2/config_v2.py` 中改 `SYMBOL = "SPY"` 并重新训练。适用于雅虎财经的任何股票代码。

**Q: 65%准确率足以盈利吗？**  
A: 小心。算上手续费、滑点和市场冲击，实际收益可能更低。用作信号确认，不要唯一决策。

**Q: 多久重新训练一次？**  
A: 推荐每月一次，或准确率明显下降时。市场条件不断变化，模型需要更新。

**Q: 最少需要多少数据？**  
A: 至少200个交易日（约1年）。更多数据更好。当前系统使用5年进行稳健训练。

**Q: 能在GPU上运行吗？**  
A: scikit-learn默认使用CPU。GPU加速可考虑XGBoost-GPU或PyTorch实现。

**Q: 想添加更多特征怎么办？**  
A: 编辑 `src/v2/data_preparation_v2.py` 添加技术指标，然后重新训练模型。

**Q: 能用于期权交易吗？**  
A: 可以，但要注意：隐含波动率、时间衰减和其他因素重要。仅用作方向指导。

**Q: 数据使用什么交易所/时区？**  
A: 雅虎财经使用NYSE时间（EST）。预测用前一天收盘价。

---

## 🎯 推荐后续步骤

1. **阅读** [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - 详细快速开始指南
2. **理解** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - 系统设计与数据流
3. **学习** [docs/V2_CLASSIFICATION.md](docs/V2_CLASSIFICATION.md) - ML方法详情
4. **尝试** [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - 所有API端点
5. **故障排除** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - 遇到问题时

---

## 💡 获得更好结果的建议

1. **使用20天周期** - 准确率和预测窗口的最佳平衡
2. **监控市场状态** - 趋势市场预测更可靠
3. **结合技术分析** - 不要仅依赖ML模型
4. **历史回测** - 用历史数据验证策略
5. **定期重新训练** - 市场条件变化，模型需要更新
6. **使用集成预测** - 组合多个信号提高鲁棒性
7. **检查与SPY相关性** - QQQ与市场高度相关
8. **管理仓位大小** - 65%准确率不代表保证盈利

---

## 📄 许可 & 免责声明

教育项目 - 免费用于学习和研究目的。

**免责声明**: 本模型仅供教育目的。不构成投资建议，不保证利润。交易和投资涉及重大损失风险。投资前务必做自己的研究并咨询合格财务顾问。

---

## 🤝 贡献

发现bug或有改进建议？联系我！

---

**准备好开始了吗？** 🚀

1. **安装**: `pip install -r requirements.txt`
2. **训练**: `python src/v2/train_v2.py` (CPU上约5分钟)
3. **预测**: `python -m uvicorn src.v2.inference_v2:app --reload --port 8000`
4. **访问**: http://localhost:8000/docs

**有问题？** 查看 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) 或阅读代码注释。

祝你预测顺利！📊
