# 量化交易系统架构文档

## 系统架构概览

本系统采用分层架构设计，从数据层到监控层，完整覆盖量化交易的各个环节。

```
┌─────────────────────────────────────────────────────────────┐
│                     监控/日志层                              │
│  (signal_logger.py) - 记录ML得分、后验PnL、持续学习         │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│                   回测/风控层                                │
│  (simulator.py, risk_manager.py) - ATR止损止盈、持仓限制    │
│  每日最大亏损、手续费/滑点                                   │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│                    AI 过滤器层                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ ML过滤器         │  │ LLM解释器        │                │
│  │ (ml_filter.py)   │  │ (signal_interpret│                │
│  │ RandomForest/    │  │ .py)             │                │
│  │ LogisticReg      │  │ 自然语言解释     │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│                     信号层                                   │
│  (signal_scorer.py) - 多因子打分系统                        │
│  trend_score, momentum_score, vol_score, volume_score       │
│  合并成候选交易信号                                          │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│                     特征层                                   │
│  (advanced_factors.py, ta_basic.py, ta_advanced.py)        │
│  技术因子：EMA、RSI、ATR、Momentum、Volatility、            │
│  Volume surge、VWAP proxy、Breakout dist、                  │
│  Price Momentum (多窗口)                                     │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│                     数据层                                   │
│  (market_data.py) - 历史K线（CSV或实时）                    │
│  Yahoo Finance / Binance API                                │
└─────────────────────────────────────────────────────────────┘
```

## 各层详细说明

### 1. 数据层 (`data/market_data.py`)

**功能**：
- 从Yahoo Finance或Binance获取历史K线数据
- 支持CSV文件导入
- 支持实时数据获取

**数据格式**：
- `datetime`: 时间戳
- `open`, `high`, `low`, `close`: OHLC价格数据
- `volume`: 成交量

### 2. 特征层 (`features/`)

#### 2.1 基础技术指标 (`ta_basic.py`)
- EMA (21, 55, 100, 200)
- RSI (14)
- ATR (14)
- 成交量移动平均
- 阻力位/支撑位

#### 2.2 高级技术指标 (`ta_advanced.py`)
- MACD
- 布林带
- 随机指标
- 威廉指标
- CCI
- ADX

#### 2.3 高级因子 (`advanced_factors.py`)
- **VWAP代理**: 使用典型价格和成交量计算
- **突破距离**: 当前价格距离阻力位/支撑位的距离百分比
- **多窗口动量**: 5、10、20、50周期的价格动量
- **波动率regime**: 基于ATR的波动率分类（LOW_VOL, MID_VOL, HIGH_VOL, EXTREME_VOL）
- **成交量爆发**: 当前成交量相对于均值的倍数
- **趋势评分**: 基于多个周期的EMA排列和价格位置（0-100）
- **动量评分**: 基于多个窗口的动量（0-100）
- **波动率评分**: 基于ATR和价格波动（0-100）
- **成交量评分**: 基于成交量比率和爆发（0-100）

### 3. 信号层 (`strategy/signal_scorer.py`)

**功能**：
- 基于多因子构造原始信号得分
- 合并成候选交易信号

**得分类型**：
- `trend_score`: 趋势得分（0-100）
- `momentum_score`: 动量得分（0-100）
- `vol_score`: 波动率得分（0-100）
- `volume_score`: 成交量得分（0-100）
- `rsi_score`: RSI得分（0-100）
- `macd_score`: MACD得分（0-100）
- `bb_score`: 布林带得分（0-100）
- `adx_score`: ADX得分（0-100）
- `composite_score`: 综合得分（加权平均）

**权重配置**：
```python
weights = {
    'trend_score': 0.25,
    'momentum_score': 0.20,
    'vol_score': 0.15,
    'volume_score': 0.15,
    'rsi_score': 0.10,
    'macd_score': 0.05,
    'bb_score': 0.05,
    'adx_score': 0.05,
}
```

### 4. AI过滤器层

#### 4.1 ML过滤器 (`strategy/ml_filter.py`)

**功能**：
- 使用监督学习模型（RandomForest/LogisticRegression）对候选信号进行筛选
- 基于历史信号的后验收益作为标签训练模型

**模型类型**：
- `RandomForestClassifier`: 随机森林分类器（默认）
- `LogisticRegression`: 逻辑回归分类器

**特征提取**：
- 趋势特征（EMA排列、EMA值）
- 动量特征（RSI、价格动量）
- 波动率特征（ATR、ATR百分比）
- 成交量特征（成交量比率、量能爆发）
- 价格位置特征
- MACD、布林带、ADX特征
- Eric Score特征
- 质量评分、LLM评分

**训练数据准备**：
- 从历史信号和交易结果中提取特征和标签
- 标签：收益>0为1，否则为0

#### 4.2 LLM解释器 (`ai_agent/signal_interpret.py`)

**功能**：
- 对候选信号输出自然语言解释与置信度
- 用于人工审核或进一步合并到策略决策
- 不是直接下单，而是提供决策支持

**输出内容**：
- 趋势结构判断
- 交易信号建议（Long/Short/Neutral）
- 评分（0-100）
- 置信度（Low/Medium/High）
- 解释说明
- 风险提示

### 5. 回测/风控层

#### 5.1 回测引擎 (`backtest/simulator.py`)

**功能**：
- 基于ATR的止损/止盈
- 支持部分止盈
- 考虑手续费和滑点
- 避免重叠交易

**参数**：
- `max_hold`: 最大持仓周期
- `atr_mult_stop`: 止损ATR倍数
- `atr_mult_target`: 止盈ATR倍数
- `min_risk_reward`: 最小盈亏比要求
- `fee_rate`: 手续费率
- `slippage`: 滑点率

#### 5.2 风险管理器 (`backtest/risk_manager.py`)

**功能**：
- **持仓限制**: 限制最大同时持仓数（默认5）
- **每日最大亏损**: 限制每日最大亏损比例（默认5%）
- **资金管理**: 每笔交易风险为总资金的1%
- **手续费/滑点**: 自动计算并扣除

**风险控制规则**：
1. 检查持仓数量：不能超过`max_positions`
2. 检查每日亏损：当日亏损不能超过`max_daily_loss`
3. 计算风险：基于止损价格计算每笔交易风险
4. 资金检查：确保有足够资金开仓

### 6. 监控/日志层 (`monitoring/signal_logger.py`)

**功能**：
- 记录每条信号的特征、ML得分、LLM输出
- 记录最终结果及后验PnL
- 用于持续学习与参数调优

**日志文件**：
- `signals_detailed.json`: 信号详细日志（JSON格式）
- `ml_scores.csv`: ML评分日志（CSV格式）
- `posterior_pnl.csv`: 后验PnL日志（CSV格式）

**记录内容**：
- 信号索引和时间
- 规则信号信息
- 市场结构标签
- 特征包（关键特征）
- LLM输出
- ML评分和预测
- 信号评分（各项得分）
- 价格信息
- 止损止盈信息
- 后验PnL（交易结果）

**训练数据生成**：
- 从日志中提取特征和标签
- 用于ML模型持续训练和优化

## 使用流程

### 1. 数据准备
```python
from data.market_data import fetch_market_data
df = fetch_market_data(symbol='BTC/USDT', timeframe='1h')
```

### 2. 特征计算
```python
from features.ta_basic import add_basic_ta
from features.ta_advanced import add_advanced_ta
from features.advanced_factors import add_advanced_factors

df = add_basic_ta(df)
df = add_advanced_ta(df)
df = add_advanced_factors(df)
```

### 3. 信号生成和评分
```python
from strategy.signal_scorer import calculate_signal_scores, merge_signal_scores

# 计算信号得分
scores = calculate_signal_scores(df, idx, signal_data)
# 合并得分到信号数据
enhanced_signal = merge_signal_scores(signal_data, scores)
```

### 4. ML过滤
```python
from strategy.ml_filter import MLSignalFilter

ml_filter = MLSignalFilter(model_type='random_forest')
ml_prob, ml_pred = ml_filter.predict(signal_data, df, idx)

if ml_prob > 0.6:  # ML置信度阈值
    # 通过ML过滤
    pass
```

### 5. LLM解释
```python
from ai_agent.signal_interpret import interpret_with_llm

llm_out = interpret_with_llm(feature_packet, provider='deepseek', model='deepseek-reasoner')
```

### 6. 回测和风控
```python
from backtest.simulator import simple_backtest
from backtest.risk_manager import RiskManager

risk_manager = RiskManager(
    max_positions=5,
    max_daily_loss=0.05,
    fee_rate=0.0005,
    slippage=0.0005
)

trades_df, metrics = simple_backtest(
    df, enhanced_signals,
    max_hold=20,
    atr_mult_stop=1.0,
    atr_mult_target=2.0,
    min_risk_reward=1.5
)
```

### 7. 日志记录
```python
from monitoring.signal_logger import SignalLogger

logger = SignalLogger(log_dir='logs')
logger.log_signal(signal_data, df, idx, ml_score=ml_prob, ml_prediction=ml_pred)
logger.log_trade_result(signal_idx, trade_record)
logger.save_logs()
```

## 持续学习流程

1. **收集数据**: 运行策略，记录所有信号和交易结果
2. **准备训练数据**: 从日志中提取特征和标签
3. **训练ML模型**: 使用历史数据训练ML过滤器
4. **评估模型**: 评估模型性能，调整参数
5. **部署模型**: 将训练好的模型部署到生产环境
6. **持续优化**: 定期重新训练模型，持续优化策略

## 配置参数

### ML过滤器配置
- `model_type`: 'random_forest' 或 'logistic_regression'
- `retrain`: 是否重新训练模型

### 风险管理配置
- `max_positions`: 最大同时持仓数（默认5）
- `max_daily_loss`: 每日最大亏损比例（默认0.05，即5%）
- `fee_rate`: 手续费率（默认0.0005，即0.05%）
- `slippage`: 滑点率（默认0.0005，即0.05%）

### 信号评分配置
- `min_composite_score`: 最小综合得分阈值（默认60.0）

## 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- `pandas`: 数据处理
- `numpy`: 数值计算
- `scikit-learn`: ML模型（RandomForest, LogisticRegression）
- `openai`: LLM API（可选）
- `ccxt`: 加密货币交易所API
- `yfinance`: Yahoo Finance数据

## 注意事项

1. **ML模型训练**: 需要足够的历史数据（建议至少100+交易记录）
2. **特征工程**: 确保特征无未来函数，可实时计算
3. **风险控制**: 严格遵循风险管理规则，避免过度交易
4. **持续学习**: 定期重新训练ML模型，适应市场变化
5. **回测验证**: 在实盘前充分回测，验证策略有效性

