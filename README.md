# AI Trading System — LLM + Quant (Enhanced)

一个结合传统量化分析与大语言模型（LLM）的智能交易系统。

## ✨ 主要功能

### 数据层
- 🌐 **真实市场数据**：支持从 Yahoo Finance 和 Binance 获取实时数据（支持永续合约）
- 📊 **历史K线数据**：支持CSV文件导入和实时数据获取
- 🛡️ **数据验证**：自动检测和修复数据问题

### 特征层（多因子）
- 📊 **基础技术指标**：EMA、RSI、ATR、成交量分析、MACD、布林带等
- 🎯 **Eric 全面策略指标**：Eric Score、Donchian 通道、EMA 眼、量能爆发、背离检测、波动预警
- 🚀 **高级技术因子**：
  - VWAP代理：使用典型价格和成交量计算
  - 突破距离：当前价格距离阻力位/支撑位的距离百分比
  - 多窗口动量：5、10、20、50周期的价格动量
  - 波动率regime：基于ATR的波动率分类（LOW_VOL, MID_VOL, HIGH_VOL, EXTREME_VOL）
  - 成交量爆发：当前成交量相对于均值的倍数
  - 价格动量（多窗口）：支持多个时间窗口的动量计算

### 信号层（规则/多因子打分）
- 🔍 **规则信号检测**：Long Structure、Breakout、RSI 背离、MACD 金叉、Eric 策略信号等
- 📈 **多因子信号评分系统**：
  - `trend_score`: 趋势得分（0-100）
  - `momentum_score`: 动量得分（0-100）
  - `vol_score`: 波动率得分（0-100）
  - `volume_score`: 成交量得分（0-100）
  - `rsi_score`: RSI得分（0-100）
  - `macd_score`: MACD得分（0-100）
  - `bb_score`: 布林带得分（0-100）
  - `adx_score`: ADX得分（0-100）
  - `composite_score`: 综合得分（加权平均）

### AI过滤器层
- 🤖 **监督ML过滤器**：使用RandomForest/LogisticRegression对候选信号进行筛选
  - 基于历史信号的后验收益作为标签训练模型
  - 自动特征提取和标准化
  - 支持模型保存和加载
- 🤖 **LLM解释器**：支持 OpenAI GPT 和 DeepSeek（包括 deepseek-reasoner 推理模型）
  - 对候选信号输出自然语言解释与置信度
  - 用于人工审核或进一步合并到策略决策
- 🛡️ **强制过滤器**：ATR过滤（避免低波动）、EMA多头排列、趋势强度>50、突破有效性验证
- 📊 **多时间周期分析**：综合1小时、4小时、日线级别信号，提高信号质量
- ⏰ **短周期入场优化**：小时级信号确认后，观察5分钟K线（优先）或3分钟K线寻找最佳入场点

### 回测/风控层
- 📈 **完整回测系统**：支持6个月+数据，200+交易回测
  - 基于ATR的止损/止盈
  - 支持部分止盈
  - 考虑手续费和滑点
  - 避免重叠交易
- 🛡️ **风险管理器**：
  - 持仓限制：限制最大同时持仓数（默认5）
  - 每日最大亏损：限制每日最大亏损比例（默认5%）
  - 资金管理：每笔交易风险为总资金的1%
  - 手续费/滑点：自动计算并扣除
- 💰 **动态盈亏比**：自动计算并调整止损止盈，确保盈亏比>=1.5

### 监控/日志层
- 📋 **完整日志系统**：记录每条信号的特征、ML得分、LLM输出、最终结果及后验PnL
  - 信号详细日志（JSON格式）
  - ML评分日志（CSV格式）
  - 后验PnL日志（CSV格式）
- 🔄 **持续学习**：从日志中提取训练数据，用于ML模型持续训练和优化
- 📉 **可视化分析**：自动生成价格图表和回测结果图表（支持中文）
- 📝 **详细报告**：生成完整的分析报告（中文）
- 🔄 **交易模式**：支持 normal（标准）、scalping（高频短单）、swing（波段）三种模式

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 数据源配置（推荐使用 Binance 获取加密货币数据）
DATA_SOURCE=binance
MARKET_SYMBOL=BTC/USDT
MARKET_TIMEFRAME=1h
MARKET_TYPE=future  # 永续合约
MARKET_LIMIT=1000

# LLM 配置（支持 OpenAI 和 DeepSeek）
LLM_PROVIDER=deepseek  # 或 'openai'
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-reasoner  # 推理模型，默认使用
# 或使用 OpenAI
# OPENAI_API_KEY=sk-your-openai-api-key
# OPENAI_MODEL=gpt-4o-mini
USE_LLM=True

# 回测配置（可选）
BACKTEST_MONTHS=6  # 回测6个月数据
```

### 3. 运行程序

**重要**: 确保在项目根目录下运行命令！

```bash
# 切换到项目目录
cd /path/to/ai_trading_full

# 运行程序
python3 main.py
```

**快速测试（不使用 LLM）**：
```bash
# 确保在项目根目录下
cd /path/to/ai_trading_full

DATA_SOURCE=yahoo MARKET_SYMBOL=BTC-USD USE_LLM=False python3 main.py
```

## 📁 项目结构

```
ai_trading_full/
├── main.py                 # 主程序入口
├── config.py              # 配置管理（支持环境变量）
├── data/
│   ├── loader.py          # 数据加载和验证
│   └── market_data.py     # 市场数据获取（Yahoo Finance, Binance）
├── features/
│   ├── ta_basic.py        # 基础技术指标
│   ├── ta_advanced.py     # 高级技术指标（MACD, 布林带等）
│   ├── advanced_factors.py # 高级技术因子（VWAP proxy、Breakout dist、多窗口动量等）
│   ├── eric_indicators.py # Eric 策略指标
│   └── divergence.py      # 背离检测
├── signal_rules.py        # 交易信号规则
├── strategy/
│   ├── strategy_runner.py # 策略执行器
│   ├── signal_filter.py   # 信号过滤器（质量评分、强制过滤）
│   ├── signal_scorer.py   # 信号评分系统（多因子打分）
│   ├── ml_filter.py       # ML过滤器（RandomForest/LogisticRegression）
│   ├── multi_timeframe_analyzer.py  # 多时间周期分析
│   └── market_structure_analyzer.py  # 市场结构分析
├── ai_agent/
│   ├── llm_client.py      # LLM 客户端（支持 OpenAI 和 DeepSeek）
│   ├── llm_prompt.py      # Prompt 模板库
│   └── signal_interpret.py # 信号解释器（改进的JSON解析）
├── backtest/
│   ├── simulator.py       # 回测模拟器
│   └── risk_manager.py    # 风险管理器（持仓限制、每日最大亏损等）
├── monitoring/
│   └── signal_logger.py   # 信号监控和日志系统（记录ML得分、后验PnL等）
├── execution/
│   └── exchange_api.py    # 交易所 API（待实现）
├── utils/
│   ├── logger.py          # 日志系统
│   ├── validators.py      # 数据验证
│   ├── visualization.py   # 可视化工具
│   ├── trading_mode.py    # 交易模式配置
│   ├── time_utils.py      # 时间工具（北京时间转换）
│   ├── entry_finder.py    # 短周期入场点查找（优先5分钟，备选3分钟）
│   ├── json_i18n.py       # JSON 中英文转换
│   └── config_validator.py # 配置验证
└── requirements.txt       # 依赖列表
```

## ⚙️ 配置说明

所有配置都可以通过环境变量或 `.env` 文件设置：

### 数据配置
- `DATA_SOURCE`: 数据源类型（`synthetic`/`csv`/`yahoo`/`binance`，默认 `synthetic`）
- `DATA_PATH`: CSV 文件路径（当 `DATA_SOURCE=csv` 时使用）
- `MARKET_SYMBOL`: 市场交易对（当 `DATA_SOURCE=yahoo` 或 `binance` 时使用，如 `BTC-USD`, `AAPL`）
- `MARKET_PERIOD`: 数据周期（Yahoo Finance，如 `1y`, `6mo`, `3mo`）
- `MARKET_INTERVAL`: 数据间隔（Yahoo Finance，如 `1h`, `1d`, `15m`）
- `MARKET_TIMEFRAME`: 时间框架（Binance，如 `1h`, `4h`, `1d`）
- `MARKET_LIMIT`: 最大数据条数（Binance，最大 1000）
- `SYNTHETIC_DATA_SIZE`: 合成数据大小（默认 1500）
- `USE_ERIC_INDICATORS`: 是否使用 Eric 全面策略指标（默认 `True`）

### LLM 配置
- `USE_LLM`: 是否启用 LLM 分析（True/False）
- `LLM_PROVIDER`: LLM 提供商（`openai` 或 `deepseek`，默认 `openai`）
- `OPENAI_API_KEY`: OpenAI API Key
- `OPENAI_MODEL`: OpenAI 模型名称（默认 `gpt-4o-mini`）
- `DEEPSEEK_API_KEY`: DeepSeek API Key
- `DEEPSEEK_MODEL`: DeepSeek 模型名称（默认 `deepseek-reasoner`，推理模型）
- `OPENAI_TEMPERATURE`: 温度参数（0-2，默认 0.0）
- `OPENAI_MAX_TOKENS`: 最大 token 数（默认 400）
- `LLM_CONCURRENT_WORKERS`: LLM并发处理线程数（默认 5，可根据API限制调整）

### 回测配置
- `BACKTEST_MAX_HOLD`: 最大持仓周期（默认 20）
- `BACKTEST_ATR_STOP_MULT`: 止损 ATR 倍数（默认 1.0）
- `BACKTEST_ATR_TARGET_MULT`: 止盈 ATR 倍数（默认 2.0）
- `BACKTEST_MONTHS`: 回测数据月数（默认 0，表示使用默认 limit）
- `BACKTEST_FULL_DATA`: 是否使用全量数据回测（True/False）
- `MIN_LLM_SCORE`: LLM 评分最低阈值（默认 40）
- `MIN_QUALITY_SCORE`: 最小质量评分（默认 50）
- `MIN_CONFIRMATIONS`: 最小确认数量（默认 2）
- `MIN_RISK_REWARD`: 最小盈亏比要求（默认 1.5）
- `MIN_ATR_PCT`: 最小 ATR 百分比（默认 0.5%，用于过滤低波动）

### ML过滤器配置
- `ML_MODEL_TYPE`: ML模型类型（`random_forest` 或 `logistic_regression`，默认 `random_forest`）
- `ML_RETRAIN`: 是否重新训练ML模型（True/False，默认 False）
- `ML_CONFIDENCE_THRESHOLD`: ML置信度阈值（0-1，默认 0.6）

### 风险管理配置
- `MAX_POSITIONS`: 最大同时持仓数（默认 5）
- `MAX_DAILY_LOSS`: 每日最大亏损比例（默认 0.05，即5%）
- `FEE_RATE`: 手续费率（默认 0.0005，即0.05%）
- `SLIPPAGE`: 滑点率（默认 0.0005，即0.05%）

### 信号评分配置
- `MIN_COMPOSITE_SCORE`: 最小综合得分阈值（默认 60.0）

### 交易模式配置
- `TRADING_MODE`: 交易模式（`normal`/`scalping`/`swing`，默认 `normal`）
  - `normal`: 标准模式，适合大多数场景
  - `scalping`: 高频短单模式，降低阈值，适合小时级及以下
  - `swing`: 波段交易模式，提高阈值，适合日线级

### 多时间周期配置
- `USE_MULTI_TIMEFRAME`: 是否使用多时间周期分析（True/False，默认 True）
- `MIN_TIMEFRAME_CONFIRMATIONS`: 多时间周期最小确认数（默认 2）

### 日志配置
- `LOG_LEVEL`: 日志级别（DEBUG/INFO/WARNING/ERROR）
- `LOG_FILE`: 日志文件路径（默认 trading.log）

## 📊 输出文件

运行后会生成以下文件：

- `sample_data.csv` - 处理后的市场数据
- `signals_log.json` - 所有检测到的信号及 LLM 分析结果
- `trades.csv` - 回测交易记录
- `trading_chart.png` - 价格图表（含信号标注）
- `backtest_results.png` - 回测结果图表
- `analysis_report.txt` - 详细分析报告
- `trading.log` - 运行日志

## 🔧 使用真实市场数据

### 方式1: 从 Yahoo Finance 获取（推荐）⭐

支持股票、加密货币、ETF、指数等，**免费且无需 API Key**！

```bash
# 获取比特币数据
export DATA_SOURCE=yahoo
export MARKET_SYMBOL=BTC-USD
export MARKET_PERIOD=1y
export MARKET_INTERVAL=1h
python main.py
```

**常用交易对**：
- 加密货币: `BTC-USD`, `ETH-USD`, `BNB-USD`
- 股票: `AAPL`, `GOOGL`, `MSFT`, `TSLA`, `AMZN`, `NVDA`
- 指数: `^GSPC` (S&P 500), `^DJI` (Dow Jones)

**时间周期选项**：
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

**数据间隔选项**：
- `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1wk`, `1mo`

### 方式2: 从 Binance 获取（加密货币专用，推荐）⭐

支持现货和永续合约，可获取最新价格数据：

```bash
# 永续合约（推荐）
export DATA_SOURCE=binance
export MARKET_SYMBOL=BTC/USDT
export MARKET_TIMEFRAME=1h
export MARKET_TYPE=future  # 永续合约
export MARKET_LIMIT=1000
python3 main.py

# 现货
export DATA_SOURCE=binance
export MARKET_SYMBOL=BTC/USDT
export MARKET_TIMEFRAME=1h
export MARKET_TYPE=spot
export MARKET_LIMIT=1000
python3 main.py
```

**支持的时间框架**：
- `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, `1M`
- **注意**：入场点查找优先使用 `5m`（更稳定），如果失败则尝试 `3m`

**回测模式（获取6个月数据）**：
```bash
export DATA_SOURCE=binance
export MARKET_SYMBOL=BTC/USDT
export MARKET_TIMEFRAME=1h
export MARKET_TYPE=future
export BACKTEST_MONTHS=6
python3 main.py
```

### 方式3: 使用本地 CSV 文件

1. 准备 CSV 文件，包含以下列：
   - `datetime` - 日期时间（会被用作索引）
   - `open`, `high`, `low`, `close` - OHLC 价格数据
   - `volume` - 成交量

2. 在 `.env` 中设置：
   ```env
   DATA_SOURCE=csv
   DATA_PATH=path/to/your/data.csv
   ```

3. 运行程序

### 方式4: 使用合成数据（测试用）

```bash
export DATA_SOURCE=synthetic
export SYNTHETIC_DATA_SIZE=1500
python main.py
```

**详细说明请查看**: [MARKET_DATA_GUIDE.md](MARKET_DATA_GUIDE.md)

## 📈 回测指标说明

- **total_trades**: 总交易次数（目标：200+）
- **win_rate**: 胜率
- **avg_return**: 平均收益率
- **total_return**: 总收益率
- **profit_factor**: 盈亏比（总盈利/总亏损）
- **max_drawdown**: 最大回撤
- **sharpe_ratio**: 夏普比率
- **max_consecutive_losses**: 最大连续亏损次数
- **avg_hold_period**: 平均持仓周期
- **risk_reward_ratio**: 风险回报比（每笔交易）

## 🛡️ 风险管理

系统提供完整的风险管理功能：

### 持仓限制
- **最大同时持仓数**：限制同时持有的仓位数量（默认5）
- **资金管理**：每笔交易风险为总资金的1%
- **资金检查**：确保有足够资金开仓（保留10%缓冲）

### 每日最大亏损
- **亏损限制**：限制每日最大亏损比例（默认5%）
- **自动停止**：当日亏损达到限制时，自动停止开新仓
- **每日重置**：每日自动重置亏损记录

### 手续费和滑点
- **手续费**：自动计算并扣除（默认0.05%）
- **滑点**：考虑买卖滑点（默认0.05%）
- **实际价格**：入场和出场价格已考虑滑点

### 使用示例
```python
from backtest.risk_manager import RiskManager

risk_manager = RiskManager(
    max_positions=5,
    max_daily_loss=0.05,
    fee_rate=0.0005,
    slippage=0.0005,
    initial_capital=100000.0
)

# 检查是否可以开仓
can_open, reason = risk_manager.can_open_position(entry_price, stop_loss)
if can_open:
    position = risk_manager.open_position(entry_idx, entry_price, stop_loss, take_profit)
```

## 📊 监控和日志系统

系统提供完整的监控和日志功能，用于持续学习与参数调优：

### 日志文件
- **signals_detailed.json**: 信号详细日志（JSON格式）
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

- **ml_scores.csv**: ML评分日志（CSV格式）
- **posterior_pnl.csv**: 后验PnL日志（CSV格式）

### 持续学习
系统支持从日志中提取训练数据，用于ML模型持续训练：

```python
from monitoring.signal_logger import SignalLogger

logger = SignalLogger(log_dir='logs')
logger.log_signal(signal_data, df, idx, ml_score=ml_prob, ml_prediction=ml_pred)
logger.log_trade_result(signal_idx, trade_record)
logger.save_logs()

# 获取训练数据
training_df = logger.get_training_data()
```

### 使用示例
```python
from monitoring.signal_logger import SignalLogger
from strategy.ml_filter import prepare_training_data, MLSignalFilter

# 记录信号
logger = SignalLogger(log_dir='logs')
logger.log_signal(signal_data, df, idx, ml_score=0.75, ml_prediction=1)

# 记录交易结果
logger.log_trade_result(signal_idx, trade_record)
logger.save_logs()

# 准备训练数据并训练ML模型
training_df = prepare_training_data(enhanced_signals, trades_df, df)
ml_filter = MLSignalFilter(model_type='random_forest', retrain=True)
ml_filter.train_model(training_df, target_col='label')
```

## 🛡️ 信号过滤系统

系统采用多层次的信号过滤机制，确保只交易高质量信号：

### 1. 多因子信号评分系统

系统对每个信号进行多维度评分，包括：

- **趋势得分** (`trend_score`): 基于EMA排列、价格位置、价格动量、RSI（0-100）
- **动量得分** (`momentum_score`): 基于多个窗口的价格动量（0-100）
- **波动率得分** (`vol_score`): 基于ATR和价格波动，评估波动率健康度（0-100）
- **成交量得分** (`volume_score`): 基于成交量比率和爆发（0-100）
- **RSI得分** (`rsi_score`): 基于RSI值判断超买超卖（0-100）
- **MACD得分** (`macd_score`): 基于MACD金叉和柱状图（0-100）
- **布林带得分** (`bb_score`): 基于价格在布林带中的位置（0-100）
- **ADX得分** (`adx_score`): 基于ADX值判断趋势强度（0-100）
- **综合得分** (`composite_score`): 加权平均所有得分（0-100）

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

### 2. ML过滤器（监督学习）

使用RandomForest或LogisticRegression对候选信号进行筛选：

- **特征提取**：自动从信号数据中提取特征（趋势、动量、波动率、成交量等）
- **模型训练**：基于历史信号的后验收益作为标签训练模型
- **预测评分**：输出信号质量概率（0-1）和预测类别（0/1）
- **持续学习**：从日志中提取训练数据，定期重新训练模型

**使用方法**：
```python
from strategy.ml_filter import MLSignalFilter

ml_filter = MLSignalFilter(model_type='random_forest')
ml_prob, ml_pred = ml_filter.predict(signal_data, df, idx)

if ml_prob > 0.6:  # ML置信度阈值
    # 通过ML过滤
    pass
```

### 3. 强制过滤器（必须满足）

- **ATR 过滤**：避免低波动市场，ATR 相对价格百分比 >= 0.5%（可配置）
- **EMA 多头排列**：强制要求 `ema21 > ema55 > ema100`
- **趋势强度**：趋势强度评分必须 > 50（基于斜率、一致性、突破、均线排列、ATR）
- **突破有效性**：如果信号包含突破，必须验证：
  - 成交量放大（vol_ratio >= 1.2）
  - 价格持续在阻力位上方（最近3根K线）或创新高
- **结构标签过滤**：只在特定市场结构下生成信号（TREND_UP, BREAKOUT_UP, REVERSAL_UP）

### 4. 信号质量评分系统（传统）

系统对每个信号进行多维度评分（总分可达 200+ 分），使用所有可用的技术指标：

#### 基础指标评分
- **趋势确认** (+20分)：EMA 多头排列
- **价格位置确认** (+15分)：价格在 EMA 上方
- **成交量确认** (+15-20分)：成交量放大
- **RSI 确认** (+5-10分)：RSI 健康区间或超卖
- **价格动量确认** (+10分)：5周期正动量
- **支撑位确认** (+15分)：接近支撑位但未跌破

#### 高级指标评分（如果可用）
- **MACD 确认** (+15分)：MACD 多头
- **布林带确认** (+10分)：价格在布林带中轨上方
- **ATR 波动率确认** (+10分 或 -10分)：波动率正常
- **随机指标确认** (+10分)：随机指标金叉且未超买
- **CCI 确认** (+10分)：CCI 健康区间或超卖
- **ADX 确认** (+15分)：ADX 显示强趋势
- **威廉指标确认** (+5分)：威廉指标在健康区间

#### Eric 策略指标评分（如果可用）
- **Eric Score 确认** (+10-15分)：Eric Score 中性或超卖
- **Donchian通道确认** (+10分)：Donchian通道上升趋势
- **EMA眼确认** (+10分)：价格接近EMA（小眼）

#### 综合评分
- **价格位置确认** (+10分)：价格在区间中上部
- **LLM 评分加权** (+10-20分)：LLM 高分

**过滤条件**：
- 最小质量评分：默认 50 分（回测模式：30 分）
- 最小确认数量：默认 2 个（回测模式：1 个）
- 最小 LLM 评分：默认 40 分（回测模式：30 分）

### 3. 动态盈亏比计算

系统自动计算并调整止损止盈，确保盈亏比 >= 1.5：

```python
# 初始止损止盈
止损 = 入场价 - ATR * 1.0
止盈 = 入场价 + ATR * 2.0

# 如果盈亏比 < 1.5，自动调整止盈
if 盈亏比 < 1.5:
    所需收益 = 风险 * 1.5
    止盈 = 入场价 + 所需收益
```

## 🔄 多时间周期分析

系统支持多时间周期综合分析，提高信号质量：

1. **数据获取**：自动获取 1小时、4小时、日线级别数据（过去7天）
2. **信号检测**：在每个时间周期上独立检测信号
3. **信号对齐**：将不同周期的信号按时间对齐（允许一定容差）
4. **多周期确认**：要求至少 2 个时间周期同时确认（可配置）
5. **综合判断**：只有多周期确认的信号才会进入下一步分析

**优势**：
- 减少假信号
- 提高信号可靠性
- 捕捉更明确的趋势

## ⏰ 短周期入场优化

当小时级信号确认后，系统会：

1. **获取短周期数据**：从信号时间点开始，获取后续的短周期K线数据
   - **优先使用5分钟数据**（更稳定，所有市场都支持）
   - 如果5分钟数据获取失败，则尝试3分钟数据
   - 从信号时间点开始，向前查找60分钟（约12个5分钟K线或20个3分钟K线）
2. **评分入场点**：对每个短周期K线进行评分，考虑：
   - 价格动作（是否回调、是否突破）
   - 成交量（是否放大）
   - 时间距离（距离信号时间越近越好）
   - K线形态（下影线/上影线、阳线/阴线）
   - 价格稳定性（波动适中）
3. **选择最佳入场点**：选择评分最高的短周期K线作为入场点

**优势**：
- 更精确的入场时机（使用5分钟数据，精度足够且更稳定）
- 减少滑点
- 提高入场价格质量
- 自动降级机制（5分钟失败时尝试3分钟）

**技术说明**：
- 虽然函数名为 `find_best_entry_point_3m`，但实际优先使用5分钟数据
- 5分钟数据在所有Binance市场都稳定支持，避免了3分钟在某些市场的不稳定性

## 🎯 Eric 全面策略指标

项目集成了完整的 Eric 全面策略指标体系，参考 TradingView 专业策略：

### 核心指标

1. **Eric Score** - 超买超卖指标
   - 基于价格在区间内的位置
   - 双重平滑和标准化
   - 一级/二级超买超卖阈值

2. **Donchian 通道** - 趋势过滤器
   - 55周期通道
   - 通道趋势判断（上升/下降/横盘）
   - 价格接近上下沿检测

3. **EMA 眼** - 支撑/压力提示
   - 价格与 EMA 的相对距离
   - 小眼（<1%）：接近 EMA，可能形成支撑/压力
   - 大眼（>3%）：远离 EMA

4. **量能爆发** - 成交量分析
   - 一级爆量：vol/sma > 1.8
   - 二级爆量：vol/sma > 3.0

5. **背离检测** - 价格 vs Eric Score
   - 牛背离：价格新低但 score 更高
   - 空背离：价格新高但 score 更低

6. **波动预警** - ATR 分析
   - ATR 相对于其均值的倍数
   - 高波动预警

### 信号类型

- **eric_long**: Eric 策略做多信号（评分 3-6）
- **eric_short**: Eric 策略做空信号（评分 3-6）

**详细说明请查看**: [ERIC_STRATEGY_GUIDE.md](ERIC_STRATEGY_GUIDE.md)

## 🤖 LLM 支持

### 支持的 LLM 提供商

1. **OpenAI**
   - 模型：`gpt-4o-mini`（默认）、`gpt-4o`、`gpt-3.5-turbo` 等
   - 配置：设置 `OPENAI_API_KEY` 和 `LLM_PROVIDER=openai`

2. **DeepSeek**（推荐，更便宜）
   - 模型：`deepseek-reasoner`（默认，推理模型）、`deepseek-chat`
   - 配置：设置 `DEEPSEEK_API_KEY` 和 `LLM_PROVIDER=deepseek`
   - 优势：价格更便宜，推理模型适合复杂分析

### 技术指标集成

系统将所有技术指标集成到 LLM 决策中，特征包包含：

#### 基础指标
- EMA排列、趋势方向
- 成交量比率、量能爆发
- ATR、ATR百分比
- RSI
- 价格动量（5周期、20周期）
- 价格位置

#### 高级指标（如果启用 USE_ADVANCED_TA）
- **MACD**：MACD线、信号线、柱状图、金叉状态
- **布林带**：上轨、中轨、下轨、宽度、价格位置
- **随机指标**：%K值、%D值
- **威廉指标**：Williams %R值
- **CCI**：商品通道指标
- **ADX**：ADX值、+DI、-DI

#### Eric 策略指标（如果启用 USE_ERIC_INDICATORS）
- **Eric Score**：原始值、平滑值
- **Donchian通道**：上轨、下轨、趋势方向
- **EMA眼**：价格与EMA的距离百分比
- **量能爆发**：爆量级别
- **背离检测**：牛背离、空背离
- **波动预警**：波动预警级别

LLM 会基于这些完整的指标数据做出更准确的交易决策。

### Prompt 模板

项目包含 10+ 个专业 Prompt 模板（`ai_agent/llm_prompt.py`）：

- 市场结构分析
- 多周期分析
- 量能分析
- 突破验证
- 假突破检测
- 风险管理
- 新闻情绪分析
- 特征重要性分析
- 交易总结
- 超参优化
- 紧急警报

### JSON 解析增强

系统采用多层 JSON 解析策略，自动处理：
- Markdown 代码块标记（```json）
- 尾随逗号
- 单引号转双引号
- 正则表达式提取 JSON 对象
- 优雅降级到 fallback 方法

## 🛠️ 技术特性

- ✅ 完整的错误处理和重试机制
- ✅ 数据验证和自动修复
- ✅ 结构化日志记录
- ✅ 环境变量配置管理
- ✅ 类型提示和文档字符串
- ✅ 可视化分析工具（支持中文）
- ✅ 详细的性能指标
- ✅ 多时间周期分析
- ✅ 短周期入场点优化（优先5分钟，备选3分钟）
- ✅ 强制信号过滤器
- ✅ 信号质量评分系统
- ✅ 动态盈亏比计算
- ✅ 市场结构分析
- ✅ 北京时间转换
- ✅ 改进的 JSON 解析（支持多种格式）
- ✅ 支持 Binance 永续合约
- ✅ 支持扩大回测周期（6个月+，200+交易）
- ✅ **ML过滤器**：监督学习模型（RandomForest/LogisticRegression）筛选信号
- ✅ **高级技术因子**：VWAP proxy、Breakout dist、多窗口动量、波动率regime等
- ✅ **多因子信号评分**：trend_score、momentum_score、vol_score、volume_score等
- ✅ **风险管理器**：持仓限制、每日最大亏损、手续费/滑点计算
- ✅ **监控/日志系统**：记录ML得分、后验PnL，支持持续学习

## 📝 注意事项

1. **API 费用**：使用 LLM 会产生 API 调用费用，建议先用 `USE_LLM=False` 测试
2. **数据质量**：确保输入数据质量，系统会自动检测和修复常见问题
3. **ML模型训练**：需要足够的历史数据（建议至少100+交易记录）才能训练有效的ML模型
4. **特征工程**：确保特征无未来函数，可实时计算
5. **风险控制**：严格遵循风险管理规则，避免过度交易
6. **持续学习**：定期重新训练ML模型，适应市场变化
7. **回测验证**：在实盘前充分回测，验证策略有效性
8. **风险提示**：本项目仅供学习和研究使用，不构成投资建议

## 📚 架构文档

详细的系统架构说明请查看 [ARCHITECTURE.md](ARCHITECTURE.md)，包括：
- 分层架构设计
- 各层详细说明
- 使用流程
- 持续学习流程
- 配置参数

## 🔄 更新日志

### v4.0 (Complete Quant Architecture)
- ✨ **ML过滤器**：实现监督学习模型（RandomForest/LogisticRegression）筛选信号
- ✨ **高级技术因子**：添加VWAP proxy、Breakout dist、多窗口动量、波动率regime等
- ✨ **多因子信号评分系统**：实现trend_score、momentum_score、vol_score、volume_score等评分
- ✨ **风险管理器**：实现持仓限制、每日最大亏损、手续费/滑点计算
- ✨ **监控/日志系统**：完整记录ML得分、后验PnL，支持持续学习
- ✨ **持续学习流程**：从日志中提取训练数据，自动训练和优化ML模型
- ✨ **架构文档**：添加完整的系统架构说明文档（ARCHITECTURE.md）

### v3.0 (Advanced Filtering & Multi-Timeframe)
- ✨ 添加强制过滤器（ATR、EMA多头排列、趋势强度、突破有效性）
- ✨ 实现多时间周期分析（1h、4h、1d综合判断）
- ✨ 添加短周期入场点优化（优先5分钟，备选3分钟，更稳定）
- ✨ 支持 DeepSeek API（包括 deepseek-reasoner 推理模型）
- ✨ 改进 JSON 解析逻辑，支持多种格式和自动修复
- ✨ 增强错误处理，优雅降级机制
- ✨ 支持扩大回测周期（6个月+，200+交易）
- ✨ 添加市场结构分析模块
- ✨ 支持 Binance 永续合约数据获取
- ✨ 添加北京时间转换功能
- ✨ 改进信号质量评分系统（新增随机指标、CCI、ADX、Donchian、EMA眼等评分项）
- ✨ 优化数据获取逻辑，确保获取最新价格
- ✨ **全面集成技术指标到决策判断**：特征包包含所有可用指标（MACD、布林带、随机指标、威廉指标、CCI、ADX、Eric Score等）
- ✨ **LLM并发处理优化**：支持多线程并发处理LLM调用，大幅提升回测速度（默认5个并发线程，可配置）

### v2.0 (Enhanced)
- ✨ 添加完整的日志系统
- ✨ 改进配置管理（支持环境变量）
- ✨ 修复 OpenAI API 调用（使用新版本 SDK）
- ✨ 添加数据验证和自动修复
- ✨ 增强回测指标（Sharpe 比率、连续亏损等）
- ✨ 添加可视化功能
- ✨ 生成详细分析报告
- ✨ 改进错误处理和重试机制

## 📄 许可证

本项目仅供学习和研究使用。
