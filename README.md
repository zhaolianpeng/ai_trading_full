# AI Trading System — LLM + Quant (Enhanced)

一个结合传统量化分析与大语言模型（LLM）的智能交易系统。

## ✨ 主要功能

- 📊 **技术指标计算**：EMA、RSI、ATR、成交量分析、MACD、布林带等
- 🎯 **Eric 全面策略指标**：Eric Score、Donchian 通道、EMA 眼、量能爆发、背离检测、波动预警
- 🔍 **规则信号检测**：Long Structure、Breakout、RSI 背离、MACD 金叉、Eric 策略信号等
- 🤖 **AI 信号解释**：支持 OpenAI GPT 和 DeepSeek（包括 deepseek-reasoner 推理模型）
- 🛡️ **强制过滤器**：ATR过滤（避免低波动）、EMA多头排列、趋势强度>50、突破有效性验证
- 📊 **多时间周期分析**：综合1小时、4小时、日线级别信号，提高信号质量
- ⏰ **3分钟入场优化**：小时级信号确认后，观察3分钟K线寻找最佳入场点
- 📈 **回测系统**：完整的回测框架，支持6个月+数据，200+交易回测
- 📉 **可视化分析**：自动生成价格图表和回测结果图表（支持中文）
- 📝 **详细报告**：生成完整的分析报告（中文）
- 🛡️ **数据验证**：自动检测和修复数据问题
- 📋 **日志记录**：完整的日志系统，便于调试和监控
- 🌐 **真实市场数据**：支持从 Yahoo Finance 和 Binance 获取实时数据（支持永续合约）
- 🎯 **信号质量评分**：多维度评分系统，自动过滤低质量信号
- 💰 **动态盈亏比**：自动计算并调整止损止盈，确保盈亏比>=1.5
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
│   ├── eric_indicators.py # Eric 策略指标
│   └── divergence.py      # 背离检测
├── signal_rules.py        # 交易信号规则
├── strategy/
│   ├── strategy_runner.py # 策略执行器
│   ├── signal_filter.py   # 信号过滤器（质量评分、强制过滤）
│   ├── multi_timeframe_analyzer.py  # 多时间周期分析
│   └── market_structure_analyzer.py  # 市场结构分析
├── ai_agent/
│   ├── llm_client.py      # LLM 客户端（支持 OpenAI 和 DeepSeek）
│   ├── llm_prompt.py      # Prompt 模板库
│   └── signal_interpret.py # 信号解释器（改进的JSON解析）
├── backtest/
│   └── simulator.py       # 回测模拟器
├── execution/
│   └── exchange_api.py    # 交易所 API（待实现）
├── utils/
│   ├── logger.py          # 日志系统
│   ├── validators.py      # 数据验证
│   ├── visualization.py   # 可视化工具
│   ├── trading_mode.py    # 交易模式配置
│   ├── time_utils.py      # 时间工具（北京时间转换）
│   ├── entry_finder.py    # 3分钟入场点查找
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
- `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`

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

## 🛡️ 信号过滤系统

系统采用多层次的信号过滤机制，确保只交易高质量信号：

### 1. 强制过滤器（必须满足）

- **ATR 过滤**：避免低波动市场，ATR 相对价格百分比 >= 0.5%（可配置）
- **EMA 多头排列**：强制要求 `ema21 > ema55 > ema100`
- **趋势强度**：趋势强度评分必须 > 50（基于斜率、一致性、突破、均线排列、ATR）
- **突破有效性**：如果信号包含突破，必须验证：
  - 成交量放大（vol_ratio >= 1.2）
  - 价格持续在阻力位上方（最近3根K线）或创新高

### 2. 信号质量评分系统

系统对每个信号进行多维度评分（总分可达 150+ 分）：

- **趋势确认** (+20分)：EMA 多头排列
- **价格位置确认** (+15分)：价格在 EMA 上方
- **成交量确认** (+15-20分)：成交量放大
- **RSI 确认** (+5-10分)：RSI 健康区间或超卖
- **MACD 确认** (+15分)：MACD 多头
- **布林带确认** (+10分)：价格在布林带中轨上方
- **ATR 波动率确认** (+10分 或 -10分)：波动率正常
- **Eric Score 确认** (+10-15分)：Eric Score 中性或超卖
- **价格动量确认** (+10分)：5周期正动量
- **支撑位确认** (+15分)：接近支撑位但未跌破
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

## ⏰ 3分钟入场优化

当小时级信号确认后，系统会：

1. **获取3分钟数据**：从信号时间点开始，获取后续的3分钟K线数据
2. **评分入场点**：对每个3分钟K线进行评分，考虑：
   - 价格动作（是否回调、是否突破）
   - 成交量（是否放大）
   - 时间距离（距离信号时间越近越好）
3. **选择最佳入场点**：选择评分最高的3分钟K线作为入场点

**优势**：
- 更精确的入场时机
- 减少滑点
- 提高入场价格质量

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
- ✅ 3分钟入场点优化
- ✅ 强制信号过滤器
- ✅ 信号质量评分系统
- ✅ 动态盈亏比计算
- ✅ 市场结构分析
- ✅ 北京时间转换
- ✅ 改进的 JSON 解析（支持多种格式）
- ✅ 支持 Binance 永续合约
- ✅ 支持扩大回测周期（6个月+，200+交易）

## 📝 注意事项

1. **API 费用**：使用 LLM 会产生 API 调用费用，建议先用 `USE_LLM=False` 测试
2. **数据质量**：确保输入数据质量，系统会自动检测和修复常见问题
3. **回测限制**：这是简化版回测，实际交易需要考虑滑点、手续费等因素
4. **风险提示**：本项目仅供学习和研究使用，不构成投资建议

## 🔄 更新日志

### v3.0 (Advanced Filtering & Multi-Timeframe)
- ✨ 添加强制过滤器（ATR、EMA多头排列、趋势强度、突破有效性）
- ✨ 实现多时间周期分析（1h、4h、1d综合判断）
- ✨ 添加3分钟入场点优化
- ✨ 支持 DeepSeek API（包括 deepseek-reasoner 推理模型）
- ✨ 改进 JSON 解析逻辑，支持多种格式和自动修复
- ✨ 增强错误处理，优雅降级机制
- ✨ 支持扩大回测周期（6个月+，200+交易）
- ✨ 添加市场结构分析模块
- ✨ 支持 Binance 永续合约数据获取
- ✨ 添加北京时间转换功能
- ✨ 改进信号质量评分系统
- ✨ 优化数据获取逻辑，确保获取最新价格

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
