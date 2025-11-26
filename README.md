# AI Trading System — LLM + Quant (Enhanced)

一个结合传统量化分析与大语言模型（LLM）的智能交易系统演示项目。

## ✨ 主要功能

- 📊 **技术指标计算**：EMA、RSI、ATR、成交量分析、MACD、布林带等
- 🎯 **Eric 全面策略指标**：Eric Score、Donchian 通道、EMA 眼、量能爆发、背离检测、波动预警
- 🔍 **规则信号检测**：Long Structure、Breakout、RSI 背离、MACD 金叉、Eric 策略信号等
- 🤖 **AI 信号解释**：使用 OpenAI GPT 分析交易信号并生成决策
- 📈 **回测系统**：完整的回测框架，包含多种性能指标
- 📉 **可视化分析**：自动生成价格图表和回测结果图表（支持中文）
- 📝 **详细报告**：生成完整的分析报告（中文）
- 🛡️ **数据验证**：自动检测和修复数据问题
- 📋 **日志记录**：完整的日志系统，便于调试和监控
- 🌐 **真实市场数据**：支持从 Yahoo Finance 和 Binance 获取实时数据

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
# 数据源配置（推荐使用 Yahoo Finance 获取真实数据）
DATA_SOURCE=yahoo
MARKET_SYMBOL=BTC-USD
MARKET_PERIOD=1y
MARKET_INTERVAL=1h

# LLM 配置（可选）
OPENAI_API_KEY=sk-your-api-key-here
USE_LLM=True
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
│   └── loader.py          # 数据加载和验证
├── features/
│   ├── ta_basic.py        # 基础技术指标
│   └── divergence.py      # 背离检测
├── signal_rules.py        # 交易信号规则
├── strategy/
│   └── strategy_runner.py # 策略执行器
├── ai_agent/
│   ├── llm_client.py      # LLM 客户端（支持重试）
│   ├── llm_prompt.py      # Prompt 模板库
│   └── signal_interpret.py # 信号解释器
├── backtest/
│   └── simulator.py       # 回测模拟器
├── execution/
│   └── exchange_api.py    # 交易所 API（待实现）
├── utils/
│   ├── logger.py          # 日志系统
│   ├── validators.py      # 数据验证
│   └── visualization.py   # 可视化工具
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
- `OPENAI_API_KEY`: OpenAI API Key
- `OPENAI_MODEL`: 模型名称（默认 gpt-4o-mini）
- `OPENAI_TEMPERATURE`: 温度参数（0-2，默认 0.0）
- `OPENAI_MAX_TOKENS`: 最大 token 数（默认 400）

### 回测配置
- `BACKTEST_MAX_HOLD`: 最大持仓周期（默认 20）
- `BACKTEST_ATR_STOP_MULT`: 止损 ATR 倍数（默认 1.0）
- `BACKTEST_ATR_TARGET_MULT`: 止盈 ATR 倍数（默认 2.0）
- `MIN_LLM_SCORE`: LLM 评分最低阈值（默认 40）

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

### 方式2: 从 Binance 获取（加密货币专用）

```bash
export DATA_SOURCE=binance
export MARKET_SYMBOL=BTC/USDT
export MARKET_TIMEFRAME=1h
export MARKET_LIMIT=1000
python main.py
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

- **total_trades**: 总交易次数
- **win_rate**: 胜率
- **avg_return**: 平均收益率
- **total_return**: 总收益率
- **profit_factor**: 盈亏比（总盈利/总亏损）
- **max_drawdown**: 最大回撤
- **sharpe_ratio**: 夏普比率
- **max_consecutive_losses**: 最大连续亏损次数
- **avg_hold_period**: 平均持仓周期

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

## 🤖 LLM Prompt 模板

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

## 🛠️ 技术特性

- ✅ 完整的错误处理和重试机制
- ✅ 数据验证和自动修复
- ✅ 结构化日志记录
- ✅ 环境变量配置管理
- ✅ 类型提示和文档字符串
- ✅ 可视化分析工具
- ✅ 详细的性能指标

## 📝 注意事项

1. **API 费用**：使用 LLM 会产生 API 调用费用，建议先用 `USE_LLM=False` 测试
2. **数据质量**：确保输入数据质量，系统会自动检测和修复常见问题
3. **回测限制**：这是简化版回测，实际交易需要考虑滑点、手续费等因素
4. **风险提示**：本项目仅供学习和研究使用，不构成投资建议

## 🔄 更新日志

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
