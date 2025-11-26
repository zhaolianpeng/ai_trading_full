# 短单交易（Scalping）使用指南

## 🎯 短单交易模式

系统现在支持**高频短单交易模式**，适合小时级及以下时间框架的频繁交易。

## 🚀 快速开始

### 启用短单交易模式

```bash
TRADING_MODE=scalping MARKET_INTERVAL=1h python3 main.py
```

### 使用 Binance 小时级数据

```bash
DATA_SOURCE=binance \
MARKET_SYMBOL=BTC/USDT \
MARKET_TIMEFRAME=1h \
TRADING_MODE=scalping \
python3 main.py
```

### 使用 Yahoo Finance 小时级数据（股票）

```bash
DATA_SOURCE=yahoo \
MARKET_SYMBOL=AAPL \
MARKET_PERIOD=1mo \
MARKET_INTERVAL=1h \
TRADING_MODE=scalping \
python3 main.py
```

## 📊 短单模式参数对比

| 参数 | 标准模式 | 短单模式 | 说明 |
|------|---------|---------|------|
| 质量评分 | 50 | **30** | 降低要求，捕捉更多机会 |
| 确认数量 | 2 | **1** | 减少确认，提高信号频率 |
| LLM评分 | 40 | **30** | 降低AI评分要求 |
| 盈亏比 | 1.5 | **1.2** | 降低盈亏比要求 |
| 最大持仓 | 20 | **10** | 缩短持仓周期 |
| 止损倍数 | 1.0 | **0.8** | 收紧止损 |
| 止盈倍数 | 2.0 | **1.5** | 降低止盈目标 |
| 成交量阈值 | 1.2 | **1.1** | 降低成交量要求 |

## ⚙️ 配置说明

### 交易模式选项

1. **scalping** - 高频短单模式
   - 适合：小时级及以下（1m, 5m, 15m, 30m, 1h）
   - 特点：交易频繁，持仓时间短

2. **normal** - 标准模式
   - 适合：小时级到日线级（1h, 4h, 1d）
   - 特点：平衡的交易频率

3. **swing** - 波段交易模式
   - 适合：日线级及以上（1d, 1w）
   - 特点：交易较少，持仓时间长

### 自动模式检测

如果使用小时级数据（`MARKET_INTERVAL=1h` 或 `MARKET_TIMEFRAME=1h`），系统会自动应用短单模式参数，即使 `TRADING_MODE=normal`。

## 📈 使用示例

### 示例 1: Binance 比特币小时级短单

```bash
DATA_SOURCE=binance \
MARKET_SYMBOL=BTC/USDT \
MARKET_TIMEFRAME=1h \
MARKET_LIMIT=1000 \
TRADING_MODE=scalping \
USE_LLM=False \
USE_ERIC_INDICATORS=True \
python3 main.py
```

### 示例 2: Yahoo Finance 股票小时级短单

```bash
DATA_SOURCE=yahoo \
MARKET_SYMBOL=AAPL \
MARKET_PERIOD=1mo \
MARKET_INTERVAL=1h \
TRADING_MODE=scalping \
USE_LLM=False \
python3 main.py
```

### 示例 3: 更激进的短单（进一步降低阈值）

```bash
TRADING_MODE=scalping \
MIN_QUALITY_SCORE=20 \
MIN_CONFIRMATIONS=1 \
MIN_LLM_SCORE=25 \
MIN_RISK_REWARD=1.1 \
BACKTEST_MAX_HOLD=5 \
python3 main.py
```

## 🎛️ 自定义参数

即使使用短单模式，你也可以通过环境变量覆盖特定参数：

```bash
TRADING_MODE=scalping \
MIN_QUALITY_SCORE=25 \      # 覆盖默认的30
MIN_RISK_REWARD=1.1 \        # 覆盖默认的1.2
BACKTEST_MAX_HOLD=8 \        # 覆盖默认的10
python3 main.py
```

## 📊 预期效果

使用短单模式后，你应该看到：

1. **更多信号**：信号数量显著增加
2. **更频繁的交易**：开单次数增加
3. **更短的持仓**：平均持仓周期缩短
4. **更快的执行**：止损止盈更紧凑

## ⚠️ 注意事项

1. **风险控制**：短单模式交易更频繁，需要严格控制风险
2. **手续费**：频繁交易会产生更多手续费，实际收益需要考虑
3. **滑点**：高频交易可能受到滑点影响
4. **数据质量**：确保数据源稳定可靠

## 🔍 监控指标

关注以下指标来评估短单模式效果：

- **交易次数**：应该显著增加
- **平均持仓周期**：应该缩短（< 10 个周期）
- **胜率**：可能略有下降，但盈亏比应该保持
- **总收益率**：综合评估

## 💡 优化建议

1. **根据市场调整**：
   - 震荡市场：使用短单模式
   - 趋势市场：使用标准或波段模式

2. **时间框架选择**：
   - 1h 时间框架：适合短单
   - 15m-30m：更适合短单
   - 1d：不适合短单

3. **参数微调**：
   - 如果信号太少：进一步降低阈值
   - 如果信号太多：适当提高阈值
   - 如果胜率太低：提高质量评分要求

## 📝 完整配置示例

`.env` 文件配置：

```env
# 数据源
DATA_SOURCE=binance
MARKET_SYMBOL=BTC/USDT
MARKET_TIMEFRAME=1h
MARKET_LIMIT=1000

# 交易模式
TRADING_MODE=scalping

# 可选：覆盖默认参数
# MIN_QUALITY_SCORE=25
# MIN_CONFIRMATIONS=1
# MIN_LLM_SCORE=25
# MIN_RISK_REWARD=1.1
# BACKTEST_MAX_HOLD=8

# 其他配置
USE_LLM=False
USE_ERIC_INDICATORS=True
USE_SIGNAL_FILTER=True
```

