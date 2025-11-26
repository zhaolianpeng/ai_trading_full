# 数据源使用指南

## 📊 支持的数据源

### 1. Yahoo Finance（股票数据推荐）

**适用场景**：股票、ETF、指数

**配置示例**：
```bash
DATA_SOURCE=yahoo MARKET_SYMBOL=AAPL MARKET_PERIOD=3mo MARKET_INTERVAL=1d python3 main.py
```

**常用股票代码**：
- `AAPL` - 苹果
- `TSLA` - 特斯拉
- `MSFT` - 微软
- `GOOGL` - 谷歌
- `AMZN` - 亚马逊

**时间周期选项**：
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

**时间间隔选项**：
- `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1wk`, `1mo`

**注意**：Yahoo Finance 对加密货币的支持不稳定，建议使用 Binance。

### 2. Binance（加密货币推荐）

**适用场景**：加密货币交易对

**配置示例**：
```bash
DATA_SOURCE=binance MARKET_SYMBOL=BTC/USDT MARKET_TIMEFRAME=1h python3 main.py
```

**常用交易对**：
- `BTC/USDT` - 比特币
- `ETH/USDT` - 以太坊
- `BNB/USDT` - 币安币
- `SOL/USDT` - Solana

**时间框架选项**：
- `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`

**数据限制**：最多获取 1000 条数据

### 3. CSV 文件

**适用场景**：本地历史数据

**配置示例**：
```bash
DATA_SOURCE=csv DATA_PATH=./data/historical_data.csv python3 main.py
```

**CSV 格式要求**：
- 必须包含列：`open`, `high`, `low`, `close`, `volume`
- 索引可以是日期时间或数字

### 4. 合成数据（测试用）

**适用场景**：测试和开发

**配置示例**：
```bash
DATA_SOURCE=synthetic SYNTHETIC_DATA_SIZE=1500 python3 main.py
```

## ⚠️ 常见问题

### 问题 1: Yahoo Finance 无法获取加密货币数据

**错误信息**：
```
No data returned for symbol BTC-USD. Check if symbol is correct.
```

**解决方案**：
1. **使用 Binance（推荐）**：
   ```bash
   DATA_SOURCE=binance MARKET_SYMBOL=BTC/USDT MARKET_TIMEFRAME=1h python3 main.py
   ```

2. **使用股票数据测试**：
   ```bash
   DATA_SOURCE=yahoo MARKET_SYMBOL=AAPL MARKET_PERIOD=3mo MARKET_INTERVAL=1d python3 main.py
   ```

3. **使用合成数据**：
   ```bash
   DATA_SOURCE=synthetic python3 main.py
   ```

### 问题 2: 数据获取失败

**可能原因**：
1. 网络连接问题
2. 符号格式错误
3. 时间周期太长
4. 时间间隔不支持

**解决方案**：
1. 检查网络连接
2. 验证符号格式
3. 缩短时间周期（如从 `1y` 改为 `3mo`）
4. 更改时间间隔（如从 `1h` 改为 `1d`）

### 问题 3: Binance 数据获取失败

**可能原因**：
1. 网络连接问题
2. 交易对格式错误
3. 时间框架不支持

**解决方案**：
1. 检查网络连接
2. 确保交易对格式正确（如 `BTC/USDT` 而不是 `BTC-USD`）
3. 使用支持的时间框架

## 🎯 推荐配置

### 股票交易（Yahoo Finance）
```bash
DATA_SOURCE=yahoo \
MARKET_SYMBOL=AAPL \
MARKET_PERIOD=6mo \
MARKET_INTERVAL=1d \
python3 main.py
```

### 加密货币交易（Binance）
```bash
DATA_SOURCE=binance \
MARKET_SYMBOL=BTC/USDT \
MARKET_TIMEFRAME=1h \
MARKET_LIMIT=1000 \
python3 main.py
```

### 快速测试（合成数据）
```bash
DATA_SOURCE=synthetic \
SYNTHETIC_DATA_SIZE=500 \
USE_LLM=False \
python3 main.py
```

## 📝 环境变量配置

在 `.env` 文件中设置：

```env
# 数据源配置
DATA_SOURCE=yahoo
MARKET_SYMBOL=AAPL
MARKET_PERIOD=3mo
MARKET_INTERVAL=1d

# 或使用 Binance
DATA_SOURCE=binance
MARKET_SYMBOL=BTC/USDT
MARKET_TIMEFRAME=1h
MARKET_LIMIT=1000
```

## 🔍 数据验证

系统会自动验证获取的数据：
- 检查必需列是否存在
- 检查数据完整性
- 修复常见数据问题（缺失值、异常值等）

如果数据验证失败，系统会尝试自动修复，并在日志中记录。

