# 市场数据获取指南

## 📊 支持的数据源

### 1. Yahoo Finance（推荐）⭐

**优点**：
- 免费，无需 API Key
- 支持股票、加密货币、ETF、指数等
- 数据质量高，更新及时
- 支持多种时间周期和间隔

**支持的资产类型**：
- **加密货币**: `BTC-USD`, `ETH-USD`, `BNB-USD` 等
- **股票**: `AAPL`, `GOOGL`, `MSFT`, `TSLA`, `AMZN`, `NVDA` 等
- **指数**: `^GSPC` (S&P 500), `^DJI` (Dow Jones), `^IXIC` (NASDAQ) 等

**配置示例**：
```bash
DATA_SOURCE=yahoo
MARKET_SYMBOL=BTC-USD
MARKET_PERIOD=1y
MARKET_INTERVAL=1h
```

**时间周期选项**：
- `1d` - 1天
- `5d` - 5天
- `1mo` - 1个月
- `3mo` - 3个月
- `6mo` - 6个月
- `1y` - 1年（推荐）
- `2y` - 2年
- `5y` - 5年
- `10y` - 10年
- `ytd` - 年初至今
- `max` - 全部历史数据

**数据间隔选项**：
- `1m`, `2m`, `5m`, `15m`, `30m` - 分钟级
- `60m`, `90m`, `1h` - 小时级（推荐）
- `1d`, `5d` - 日级
- `1wk`, `1mo`, `3mo` - 周/月级

### 2. Binance（加密货币专用）

**优点**：
- 专业的加密货币交易所数据
- 数据精度高
- 支持实时数据

**配置示例**：
```bash
DATA_SOURCE=binance
MARKET_SYMBOL=BTC/USDT
MARKET_TIMEFRAME=1h
MARKET_LIMIT=1000
```

**时间框架选项**：
- `1m`, `5m`, `15m`, `30m` - 分钟级
- `1h`, `4h` - 小时级（推荐）
- `1d`, `1w` - 日/周级

**注意**：`MARKET_LIMIT` 最大为 1000 条

### 3. 本地 CSV 文件

**配置示例**：
```bash
DATA_SOURCE=csv
DATA_PATH=data/historical_data.csv
```

CSV 文件必须包含以下列：
- `datetime` - 日期时间（会被用作索引）
- `open`, `high`, `low`, `close` - OHLC 价格数据
- `volume` - 成交量

### 4. 合成数据（测试用）

**配置示例**：
```bash
DATA_SOURCE=synthetic
SYNTHETIC_DATA_SIZE=1500
```

## 🚀 快速开始

### 获取比特币数据（Yahoo Finance）

```bash
# 方式1: 使用环境变量
export DATA_SOURCE=yahoo
export MARKET_SYMBOL=BTC-USD
export MARKET_PERIOD=1y
export MARKET_INTERVAL=1h
python3 main.py

# 方式2: 使用 .env 文件
# 编辑 .env 文件，设置：
# DATA_SOURCE=yahoo
# MARKET_SYMBOL=BTC-USD
# MARKET_PERIOD=1y
# MARKET_INTERVAL=1h
python3 main.py
```

### 获取股票数据（Yahoo Finance）

```bash
export DATA_SOURCE=yahoo
export MARKET_SYMBOL=AAPL
export MARKET_PERIOD=6mo
export MARKET_INTERVAL=1d
python3 main.py
```

### 获取 Binance 数据

```bash
export DATA_SOURCE=binance
export MARKET_SYMBOL=BTC/USDT
export MARKET_TIMEFRAME=1h
export MARKET_LIMIT=1000
python3 main.py
```

## 📋 常用交易对符号

### 加密货币（Yahoo Finance）
- `BTC-USD` - 比特币
- `ETH-USD` - 以太坊
- `BNB-USD` - 币安币
- `ADA-USD` - 卡尔达诺
- `SOL-USD` - Solana
- `XRP-USD` - 瑞波币

### 加密货币（Binance）
- `BTC/USDT` - 比特币/USDT
- `ETH/USDT` - 以太坊/USDT
- `BNB/USDT` - 币安币/USDT
- `ADA/USDT` - 卡尔达诺/USDT

### 美股
- `AAPL` - 苹果
- `GOOGL` - 谷歌
- `MSFT` - 微软
- `TSLA` - 特斯拉
- `AMZN` - 亚马逊
- `NVDA` - 英伟达
- `META` - Meta (Facebook)
- `NFLX` - Netflix

### 指数
- `^GSPC` - S&P 500
- `^DJI` - 道琼斯工业平均指数
- `^IXIC` - NASDAQ 综合指数

## ⚙️ 配置建议

### 对于加密货币交易
```bash
DATA_SOURCE=yahoo
MARKET_SYMBOL=BTC-USD
MARKET_PERIOD=1y      # 1年数据
MARKET_INTERVAL=1h     # 1小时K线
```

### 对于股票交易
```bash
DATA_SOURCE=yahoo
MARKET_SYMBOL=AAPL
MARKET_PERIOD=6mo      # 6个月数据
MARKET_INTERVAL=1d     # 日K线
```

### 对于短期交易
```bash
DATA_SOURCE=yahoo
MARKET_SYMBOL=BTC-USD
MARKET_PERIOD=1mo      # 1个月数据
MARKET_INTERVAL=15m    # 15分钟K线
```

## 🔧 故障排除

### 问题1: 无法获取数据
- 检查网络连接
- 确认交易对符号正确
- 对于 Yahoo Finance，某些符号可能需要使用不同的格式

### 问题2: 数据为空
- 检查时间周期和间隔的组合是否有效
- 某些资产可能没有足够的历史数据
- 尝试使用更长的时间周期

### 问题3: 数据获取缓慢
- Yahoo Finance 有时会限流，稍等片刻重试
- 减少数据量（使用更短的时间周期）
- 使用 Binance 可能更快（但仅限加密货币）

## 📝 注意事项

1. **数据延迟**: Yahoo Finance 和 Binance 的数据可能有 15-20 分钟的延迟
2. **API 限制**: Yahoo Finance 虽然没有官方限制，但频繁请求可能被限流
3. **数据质量**: 建议使用 Yahoo Finance 获取股票数据，Binance 获取加密货币数据
4. **时区**: 所有数据使用 UTC 时区
5. **数据验证**: 系统会自动验证和修复数据问题

## 🎯 最佳实践

1. **首次使用**: 先用合成数据测试系统是否正常工作
2. **数据量**: 建议使用 1000-2000 条数据，太少可能信号不足，太多可能计算缓慢
3. **时间周期**: 根据交易策略选择合适的时间周期
4. **缓存**: 可以考虑将获取的数据保存为 CSV，避免重复下载

