# 使用示例

## 📊 获取真实市场数据示例

### 示例1: 获取比特币数据（Yahoo Finance）

```bash
DATA_SOURCE=yahoo \
MARKET_SYMBOL=BTC-USD \
MARKET_PERIOD=1y \
MARKET_INTERVAL=1h \
USE_LLM=False \
python3 main.py
```

**结果**：
- 获取最近1年的比特币小时数据
- 自动进行信号检测和回测
- 生成图表和报告

### 示例2: 获取苹果股票数据

```bash
DATA_SOURCE=yahoo \
MARKET_SYMBOL=AAPL \
MARKET_PERIOD=6mo \
MARKET_INTERVAL=1d \
USE_LLM=False \
python3 main.py
```

### 示例3: 获取以太坊数据（Binance）

```bash
DATA_SOURCE=binance \
MARKET_SYMBOL=ETH/USDT \
MARKET_TIMEFRAME=1h \
MARKET_LIMIT=1000 \
USE_LLM=False \
python3 main.py
```

### 示例4: 使用 .env 文件配置

创建 `.env` 文件：

```env
# 数据源配置
DATA_SOURCE=yahoo
MARKET_SYMBOL=BTC-USD
MARKET_PERIOD=1y
MARKET_INTERVAL=1h

# LLM 配置（可选）
USE_LLM=True
OPENAI_API_KEY=sk-your-key-here

# 其他配置
USE_ADVANCED_TA=True
BACKTEST_MAX_HOLD=20
```

然后直接运行：

```bash
python3 main.py
```

## 🎯 不同资产类型的配置

### 加密货币（推荐使用 Yahoo Finance）

```bash
# 比特币
DATA_SOURCE=yahoo MARKET_SYMBOL=BTC-USD MARKET_PERIOD=1y MARKET_INTERVAL=1h python3 main.py

# 以太坊
DATA_SOURCE=yahoo MARKET_SYMBOL=ETH-USD MARKET_PERIOD=6mo MARKET_INTERVAL=1h python3 main.py

# Solana
DATA_SOURCE=yahoo MARKET_SYMBOL=SOL-USD MARKET_PERIOD=3mo MARKET_INTERVAL=1h python3 main.py
```

### 美股

```bash
# 苹果
DATA_SOURCE=yahoo MARKET_SYMBOL=AAPL MARKET_PERIOD=1y MARKET_INTERVAL=1d python3 main.py

# 特斯拉
DATA_SOURCE=yahoo MARKET_SYMBOL=TSLA MARKET_PERIOD=6mo MARKET_INTERVAL=1d python3 main.py

# 英伟达
DATA_SOURCE=yahoo MARKET_SYMBOL=NVDA MARKET_PERIOD=1y MARKET_INTERVAL=1d python3 main.py
```

### 指数

```bash
# S&P 500
DATA_SOURCE=yahoo MARKET_SYMBOL=^GSPC MARKET_PERIOD=1y MARKET_INTERVAL=1d python3 main.py

# NASDAQ
DATA_SOURCE=yahoo MARKET_SYMBOL=^IXIC MARKET_PERIOD=1y MARKET_INTERVAL=1d python3 main.py
```

## ⚡ 快速测试命令

### 快速测试（不使用 LLM，使用真实数据）

```bash
DATA_SOURCE=yahoo MARKET_SYMBOL=BTC-USD MARKET_PERIOD=3mo MARKET_INTERVAL=1h USE_LLM=False python3 main.py
```

### 快速测试（使用合成数据）

```bash
DATA_SOURCE=synthetic SYNTHETIC_DATA_SIZE=1500 USE_LLM=False python3 main.py
```

## 📈 不同时间周期的配置

### 短期交易（日内）

```bash
DATA_SOURCE=yahoo \
MARKET_SYMBOL=BTC-USD \
MARKET_PERIOD=1mo \
MARKET_INTERVAL=15m \
python3 main.py
```

### 中期交易（几天到几周）

```bash
DATA_SOURCE=yahoo \
MARKET_SYMBOL=BTC-USD \
MARKET_PERIOD=3mo \
MARKET_INTERVAL=1h \
python3 main.py
```

### 长期交易（几周到几个月）

```bash
DATA_SOURCE=yahoo \
MARKET_SYMBOL=BTC-USD \
MARKET_PERIOD=1y \
MARKET_INTERVAL=1d \
python3 main.py
```

## 🔧 高级配置示例

### 完整配置示例

```bash
# 数据源
DATA_SOURCE=yahoo
MARKET_SYMBOL=BTC-USD
MARKET_PERIOD=1y
MARKET_INTERVAL=1h

# LLM 配置
USE_LLM=True
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.0
OPENAI_MAX_TOKENS=400

# 技术指标
USE_ADVANCED_TA=True

# 回测配置
BACKTEST_MAX_HOLD=20
BACKTEST_ATR_STOP_MULT=1.0
BACKTEST_ATR_TARGET_MULT=2.0
MIN_LLM_SCORE=40

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=trading.log

# 输出配置
OUTPUT_DIR=./output

python3 main.py
```

## 💡 提示

1. **首次使用**: 建议先用 `USE_LLM=False` 测试，确认数据获取和信号检测正常
2. **数据量**: 建议使用 1000-3000 条数据，太少信号不足，太多计算缓慢
3. **时间周期**: 根据你的交易策略选择合适的时间周期
4. **网络**: 确保网络连接正常，Yahoo Finance 需要访问外网
5. **缓存**: 可以考虑将获取的数据保存为 CSV，避免重复下载

