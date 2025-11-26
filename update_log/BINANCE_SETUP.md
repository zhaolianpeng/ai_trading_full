# Binance 永续合约数据配置指南

## 🔍 价格问题说明

### 问题：价格不匹配

**现象**：当前 BTC/USDT 永续价格是 87780，但查询出来的数据是 114926

**原因分析**：
1. **市场类型错误**：可能使用了现货（spot）而不是永续合约（future）
2. **历史数据**：获取的是历史数据，不是最新价格
3. **数据源问题**：Binance 现货和永续价格不同

### 解决方案

#### 1. 使用永续合约数据（推荐）

```bash
DATA_SOURCE=binance \
MARKET_SYMBOL=BTC/USDT \
MARKET_TIMEFRAME=1h \
MARKET_TYPE=future \
python3 main.py
```

**关键配置**：
- `MARKET_TYPE=future` - 使用永续合约市场
- 默认情况下，如果符号包含 `USDT`，会自动使用 `future`

#### 2. 验证当前价格

运行程序后，查看日志中的最新价格：
```
最新价格: $87712.80 (时间: 2025-11-26 07:00:00)
```

如果价格接近当前市场价格（87780），说明数据正确。

#### 3. 价格差异的原因

- **时间差**：数据是整点数据，可能有几分钟延迟
- **市场类型**：现货和永续价格可能略有差异
- **数据延迟**：Binance API 返回的数据可能有轻微延迟

## 📊 倒推1周信号功能

### 功能说明

系统现在支持**倒推指定天数内的交易信号**，默认倒推7天（1周）。

### 使用方法

```bash
# 倒推7天内的信号（默认）
SIGNAL_LOOKBACK_DAYS=7 python3 main.py

# 倒推3天内的信号
SIGNAL_LOOKBACK_DAYS=3 python3 main.py

# 倒推14天内的信号（2周）
SIGNAL_LOOKBACK_DAYS=14 python3 main.py
```

### 工作原理

1. **时间索引检测**：如果数据有时间索引，按时间倒推
2. **数据条数估算**：如果没有时间索引，按数据条数估算
   - 小时级数据：1周 = 7 * 24 = 168 条
   - 日线级数据：1周 = 7 条

3. **信号检测**：只分析最近 N 天的数据
4. **索引映射**：将信号索引映射回完整数据集的索引

### 日志输出示例

```
倒推 7 天内的交易信号（从 2025-11-19 07:00:00 到 2025-11-26 07:00:00）
数据范围: 200 条，将分析最近 169 条数据
发现 10 个原始信号（在最近 7 天内）
```

## 🚀 完整配置示例

### Binance 永续合约 + 小时级 + 倒推1周

```bash
DATA_SOURCE=binance \
MARKET_SYMBOL=BTC/USDT \
MARKET_TIMEFRAME=1h \
MARKET_LIMIT=200 \
MARKET_TYPE=future \
SIGNAL_LOOKBACK_DAYS=7 \
TRADING_MODE=scalping \
USE_LLM=False \
USE_ERIC_INDICATORS=True \
python3 main.py
```

### .env 文件配置

```env
# 数据源配置
DATA_SOURCE=binance
MARKET_SYMBOL=BTC/USDT
MARKET_TIMEFRAME=1h
MARKET_LIMIT=200
MARKET_TYPE=future

# 信号倒推配置
SIGNAL_LOOKBACK_DAYS=7

# 交易模式
TRADING_MODE=scalping
MIN_QUALITY_SCORE=20
MIN_CONFIRMATIONS=1
MIN_LLM_SCORE=30
MIN_RISK_REWARD=1.2
```

## 📈 数据验证

### 检查获取的数据

运行程序后，查看日志：

```
使用 Binance future 市场获取数据
Fetched 200 rows from Binance (future)
Date range: 2025-11-18 00:00:00 to 2025-11-26 07:00:00
Price range: $82188.70 - $93447.30
最新价格: $87712.80 (时间: 2025-11-26 07:00:00)
```

**验证点**：
1. ✅ 市场类型：`future`（永续合约）
2. ✅ 最新价格：接近当前市场价格
3. ✅ 时间范围：包含最近的数据
4. ✅ 数据条数：足够进行分析

## ⚠️ 常见问题

### 问题 1: 价格仍然不匹配

**可能原因**：
1. 使用了现货市场而不是永续合约
2. 数据延迟
3. 时间框架不同

**解决方案**：
```bash
# 明确指定使用永续合约
MARKET_TYPE=future python3 main.py

# 增加数据量，获取更多历史数据
MARKET_LIMIT=1000 python3 main.py
```

### 问题 2: 倒推功能不工作

**检查**：
1. 数据是否有时间索引
2. `SIGNAL_LOOKBACK_DAYS` 是否设置正确
3. 查看日志中的倒推信息

### 问题 3: 信号太少

**解决方案**：
1. 降低质量评分阈值：`MIN_QUALITY_SCORE=20`
2. 降低确认数量：`MIN_CONFIRMATIONS=1`
3. 使用短单模式：`TRADING_MODE=scalping`
4. 增加倒推天数：`SIGNAL_LOOKBACK_DAYS=14`

## 💡 最佳实践

1. **使用永续合约**：对于 BTC/USDT 等，使用 `MARKET_TYPE=future`
2. **验证价格**：运行后检查日志中的最新价格
3. **合理设置倒推天数**：
   - 小时级数据：7-14 天
   - 日线级数据：30-90 天
4. **监控数据质量**：确保数据完整且价格合理

