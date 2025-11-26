# 快速开始指南

## ⚠️ 重要提示

**运行前请确保在项目根目录下！**

```bash
# 切换到项目目录
cd /Users/zhaolianpeng/code/Goproject/src/ai_trading_full

# 或者使用相对路径
cd path/to/ai_trading_full
```

## 🚀 最简单的使用方式

### 方式0: 使用运行脚本（最简单）⭐

```bash
# 1. 切换到项目目录
cd /Users/zhaolianpeng/code/Goproject/src/ai_trading_full

# 2. 运行脚本（会自动切换到正确目录）
./run.sh

# 或者使用自定义参数
DATA_SOURCE=yahoo MARKET_SYMBOL=BTC-USD ./run.sh
```

### 方式1: 使用真实市场数据（推荐）

```bash
# 1. 先切换到项目目录
cd /Users/zhaolianpeng/code/Goproject/src/ai_trading_full

# 2. 获取比特币数据并运行回测
DATA_SOURCE=yahoo \
MARKET_SYMBOL=BTC-USD \
MARKET_PERIOD=1y \
MARKET_INTERVAL=1h \
USE_LLM=False \
python3 main.py
```

### 方式2: 使用合成数据（快速测试）

```bash
# 1. 先切换到项目目录
cd /Users/zhaolianpeng/code/Goproject/src/ai_trading_full

# 2. 使用合成数据，快速测试系统
DATA_SOURCE=synthetic \
USE_LLM=False \
python3 main.py
```

### 方式3: 使用 .env 文件

1. 切换到项目目录：
```bash
cd /Users/zhaolianpeng/code/Goproject/src/ai_trading_full
```

2. 创建 `.env` 文件：
```bash
cat > .env << EOF
DATA_SOURCE=yahoo
MARKET_SYMBOL=BTC-USD
MARKET_PERIOD=1y
MARKET_INTERVAL=1h
USE_LLM=False
EOF
```

3. 运行：
```bash
python3 main.py
```

## 📋 常用命令

### 获取不同资产的数据

**注意**: 以下所有命令都需要先在项目目录下执行 `cd` 命令！

```bash
# 先切换到项目目录
cd /Users/zhaolianpeng/code/Goproject/src/ai_trading_full

# 比特币
DATA_SOURCE=yahoo MARKET_SYMBOL=BTC-USD MARKET_PERIOD=1y MARKET_INTERVAL=1h python3 main.py

# 以太坊
DATA_SOURCE=yahoo MARKET_SYMBOL=ETH-USD MARKET_PERIOD=6mo MARKET_INTERVAL=1h python3 main.py

# 苹果股票
DATA_SOURCE=yahoo MARKET_SYMBOL=AAPL MARKET_PERIOD=1y MARKET_INTERVAL=1d python3 main.py

# 特斯拉
DATA_SOURCE=yahoo MARKET_SYMBOL=TSLA MARKET_PERIOD=6mo MARKET_INTERVAL=1d python3 main.py
```

## ⚠️ 注意事项

1. **Python 版本**: 确保使用 `python3` 而不是 `python`
2. **网络连接**: 获取线上数据需要网络连接
3. **首次运行**: 建议先用 `USE_LLM=False` 测试
4. **数据量**: 建议使用 1000-3000 条数据

## 🔧 如果遇到问题

### 问题1: `python: command not found`
**解决**: 使用 `python3` 替代 `python`

### 问题2: `can't open file 'main.py': [Errno 2] No such file or directory`
**解决**: 
```bash
# 确保在项目根目录下
cd /Users/zhaolianpeng/code/Goproject/src/ai_trading_full
# 或者使用你的实际路径
cd path/to/ai_trading_full

# 然后运行
python3 main.py
```

### 问题3: 无法获取数据
**解决**: 
- 检查网络连接
- 确认交易对符号正确
- 尝试使用更短的时间周期

### 问题3: 权限错误
**解决**: 
- 系统会自动处理，只使用控制台输出
- 或者设置 `LOG_FILE=` 禁用文件日志

## 📊 输出文件

运行成功后会生成：
- `trading_chart.png` - 价格图表
- `backtest_results.png` - 回测结果图表
- `analysis_report.txt` - 分析报告
- `trades.csv` - 交易记录
- `signals_log.json` - 信号日志
- `sample_data.csv` - 处理后的数据

