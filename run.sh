#!/bin/bash
# 快速运行脚本
# 使用方法: ./run.sh 或 bash run.sh

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查 main.py 是否存在
if [ ! -f "main.py" ]; then
    echo "错误: 找不到 main.py 文件"
    echo "请确保在项目根目录下运行此脚本"
    exit 1
fi

# 设置默认参数（如果未设置）
export DATA_SOURCE=${DATA_SOURCE:-yahoo}
export MARKET_SYMBOL=${MARKET_SYMBOL:-BTC-USD}
export MARKET_PERIOD=${MARKET_PERIOD:-1y}
export MARKET_INTERVAL=${MARKET_INTERVAL:-1h}
export USE_LLM=${USE_LLM:-False}

echo "=========================================="
echo "AI Trading System"
echo "=========================================="
echo "项目目录: $SCRIPT_DIR"
echo "数据源: $DATA_SOURCE"
echo "交易对: $MARKET_SYMBOL"
echo "=========================================="
echo ""

# 运行主程序
python3 main.py

