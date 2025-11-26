# config.py — 全局配置
import os
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# ==================== 数据配置 ====================
DATA_SOURCE = os.getenv('DATA_SOURCE', 'synthetic')  # 数据源: 'synthetic', 'csv', 'yahoo', 'binance'
DATA_PATH = os.getenv('DATA_PATH', None)  # CSV文件路径（当DATA_SOURCE='csv'时使用）
MARKET_SYMBOL = os.getenv('MARKET_SYMBOL', 'BTC-USD')  # 市场交易对（当DATA_SOURCE='yahoo'或'binance'时使用）
MARKET_PERIOD = os.getenv('MARKET_PERIOD', '1y')  # 数据周期（Yahoo Finance: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'）
MARKET_INTERVAL = os.getenv('MARKET_INTERVAL', '1h')  # 数据间隔（Yahoo Finance: '1m', '5m', '15m', '30m', '1h', '1d'等）
MARKET_TIMEFRAME = os.getenv('MARKET_TIMEFRAME', '1h')  # 时间框架（Binance: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'）
MARKET_LIMIT = int(os.getenv('MARKET_LIMIT', '1000'))  # 最大数据条数（Binance，最大1000）
SYNTHETIC_DATA_SIZE = int(os.getenv('SYNTHETIC_DATA_SIZE', '1500'))  # 合成数据大小
USE_ADVANCED_TA = os.getenv('USE_ADVANCED_TA', 'True').lower() == 'true'  # 是否使用高级技术指标
USE_ERIC_INDICATORS = os.getenv('USE_ERIC_INDICATORS', 'True').lower() == 'true'  # 是否使用 Eric 策略指标

# ==================== LLM 配置 ====================
USE_LLM = os.getenv('USE_LLM', 'True').lower() == 'true'  # 是否启用 LLM 分析
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')  # 根据你账户可用模型修改
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.0'))  # LLM 温度参数
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '400'))  # LLM 最大token数

# ==================== 回测配置 ====================
BACKTEST_MAX_HOLD = int(os.getenv('BACKTEST_MAX_HOLD', '20'))  # 最大持仓周期
BACKTEST_ATR_STOP_MULT = float(os.getenv('BACKTEST_ATR_STOP_MULT', '1.0'))  # 止损 ATR 倍数
BACKTEST_ATR_TARGET_MULT = float(os.getenv('BACKTEST_ATR_TARGET_MULT', '2.0'))  # 止盈 ATR 倍数
MIN_LLM_SCORE = int(os.getenv('MIN_LLM_SCORE', '40'))  # LLM 评分最低阈值

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = os.getenv('LOG_FILE', 'trading.log')  # 日志文件路径，None 表示不写入文件

# ==================== 输出配置 ====================
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '.')  # 输出文件目录
