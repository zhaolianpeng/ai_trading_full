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
SIGNAL_LOOKBACK_DAYS = int(os.getenv('SIGNAL_LOOKBACK_DAYS', '7'))  # 信号倒推天数（默认7天，即1周）
SYNTHETIC_DATA_SIZE = int(os.getenv('SYNTHETIC_DATA_SIZE', '1500'))  # 合成数据大小
USE_ADVANCED_TA = os.getenv('USE_ADVANCED_TA', 'True').lower() == 'true'  # 是否使用高级技术指标
USE_ERIC_INDICATORS = os.getenv('USE_ERIC_INDICATORS', 'True').lower() == 'true'  # 是否使用 Eric 策略指标

# ==================== LLM 配置 ====================
USE_LLM = os.getenv('USE_LLM', 'True').lower() == 'true'  # 是否启用 LLM 分析
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')  # LLM 提供商: 'openai' 或 'deepseek'
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')  # OpenAI 模型（如 'gpt-4o-mini', 'gpt-4o'）
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner')  # DeepSeek 模型（如 'deepseek-chat', 'deepseek-reasoner'）
OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.0'))  # LLM 温度参数
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '400'))  # LLM 最大token数
LLM_CONCURRENT_WORKERS = int(os.getenv('LLM_CONCURRENT_WORKERS', '5'))  # LLM并发处理线程数（默认5，可根据API限制调整）

# ==================== 回测配置 ====================
BACKTEST_MAX_HOLD = int(os.getenv('BACKTEST_MAX_HOLD', '20'))  # 最大持仓周期
BACKTEST_ATR_STOP_MULT = float(os.getenv('BACKTEST_ATR_STOP_MULT', '1.0'))  # 止损 ATR 倍数
BACKTEST_ATR_TARGET_MULT = float(os.getenv('BACKTEST_ATR_TARGET_MULT', '2.0'))  # 止盈 ATR 倍数
BACKTEST_PARTIAL_TP_RATIO = float(os.getenv('BACKTEST_PARTIAL_TP_RATIO', '0.5'))  # 部分止盈比例（0-1，0表示不使用部分止盈）
BACKTEST_PARTIAL_TP_MULT = float(os.getenv('BACKTEST_PARTIAL_TP_MULT', '1.0'))  # 部分止盈 ATR 倍数
MIN_LLM_SCORE = int(os.getenv('MIN_LLM_SCORE', '40'))  # LLM 评分最低阈值
MIN_RISK_REWARD = float(os.getenv('MIN_RISK_REWARD', '1.5'))  # 最小盈亏比要求
MIN_QUALITY_SCORE = int(os.getenv('MIN_QUALITY_SCORE', '50'))  # 最小质量评分
MIN_CONFIRMATIONS = int(os.getenv('MIN_CONFIRMATIONS', '2'))  # 最小确认数量
USE_SIGNAL_FILTER = os.getenv('USE_SIGNAL_FILTER', 'True').lower() == 'true'  # 是否使用信号过滤器

# ==================== 短单交易配置 ====================
TRADING_MODE = os.getenv('TRADING_MODE', 'normal')  # 交易模式: 'normal', 'scalping', 'swing'
# scalping: 高频短单，降低所有阈值，适合小时级及以下
# swing: 波段交易，提高阈值，适合日线级
# normal: 标准模式

# ==================== 高频交易配置 ====================
USE_HIGH_FREQUENCY = os.getenv('USE_HIGH_FREQUENCY', 'True').lower() == 'true'  # 是否启用高频交易策略
HF_MIN_CONSECUTIVE_OVERBOUGHT = int(os.getenv('HF_MIN_CONSECUTIVE_OVERBOUGHT', '3'))  # 最小连续超买次数（日线/4小时）
HF_MIN_CONSECUTIVE_OVERSOLD = int(os.getenv('HF_MIN_CONSECUTIVE_OVERSOLD', '3'))  # 最小连续超卖次数（日线/4小时）
ALLOW_MULTIPLE_TRADES_PER_DAY = os.getenv('ALLOW_MULTIPLE_TRADES_PER_DAY', 'True').lower() == 'true'  # 是否允许一天多次交易

# ==================== 合约交易配置 ====================
FUTURES_LEVERAGE = int(os.getenv('FUTURES_LEVERAGE', '3'))  # 杠杆倍数（默认3倍，建议1-10倍）
FUTURES_RISK_PER_TRADE = float(os.getenv('FUTURES_RISK_PER_TRADE', '0.02'))  # 每笔交易风险比例（默认2%）
FUTURES_MARGIN_RATE = float(os.getenv('FUTURES_MARGIN_RATE', '0.01'))  # 保证金率（默认1%，即100倍杠杆需要1%保证金）
FUTURES_MAINTENANCE_MARGIN_RATE = float(os.getenv('FUTURES_MAINTENANCE_MARGIN_RATE', '0.5'))  # 维持保证金率（默认50%）
FUTURES_MIN_STOP_PCT = float(os.getenv('FUTURES_MIN_STOP_PCT', '0.005'))  # 最小止损百分比（默认0.5%）
FUTURES_MAX_STOP_PCT = float(os.getenv('FUTURES_MAX_STOP_PCT', '0.05'))  # 最大止损百分比（默认5%）
FUTURES_MAX_PROFIT_PCT = float(os.getenv('FUTURES_MAX_PROFIT_PCT', '0.20'))  # 最大止盈百分比（默认20%）
FUTURES_USE_ENHANCED_STRATEGY = os.getenv('FUTURES_USE_ENHANCED_STRATEGY', 'True').lower() == 'true'  # 是否使用增强的合约交易策略

# ==================== 日志配置 ====================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = os.getenv('LOG_FILE', 'trading.log')  # 日志文件路径，None 表示不写入文件

# ==================== 输出配置 ====================
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '.')  # 输出文件目录
