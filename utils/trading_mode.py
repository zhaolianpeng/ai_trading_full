# utils/trading_mode.py
"""
交易模式配置 - 根据交易模式自动调整参数
支持：normal（标准）、scalping（高频短单）、swing（波段）
"""
import os
from utils.logger import logger

def get_trading_mode_config(trading_mode='normal', data_interval='1h'):
    """
    根据交易模式和数据间隔获取配置参数
    
    Args:
        trading_mode: 交易模式 ('normal', 'scalping', 'swing')
        data_interval: 数据间隔 ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
    
    Returns:
        配置字典
    """
    # 判断是否为小时级或更短的时间框架
    is_short_timeframe = any(x in data_interval.lower() for x in ['1m', '5m', '15m', '30m', '1h'])
    
    if trading_mode == 'scalping' or (trading_mode == 'normal' and is_short_timeframe):
        # 高频短单模式：降低阈值，增加交易频率
        logger.info("使用高频短单模式（适合小时级及以下时间框架）")
        config = {
            'min_quality_score': 30,      # 降低质量评分要求（从50降到30）
            'min_confirmations': 1,       # 降低确认数量（从2降到1）
            'min_llm_score': 30,         # 降低LLM评分要求（从40降到30）
            'min_risk_reward': 1.2,       # 降低盈亏比要求（从1.5降到1.2）
            'max_hold': 10,               # 缩短持仓周期（从20降到10）
            'atr_stop_mult': 1.5,         # 放宽止损（从0.8提高到1.5），避免被短期波动触发
            'atr_target_mult': 1.5,       # 降低止盈目标（从2.0降到1.5）
            'partial_tp_mult': 0.8,      # 降低部分止盈目标
            'volume_threshold': 1.1,      # 降低成交量要求
        }
    elif trading_mode == 'swing':
        # 波段交易模式：提高阈值，减少交易频率
        logger.info("使用波段交易模式（适合日线级时间框架）")
        config = {
            'min_quality_score': 60,
            'min_confirmations': 3,
            'min_llm_score': 50,
            'min_risk_reward': 2.0,
            'max_hold': 30,
            'atr_stop_mult': 1.2,
            'atr_target_mult': 2.5,
            'partial_tp_mult': 1.2,
            'volume_threshold': 1.3,
        }
    else:
        # 标准模式
        logger.info("使用标准交易模式")
        config = {
            'min_quality_score': 50,
            'min_confirmations': 2,
            'min_llm_score': 40,
            'min_risk_reward': 1.5,
            'max_hold': 20,
            'atr_stop_mult': 1.5,         # 放宽止损（从1.0提高到1.5），避免被短期波动触发
            'atr_target_mult': 2.0,
            'partial_tp_mult': 1.0,
            'volume_threshold': 1.2,
        }
    
    return config

def apply_trading_mode_config():
    """
    应用交易模式配置到环境变量（如果未设置）
    """
    trading_mode = os.getenv('TRADING_MODE', 'normal')
    data_interval = os.getenv('MARKET_INTERVAL', os.getenv('MARKET_TIMEFRAME', '1h'))
    
    config = get_trading_mode_config(trading_mode, data_interval)
    
    # 如果环境变量未设置，则设置默认值
    if 'MIN_QUALITY_SCORE' not in os.environ:
        os.environ['MIN_QUALITY_SCORE'] = str(config['min_quality_score'])
    if 'MIN_CONFIRMATIONS' not in os.environ:
        os.environ['MIN_CONFIRMATIONS'] = str(config['min_confirmations'])
    if 'MIN_LLM_SCORE' not in os.environ:
        os.environ['MIN_LLM_SCORE'] = str(config['min_llm_score'])
    if 'MIN_RISK_REWARD' not in os.environ:
        os.environ['MIN_RISK_REWARD'] = str(config['min_risk_reward'])
    if 'BACKTEST_MAX_HOLD' not in os.environ:
        os.environ['BACKTEST_MAX_HOLD'] = str(config['max_hold'])
    if 'BACKTEST_ATR_STOP_MULT' not in os.environ:
        os.environ['BACKTEST_ATR_STOP_MULT'] = str(config['atr_stop_mult'])
    if 'BACKTEST_ATR_TARGET_MULT' not in os.environ:
        os.environ['BACKTEST_ATR_TARGET_MULT'] = str(config['atr_target_mult'])
    if 'BACKTEST_PARTIAL_TP_MULT' not in os.environ:
        os.environ['BACKTEST_PARTIAL_TP_MULT'] = str(config['partial_tp_mult'])
    
    return config

