# utils/config_validator.py
"""
配置验证工具
"""
import os
from typing import List, Tuple
from utils.logger import logger

def validate_config() -> Tuple[bool, List[str]]:
    """
    验证配置是否有效
    
    Returns:
        (is_valid, error_messages): 是否有效和错误消息列表
    """
    errors = []
    
    # 检查 LLM 配置
    use_llm = os.getenv('USE_LLM', 'True').lower() == 'true'
    if use_llm:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("USE_LLM is True but OPENAI_API_KEY is not set. Disabling LLM features.")
            os.environ['USE_LLM'] = 'False'  # 自动禁用
        elif not api_key.startswith('sk-'):
            logger.warning("OPENAI_API_KEY format seems invalid. Disabling LLM features.")
            os.environ['USE_LLM'] = 'False'  # 自动禁用
    
    # 检查数据源配置
    data_source = os.getenv('DATA_SOURCE', 'synthetic')
    
    if data_source == 'csv':
        data_path = os.getenv('DATA_PATH')
        if not data_path:
            errors.append("DATA_SOURCE is 'csv' but DATA_PATH is not set")
        elif not os.path.exists(data_path):
            errors.append(f"DATA_PATH specified but file does not exist: {data_path}")
    elif data_source == 'yahoo':
        symbol = os.getenv('MARKET_SYMBOL', 'BTC-USD')
        if not symbol:
            errors.append("DATA_SOURCE is 'yahoo' but MARKET_SYMBOL is not set")
    elif data_source == 'binance':
        symbol = os.getenv('MARKET_SYMBOL', 'BTC/USDT')
        if not symbol:
            errors.append("DATA_SOURCE is 'binance' but MARKET_SYMBOL is not set")
    
    # 检查数值配置范围
    try:
        synthetic_size = int(os.getenv('SYNTHETIC_DATA_SIZE', '1500'))
        if synthetic_size < 100:
            errors.append("SYNTHETIC_DATA_SIZE should be at least 100")
        if synthetic_size > 100000:
            errors.append("SYNTHETIC_DATA_SIZE is too large (max 100000)")
    except ValueError:
        errors.append("SYNTHETIC_DATA_SIZE must be a valid integer")
    
    try:
        temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.0'))
        if temperature < 0 or temperature > 2:
            errors.append("OPENAI_TEMPERATURE must be between 0 and 2")
    except ValueError:
        errors.append("OPENAI_TEMPERATURE must be a valid float")
    
    try:
        max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '400'))
        if max_tokens < 1 or max_tokens > 4000:
            errors.append("OPENAI_MAX_TOKENS must be between 1 and 4000")
    except ValueError:
        errors.append("OPENAI_MAX_TOKENS must be a valid integer")
    
    try:
        max_hold = int(os.getenv('BACKTEST_MAX_HOLD', '20'))
        if max_hold < 1:
            errors.append("BACKTEST_MAX_HOLD must be at least 1")
    except ValueError:
        errors.append("BACKTEST_MAX_HOLD must be a valid integer")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def print_config_summary():
    """打印配置摘要"""
    logger.info("=" * 60)
    logger.info("Configuration Summary")
    logger.info("=" * 60)
    data_source = os.getenv('DATA_SOURCE', 'synthetic')
    logger.info(f"DATA_SOURCE: {data_source}")
    if data_source == 'csv':
        logger.info(f"DATA_PATH: {os.getenv('DATA_PATH', 'Not set')}")
    elif data_source in ['yahoo', 'binance']:
        logger.info(f"MARKET_SYMBOL: {os.getenv('MARKET_SYMBOL', 'Not set')}")
        if data_source == 'yahoo':
            logger.info(f"MARKET_PERIOD: {os.getenv('MARKET_PERIOD', '1y')}")
            logger.info(f"MARKET_INTERVAL: {os.getenv('MARKET_INTERVAL', '1h')}")
        else:
            logger.info(f"MARKET_TIMEFRAME: {os.getenv('MARKET_TIMEFRAME', '1h')}")
            logger.info(f"MARKET_LIMIT: {os.getenv('MARKET_LIMIT', '1000')}")
    logger.info(f"USE_LLM: {os.getenv('USE_LLM', 'True')}")
    logger.info(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    logger.info(f"SYNTHETIC_DATA_SIZE: {os.getenv('SYNTHETIC_DATA_SIZE', '1500')}")
    logger.info(f"BACKTEST_MAX_HOLD: {os.getenv('BACKTEST_MAX_HOLD', '20')}")
    logger.info(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'INFO')}")
    logger.info("=" * 60)
