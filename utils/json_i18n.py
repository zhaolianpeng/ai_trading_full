# utils/json_i18n.py
"""
JSON 关键字中文化工具
将 signals_log.json 中的英文关键字转换为中文
"""
from typing import Dict, Any, List, Union

# 关键字映射表（英文 -> 中文）
KEY_MAPPING = {
    # rule 对象
    'rule': '规则信号',
    'type': '类型',
    'score': '评分',
    'confidence': '置信度',
    'idx': '索引',
    
    # feature_packet 对象
    'feature_packet': '特征包',
    'trend': '趋势',
    'ema_alignment': 'EMA排列',
    'higher_highs': '更高高点',
    'higher_lows': '更高低点',
    'volume_spike': '量能爆发',
    'breakout': '突破',
    'rsi_divergence': 'RSI背离',
    'atr': 'ATR波动率',
    'vol_ratio': '成交量比率',
    'close': '收盘价',
    
    # llm 对象
    'llm': 'AI决策',
    'trend_structure': '趋势结构',
    'signal': '信号',
    'explanation': '解释',
    'risk': '风险',
    
    # 时间相关
    'signal_time': '信号时间',
    'entry_time': '开单时间',
    'exit_time': '平仓时间',
    'partial_exit_time': '部分止盈时间',
    '交易时间': '交易时间',
    
    # 入场点相关
    'best_entry_3m': '3分钟最佳入场点',
    'entry_price': '入场价格',
    'entry_reason': '入场原因',
    'entry_score': '入场评分',
}

# 反向映射（中文 -> 英文，用于读取）
REVERSE_MAPPING = {v: k for k, v in KEY_MAPPING.items()}

def get_value_safe(data: Dict, key: str, default: Any = None) -> Any:
    """
    安全获取值，支持中英文关键字
    
    Args:
        data: 字典
        key: 英文关键字
        default: 默认值
    
    Returns:
        值，如果中英文关键字都不存在则返回默认值
    """
    # 先尝试英文关键字
    if key in data:
        return data[key]
    # 再尝试中文关键字
    chinese_key = KEY_MAPPING.get(key)
    if chinese_key and chinese_key in data:
        return data[chinese_key]
    return default

def translate_keys_to_chinese(data: Any) -> Any:
    """
    将字典中的英文关键字转换为中文
    
    Args:
        data: 可以是字典、列表或基本类型
    
    Returns:
        转换后的数据
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # 转换关键字
            chinese_key = KEY_MAPPING.get(key, key)
            # 递归转换值
            result[chinese_key] = translate_keys_to_chinese(value)
        return result
    elif isinstance(data, list):
        return [translate_keys_to_chinese(item) for item in data]
    else:
        return data

def translate_keys_to_english(data: Any) -> Any:
    """
    将字典中的中文关键字转换回英文（用于读取）
    
    Args:
        data: 可以是字典、列表或基本类型
    
    Returns:
        转换后的数据
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # 转换关键字
            english_key = REVERSE_MAPPING.get(key, key)
            # 递归转换值
            result[english_key] = translate_keys_to_english(value)
        return result
    elif isinstance(data, list):
        return [translate_keys_to_english(item) for item in data]
    else:
        return data

