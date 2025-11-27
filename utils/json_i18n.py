# utils/json_i18n.py
"""
JSON 关键字中文化工具
将 signals_log.json 中的英文关键字转换为中文
"""
from typing import Dict, Any, List, Union

# 关键字映射表（英文 -> 中文）
KEY_MAPPING = {
    # 基础字段
    'index': '索引',
    'quality_score': '质量评分',
    'quality_reasons': '质量原因',
    'risk_reward_ratio': '盈亏比',
    'stop_loss': '止损价',
    'take_profit': '全部止盈价',
    'adjusted_target': '已调整目标',
    
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
    'atr_pct': 'ATR百分比',
    'vol_ratio': '成交量比率',
    'close': '收盘价',
    'open': '开盘价',
    'high': '最高价',
    'low': '最低价',
    
    # 技术指标 - RSI
    'rsi14': 'RSI14',
    
    # 技术指标 - MACD
    'macd': 'MACD',
    'macd_signal': 'MACD信号线',
    'macd_hist': 'MACD柱状图',
    'macd_bullish': 'MACD多头',
    
    # 技术指标 - 布林带
    'bb_upper': '布林带上轨',
    'bb_middle': '布林带中轨',
    'bb_lower': '布林带下轨',
    'bb_width': '布林带宽度',
    'price_above_bb_mid': '价格在布林带中轨上方',
    
    # 技术指标 - 随机指标
    'stoch_k': '随机指标K',
    'stoch_d': '随机指标D',
    
    # 技术指标 - 威廉指标
    'williams_r': '威廉指标',
    
    # 技术指标 - CCI
    'cci': 'CCI',
    
    # 技术指标 - ADX
    'adx': 'ADX',
    'plus_di': '正DI',
    'minus_di': '负DI',
    
    # Eric 策略指标
    'eric_score': 'Eric评分',
    'eric_score_smoothed': 'Eric评分平滑',
    'donchian_upper': 'Donchian上轨',
    'donchian_lower': 'Donchian下轨',
    'donchian_trend': 'Donchian趋势',
    'ema_eye': 'EMA眼',
    
    # 价格动量
    'price_momentum_5': '5周期价格动量',
    'price_momentum_20': '20周期价格动量',
    'price_position': '价格位置',
    
    # llm 对象
    'llm': 'AI决策',
    'trend_structure': '趋势结构',
    'signal': '信号',
    'explanation': '解释',
    'risk': '风险',
    
    # 时间相关
    'signal_time': '信号时间',
    'entry_time': '开单时间',  # 顶层信号对象的开单时间
    'exit_time': '平仓时间',
    'partial_exit_time': '部分止盈时间',
    '交易时间': '交易时间',
    
    # 入场点相关
    'best_entry_3m': '短周期最佳入场点',
    'entry_price': '入场价格',
    'entry_reason': '入场原因',
    'entry_score': '入场评分',
    # 注意：best_entry_3m 对象中的 entry_time 也会被翻译为 '开单时间'
    # 如果需要区分，可以在 best_entry_3m 对象中直接使用中文关键字
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

