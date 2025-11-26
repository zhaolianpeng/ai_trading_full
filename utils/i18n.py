# utils/i18n.py
"""
国际化支持 - 中文翻译
"""
from typing import Dict

# 指标名称中文映射
METRICS_CN: Dict[str, str] = {
    'total_trades': '总交易次数',
    'win_rate': '胜率',
    'avg_return': '平均收益率',
    'total_return': '总收益率',
    'profit_factor': '盈亏比',
    'max_drawdown': '最大回撤',
    'sharpe_ratio': '夏普比率',
    'max_consecutive_losses': '最大连续亏损次数',
    'avg_hold_period': '平均持仓周期',
    'gross_profit': '总盈利',
    'gross_loss': '总亏损',
}

# 信号类型中文映射
SIGNAL_TYPES_CN: Dict[str, str] = {
    'long_structure': '多头结构',
    'breakout_long': '突破做多',
    'rsi_positive_divergence': 'RSI正背离',
    'rsi_negative_divergence': 'RSI负背离',
    'macd_cross_up': 'MACD金叉',
    'bb_bounce': '布林带反弹',
    'rsi_oversold_bounce': 'RSI超卖反弹',
    'eric_long': 'Eric策略做多',
    'eric_short': 'Eric策略做空',
    'unknown': '未知',
}

# LLM信号中文映射
LLM_SIGNALS_CN: Dict[str, str] = {
    'Long': '做多',
    'Short': '做空',
    'Neutral': '中性',
    'Hold': '持有',
}

def get_metric_name_cn(key: str) -> str:
    """获取指标的中文名称"""
    return METRICS_CN.get(key, key)

def get_signal_type_cn(key: str) -> str:
    """获取信号类型的中文名称"""
    return SIGNAL_TYPES_CN.get(key, key)

def get_llm_signal_cn(key: str) -> str:
    """获取LLM信号的中文名称"""
    return LLM_SIGNALS_CN.get(key, key)

def format_metric_value(key: str, value) -> str:
    """格式化指标值（带中文名称）"""
    name_cn = get_metric_name_cn(key)
    
    if isinstance(value, float):
        if 'rate' in key.lower() or 'ratio' in key.lower() or 'factor' in key.lower():
            return f"{name_cn}: {value:.2%}"
        elif 'return' in key.lower() or 'drawdown' in key.lower() or 'profit' in key.lower() or 'loss' in key.lower():
            return f"{name_cn}: {value:.4f} ({value:.2%})"
        else:
            return f"{name_cn}: {value:.4f}"
    elif isinstance(value, int):
        return f"{name_cn}: {value}"
    else:
        return f"{name_cn}: {value}"

