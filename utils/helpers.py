# utils/helpers.py
"""
辅助工具函数
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from utils.logger import logger

def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    格式化回测指标为可读字符串
    
    Args:
        metrics: 指标字典
    
    Returns:
        格式化后的字符串
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key.lower() or 'ratio' in key.lower():
                lines.append(f"{key}: {value:.2%}")
            elif 'return' in key.lower() or 'drawdown' in key.lower():
                lines.append(f"{key}: {value:.4f} ({value:.2%})")
            else:
                lines.append(f"{key}: {value:.4f}")
        elif isinstance(value, int):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)

def calculate_portfolio_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    计算投资组合指标
    
    Args:
        returns: 收益率序列
    
    Returns:
        指标字典
    """
    if returns.empty:
        return {}
    
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值（当分母为0时返回）
    
    Returns:
        除法结果
    """
    if abs(denominator) < 1e-9:
        return default
    return numerator / denominator
