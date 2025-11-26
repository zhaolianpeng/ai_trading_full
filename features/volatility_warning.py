# features/volatility_warning.py
"""
波动预警（基于 ATR）
参考 TradingView "Eric 全面策略"
"""
import pandas as pd
import numpy as np

def volatility_warning(df, atr_len=14, atr_mult=2.5):
    """
    基于 ATR 的波动预警
    
    Args:
        df: 价格数据 DataFrame（需包含 atr14 或需要先计算 ATR）
        atr_len: ATR 周期（默认14）
        atr_mult: 波动预警阈值倍数（默认2.5，即 ATR > 2.5 * sma(ATR)）
    
    Returns:
        DataFrame 添加以下列：
        - atr_sma: ATR 的移动平均
        - volatility_warning: 波动预警标志
    """
    df = df.copy()
    
    # 如果没有 ATR，先计算
    if 'atr14' not in df.columns:
        from features.ta_basic import atr
        df = df.copy()
        df['atr14'] = atr(df, atr_len)
    
    atr_sma = df['atr14'].rolling(atr_len).mean()
    atr_sma = atr_sma.replace(0, 0.00001)
    
    df['atr_sma'] = atr_sma
    df['volatility_warning'] = (df['atr14'] / atr_sma) > atr_mult
    
    return df

