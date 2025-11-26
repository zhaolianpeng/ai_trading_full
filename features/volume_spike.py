# features/volume_spike.py
"""
量能爆发检测
参考 TradingView "Eric 全面策略"
"""
import pandas as pd
import numpy as np

def volume_spike(df, vol_len=20, threshold1=1.8, threshold2=3.0):
    """
    检测量能爆发
    
    Args:
        df: 价格数据 DataFrame（需包含 volume）
        vol_len: 量能基准周期（默认20）
        threshold1: 一级爆量阈值（默认1.8）
        threshold2: 二级爆量阈值（默认3.0）
    
    Returns:
        DataFrame 添加以下列：
        - vol_ratio: 成交量比率 (volume / sma(volume))
        - vol_spike1: 一级爆量（vol_ratio > threshold1）
        - vol_spike2: 二级爆量（vol_ratio > threshold2）
    """
    df = df.copy()
    
    vol_sma = df['volume'].rolling(vol_len).mean()
    vol_sma = vol_sma.replace(0, 1)  # 避免除零
    
    df['vol_ratio'] = df['volume'] / vol_sma
    df['vol_spike1'] = df['vol_ratio'] > threshold1
    df['vol_spike2'] = df['vol_ratio'] > threshold2
    
    return df

