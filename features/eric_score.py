# features/eric_score.py
"""
Eric Score 指标 - 基于价格在区间内位置的超买超卖指标
参考 TradingView "Eric 全面策略"
"""
import pandas as pd
import numpy as np

def eric_score(df, len_period=14, zlen=20, smooth1=3, smooth2=5):
    """
    计算 Eric Score 指标
    
    Args:
        df: 价格数据 DataFrame（需包含 high, low, close）
        len_period: 区间周期（默认14）
        zlen: 标准化周期（默认20）
        smooth1: 第一层平滑（默认3）
        smooth2: 第二层平滑（默认5）
    
    Returns:
        DataFrame 添加 eric_score, eric_score_smoothed 列
    """
    df = df.copy()
    
    # 计算价格在区间内的位置 (0-1)
    lowest = df['low'].rolling(len_period).min()
    highest = df['high'].rolling(len_period).max()
    range_size = highest - lowest
    range_size = range_size.replace(0, 0.00001)  # 避免除零
    
    pos = (df['close'] - lowest) / range_size
    
    # 双重平滑
    pos_smooth1 = pos.ewm(span=smooth1, adjust=False).mean()
    pos_smooth = pos_smooth1.ewm(span=smooth2, adjust=False).mean()
    
    # 标准化：计算 z-score
    mean_pos = pos_smooth.rolling(zlen).mean()
    stdev_pos = pos_smooth.rolling(zlen).std()
    stdev_pos = stdev_pos.replace(0, 0.00001)  # 避免除零
    
    score = (pos_smooth - mean_pos) / stdev_pos
    
    # 最终平滑
    score_smoothed = score.ewm(span=smooth2, adjust=False).mean()
    
    df['eric_score'] = score
    df['eric_score_smoothed'] = score_smoothed
    
    return df

def eric_oversold_overbought(df, ob_lev=0.7, os_lev=-0.7, ob_lev2_mult=1.25, os_lev2_mult=1.25):
    """
    判断 Eric Score 的超买超卖状态
    
    Args:
        df: 包含 eric_score_smoothed 的 DataFrame
        ob_lev: 超买阈值（默认0.7）
        os_lev: 超卖阈值（默认-0.7）
        ob_lev2_mult: 二级超买阈值倍数（默认1.25）
        os_lev2_mult: 二级超卖阈值倍数（默认1.25）
    
    Returns:
        DataFrame 添加 eric_oversold, eric_overbought, eric_oversold2, eric_overbought2 列
    """
    df = df.copy()
    
    ob_lev2 = ob_lev * ob_lev2_mult
    os_lev2 = os_lev * os_lev2_mult
    
    df['eric_oversold'] = df['eric_score_smoothed'] < os_lev
    df['eric_overbought'] = df['eric_score_smoothed'] > ob_lev
    df['eric_oversold2'] = df['eric_score_smoothed'] < os_lev2
    df['eric_overbought2'] = df['eric_score_smoothed'] > ob_lev2
    
    return df

