# features/ema_eye.py
"""
EMA 眼 - 价格与EMA的距离分析（支撑/压力提示）
参考 TradingView "Eric 全面策略"
"""
import pandas as pd
import numpy as np

def ema_eye(df, eye_small=0.01, eye_big=0.03):
    """
    计算 EMA 眼（价格与EMA的相对距离）
    
    Args:
        df: 价格数据 DataFrame（需包含 close 和 EMA 列：ema21, ema55, ema100, ema200）
        eye_small: 小眼阈值（相对距离 < 1%，视为接近EMA）
        eye_big: 大眼阈值（相对距离 > 3%，视为远离EMA）
    
    Returns:
        DataFrame 添加以下列：
        - eye21, eye55, eye100, eye200: 0=正常, 1=小眼(接近), 2=大眼(远离)
        - eye21_support: EMA21是否提供支撑（价格在EMA上方且接近）
        - eye21_pressure: EMA21是否形成压力（价格在EMA下方且接近）
        - eye55_support, eye55_pressure: 同上
    """
    df = df.copy()
    
    ema_cols = ['ema21', 'ema55', 'ema100', 'ema200']
    
    for ema_col in ema_cols:
        if ema_col not in df.columns:
            continue
        
        # 计算相对距离
        rel_dist = np.abs(df['close'] - df[ema_col]) / df[ema_col]
        
        # 判断眼的状态
        eye_col = ema_col.replace('ema', 'eye')
        df[eye_col] = 0
        df.loc[rel_dist < eye_small, eye_col] = 1  # 小眼（接近）
        df.loc[rel_dist > eye_big, eye_col] = 2    # 大眼（远离）
        
        # 判断支撑/压力
        support_col = f"{eye_col}_support"
        pressure_col = f"{eye_col}_pressure"
        
        df[support_col] = (df['close'] > df[ema_col]) & (df[eye_col] == 1)
        df[pressure_col] = (df['close'] < df[ema_col]) & (df[eye_col] == 1)
    
    return df

