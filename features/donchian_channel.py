# features/donchian_channel.py
"""
Donchian 通道 - 趋势过滤器
参考 TradingView "Eric 全面策略"
"""
import pandas as pd
import numpy as np

def donchian_channel(df, length=55, tolerance_pct=0.15):
    """
    计算 Donchian 通道
    
    Args:
        df: 价格数据 DataFrame（需包含 high, low, close）
        length: 通道周期（默认55）
        tolerance_pct: 接近通道上下沿的容差百分比（默认0.15，即15%）
    
    Returns:
        DataFrame 添加以下列：
        - donchian_high: 通道上沿
        - donchian_low: 通道下沿
        - donchian_mid: 通道中轨
        - donchian_height: 通道高度
        - near_upper: 是否接近上沿
        - near_lower: 是否接近下沿
        - channel_trend: 通道趋势（'bullish', 'bearish', 'sideways'）
    """
    df = df.copy()
    
    # 计算通道
    df['donchian_high'] = df['high'].rolling(length).max()
    df['donchian_low'] = df['low'].rolling(length).min()
    df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
    df['donchian_height'] = df['donchian_high'] - df['donchian_low']
    df['donchian_height'] = df['donchian_height'].replace(0, 0.00001)
    
    # 判断是否接近上下沿
    df['near_upper'] = df['close'] >= (df['donchian_high'] - df['donchian_height'] * tolerance_pct)
    df['near_lower'] = df['close'] <= (df['donchian_low'] + df['donchian_height'] * tolerance_pct)
    
    # 判断通道趋势（通过比较当前和之前的通道边沿）
    ref_offset = max(1, length // 4)
    
    # 通道边沿是否上升/下降
    up_edge_rising = df['donchian_high'] > df['donchian_high'].shift(ref_offset)
    low_edge_rising = df['donchian_low'] > df['donchian_low'].shift(ref_offset)
    up_edge_falling = df['donchian_high'] < df['donchian_high'].shift(ref_offset)
    low_edge_falling = df['donchian_low'] < df['donchian_low'].shift(ref_offset)
    
    # 结合EMA排列判断（简化版，需要EMA数据）
    # 这里先只基于通道边沿判断
    df['channel_trend'] = 'sideways'
    df.loc[(up_edge_rising & low_edge_rising) | 
           ((df['close'] > df['donchian_mid']) & up_edge_rising), 'channel_trend'] = 'bullish'
    df.loc[(up_edge_falling & low_edge_falling) | 
           ((df['close'] < df['donchian_mid']) & up_edge_falling), 'channel_trend'] = 'bearish'
    
    return df

