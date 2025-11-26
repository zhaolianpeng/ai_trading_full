# features/ta_advanced.py
"""
高级技术指标
"""
import pandas as pd
import numpy as np

def macd(series, fast=12, slow=26, signal=9):
    """
    MACD 指标
    
    Args:
        series: 价格序列
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期
    
    Returns:
        (macd_line, signal_line, histogram): MACD线、信号线、柱状图
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series, n=20, num_std=2):
    """
    布林带
    
    Args:
        series: 价格序列
        n: 周期
        num_std: 标准差倍数
    
    Returns:
        (upper, middle, lower): 上轨、中轨、下轨
    """
    middle = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def stochastic(high, low, close, k_period=14, d_period=3):
    """
    随机指标 (Stochastic Oscillator)
    
    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        k_period: %K 周期
        d_period: %D 周期
    
    Returns:
        (k, d): %K 和 %D 值
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-9))
    d = k.rolling(d_period).mean()
    return k, d

def williams_r(high, low, close, period=14):
    """
    威廉指标 (Williams %R)
    
    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期
    
    Returns:
        Williams %R 值
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-9))
    return wr

def cci(high, low, close, period=20):
    """
    商品通道指标 (Commodity Channel Index)
    
    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期
    
    Returns:
        CCI 值
    """
    tp = (high + low + close) / 3  # 典型价格
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mad + 1e-9)
    return cci

def adx(high, low, close, period=14):
    """
    平均趋向指标 (Average Directional Index)
    
    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期
    
    Returns:
        ADX 值
    """
    # 计算 +DM 和 -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # 计算 TR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 平滑处理
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9))
    
    # 计算 DX 和 ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx, plus_di, minus_di

def add_advanced_ta(df):
    """
    添加高级技术指标到 DataFrame
    
    Args:
        df: 价格数据 DataFrame
    
    Returns:
        添加了高级指标的 DataFrame
    """
    df = df.copy()
    
    # MACD
    macd_line, signal_line, histogram = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = histogram
    
    # 布林带
    bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle  # 布林带宽度
    
    # 随机指标
    stoch_k, stoch_d = stochastic(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    # 威廉指标
    df['williams_r'] = williams_r(df['high'], df['low'], df['close'])
    
    # CCI
    df['cci'] = cci(df['high'], df['low'], df['close'])
    
    # ADX
    adx_val, plus_di, minus_di = adx(df['high'], df['low'], df['close'])
    df['adx'] = adx_val
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    return df
