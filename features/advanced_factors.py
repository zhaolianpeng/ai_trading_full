# features/advanced_factors.py
"""
高级技术因子
包括：VWAP proxy、Breakout distance、多窗口动量、波动率因子等
"""
import pandas as pd
import numpy as np

def vwap_proxy(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    VWAP代理：使用典型价格和成交量计算
    典型价格 = (high + low + close) / 3
    
    Args:
        df: 价格数据DataFrame
        period: 计算周期
    
    Returns:
        VWAP代理值
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(period, min_periods=1).sum() / df['volume'].rolling(period, min_periods=1).sum()
    return vwap

def breakout_distance(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    突破距离：当前价格距离阻力位/支撑位的距离百分比
    
    Args:
        df: 价格数据DataFrame
        period: 计算周期
    
    Returns:
        突破距离（正数表示在阻力位上方，负数表示在支撑位下方）
    """
    resistance = df['close'].rolling(period, min_periods=1).max()
    support = df['close'].rolling(period, min_periods=1).min()
    
    # 计算距离阻力位和支撑位的距离
    dist_to_resistance = (df['close'] - resistance) / resistance * 100
    dist_to_support = (df['close'] - support) / support * 100
    
    # 返回更接近的距离
    breakout_dist = np.where(
        abs(dist_to_resistance) < abs(dist_to_support),
        dist_to_resistance,
        dist_to_support
    )
    
    return pd.Series(breakout_dist, index=df.index)

def price_momentum_multi_window(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """
    多窗口价格动量
    
    Args:
        df: 价格数据DataFrame
        windows: 动量窗口列表
    
    Returns:
        包含各窗口动量的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    for window in windows:
        if window < len(df):
            momentum = (df['close'] - df['close'].shift(window)) / df['close'].shift(window) * 100
            result[f'momentum_{window}'] = momentum
    
    return result

def volatility_regime(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    波动率regime：基于ATR的波动率分类
    
    Args:
        df: 价格数据DataFrame
        period: 计算周期
    
    Returns:
        波动率regime（0=LOW_VOL, 1=MID_VOL, 2=HIGH_VOL, 3=EXTREME_VOL）
    """
    if 'atr14' not in df.columns:
        # 如果没有ATR，使用价格范围估算
        price_range = (df['high'] - df['low']) / df['close']
        vol_metric = price_range.rolling(period, min_periods=1).mean()
    else:
        vol_metric = (df['atr14'] / df['close']).rolling(period, min_periods=1).mean()
    
    # 计算分位数
    vol_p25 = vol_metric.rolling(period * 2, min_periods=1).quantile(0.25)
    vol_p50 = vol_metric.rolling(period * 2, min_periods=1).quantile(0.50)
    vol_p75 = vol_metric.rolling(period * 2, min_periods=1).quantile(0.75)
    
    # 分类
    regime = np.where(vol_metric < vol_p25, 0,  # LOW_VOL
            np.where(vol_metric < vol_p50, 1,    # MID_VOL
            np.where(vol_metric < vol_p75, 2,     # HIGH_VOL
                     3)))                         # EXTREME_VOL
    
    return pd.Series(regime, index=df.index)

def volume_surge(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    成交量爆发：当前成交量相对于均值的倍数
    
    Args:
        df: 价格数据DataFrame
        period: 计算周期
    
    Returns:
        成交量爆发倍数
    """
    vol_ma = df['volume'].rolling(period, min_periods=1).mean()
    surge = df['volume'] / (vol_ma + 1e-9)
    return surge

def trend_score(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.Series:
    """
    趋势评分：基于多个周期的EMA排列和价格位置
    
    Args:
        df: 价格数据DataFrame
        periods: EMA周期列表
    
    Returns:
        趋势评分（0-100）
    """
    score = pd.Series(0.0, index=df.index)
    
    # 计算各周期EMA
    emas = {}
    for period in periods:
        if period <= len(df):
            emas[period] = df['close'].ewm(span=period, adjust=False).mean()
    
    # 检查EMA排列（多头排列加分）
    for i in range(len(df)):
        if i < max(periods):
            continue
        
        alignment_score = 0
        # 检查EMA是否按顺序排列
        if len(emas) >= 2:
            periods_sorted = sorted(periods)
            is_bullish = True
            for j in range(len(periods_sorted) - 1):
                if emas[periods_sorted[j]].iloc[i] <= emas[periods_sorted[j+1]].iloc[i]:
                    is_bullish = False
                    break
            if is_bullish:
                alignment_score += 30
        
        # 检查价格是否在EMA上方
        if len(emas) > 0:
            price_above_ema = sum(1 for period in periods if df['close'].iloc[i] > emas[period].iloc[i])
            alignment_score += (price_above_ema / len(periods)) * 20
        
        # 检查价格动量
        if i >= 20:
            momentum = (df['close'].iloc[i] - df['close'].iloc[i-20]) / df['close'].iloc[i-20] * 100
            alignment_score += min(30, max(0, momentum * 10))
        
        # 检查RSI
        if 'rsi14' in df.columns and not pd.isna(df['rsi14'].iloc[i]):
            rsi_val = df['rsi14'].iloc[i]
            if 40 < rsi_val < 70:  # 健康区间
                alignment_score += 20
            elif rsi_val > 70:  # 超买，扣分
                alignment_score -= 10
        
        score.iloc[i] = min(100, max(0, alignment_score))
    
    return score

def momentum_score(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.Series:
    """
    动量评分：基于多个窗口的动量
    
    Args:
        df: 价格数据DataFrame
        windows: 动量窗口列表
    
    Returns:
        动量评分（0-100）
    """
    score = pd.Series(0.0, index=df.index)
    
    for i in range(len(df)):
        if i < max(windows):
            continue
        
        momentum_sum = 0
        positive_count = 0
        
        for window in windows:
            if i >= window:
                momentum = (df['close'].iloc[i] - df['close'].iloc[i-window]) / df['close'].iloc[i-window] * 100
                momentum_sum += momentum
                if momentum > 0:
                    positive_count += 1
        
        # 评分：正动量窗口数量 + 平均动量
        avg_momentum = momentum_sum / len(windows) if len(windows) > 0 else 0
        score.iloc[i] = min(100, max(0, (positive_count / len(windows)) * 50 + avg_momentum * 10))
    
    return score

def vol_score(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    波动率评分：基于ATR和价格波动
    
    Args:
        df: 价格数据DataFrame
        period: 计算周期
    
    Returns:
        波动率评分（0-100，越高表示波动率越健康）
    """
    if 'atr14' not in df.columns:
        return pd.Series(50.0, index=df.index)  # 默认中等评分
    
    atr_pct = (df['atr14'] / df['close']) * 100
    
    # 计算波动率的健康区间（0.5% - 3%）
    score = pd.Series(0.0, index=df.index)
    
    for i in range(len(df)):
        if pd.isna(atr_pct.iloc[i]):
            score.iloc[i] = 50.0
            continue
        
        vol_val = atr_pct.iloc[i]
        
        if 0.5 <= vol_val <= 3.0:  # 健康区间
            score.iloc[i] = 100 - abs(vol_val - 1.5) * 20
        elif vol_val < 0.5:  # 波动率过低
            score.iloc[i] = vol_val * 100
        else:  # 波动率过高
            score.iloc[i] = max(0, 100 - (vol_val - 3.0) * 10)
    
    return score

def volume_score(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    成交量评分：基于成交量比率和爆发
    
    Args:
        df: 价格数据DataFrame
        period: 计算周期
    
    Returns:
        成交量评分（0-100）
    """
    vol_ma = df['volume'].rolling(period, min_periods=1).mean()
    vol_ratio = df['volume'] / (vol_ma + 1e-9)
    
    # 评分：成交量放大加分，但不要过度放大（可能是异常）
    score = pd.Series(0.0, index=df.index)
    
    for i in range(len(df)):
        ratio = vol_ratio.iloc[i]
        
        if 1.0 <= ratio <= 3.0:  # 健康放大
            score.iloc[i] = min(100, 50 + (ratio - 1.0) * 25)
        elif ratio > 3.0:  # 过度放大，可能是异常
            score.iloc[i] = max(50, 100 - (ratio - 3.0) * 10)
        else:  # 成交量萎缩
            score.iloc[i] = ratio * 50
    
    return score

def add_advanced_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加所有高级因子到DataFrame
    
    Args:
        df: 价格数据DataFrame
    
    Returns:
        添加了高级因子的DataFrame
    """
    df = df.copy()
    
    # VWAP代理
    df['vwap_proxy'] = vwap_proxy(df, period=20)
    
    # 突破距离
    df['breakout_dist'] = breakout_distance(df, period=50)
    
    # 多窗口动量
    momentum_df = price_momentum_multi_window(df, windows=[5, 10, 20, 50])
    for col in momentum_df.columns:
        df[col] = momentum_df[col]
    
    # 波动率regime
    df['volatility_regime'] = volatility_regime(df, period=20)
    
    # 成交量爆发
    df['volume_surge'] = volume_surge(df, period=20)
    
    # 信号评分
    df['trend_score'] = trend_score(df, periods=[5, 10, 20, 50])
    df['momentum_score'] = momentum_score(df, windows=[5, 10, 20])
    df['vol_score'] = vol_score(df, period=20)
    df['volume_score'] = volume_score(df, period=50)
    
    return df

