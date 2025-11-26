# features/eric_indicators.py
"""
Eric 全面策略指标整合
将所有 Eric 策略相关的指标整合在一起
"""
import pandas as pd
from .ta_basic import add_basic_ta, add_vegas_ema
from .eric_score import eric_score, eric_oversold_overbought
from .donchian_channel import donchian_channel
from .ema_eye import ema_eye
from .volume_spike import volume_spike
from .eric_divergence import eric_divergence
from .volatility_warning import volatility_warning

def add_eric_indicators(df, 
                       use_eric_score=True,
                       use_donchian=True,
                       use_ema_eye=True,
                       use_volume_spike=True,
                       use_divergence=True,
                       use_volatility_warning=True,
                       use_vegas=False,
                       **kwargs):
    """
    添加所有 Eric 策略相关的指标
    
    Args:
        df: 价格数据 DataFrame
        use_eric_score: 是否使用 Eric Score
        use_donchian: 是否使用 Donchian 通道
        use_ema_eye: 是否使用 EMA 眼
        use_volume_spike: 是否使用量能爆发
        use_divergence: 是否使用背离检测
        use_volatility_warning: 是否使用波动预警
        use_vegas: 是否使用维加斯通道（EMA144/169）
        **kwargs: 其他参数（传递给各个指标函数）
    
    Returns:
        添加了所有指标的 DataFrame
    """
    df = df.copy()
    
    # 1. 基础技术指标（EMA, RSI, ATR等）
    df = add_basic_ta(df)
    
    # 2. 维加斯通道（可选）
    if use_vegas:
        df = add_vegas_ema(df)
    
    # 3. Eric Score
    if use_eric_score:
        eric_len = kwargs.get('eric_len', 14)
        eric_zlen = kwargs.get('eric_zlen', 20)
        eric_smooth1 = kwargs.get('eric_smooth1', 3)
        eric_smooth2 = kwargs.get('eric_smooth2', 5)
        df = eric_score(df, eric_len, eric_zlen, eric_smooth1, eric_smooth2)
        
        # 超买超卖判断
        ob_lev = kwargs.get('ob_lev', 0.7)
        os_lev = kwargs.get('os_lev', -0.7)
        ob_lev2_mult = kwargs.get('ob_lev2_mult', 1.25)
        os_lev2_mult = kwargs.get('os_lev2_mult', 1.25)
        df = eric_oversold_overbought(df, ob_lev, os_lev, ob_lev2_mult, os_lev2_mult)
    
    # 4. Donchian 通道
    if use_donchian:
        chan_len = kwargs.get('chan_len', 55)
        chan_tol = kwargs.get('chan_tol', 0.15)
        df = donchian_channel(df, chan_len, chan_tol)
    
    # 5. EMA 眼
    if use_ema_eye:
        eye_small = kwargs.get('eye_small', 0.01)
        eye_big = kwargs.get('eye_big', 0.03)
        df = ema_eye(df, eye_small, eye_big)
    
    # 6. 量能爆发
    if use_volume_spike:
        vol_len = kwargs.get('vol_len', 20)
        vol_threshold1 = kwargs.get('vol_threshold1', 1.8)
        vol_threshold2 = kwargs.get('vol_threshold2', 3.0)
        df = volume_spike(df, vol_len, vol_threshold1, vol_threshold2)
    
    # 7. 背离检测（需要 Eric Score）
    if use_divergence and use_eric_score and 'eric_score_smoothed' in df.columns:
        div_lookback = kwargs.get('div_lookback', 10)
        df = eric_divergence(df, div_lookback)
    
    # 8. 波动预警
    if use_volatility_warning:
        atr_mult = kwargs.get('atr_mult', 2.5)
        df = volatility_warning(df, atr_len=14, atr_mult=atr_mult)
    
    return df

