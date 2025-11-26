# signal_rules_eric.py
"""
基于 Eric 全面策略的信号检测
参考 TradingView "Eric 全面策略（趋势过滤器 + EMA组 + EMA眼 + 超买超卖 + 背离 + 爆量 + 波动预警）"
"""
import pandas as pd
import numpy as np
from features.eric_indicators import add_eric_indicators
from utils.logger import logger

def detect_eric_signals(df, 
                        use_eric_score=True,
                        use_donchian=True,
                        use_ema_eye=True,
                        use_volume_spike=True,
                        use_divergence=True,
                        use_volatility_warning=True,
                        use_vegas=False,
                        confirm_bars=3,
                        confirm_body_mult=1.2,
                        **kwargs):
    """
    基于 Eric 策略的信号检测
    
    Args:
        df: 价格数据 DataFrame
        use_eric_score: 是否使用 Eric Score
        use_donchian: 是否使用 Donchian 通道
        use_ema_eye: 是否使用 EMA 眼
        use_volume_spike: 是否使用量能爆发
        use_divergence: 是否使用背离检测
        use_volatility_warning: 是否使用波动预警
        use_vegas: 是否使用维加斯通道
        confirm_bars: 偏右确认的K线数量
        confirm_body_mult: 确认烛实体倍数
        **kwargs: 传递给 add_eric_indicators 的参数
    
    Returns:
        (df, signals): 增强后的 df 和信号列表
    """
    logger.info("添加 Eric 策略指标...")
    df = add_eric_indicators(
        df,
        use_eric_score=use_eric_score,
        use_donchian=use_donchian,
        use_ema_eye=use_ema_eye,
        use_volume_spike=use_volume_spike,
        use_divergence=use_divergence,
        use_volatility_warning=use_volatility_warning,
        use_vegas=use_vegas,
        **kwargs
    )
    
    logger.info("检测 Eric 策略信号...")
    signals = []
    
    # 计算确认烛（大实体）
    body = np.abs(df['close'] - df['open'])
    avg_body = body.rolling(10).mean()
    is_big_bull = (df['close'] > df['open']) & (body > avg_body * confirm_body_mult)
    is_big_bear = (df['close'] < df['open']) & (body > avg_body * confirm_body_mult)
    
    # 检查最近 confirm_bars 内是否有确认烛
    recent_big_bull = pd.Series(False, index=df.index)
    recent_big_bear = pd.Series(False, index=df.index)
    
    for i in range(len(df)):
        if i < confirm_bars:
            continue
        # 检查过去 confirm_bars 根K线
        window_bull = is_big_bull.iloc[max(0, i-confirm_bars):i+1]
        window_bear = is_big_bear.iloc[max(0, i-confirm_bars):i+1]
        recent_big_bull.iloc[i] = window_bull.any()
        recent_big_bear.iloc[i] = window_bear.any()
    
    # 多头信号条件
    if use_eric_score and 'eric_oversold' in df.columns:
        # 基础条件：Eric 超卖 + 接近通道下沿 + 通道非强空
        cand_long_base = (
            (df['eric_oversold'] | df.get('eric_oversold2', pd.Series(False, index=df.index))) &
            df.get('near_lower', pd.Series(False, index=df.index)) &
            (df.get('channel_trend', 'sideways') != 'bearish')
        )
        
        # EMA 支撑条件
        cand_long_ema = (
            (df['close'] > df['ema55']) |
            (df['ema21'] > df['ema55']) |
            (df.get('eye21_support', pd.Series(False, index=df.index)) & (df['close'] > df['ema21'])) |
            (df.get('eye55', pd.Series(0, index=df.index)) == 1)
        )
        
        # 背离增强
        cand_long_div = df.get('bull_divergence', pd.Series(False, index=df.index))
        
        long_candidate = cand_long_base & (cand_long_ema | cand_long_div)
        can_enter_long = long_candidate & recent_big_bull
        
        # 防止重复信号
        can_enter_long = can_enter_long & ~(long_candidate.shift(1) & recent_big_bull.shift(1))
        
        for idx in df[can_enter_long].index:
            i = df.index.get_loc(idx)
            # 计算信号强度
            score = 3  # 基础分数
            if df.loc[idx, 'eric_oversold2'] if 'eric_oversold2' in df.columns else False:
                score += 1  # 二级超卖加分
            if df.loc[idx, 'bull_divergence'] if 'bull_divergence' in df.columns else False:
                score += 1  # 背离加分
            if df.loc[idx, 'vol_spike2'] if 'vol_spike2' in df.columns else False:
                score += 1  # 二级爆量加分
            
            signals.append({
                'type': 'eric_long',
                'score': score,
                'confidence': 'high' if score >= 5 else 'medium',
                'idx': i
            })
    
    # 空头信号条件
    if use_eric_score and 'eric_overbought' in df.columns:
        # 基础条件：Eric 超买 + 接近通道上沿 + 通道非强多
        cand_short_base = (
            (df['eric_overbought'] | df.get('eric_overbought2', pd.Series(False, index=df.index))) &
            df.get('near_upper', pd.Series(False, index=df.index)) &
            (df.get('channel_trend', 'sideways') != 'bullish')
        )
        
        # EMA 压力条件
        cand_short_ema = (
            (df['close'] < df['ema55']) |
            (df['ema21'] < df['ema55']) |
            (df.get('eye21_pressure', pd.Series(False, index=df.index)) & (df['close'] < df['ema21'])) |
            (df.get('eye55', pd.Series(0, index=df.index)) == 1)
        )
        
        # 背离增强
        cand_short_div = df.get('bear_divergence', pd.Series(False, index=df.index))
        
        short_candidate = cand_short_base & (cand_short_ema | cand_short_div)
        can_enter_short = short_candidate & recent_big_bear
        
        # 防止重复信号
        can_enter_short = can_enter_short & ~(short_candidate.shift(1) & recent_big_bear.shift(1))
        
        for idx in df[can_enter_short].index:
            i = df.index.get_loc(idx)
            # 计算信号强度
            score = 3  # 基础分数
            if df.loc[idx, 'eric_overbought2'] if 'eric_overbought2' in df.columns else False:
                score += 1  # 二级超买加分
            if df.loc[idx, 'bear_divergence'] if 'bear_divergence' in df.columns else False:
                score += 1  # 背离加分
            if df.loc[idx, 'vol_spike2'] if 'vol_spike2' in df.columns else False:
                score += 1  # 二级爆量加分
            
            signals.append({
                'type': 'eric_short',
                'score': score,
                'confidence': 'high' if score >= 5 else 'medium',
                'idx': i
            })
    
    # 去重
    seen = set()
    unique_signals = []
    for sig in signals:
        key = (sig['idx'], sig['type'])
        if key not in seen:
            seen.add(key)
            unique_signals.append(sig)
    
    logger.info(f"检测到 {len(unique_signals)} 个 Eric 策略信号")
    return df, unique_signals

