# signal_rules.py
from features.ta_basic import add_basic_ta
from features.ta_advanced import add_advanced_ta
from features.divergence import detect_rsi_divergence
import numpy as np
import pandas as pd
import os
from utils.logger import logger

def detect_rules(df, use_advanced_ta=True, use_eric_indicators=False):
    """
    基于规则的信号检测：long_structure / breakout / rsi divergence / macd / bollinger / eric
    使用向量化操作提升性能
    
    Args:
        df: 价格数据 DataFrame
        use_advanced_ta: 是否使用高级技术指标
        use_eric_indicators: 是否使用 Eric 策略指标
    
    Returns:
        (df, signals): 增强后的 df（含指标列）与 signals 列表
    """
    logger.info("添加技术指标...")
    df = add_basic_ta(df)
    if use_advanced_ta:
        df = add_advanced_ta(df)
    
    # 如果使用 Eric 指标，添加相关指标
    if use_eric_indicators:
        from features.eric_indicators import add_eric_indicators
        df = add_eric_indicators(df, use_eric_score=True, use_donchian=True, 
                                use_ema_eye=True, use_volume_spike=True,
                                use_divergence=True, use_volatility_warning=True)
    
    logger.info("检测交易信号...")
    signals = []
    
    # 向量化检测 long_structure（优化性能）
    if len(df) > 100:
        # EMA 条件
        ema_condition = (df['ema21'] > df['ema55']) & (df['ema55'] > df['ema100'])
        
        # 价格突破条件（使用 rolling）
        close_20max = df['close'].rolling(20, min_periods=20).max()
        close_20min = df['close'].rolling(20, min_periods=20).min()
        close_40mean = df['close'].rolling(40, min_periods=40).mean()
        price_condition = (df['close'] == close_20max) & (close_20min > close_40mean.shift(1))
        
        # 成交量条件（根据交易模式调整阈值）
        volume_threshold = float(os.getenv('VOLUME_THRESHOLD', '1.2'))
        volume_condition = df['volume'] > (df['vol_ma50'] * volume_threshold)
        
        # 组合条件
        long_structure_mask = ema_condition & price_condition & volume_condition
        
        for idx in df[long_structure_mask].index:
            i = df.index.get_loc(idx)
            signals.append({'type': 'long_structure', 'score': 4, 'confidence': 'high', 'idx': i})
    
    # 向量化检测 breakout_long
    if 'res50' in df.columns:
        breakout_mask = df['close'] > df['res50']
        for idx in df[breakout_mask].index:
            i = df.index.get_loc(idx)
            if not pd.isna(df.loc[idx, 'res50']):
                signals.append({'type': 'breakout_long', 'score': 3, 'confidence': 'medium', 'idx': i})
    
    # MACD 信号（如果可用）
    if use_advanced_ta and 'macd' in df.columns:
        # MACD 金叉
        macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        for idx in df[macd_cross_up].index:
            i = df.index.get_loc(idx)
            if i >= 26:  # MACD 需要足够的数据
                signals.append({'type': 'macd_cross_up', 'score': 3, 'confidence': 'medium', 'idx': i})
    
    # 布林带信号（如果可用）
    if use_advanced_ta and 'bb_lower' in df.columns:
        # 价格触及下轨后反弹
        bb_bounce = (df['close'] <= df['bb_lower']) & (df['close'].shift(1) > df['bb_lower'].shift(1))
        for idx in df[bb_bounce].index:
            i = df.index.get_loc(idx)
            if i >= 20:  # 布林带需要足够的数据
                signals.append({'type': 'bb_bounce', 'score': 2, 'confidence': 'medium', 'idx': i})
    
    # RSI 超卖反弹（向量化）
    if 'rsi14' in df.columns:
        rsi_oversold = (df['rsi14'] < 30) & (df['rsi14'].shift(1) >= 30)
        for idx in df[rsi_oversold].index:
            i = df.index.get_loc(idx)
            if i >= 14:
                signals.append({'type': 'rsi_oversold_bounce', 'score': 2, 'confidence': 'medium', 'idx': i})
    
    # RSI 背离检测（需要逐个检查，但可以优化）
    logger.debug("检测 RSI 背离...")
    divergence_signals = []
    # 只检查最近的数据点以减少计算量
    check_range = min(200, len(df))  # 只检查最近200个点
    start_idx = max(0, len(df) - check_range)
    
    for i in range(start_idx, len(df)):
        if i < 20:  # 需要足够的数据
            continue
        df_at_idx = df.iloc[:i+1]
        divs = detect_rsi_divergence(df_at_idx, order=3)
        for d in divs:
            if d[0] == i:
                tp = 'rsi_positive_divergence' if d[1] == 'positive' else 'rsi_negative_divergence'
                divergence_signals.append({'type': tp, 'score': 2, 'confidence': 'medium', 'idx': i})
    
    signals.extend(divergence_signals)
    
    # Eric 策略信号（如果启用了 Eric 指标）
    if use_eric_indicators:
        from signal_rules_eric import detect_eric_signals
        df, eric_signals = detect_eric_signals(df, use_eric_score=True, use_donchian=True,
                                               use_ema_eye=True, use_volume_spike=True,
                                               use_divergence=True, use_volatility_warning=True)
        signals.extend(eric_signals)
    
    # 去重：同一索引的相同类型信号只保留一个
    seen = set()
    unique_signals = []
    for sig in signals:
        key = (sig['idx'], sig['type'])
        if key not in seen:
            seen.add(key)
            unique_signals.append(sig)
    
    logger.info(f"检测到 {len(unique_signals)} 个唯一信号")
    return df, unique_signals
