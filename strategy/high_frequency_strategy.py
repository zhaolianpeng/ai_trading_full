# strategy/high_frequency_strategy.py
"""
高频交易策略
基于多时间周期超买/超卖判断，在小周期寻找反向交易机会

策略逻辑：
1. 日线或4小时线连续处于超买状态 -> 在小时线和5分钟线找机会做空
2. 日线或4小时线连续处于超卖状态 -> 在小时线和5分钟线找机会做多
3. 分钟级高频短单，一天可以多次交易
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from utils.logger import logger
from data.market_data import fetch_market_data
from config import DATA_SOURCE, MARKET_SYMBOL, MARKET_TYPE

def detect_overbought_oversold(df: pd.DataFrame, period: int = 14, 
                                overbought_threshold: float = 70.0,
                                oversold_threshold: float = 30.0) -> pd.Series:
    """
    检测超买超卖状态
    
    Args:
        df: 价格数据DataFrame
        period: RSI周期（默认14）
        overbought_threshold: 超买阈值（默认70）
        oversold_threshold: 超卖阈值（默认30）
    
    Returns:
        超买超卖状态Series（'OVERBOUGHT', 'OVERSOLD', 'NEUTRAL'）
    """
    df = df.copy()
    if 'rsi14' not in df.columns:
        # 如果没有RSI，计算RSI
        from features.ta_basic import rsi
        df['rsi14'] = rsi(df['close'], period)
    
    # 使用numpy.select进行条件判断
    conditions = [
        df['rsi14'] >= overbought_threshold,  # 超买
        df['rsi14'] <= oversold_threshold,    # 超卖
    ]
    choices = ['OVERBOUGHT', 'OVERSOLD']
    
    # 默认中性
    result = pd.Series(np.select(conditions, choices, default='NEUTRAL'), index=df.index)
    
    return result

def check_consecutive_condition(df: pd.DataFrame, condition_col: str, 
                               target_value: str, min_consecutive: int = 3) -> pd.Series:
    """
    检查连续条件
    
    Args:
        df: 价格数据DataFrame
        condition_col: 条件列名
        target_value: 目标值
        min_consecutive: 最小连续次数
    
    Returns:
        布尔Series，表示是否满足连续条件
    """
    if condition_col not in df.columns:
        return pd.Series(False, index=df.index)
    
    # 计算连续次数
    consecutive = (df[condition_col] == target_value).astype(int)
    consecutive_count = consecutive.groupby((consecutive != consecutive.shift()).cumsum()).cumcount() + 1
    consecutive_count = consecutive_count * consecutive  # 只保留满足条件的连续计数
    
    return consecutive_count >= min_consecutive

def find_long_opportunity_h1(df_1h: pd.DataFrame, idx: int, 
                             lookback: int = 20) -> Optional[Dict]:
    """
    在1小时线寻找做多机会
    
    Args:
        df_1h: 1小时K线数据
        idx: 当前索引
        lookback: 回看周期
    
    Returns:
        做多机会字典，如果没有机会则返回None
    """
    if idx < lookback or idx >= len(df_1h):
        return None
    
    row = df_1h.iloc[idx]
    
    # 1. 检查趋势：价格在EMA上方（上升趋势）
    ema_bullish = False
    if all(col in df_1h.columns for col in ['ema21', 'ema55']):
        ema_bullish = row['ema21'] > row['ema55']
    
    # 2. 检查RSI：RSI从低位反弹或处于低位
    rsi_condition = False
    if 'rsi14' in df_1h.columns and not pd.isna(row['rsi14']):
        rsi_val = row['rsi14']
        # RSI < 40 或 RSI从低位反弹
        if rsi_val < 40:
            rsi_condition = True
        elif idx > 0:
            prev_rsi = df_1h['rsi14'].iloc[idx-1]
            if prev_rsi < rsi_val and prev_rsi < 35:  # 从低位反弹
                rsi_condition = True
    
    # 3. 检查价格动量：正动量
    momentum_positive = False
    if idx >= 5:
        momentum = (row['close'] - df_1h['close'].iloc[idx-5]) / df_1h['close'].iloc[idx-5]
        if momentum > 0.001:  # 上涨超过0.1%
            momentum_positive = True
    
    # 4. 检查成交量：成交量放大（确认上涨）
    volume_confirmation = False
    if 'vol_ma50' in df_1h.columns and not pd.isna(row.get('vol_ma50', np.nan)):
        vol_ratio = row['volume'] / row['vol_ma50'] if row['vol_ma50'] > 0 else 1.0
        if vol_ratio > 1.1:  # 成交量放大
            volume_confirmation = True
    
    # 5. 检查价格位置：接近支撑位或从低位反弹
    price_position = False
    if 'sup50' in df_1h.columns and not pd.isna(row.get('sup50', np.nan)):
        dist_to_support = (row['close'] - row['sup50']) / row['sup50']
        if -0.02 < dist_to_support < 0.01:  # 接近支撑位
            price_position = True
    
    # 综合评分
    score = 0
    reasons = []
    
    if ema_bullish:
        score += 30
        reasons.append("EMA多头排列")
    
    if rsi_condition:
        score += 25
        reasons.append("RSI低位或反弹")
    
    if momentum_positive:
        score += 20
        reasons.append("正动量")
    
    if volume_confirmation:
        score += 15
        reasons.append("成交量放大")
    
    if price_position:
        score += 10
        reasons.append("接近支撑位")
    
    # 需要至少60分才认为是有效机会
    if score >= 60:
        return {
            'entry_idx': idx,
            'entry_price': float(row['close']),
            'entry_time': df_1h.index[idx] if isinstance(df_1h.index, pd.DatetimeIndex) else None,
            'direction': 'Long',
            'score': score,
            'reasons': reasons,
            'timeframe': '1h'
        }
    
    return None

def find_short_opportunity_h1(df_1h: pd.DataFrame, idx: int, 
                              lookback: int = 20) -> Optional[Dict]:
    """
    在1小时线寻找做空机会
    
    Args:
        df_1h: 1小时K线数据
        idx: 当前索引
        lookback: 回看周期
    
    Returns:
        做空机会字典，如果没有机会则返回None
    """
    if idx < lookback or idx >= len(df_1h):
        return None
    
    row = df_1h.iloc[idx]
    
    # 1. 检查趋势：价格在EMA下方（下降趋势）
    ema_bearish = False
    if all(col in df_1h.columns for col in ['ema21', 'ema55']):
        ema_bearish = row['ema21'] < row['ema55']
    
    # 2. 检查RSI：RSI从高位回落或处于高位
    rsi_condition = False
    if 'rsi14' in df_1h.columns and not pd.isna(row['rsi14']):
        rsi_val = row['rsi14']
        # RSI > 60 或 RSI从高位回落
        if rsi_val > 60:
            rsi_condition = True
        elif idx > 0:
            prev_rsi = df_1h['rsi14'].iloc[idx-1]
            if prev_rsi > rsi_val and prev_rsi > 65:  # 从高位回落
                rsi_condition = True
    
    # 3. 检查价格动量：负动量
    momentum_negative = False
    if idx >= 5:
        momentum = (row['close'] - df_1h['close'].iloc[idx-5]) / df_1h['close'].iloc[idx-5]
        if momentum < -0.001:  # 下跌超过0.1%
            momentum_negative = True
    
    # 4. 检查成交量：成交量放大（确认下跌）
    volume_confirmation = False
    if 'vol_ma50' in df_1h.columns and not pd.isna(row.get('vol_ma50', np.nan)):
        vol_ratio = row['volume'] / row['vol_ma50'] if row['vol_ma50'] > 0 else 1.0
        if vol_ratio > 1.1:  # 成交量放大
            volume_confirmation = True
    
    # 5. 检查价格位置：接近阻力位或从高位回落
    price_position = False
    if 'res50' in df_1h.columns and not pd.isna(row.get('res50', np.nan)):
        dist_to_resistance = (row['close'] - row['res50']) / row['res50']
        if -0.01 < dist_to_resistance < 0.02:  # 接近阻力位
            price_position = True
    
    # 综合评分
    score = 0
    reasons = []
    
    if ema_bearish:
        score += 30
        reasons.append("EMA空头排列")
    
    if rsi_condition:
        score += 25
        reasons.append("RSI高位或回落")
    
    if momentum_negative:
        score += 20
        reasons.append("负动量")
    
    if volume_confirmation:
        score += 15
        reasons.append("成交量放大")
    
    if price_position:
        score += 10
        reasons.append("接近阻力位")
    
    # 需要至少60分才认为是有效机会
    if score >= 60:
        return {
            'entry_idx': idx,
            'entry_price': float(row['close']),
            'entry_time': df_1h.index[idx] if isinstance(df_1h.index, pd.DatetimeIndex) else None,
            'direction': 'Short',
            'score': score,
            'reasons': reasons,
            'timeframe': '1h'
        }
    
    return None

def find_long_opportunity_5m(df_5m: pd.DataFrame, signal_time: datetime,
                              lookforward_minutes: int = 60) -> Optional[Dict]:
    """
    在5分钟线寻找做多机会
    
    Args:
        df_5m: 5分钟K线数据
        signal_time: 信号时间（从1小时线来的）
        lookforward_minutes: 向前查找的分钟数（默认60分钟）
    
    Returns:
        做多机会字典，如果没有机会则返回None
    """
    if df_5m.empty:
        return None
    
    # 找到信号时间对应的K线
    signal_idx = None
    try:
        if isinstance(df_5m.index, pd.DatetimeIndex):
            time_diffs = abs(df_5m.index - signal_time)
            min_idx = np.argmin(time_diffs.values) if hasattr(time_diffs, 'values') else np.argmin(np.array(time_diffs))
            signal_idx = int(min_idx)
    except:
        signal_idx = len(df_5m) - 1
    
    if signal_idx is None or signal_idx >= len(df_5m) or signal_idx < 0:
        return None
    
    # 从信号时间开始，向前查找（未来）
    best_opportunity = None
    best_score = -float('inf')
    
    # 查找范围：最多12个5分钟K线（60分钟）
    search_range = min(len(df_5m) - signal_idx, 12)
    
    for i in range(signal_idx, min(len(df_5m), signal_idx + search_range)):
        if i >= len(df_5m):
            continue
        
        row = df_5m.iloc[i]
        entry_time = df_5m.index[i]
        
        # 评分因素
        score = 0
        reasons = []
        
        # 1. 价格回调（做多的好时机）
        if signal_idx > 0:
            price_change = (row['close'] - df_5m['close'].iloc[signal_idx]) / df_5m['close'].iloc[signal_idx]
            if -0.01 < price_change <= 0:  # 0-1%的回调
                score += 35
                reasons.append(f"价格回调{abs(price_change)*100:.2f}%")
            elif -0.02 < price_change <= -0.01:  # 1-2%的回调
                score += 25
                reasons.append(f"价格回调{abs(price_change)*100:.2f}%")
        
        # 2. 成交量（成交量放大更好）
        if 'volume' in row and 'vol_ma50' in df_5m.columns:
            vol_ratio = row['volume'] / df_5m['vol_ma50'].iloc[i] if df_5m['vol_ma50'].iloc[i] > 0 else 1.0
            if vol_ratio > 1.5:
                score += 25
                reasons.append(f"成交量放大{vol_ratio:.2f}x")
            elif vol_ratio > 1.2:
                score += 15
                reasons.append(f"成交量放大{vol_ratio:.2f}x")
        
        # 3. K线形态（下影线长表示有支撑）
        if row['close'] > row['open']:  # 阳线
            score += 10
            reasons.append("阳线")
        
        lower_shadow = min(row['open'], row['close']) - row['low']
        body = abs(row['close'] - row['open'])
        if body > 0 and lower_shadow / body > 1.5:  # 下影线是实体的1.5倍以上
            score += 15
            reasons.append("长下影线支撑")
        
        # 4. 时间距离（越近越好）
        time_diff = (entry_time - signal_time).total_seconds() / 60 if isinstance(entry_time, datetime) else 0
        if 0 <= time_diff <= 15:  # 0-15分钟
            score += 20
            reasons.append(f"距离信号{time_diff:.0f}分钟")
        elif 15 < time_diff <= 30:
            score += 15
            reasons.append(f"距离信号{time_diff:.0f}分钟")
        
        # 5. 价格稳定性（波动适中）
        price_range = (row['high'] - row['low']) / row['close']
        if 0.003 < price_range < 0.01:  # 0.3%-1%的波动
            score += 10
            reasons.append("波动适中")
        
        # 更新最佳机会
        if score > best_score:
            best_score = score
            best_opportunity = {
                'entry_idx': i,
                'entry_price': float(min(row['close'], row['low'])),  # 做多用最低价
                'entry_time': entry_time,
                'direction': 'Long',
                'score': score,
                'reasons': reasons,
                'timeframe': '5m'
            }
    
    if best_opportunity and best_score >= 50:
        return best_opportunity
    
    return None

def find_short_opportunity_5m(df_5m: pd.DataFrame, signal_time: datetime,
                              lookforward_minutes: int = 60) -> Optional[Dict]:
    """
    在5分钟线寻找做空机会
    
    Args:
        df_5m: 5分钟K线数据
        signal_time: 信号时间（从1小时线来的）
        lookforward_minutes: 向前查找的分钟数（默认60分钟）
    
    Returns:
        做空机会字典，如果没有机会则返回None
    """
    if df_5m.empty:
        return None
    
    # 找到信号时间对应的K线
    signal_idx = None
    try:
        if isinstance(df_5m.index, pd.DatetimeIndex):
            time_diffs = abs(df_5m.index - signal_time)
            min_idx = np.argmin(time_diffs.values) if hasattr(time_diffs, 'values') else np.argmin(np.array(time_diffs))
            signal_idx = int(min_idx)
    except:
        signal_idx = len(df_5m) - 1
    
    if signal_idx is None or signal_idx >= len(df_5m) or signal_idx < 0:
        return None
    
    # 从信号时间开始，向前查找（未来）
    best_opportunity = None
    best_score = -float('inf')
    
    # 查找范围：最多12个5分钟K线（60分钟）
    search_range = min(len(df_5m) - signal_idx, 12)
    
    for i in range(signal_idx, min(len(df_5m), signal_idx + search_range)):
        if i >= len(df_5m):
            continue
        
        row = df_5m.iloc[i]
        entry_time = df_5m.index[i]
        
        # 评分因素
        score = 0
        reasons = []
        
        # 1. 价格反弹（做空的好时机）
        if signal_idx > 0:
            price_change = (row['close'] - df_5m['close'].iloc[signal_idx]) / df_5m['close'].iloc[signal_idx]
            if 0 < price_change < 0.01:  # 0-1%的反弹
                score += 35
                reasons.append(f"价格反弹{price_change*100:.2f}%")
            elif 0.01 <= price_change < 0.02:  # 1-2%的反弹
                score += 25
                reasons.append(f"价格反弹{price_change*100:.2f}%")
        
        # 2. 成交量（成交量放大更好）
        if 'volume' in row and 'vol_ma50' in df_5m.columns:
            vol_ratio = row['volume'] / df_5m['vol_ma50'].iloc[i] if df_5m['vol_ma50'].iloc[i] > 0 else 1.0
            if vol_ratio > 1.5:
                score += 25
                reasons.append(f"成交量放大{vol_ratio:.2f}x")
            elif vol_ratio > 1.2:
                score += 15
                reasons.append(f"成交量放大{vol_ratio:.2f}x")
        
        # 3. K线形态（上影线长表示有压力）
        if row['close'] < row['open']:  # 阴线
            score += 10
            reasons.append("阴线")
        
        upper_shadow = row['high'] - max(row['open'], row['close'])
        body = abs(row['close'] - row['open'])
        if body > 0 and upper_shadow / body > 1.5:  # 上影线是实体的1.5倍以上
            score += 15
            reasons.append("长上影线压力")
        
        # 4. 时间距离（越近越好）
        time_diff = (entry_time - signal_time).total_seconds() / 60 if isinstance(entry_time, datetime) else 0
        if 0 <= time_diff <= 15:  # 0-15分钟
            score += 20
            reasons.append(f"距离信号{time_diff:.0f}分钟")
        elif 15 < time_diff <= 30:
            score += 15
            reasons.append(f"距离信号{time_diff:.0f}分钟")
        
        # 5. 价格稳定性（波动适中）
        price_range = (row['high'] - row['low']) / row['close']
        if 0.003 < price_range < 0.01:  # 0.3%-1%的波动
            score += 10
            reasons.append("波动适中")
        
        # 更新最佳机会
        if score > best_score:
            best_score = score
            best_opportunity = {
                'entry_idx': i,
                'entry_price': float(max(row['close'], row['high'])),  # 做空用最高价
                'entry_time': entry_time,
                'direction': 'Short',
                'score': score,
                'reasons': reasons,
                'timeframe': '5m'
            }
    
    if best_opportunity and best_score >= 50:
        return best_opportunity
    
    return None

def detect_high_frequency_signals(df_daily: pd.DataFrame = None,
                                  df_4h: pd.DataFrame = None,
                                  df_1h: pd.DataFrame = None,
                                  min_consecutive_overbought: int = 3,
                                  min_consecutive_oversold: int = 3) -> List[Dict]:
    """
    检测高频交易信号
    
    策略：
    1. 日线或4小时线连续超买 -> 在1小时和5分钟线找做空机会
    2. 日线或4小时线连续超卖 -> 在1小时和5分钟线找做多机会
    
    Args:
        df_daily: 日线数据（可选）
        df_4h: 4小时线数据（可选）
        df_1h: 1小时线数据（必需）
        min_consecutive_overbought: 最小连续超买次数
        min_consecutive_oversold: 最小连续超卖次数
    
    Returns:
        高频交易信号列表
    """
    signals = []
    
    if df_1h is None or len(df_1h) < 50:
        logger.warning("1小时线数据不足，无法进行高频交易分析")
        return signals
    
    # 1. 检测大周期超买超卖状态
    daily_overbought = False
    h4_overbought = False
    
    if df_daily is not None and len(df_daily) >= min_consecutive_overbought:
        # 确保有RSI指标
        if 'rsi14' not in df_daily.columns:
            from features.ta_basic import add_basic_ta
            df_daily = add_basic_ta(df_daily)
        
        daily_rsi_status = detect_overbought_oversold(df_daily)
        df_daily['rsi_status'] = daily_rsi_status
        
        daily_consecutive = check_consecutive_condition(
            df_daily, 'rsi_status', 'OVERBOUGHT', min_consecutive_overbought
        )
        if daily_consecutive:
            daily_overbought = True
            logger.info(f"日线连续{min_consecutive_overbought}根K线超买，寻找做空机会")
    
    if df_4h is not None and len(df_4h) >= min_consecutive_overbought:
        # 确保有RSI指标
        if 'rsi14' not in df_4h.columns:
            from features.ta_basic import add_basic_ta
            df_4h = add_basic_ta(df_4h)
        
        h4_rsi_status = detect_overbought_oversold(df_4h)
        df_4h['rsi_status'] = h4_rsi_status
        
        h4_consecutive = check_consecutive_condition(
            df_4h, 'rsi_status', 'OVERBOUGHT', min_consecutive_overbought
        )
        if h4_consecutive:
            h4_overbought = True
            logger.info(f"4小时线连续{min_consecutive_overbought}根K线超买，寻找做空机会")
    
    # 2. 如果大周期超买，在1小时线找做空机会
    if daily_overbought or h4_overbought:
        # 确保1小时线有必要的指标
        if 'rsi14' not in df_1h.columns:
            from features.ta_basic import add_basic_ta
            df_1h = add_basic_ta(df_1h)
        
        # 从最近50根K线开始查找
        start_idx = max(50, len(df_1h) - 50)
        for idx in range(start_idx, len(df_1h)):
            opportunity = find_short_opportunity_h1(df_1h, idx)
            if opportunity:
                # 添加大周期信息
                opportunity['higher_timeframe_signal'] = {
                    'daily_overbought': daily_overbought,
                    'h4_overbought': h4_overbought,
                    'signal_type': 'HIGH_FREQ_SHORT'
                }
                signals.append(opportunity)
                logger.info(f"1小时线找到做空机会: 价格={opportunity['entry_price']:.2f}, "
                           f"评分={opportunity['score']}, 原因={', '.join(opportunity['reasons'])}")
    
    # 3. 检测大周期超卖（做多机会）
    daily_oversold = False
    h4_oversold = False
    
    if df_daily is not None and len(df_daily) >= min_consecutive_oversold:
        # 确保有RSI指标
        if 'rsi14' not in df_daily.columns:
            from features.ta_basic import add_basic_ta
            df_daily = add_basic_ta(df_daily)
        
        daily_rsi_status = detect_overbought_oversold(df_daily)
        if 'rsi_status' not in df_daily.columns:
            df_daily['rsi_status'] = daily_rsi_status
        
        daily_consecutive = check_consecutive_condition(
            df_daily, 'rsi_status', 'OVERSOLD', min_consecutive_oversold
        )
        if daily_consecutive:
            daily_oversold = True
            logger.info(f"日线连续{min_consecutive_oversold}根K线超卖，寻找做多机会")
    
    if df_4h is not None and len(df_4h) >= min_consecutive_oversold:
        # 确保有RSI指标
        if 'rsi14' not in df_4h.columns:
            from features.ta_basic import add_basic_ta
            df_4h = add_basic_ta(df_4h)
        
        h4_rsi_status = detect_overbought_oversold(df_4h)
        if 'rsi_status' not in df_4h.columns:
            df_4h['rsi_status'] = h4_rsi_status
        
        h4_consecutive = check_consecutive_condition(
            df_4h, 'rsi_status', 'OVERSOLD', min_consecutive_oversold
        )
        if h4_consecutive:
            h4_oversold = True
            logger.info(f"4小时线连续{min_consecutive_oversold}根K线超卖，寻找做多机会")
    
    # 4. 如果大周期超卖，在1小时线找做多机会
    if daily_oversold or h4_oversold:
        # 确保1小时线有必要的指标
        if 'rsi14' not in df_1h.columns:
            from features.ta_basic import add_basic_ta
            df_1h = add_basic_ta(df_1h)
        
        # 从最近50根K线开始查找
        start_idx = max(50, len(df_1h) - 50)
        for idx in range(start_idx, len(df_1h)):
            opportunity = find_long_opportunity_h1(df_1h, idx)
            if opportunity:
                # 添加大周期信息
                opportunity['higher_timeframe_signal'] = {
                    'daily_oversold': daily_oversold,
                    'h4_oversold': h4_oversold,
                    'signal_type': 'HIGH_FREQ_LONG'
                }
                signals.append(opportunity)
                logger.info(f"1小时线找到做多机会: 价格={opportunity['entry_price']:.2f}, "
                           f"评分={opportunity['score']}, 原因={', '.join(opportunity['reasons'])}")
    
    return signals

def enhance_with_5m_entry(signal: Dict, df_5m: pd.DataFrame = None) -> Dict:
    """
    使用5分钟线优化入场点
    
    Args:
        signal: 1小时线信号
        df_5m: 5分钟线数据
    
    Returns:
        增强后的信号（包含5分钟入场点）
    """
    if df_5m is None or df_5m.empty:
        return signal
    
    entry_time = signal.get('entry_time')
    if entry_time is None:
        return signal
    
    if signal['direction'] == 'Short':
        # 做空：在5分钟线找反弹点
        opportunity_5m = find_short_opportunity_5m(df_5m, entry_time)
        if opportunity_5m:
            signal['best_entry_5m'] = opportunity_5m
            signal['entry_price'] = opportunity_5m['entry_price']  # 使用5分钟入场价
            signal['entry_time'] = opportunity_5m['entry_time']
            logger.info(f"5分钟线优化入场点: 价格={opportunity_5m['entry_price']:.2f}, "
                       f"评分={opportunity_5m['score']}")
    else:
        # 做多：在5分钟线找回调点
        opportunity_5m = find_long_opportunity_5m(df_5m, entry_time)
        if opportunity_5m:
            signal['best_entry_5m'] = opportunity_5m
            signal['entry_price'] = opportunity_5m['entry_price']  # 使用5分钟入场价
            signal['entry_time'] = opportunity_5m['entry_time']
            logger.info(f"5分钟线优化入场点: 价格={opportunity_5m['entry_price']:.2f}, "
                       f"评分={opportunity_5m['score']}")
    
    return signal

