# strategy/market_structure_analyzer.py
"""
量化市场结构分析模块
根据K线数据判断市场结构、趋势强度、市场情绪等
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import logger

def calculate_trend_strength(df: pd.DataFrame, n: int = 50) -> float:
    """
    根据过去 n 根 K 线，给出趋势强度评分（0-100）
    评分基于斜率、一致性、突破、均线排列、ATR
    
    Args:
        df: K线数据 DataFrame
        n: 使用的K线数量
    
    Returns:
        趋势强度评分（0-100）
    """
    if len(df) < n:
        n = len(df)
    
    recent_df = df.iloc[-n:].copy()
    
    # 1. 斜率评分（0-25分）
    if len(recent_df) > 1:
        price_change = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc[0]
        slope_score = min(25, abs(price_change) * 1000)  # 每1%变化得10分，最高25分
    else:
        slope_score = 0
    
    # 2. 一致性评分（0-25分）
    # 计算价格是否持续朝一个方向移动
    closes = recent_df['close'].values
    up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
    down_days = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
    consistency = max(up_days, down_days) / len(closes) if len(closes) > 1 else 0
    consistency_score = consistency * 25
    
    # 3. 突破评分（0-20分）
    # 检查是否有明显的突破
    if len(recent_df) > 20:
        high_20 = recent_df['high'].iloc[-20:].max()
        low_20 = recent_df['low'].iloc[-20:].min()
        current_price = recent_df['close'].iloc[-1]
        
        # 突破高点或低点
        if current_price > high_20 * 0.99:
            breakout_score = 20
        elif current_price < low_20 * 1.01:
            breakout_score = 15
        else:
            breakout_score = 5
    else:
        breakout_score = 0
    
    # 4. 均线排列评分（0-20分）
    if 'ema21' in recent_df.columns and 'ema55' in recent_df.columns:
        ema21 = recent_df['ema21'].iloc[-1]
        ema55 = recent_df['ema55'].iloc[-1]
        current_price = recent_df['close'].iloc[-1]
        
        # 多头排列：价格 > EMA21 > EMA55
        if current_price > ema21 > ema55:
            ma_score = 20
        # 空头排列：价格 < EMA21 < EMA55
        elif current_price < ema21 < ema55:
            ma_score = 15
        else:
            ma_score = 5
    else:
        ma_score = 0
    
    # 5. ATR评分（0-10分）
    # 波动率适中时趋势更可靠
    if 'atr14' in recent_df.columns:
        atr = recent_df['atr14'].iloc[-1]
        price = recent_df['close'].iloc[-1]
        atr_pct = (atr / price) * 100 if price > 0 else 0
        
        # ATR在1-3%之间时得分最高
        if 1 <= atr_pct <= 3:
            atr_score = 10
        elif 0.5 <= atr_pct < 1 or 3 < atr_pct <= 5:
            atr_score = 7
        else:
            atr_score = 3
    else:
        atr_score = 0
    
    total_score = slope_score + consistency_score + breakout_score + ma_score + atr_score
    return min(100, max(0, total_score))

def classify_market_regime(df: pd.DataFrame, n: int = 50) -> str:
    """
    将市场分为：TREND、MEAN_REVERT、HIGH_VOL、LOW_VOL、CHAOS
    
    Args:
        df: K线数据 DataFrame
        n: 使用的K线数量
    
    Returns:
        市场类型标签
    """
    if len(df) < n:
        n = len(df)
    
    recent_df = df.iloc[-n:].copy()
    
    # 计算波动率
    if 'atr14' in recent_df.columns:
        atr = recent_df['atr14'].iloc[-1]
        price = recent_df['close'].iloc[-1]
        vol_pct = (atr / price) * 100 if price > 0 else 0
    else:
        # 使用价格范围估算波动率
        price_range = (recent_df['high'].max() - recent_df['low'].min()) / recent_df['close'].mean()
        vol_pct = price_range * 100
    
    # 计算趋势强度
    trend_strength = calculate_trend_strength(df, n)
    
    # 计算成交量
    if 'volume' in recent_df.columns:
        avg_volume = recent_df['volume'].mean()
        recent_volume = recent_df['volume'].iloc[-5:].mean()
        vol_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    else:
        vol_ratio = 1
    
    # 判断市场类型
    if vol_pct > 5:
        return "HIGH_VOL"
    elif vol_pct < 0.5:
        return "LOW_VOL"
    elif trend_strength > 60:
        return "TREND"
    elif trend_strength < 30 and vol_pct < 2:
        return "MEAN_REVERT"
    else:
        return "CHAOS"

def analyze_market_structure(df: pd.DataFrame, n: int = 50) -> str:
    """
    根据 n 根 K 线数据，判断当前的市场结构
    
    Args:
        df: K线数据 DataFrame
        n: 使用的K线数量（默认50）
    
    Returns:
        市场结构标签：TREND_UP、TREND_DOWN、RANGE、REVERSAL_UP、REVERSAL_DOWN、
                     BREAKOUT_UP、BREAKOUT_DOWN、LOW_VOL、HIGH_VOL
    """
    if len(df) < n:
        n = len(df)
    
    recent_df = df.iloc[-n:].copy()
    
    # 计算基本指标
    closes = recent_df['close'].values
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    
    # 1. 趋势判断
    price_change = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
    trend_strength = calculate_trend_strength(df, n)
    
    # 2. 波动率判断
    if 'atr14' in recent_df.columns:
        atr = recent_df['atr14'].iloc[-1]
        price = recent_df['close'].iloc[-1]
        vol_pct = (atr / price) * 100 if price > 0 else 0
    else:
        price_range = (highs.max() - lows.min()) / closes.mean()
        vol_pct = price_range * 100
    
    # 3. 成交量判断
    if 'volume' in recent_df.columns:
        avg_volume = recent_df['volume'].mean()
        recent_volume = recent_df['volume'].iloc[-10:].mean()
        vol_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    else:
        vol_ratio = 1
    
    # 4. 突破判断
    if len(recent_df) > 20:
        high_20 = highs[-20:].max()
        low_20 = lows[-20:].min()
        current_price = closes[-1]
        range_size = high_20 - low_20
        
        # 突破高点
        if current_price > high_20 * 0.995:
            if vol_ratio > 1.2:
                return "BREAKOUT_UP"
            else:
                return "TREND_UP"
        # 突破低点
        elif current_price < low_20 * 1.005:
            if vol_ratio > 1.2:
                return "BREAKOUT_DOWN"
            else:
                return "TREND_DOWN"
    
    # 5. 反转判断
    if len(recent_df) > 10:
        # 检查最近是否有反转迹象
        recent_trend = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] > 0 else 0
        earlier_trend = (closes[-10] - closes[-20]) / closes[-20] if len(closes) > 20 and closes[-20] > 0 else 0
        
        # 从下跌转为上涨
        if earlier_trend < -0.02 and recent_trend > 0.01:
            return "REVERSAL_UP"
        # 从上涨转为下跌
        elif earlier_trend > 0.02 and recent_trend < -0.01:
            return "REVERSAL_DOWN"
    
    # 6. 趋势判断
    if trend_strength > 50:
        if price_change > 0.01:
            return "TREND_UP"
        elif price_change < -0.01:
            return "TREND_DOWN"
    
    # 7. 波动率判断
    if vol_pct > 5:
        return "HIGH_VOL"
    elif vol_pct < 0.5:
        return "LOW_VOL"
    
    # 8. 默认：震荡
    return "RANGE"

def analyze_market_sentiment(df: pd.DataFrame, n: int = 100) -> str:
    """
    将最近 n 根 K 线的市场情绪分类
    
    Args:
        df: K线数据 DataFrame
        n: 使用的K线数量（默认100）
    
    Returns:
        市场情绪标签：FEAR、GREED、PANIC、EUPHORIA、NEUTRAL
    """
    if len(df) < n:
        n = len(df)
    
    recent_df = df.iloc[-n:].copy()
    
    # 计算价格变化
    price_change = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc[0] if recent_df['close'].iloc[0] > 0 else 0
    
    # 计算波动率
    if 'atr14' in recent_df.columns:
        atr = recent_df['atr14'].iloc[-1]
        price = recent_df['close'].iloc[-1]
        vol_pct = (atr / price) * 100 if price > 0 else 0
    else:
        price_range = (recent_df['high'].max() - recent_df['low'].min()) / recent_df['close'].mean()
        vol_pct = price_range * 100
    
    # 计算最大回撤
    if len(recent_df) > 1:
        cumulative = (1 + recent_df['close'].pct_change()).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0
    
    # 计算成交量
    if 'volume' in recent_df.columns:
        avg_volume = recent_df['volume'].mean()
        recent_volume = recent_df['volume'].iloc[-10:].mean()
        vol_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    else:
        vol_ratio = 1
    
    # 判断情绪
    # PANIC: 大幅下跌 + 高波动 + 高成交量
    if price_change < -0.1 and vol_pct > 3 and vol_ratio > 1.5:
        return "PANIC"
    # FEAR: 下跌 + 高波动
    elif price_change < -0.05 and vol_pct > 2:
        return "FEAR"
    # EUPHORIA: 大幅上涨 + 高波动 + 高成交量
    elif price_change > 0.1 and vol_pct > 3 and vol_ratio > 1.5:
        return "EUPHORIA"
    # GREED: 上涨 + 高波动
    elif price_change > 0.05 and vol_pct > 2:
        return "GREED"
    else:
        return "NEUTRAL"

def calculate_reversal_probability(df: pd.DataFrame, n: int = 10) -> float:
    """
    给出最近 n 根 K 线反转概率评分（0-1）
    
    Args:
        df: K线数据 DataFrame
        n: 使用的K线数量（默认10）
    
    Returns:
        反转概率（0-1）
    """
    if len(df) < n:
        n = len(df)
    
    recent_df = df.iloc[-n:].copy()
    closes = recent_df['close'].values
    
    if len(closes) < 3:
        return 0.0
    
    # 1. 趋势强度（趋势越强，反转概率越低）
    price_change = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0
    trend_strength = abs(price_change) * 10  # 转换为0-1范围
    reversal_from_trend = min(1.0, trend_strength)
    
    # 2. RSI背离（如果有RSI指标）
    if 'rsi14' in recent_df.columns:
        rsi = recent_df['rsi14'].values
        # 价格创新高但RSI未创新高（看跌背离）
        if closes[-1] > closes[:-1].max() and rsi[-1] < rsi[:-1].max():
            rsi_divergence = 0.3
        # 价格创新低但RSI未创新低（看涨背离）
        elif closes[-1] < closes[:-1].min() and rsi[-1] > rsi[:-1].min():
            rsi_divergence = 0.3
        else:
            rsi_divergence = 0.0
    else:
        rsi_divergence = 0.0
    
    # 3. 超买超卖
    if 'rsi14' in recent_df.columns:
        rsi = recent_df['rsi14'].iloc[-1]
        if rsi > 70:  # 超买
            overbought_oversold = 0.4
        elif rsi < 30:  # 超卖
            overbought_oversold = 0.4
        else:
            overbought_oversold = 0.1
    else:
        overbought_oversold = 0.0
    
    # 4. 成交量萎缩（趋势可能结束）
    if 'volume' in recent_df.columns:
        avg_volume = recent_df['volume'].iloc[:-1].mean() if len(recent_df) > 1 else recent_df['volume'].iloc[0]
        recent_volume = recent_df['volume'].iloc[-1]
        if recent_volume < avg_volume * 0.7:
            volume_decline = 0.2
        else:
            volume_decline = 0.0
    else:
        volume_decline = 0.0
    
    # 5. 价格接近支撑/阻力
    if len(recent_df) > 20:
        high_20 = recent_df['high'].iloc[-20:].max()
        low_20 = recent_df['low'].iloc[-20:].min()
        current_price = recent_df['close'].iloc[-1]
        range_size = high_20 - low_20
        
        # 接近阻力
        if (high_20 - current_price) / range_size < 0.1:
            near_level = 0.3
        # 接近支撑
        elif (current_price - low_20) / range_size < 0.1:
            near_level = 0.3
        else:
            near_level = 0.0
    else:
        near_level = 0.0
    
    # 综合反转概率
    reversal_prob = min(1.0, reversal_from_trend + rsi_divergence + overbought_oversold + volume_decline + near_level)
    return reversal_prob

def detect_structure_switch(df: pd.DataFrame, n: int = 100) -> str:
    """
    判断市场是否处于结构切换（trend→range 或 range→trend）
    
    Args:
        df: K线数据 DataFrame
        n: 使用的K线数量（默认100）
    
    Returns:
        "YES" 或 "NO"
    """
    if len(df) < n:
        n = len(df)
    
    recent_df = df.iloc[-n:].copy()
    
    # 将数据分为两段：前半段和后半段
    mid_point = n // 2
    first_half = recent_df.iloc[:mid_point]
    second_half = recent_df.iloc[mid_point:]
    
    # 判断前半段的市场类型
    first_strength = calculate_trend_strength(first_half, len(first_half))
    first_structure = "TREND" if first_strength > 50 else "RANGE"
    
    # 判断后半段的市场类型
    second_strength = calculate_trend_strength(second_half, len(second_half))
    second_structure = "TREND" if second_strength > 50 else "RANGE"
    
    # 如果前后两段的市场类型不同，说明发生了切换
    if first_structure != second_structure:
        return "YES"
    else:
        return "NO"

def generate_quantitative_features(df: pd.DataFrame, n: int = 50) -> Dict[str, float]:
    """
    针对数据集，生成 10 个可量化的、非未来函数的特征
    
    Args:
        df: K线数据 DataFrame
        n: 使用的K线数量（默认50）
    
    Returns:
        包含10个特征的字典
    """
    if len(df) < n:
        n = len(df)
    
    recent_df = df.iloc[-n:].copy()
    
    features = {}
    
    # 1. 价格动量（最近5根K线的平均收益率）
    if len(recent_df) >= 5:
        returns = recent_df['close'].pct_change().dropna()
        features['momentum_5'] = returns.iloc[-5:].mean() if len(returns) >= 5 else 0.0
    else:
        features['momentum_5'] = 0.0
    
    # 2. 波动率（ATR相对于价格的百分比）
    if 'atr14' in recent_df.columns:
        atr = recent_df['atr14'].iloc[-1]
        price = recent_df['close'].iloc[-1]
        features['volatility_atr'] = (atr / price) * 100 if price > 0 else 0.0
    else:
        price_range = (recent_df['high'].max() - recent_df['low'].min()) / recent_df['close'].mean()
        features['volatility_atr'] = price_range * 100
    
    # 3. 趋势强度（基于线性回归斜率）
    if len(recent_df) > 1:
        x = np.arange(len(recent_df))
        y = recent_df['close'].values
        slope = np.polyfit(x, y, 1)[0]
        price_mean = recent_df['close'].mean()
        features['trend_slope'] = (slope / price_mean) * 100 if price_mean > 0 else 0.0
    else:
        features['trend_slope'] = 0.0
    
    # 4. 均线距离（价格与EMA21的距离百分比）
    if 'ema21' in recent_df.columns:
        price = recent_df['close'].iloc[-1]
        ema21 = recent_df['ema21'].iloc[-1]
        features['ema21_distance'] = ((price - ema21) / ema21) * 100 if ema21 > 0 else 0.0
    else:
        features['ema21_distance'] = 0.0
    
    # 5. 成交量比率（最近10根K线平均成交量 / 整体平均成交量）
    if 'volume' in recent_df.columns:
        avg_volume = recent_df['volume'].mean()
        recent_volume = recent_df['volume'].iloc[-10:].mean() if len(recent_df) >= 10 else recent_df['volume'].iloc[-1]
        features['volume_ratio'] = recent_volume / avg_volume if avg_volume > 0 else 1.0
    else:
        features['volume_ratio'] = 1.0
    
    # 6. RSI值（如果有）
    if 'rsi14' in recent_df.columns:
        features['rsi14'] = recent_df['rsi14'].iloc[-1]
    else:
        features['rsi14'] = 50.0  # 中性值
    
    # 7. 价格位置（当前价格在最近n根K线价格区间中的位置，0-100）
    high_n = recent_df['high'].max()
    low_n = recent_df['low'].min()
    current_price = recent_df['close'].iloc[-1]
    if high_n > low_n:
        features['price_position'] = ((current_price - low_n) / (high_n - low_n)) * 100
    else:
        features['price_position'] = 50.0
    
    # 8. 连续上涨/下跌天数
    closes = recent_df['close'].values
    consecutive = 0
    direction = 0  # 1 for up, -1 for down
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i-1]:
            if direction == 1:
                consecutive += 1
            else:
                consecutive = 1
                direction = 1
        elif closes[i] < closes[i-1]:
            if direction == -1:
                consecutive += 1
            else:
                consecutive = 1
                direction = -1
        else:
            break
    features['consecutive_days'] = consecutive * direction  # 正数表示连续上涨，负数表示连续下跌
    
    # 9. 最大回撤（最近n根K线的最大回撤百分比）
    if len(recent_df) > 1:
        cumulative = (1 + recent_df['close'].pct_change()).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        features['max_drawdown'] = abs(drawdown.min()) * 100
    else:
        features['max_drawdown'] = 0.0
    
    # 10. 价格效率（价格变化 / 价格波动范围，衡量趋势的"干净"程度）
    price_change = abs(recent_df['close'].iloc[-1] - recent_df['close'].iloc[0])
    price_range = recent_df['high'].max() - recent_df['low'].min()
    features['price_efficiency'] = (price_change / price_range) * 100 if price_range > 0 else 0.0
    
    return features

def comprehensive_market_analysis(df: pd.DataFrame, n: int = 50) -> Dict:
    """
    综合市场分析，返回所有分析结果
    
    Args:
        df: K线数据 DataFrame
        n: 使用的K线数量（默认50）
    
    Returns:
        包含所有分析结果的字典
    """
    result = {
        'market_structure': analyze_market_structure(df, n),
        'trend_strength': calculate_trend_strength(df, n),
        'market_regime': classify_market_regime(df, n),
        'market_sentiment': analyze_market_sentiment(df, min(100, len(df))),
        'reversal_probability': calculate_reversal_probability(df, min(10, len(df))),
        'structure_switch': detect_structure_switch(df, min(100, len(df))),
        'quantitative_features': generate_quantitative_features(df, n)
    }
    
    return result

