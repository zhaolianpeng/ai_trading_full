# strategy/signal_scorer.py
"""
信号评分系统
基于多因子构造原始信号得分（trend_score、momentum_score、vol_score、volume_score等）
合并成候选交易信号
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.logger import logger

def calculate_signal_scores(df: pd.DataFrame, idx: int, signal_data: Dict) -> Dict[str, float]:
    """
    计算信号的各项得分
    
    Args:
        df: 价格数据DataFrame
        idx: 信号索引
        signal_data: 信号数据字典
    
    Returns:
        包含各项得分的字典
    """
    if idx >= len(df):
        return {}
    
    row = df.iloc[idx]
    packet = signal_data.get('feature_packet', {})
    
    scores = {}
    
    # 1. 趋势得分
    if 'trend_score' in df.columns:
        scores['trend_score'] = float(row['trend_score']) if not pd.isna(row['trend_score']) else 0.0
    else:
        # 如果没有trend_score，使用EMA排列估算
        if all(col in df.columns for col in ['ema21', 'ema55', 'ema100']):
            ema_score = 0.0
            if row['ema21'] > row['ema55'] > row['ema100']:
                ema_score += 30
            if row['close'] > row['ema21']:
                ema_score += 20
            scores['trend_score'] = ema_score
        else:
            scores['trend_score'] = 0.0
    
    # 2. 动量得分
    if 'momentum_score' in df.columns:
        scores['momentum_score'] = float(row['momentum_score']) if not pd.isna(row['momentum_score']) else 0.0
    else:
        # 如果没有momentum_score，使用价格动量估算
        momentum_5 = packet.get('price_momentum_5', 0)
        momentum_20 = packet.get('price_momentum_20', 0)
        momentum_score = 0.0
        if momentum_5 and momentum_5 > 0:
            momentum_score += 25
        if momentum_20 and momentum_20 > 0:
            momentum_score += 25
        scores['momentum_score'] = momentum_score
    
    # 3. 波动率得分
    if 'vol_score' in df.columns:
        scores['vol_score'] = float(row['vol_score']) if not pd.isna(row['vol_score']) else 0.0
    else:
        # 如果没有vol_score，使用ATR估算
        atr_pct = packet.get('atr_pct', 0)
        if atr_pct:
            if 0.5 <= atr_pct <= 3.0:
                vol_score = 100 - abs(atr_pct - 1.5) * 20
            else:
                vol_score = max(0, 50 - abs(atr_pct - 1.5) * 10)
            scores['vol_score'] = vol_score
        else:
            scores['vol_score'] = 50.0
    
    # 4. 成交量得分
    if 'volume_score' in df.columns:
        scores['volume_score'] = float(row['volume_score']) if not pd.isna(row['volume_score']) else 0.0
    else:
        # 如果没有volume_score，使用vol_ratio估算
        vol_ratio = packet.get('vol_ratio', 1.0)
        if vol_ratio:
            if 1.0 <= vol_ratio <= 3.0:
                volume_score = 50 + (vol_ratio - 1.0) * 25
            elif vol_ratio > 3.0:
                volume_score = max(50, 100 - (vol_ratio - 3.0) * 10)
            else:
                volume_score = vol_ratio * 50
            scores['volume_score'] = volume_score
        else:
            scores['volume_score'] = 50.0
    
    # 5. RSI得分
    if 'rsi14' in df.columns and not pd.isna(row['rsi14']):
        rsi_val = row['rsi14']
        if 40 <= rsi_val <= 70:
            scores['rsi_score'] = 100.0
        elif 30 <= rsi_val < 40 or 70 < rsi_val <= 80:
            scores['rsi_score'] = 70.0
        elif rsi_val < 30:
            scores['rsi_score'] = 50.0  # 超卖，可能反弹
        else:
            scores['rsi_score'] = 30.0  # 超买
    else:
        scores['rsi_score'] = 50.0
    
    # 6. MACD得分（如果可用）
    if packet.get('macd_bullish'):
        scores['macd_score'] = 80.0
    elif packet.get('macd_hist') and packet.get('macd_hist') > 0:
        scores['macd_score'] = 60.0
    else:
        scores['macd_score'] = 40.0
    
    # 7. 布林带得分（如果可用）
    if packet.get('price_above_bb_mid'):
        scores['bb_score'] = 70.0
    else:
        scores['bb_score'] = 30.0
    
    # 8. ADX得分（如果可用）
    if packet.get('adx'):
        adx_val = packet.get('adx')
        if adx_val > 25:
            scores['adx_score'] = 100.0
        elif adx_val > 20:
            scores['adx_score'] = 70.0
        else:
            scores['adx_score'] = 40.0
    else:
        scores['adx_score'] = 50.0
    
    # 9. 综合得分（加权平均）
    weights = {
        'trend_score': 0.25,
        'momentum_score': 0.20,
        'vol_score': 0.15,
        'volume_score': 0.15,
        'rsi_score': 0.10,
        'macd_score': 0.05,
        'bb_score': 0.05,
        'adx_score': 0.05,
    }
    
    composite_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())
    scores['composite_score'] = composite_score
    
    return scores

def merge_signal_scores(signal_data: Dict, scores: Dict[str, float]) -> Dict:
    """
    将得分合并到信号数据中
    
    Args:
        signal_data: 原始信号数据
        scores: 得分字典
    
    Returns:
        合并后的信号数据
    """
    enhanced = signal_data.copy()
    enhanced['signal_scores'] = scores
    enhanced['composite_score'] = scores.get('composite_score', 0.0)
    return enhanced

def filter_by_composite_score(signals: List[Dict], min_score: float = 60.0) -> List[Dict]:
    """
    根据综合得分过滤信号
    
    Args:
        signals: 信号列表
        min_score: 最小综合得分
    
    Returns:
        过滤后的信号列表
    """
    filtered = []
    for signal in signals:
        composite_score = signal.get('composite_score', 0.0)
        if composite_score >= min_score:
            filtered.append(signal)
        else:
            logger.debug(f"信号 {signal.get('rule', {}).get('idx', -1)} 综合得分 {composite_score:.2f} < {min_score}，已过滤")
    
    logger.info(f"综合得分过滤: {len(signals)} -> {len(filtered)} (阈值={min_score})")
    return filtered

