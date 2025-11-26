# strategy/signal_filter.py
"""
信号过滤和评分系统 - 提升胜率
"""
import pandas as pd
import numpy as np
from utils.logger import logger
from utils.json_i18n import get_value_safe

def calculate_risk_reward_ratio(df, idx, entry_price, stop_loss, take_profit):
    """
    计算风险回报比（盈亏比）
    
    Args:
        df: 价格数据
        idx: 信号索引
        entry_price: 入场价格
        stop_loss: 止损价格
        take_profit: 止盈价格
    
    Returns:
        风险回报比（盈亏比）
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    if risk <= 0:
        return 0
    
    return reward / risk

def calculate_dynamic_stop_target(df, idx, entry_price, atr, 
                                 atr_mult_stop=1.0, atr_mult_target=2.0,
                                 min_risk_reward=1.5):
    """
    动态计算止损和止盈，确保盈亏比 >= min_risk_reward
    
    Args:
        df: 价格数据
        idx: 信号索引
        entry_price: 入场价格
        atr: ATR 值
        atr_mult_stop: 止损 ATR 倍数
        atr_mult_target: 止盈 ATR 倍数（初始值）
        min_risk_reward: 最小盈亏比要求
    
    Returns:
        (stop_loss, take_profit, risk_reward_ratio, adjusted)
        adjusted: 是否调整了止盈以满足最小盈亏比
    """
    # 初始止损和止盈
    stop_loss = entry_price - atr * atr_mult_stop
    take_profit = entry_price + atr * atr_mult_target
    
    # 计算初始盈亏比
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    
    if risk <= 0:
        return stop_loss, take_profit, 0, False
    
    initial_rr = reward / risk
    
    # 如果初始盈亏比不满足要求，调整止盈
    if initial_rr < min_risk_reward:
        # 计算满足最小盈亏比所需的止盈
        required_reward = risk * min_risk_reward
        take_profit = entry_price + required_reward
        adjusted = True
    else:
        adjusted = False
    
    final_rr = abs(take_profit - entry_price) / risk
    
    return stop_loss, take_profit, final_rr, adjusted

def filter_signal_quality(df, idx, signal_data, min_confirmations=2):
    """
    过滤信号质量，提升胜率
    
    Args:
        df: 价格数据 DataFrame
        idx: 信号索引
        signal_data: 信号数据（包含 rule, feature_packet, llm）
        min_confirmations: 最小确认数量
    
    Returns:
        (is_valid, score, reasons): 是否有效、评分、原因列表
    """
    if idx >= len(df):
        return False, 0, ["索引超出范围"]
    
    row = df.iloc[idx]
    reasons = []
    score = 0
    
    # 1. 趋势确认（+20分）
    if 'ema21' in df.columns and 'ema55' in df.columns and 'ema100' in df.columns:
        ema_bull = row['ema21'] > row['ema55'] > row['ema100']
        if ema_bull:
            score += 20
            reasons.append("EMA多头排列")
    
    # 2. 价格位置确认（+15分）
    if 'ema21' in df.columns and 'ema55' in df.columns:
        price_above_ema = row['close'] > row['ema21'] > row['ema55']
        if price_above_ema:
            score += 15
            reasons.append("价格在EMA上方")
    
    # 3. 成交量确认（+15分）
    # 根据交易模式调整成交量阈值
    import os
    volume_threshold = float(os.getenv('VOLUME_THRESHOLD', '1.2'))
    if 'vol_ma50' in df.columns and not pd.isna(row['vol_ma50']):
        vol_ratio = row['volume'] / row['vol_ma50'] if row['vol_ma50'] > 0 else 0
        if vol_ratio > volume_threshold:
            score += 15
            reasons.append(f"成交量放大({vol_ratio:.2f}x)")
        elif vol_ratio > (volume_threshold + 0.3):
            score += 5  # 额外加分
            reasons.append(f"成交量显著放大({vol_ratio:.2f}x)")
    
    # 4. RSI 确认（+10分）
    if 'rsi14' in df.columns and not pd.isna(row['rsi14']):
        rsi = row['rsi14']
        if 40 < rsi < 70:  # 既不过热也不过冷
            score += 10
            reasons.append(f"RSI健康({rsi:.1f})")
        elif rsi < 30:
            score += 5  # 超卖反弹
            reasons.append(f"RSI超卖({rsi:.1f})")
    
    # 5. MACD 确认（+15分）
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        if not pd.isna(row['macd']) and not pd.isna(row['macd_signal']):
            macd_bull = row['macd'] > row['macd_signal']
            if macd_bull:
                score += 15
                reasons.append("MACD多头")
    
    # 6. 布林带确认（+10分）
    if 'bb_lower' in df.columns and 'bb_middle' in df.columns:
        if not pd.isna(row['bb_lower']) and not pd.isna(row['bb_middle']):
            price_above_bb_mid = row['close'] > row['bb_middle']
            if price_above_bb_mid:
                score += 10
                reasons.append("价格在布林带中轨上方")
    
    # 7. ATR 波动率确认（+10分，避免高波动）
    if 'atr14' in df.columns and not pd.isna(row['atr14']):
        atr_sma = df['atr14'].rolling(20).mean().iloc[idx] if idx >= 20 else row['atr14']
        if not pd.isna(atr_sma) and atr_sma > 0:
            atr_ratio = row['atr14'] / atr_sma
            if atr_ratio < 1.5:  # 波动率不是异常高
                score += 10
                reasons.append(f"波动率正常({atr_ratio:.2f}x)")
            elif atr_ratio > 2.0:
                score -= 10  # 波动率过高，扣分
                reasons.append(f"波动率过高({atr_ratio:.2f}x)")
    
    # 8. Eric Score 确认（如果可用，+15分）
    if 'eric_score_smoothed' in df.columns and not pd.isna(row.get('eric_score_smoothed', np.nan)):
        eric_score = row['eric_score_smoothed']
        if -0.5 < eric_score < 0.5:  # 既不过买也不过卖
            score += 15
            reasons.append(f"Eric Score中性({eric_score:.2f})")
        elif eric_score < -0.7:
            score += 10  # 超卖
            reasons.append(f"Eric Score超卖({eric_score:.2f})")
    
    # 9. 价格动量确认（+10分）
    if idx >= 5:
        price_momentum = (row['close'] - df['close'].iloc[idx-5]) / df['close'].iloc[idx-5]
        if price_momentum > 0:
            score += 10
            reasons.append(f"5周期正动量({price_momentum:.2%})")
    
    # 10. 支撑位确认（+15分）
    if idx >= 20:
        recent_low = df['low'].iloc[max(0, idx-20):idx+1].min()
        support_distance = (row['close'] - recent_low) / recent_low
        if 0 < support_distance < 0.02:  # 接近支撑位但未跌破
            score += 15
            reasons.append(f"接近支撑位({support_distance:.2%})")
    
    # 11. LLM 评分加权（如果可用）
    llm = get_value_safe(signal_data, 'llm', {})
    if isinstance(llm, dict):
        llm_score = get_value_safe(llm, 'score', 0)
        try:
            llm_score = int(float(llm_score))
            if llm_score >= 70:
                score += 20
                reasons.append(f"LLM高分({llm_score})")
            elif llm_score >= 60:
                score += 10
                reasons.append(f"LLM中高分({llm_score})")
        except:
            pass
    
    # 判断是否有效（至少需要 min_confirmations 个确认）
    confirmations = len(reasons)
    is_valid = confirmations >= min_confirmations and score >= 50
    
    return is_valid, score, reasons

def apply_signal_filters(df, enhanced_signals, 
                         min_quality_score=50,
                         min_confirmations=2,
                         min_risk_reward=1.5,
                         min_llm_score=40):
    """
    应用信号过滤器，提升胜率
    
    Args:
        df: 价格数据 DataFrame
        enhanced_signals: 增强信号列表
        min_quality_score: 最小质量评分
        min_confirmations: 最小确认数量
        min_risk_reward: 最小盈亏比要求
        min_llm_score: 最小 LLM 评分
    
    Returns:
        过滤后的信号列表（包含质量评分和盈亏比信息）
    """
    logger.info(f"应用信号过滤器（最小质量评分={min_quality_score}, 最小盈亏比={min_risk_reward}）...")
    
    filtered_signals = []
    skipped_count = 0
    
    for item in enhanced_signals:
        s = get_value_safe(item, 'rule', {})
        idx = get_value_safe(s, 'idx', 0)
        
        # 基本检查
        if idx >= len(df) or idx + 1 >= len(df):
            skipped_count += 1
            continue
        
        # LLM 信号检查
        llm = get_value_safe(item, 'llm', {})
        signal = get_value_safe(llm, 'signal', 'Neutral') if isinstance(llm, dict) else 'Neutral'
        raw_score = get_value_safe(llm, 'score', 0)
        try:
            llm_score = int(float(raw_score))
        except:
            llm_score = 0
        
        # LLM 信号检查（如果 USE_LLM=False，signal 可能是 'Neutral'，需要特殊处理）
        if signal != 'Long':
            # 如果未使用 LLM，fallback 可能返回 'Neutral'，这种情况下如果评分足够高，也允许通过
            if signal == 'Neutral' and llm_score >= min_llm_score:
                # Neutral 但评分足够，可以继续
                pass
            else:
                skipped_count += 1
                continue
        elif llm_score < min_llm_score:
            skipped_count += 1
            continue
        
        # 质量评分检查
        is_valid, quality_score, reasons = filter_signal_quality(df, idx, item, min_confirmations)
        
        if not is_valid or quality_score < min_quality_score:
            skipped_count += 1
            continue
        
        # 计算入场价格和 ATR
        entry_price = df['close'].iloc[idx + 1]
        atr = df['atr14'].iloc[idx + 1] if 'atr14' in df.columns and not pd.isna(df['atr14'].iloc[idx + 1]) else entry_price * 0.01
        
        # 计算动态止损止盈，确保盈亏比 >= min_risk_reward
        stop_loss, take_profit, risk_reward_ratio, adjusted = calculate_dynamic_stop_target(
            df, idx, entry_price, atr,
            atr_mult_stop=1.0,
            atr_mult_target=2.0,
            min_risk_reward=min_risk_reward
        )
        
        # 盈亏比检查
        if risk_reward_ratio < min_risk_reward:
            skipped_count += 1
            continue
        
        # 添加到过滤后的信号列表
        filtered_item = item.copy()
        filtered_item['quality_score'] = quality_score
        filtered_item['quality_reasons'] = reasons
        filtered_item['risk_reward_ratio'] = risk_reward_ratio
        filtered_item['stop_loss'] = stop_loss
        filtered_item['take_profit'] = take_profit
        filtered_item['adjusted_target'] = adjusted
        
        filtered_signals.append(filtered_item)
    
    logger.info(f"信号过滤完成: {len(enhanced_signals)} -> {len(filtered_signals)} (跳过 {skipped_count} 个)")
    
    return filtered_signals

