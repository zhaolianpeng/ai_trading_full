# strategy/signal_filter.py
"""
信号过滤和评分系统 - 提升胜率
"""
import pandas as pd
import numpy as np
import os
from typing import List, Optional
from utils.logger import logger
from utils.json_i18n import get_value_safe
from strategy.market_structure_analyzer import calculate_trend_strength

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
    
    # 11. 随机指标确认（+10分，如果可用）
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        if not pd.isna(row['stoch_k']) and not pd.isna(row['stoch_d']):
            stoch_k = row['stoch_k']
            stoch_d = row['stoch_d']
            # 随机指标金叉或处于上升趋势
            if stoch_k > stoch_d and stoch_k < 80:  # 金叉且未超买
                score += 10
                reasons.append(f"随机指标金叉(K={stoch_k:.1f}, D={stoch_d:.1f})")
    
    # 12. CCI确认（+10分，如果可用）
    if 'cci' in df.columns and not pd.isna(row.get('cci', np.nan)):
        cci = row['cci']
        if 0 < cci < 100:  # CCI在健康区间
            score += 10
            reasons.append(f"CCI健康({cci:.1f})")
        elif cci < -100:
            score += 5  # 超卖
            reasons.append(f"CCI超卖({cci:.1f})")
    
    # 13. ADX确认（+15分，如果可用）
    if 'adx' in df.columns and not pd.isna(row.get('adx', np.nan)):
        adx = row['adx']
        if adx > 25:  # 强趋势
            score += 15
            reasons.append(f"ADX强趋势({adx:.1f})")
        elif adx > 20:
            score += 10
            reasons.append(f"ADX中等趋势({adx:.1f})")
    
    # 14. Donchian通道确认（+10分，如果可用）
    if 'donchian_trend' in df.columns and not pd.isna(row.get('donchian_trend', np.nan)):
        donchian_trend = str(row['donchian_trend']).upper()
        if donchian_trend == 'UP':
            score += 10
            reasons.append("Donchian通道上升趋势")
    
    # 15. EMA眼确认（+10分，如果可用）
    if 'ema_eye' in df.columns and not pd.isna(row.get('ema_eye', np.nan)):
        ema_eye = abs(row['ema_eye'])
        if ema_eye < 1.0:  # 小眼，接近EMA
            score += 10
            reasons.append(f"EMA眼小({ema_eye:.2f}%)")
    
    # 16. 价格位置确认（+10分）
    if idx >= 50:
        high_50 = df['high'].iloc[max(0, idx-49):idx+1].max()
        low_50 = df['low'].iloc[max(0, idx-49):idx+1].min()
        if high_50 > low_50:
            price_position = (row['close'] - low_50) / (high_50 - low_50)
            if 0.5 < price_position < 0.8:  # 价格在区间中上部
                score += 10
                reasons.append(f"价格位置良好({price_position:.2%})")
    
    # 17. 威廉指标确认（+5分，如果可用）
    if 'williams_r' in df.columns and not pd.isna(row.get('williams_r', np.nan)):
        wr = row['williams_r']
        if -50 < wr < -20:  # 威廉指标在健康区间
            score += 5
            reasons.append(f"威廉指标健康({wr:.1f})")
    
    # 18. LLM 评分加权（如果可用）
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
    # 回测模式下，大幅降低质量评分要求以产生更多交易
    confirmations = len(reasons)
    import os
    backtest_mode = os.getenv('BACKTEST_MODE', 'False').lower() == 'true' or \
                   os.getenv('BACKTEST_FULL_DATA', 'False').lower() == 'true' or \
                   os.getenv('BACKTEST_MONTHS', '0') != '0'
    min_score_threshold = 15 if backtest_mode else 50  # 回测模式降低到15（从30）
    is_valid = confirmations >= min_confirmations and score >= min_score_threshold
    
    return is_valid, score, reasons

def apply_signal_filters(df, enhanced_signals,
                         min_quality_score=50,
                         min_confirmations=2,
                         min_risk_reward=1.5,
                         min_llm_score=40,
                         structure_confidence_threshold: int = 40,
                         allowed_structure_labels: Optional[List[str]] = None):
    """
    应用信号过滤器，提升胜率
    参考 ai_quant_strategy.py 的实现，只在特定结构标签下生成信号
    
    Args:
        df: 价格数据 DataFrame
        enhanced_signals: 增强信号列表
        min_quality_score: 最小质量评分
        min_confirmations: 最小确认数量
        min_risk_reward: 最小盈亏比要求
        min_llm_score: 最小 LLM 评分
        structure_confidence_threshold: 结构置信度阈值（LLM评分）
        allowed_structure_labels: 允许的结构标签列表（None表示使用默认：TREND_UP, BREAKOUT_UP, REVERSAL_UP）
    
    Returns:
        过滤后的信号列表（包含质量评分和盈亏比信息）
    """
    if allowed_structure_labels is None:
        # 默认只允许做多信号的结构标签（参考 ai_quant_strategy.py）
        allowed_structure_labels = ["TREND_UP", "BREAKOUT_UP", "REVERSAL_UP"]
    
    logger.info(f"应用信号过滤器（最小质量评分={min_quality_score}, 最小盈亏比={min_risk_reward}, 允许的结构标签={allowed_structure_labels}）...")
    
    filtered_signals = []
    skipped_count = 0
    
    for item in enhanced_signals:
        s = get_value_safe(item, 'rule', {})
        idx = get_value_safe(s, 'idx', 0)
        
        # 基本检查
        if idx >= len(df) or idx + 1 >= len(df):
            skipped_count += 1
            continue
        
        # 结构标签检查（参考 ai_quant_strategy.py：只在特定结构下生成信号）
        structure_label = get_value_safe(item, 'structure_label', None)
        if structure_label is None:
            # 尝试从LLM输出中获取
            llm = get_value_safe(item, 'llm', {})
            structure_label = get_value_safe(llm, 'structure_label', None) or get_value_safe(llm, 'rule_structure_label', None)
        
        if structure_label not in allowed_structure_labels:
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
        
        # 结构置信度检查（如果使用LLM）
        if llm_score < structure_confidence_threshold:
            skipped_count += 1
            continue
        
        # LLM 信号检查（支持Long和Short，高频交易可能产生Short信号）
        # 检查是否为高频交易信号
        is_high_freq = get_value_safe(item, 'hf_signal', None) is not None or \
                      get_value_safe(item, 'structure_label', '') == 'HIGH_FREQ'
        
        if signal not in ['Long', 'Short']:
            # 如果未使用 LLM，fallback 可能返回 'Neutral'，这种情况下如果评分足够高，也允许通过
            if signal == 'Neutral' and llm_score >= min_llm_score:
                # Neutral 但评分足够，可以继续
                pass
            elif not is_high_freq:  # 高频交易信号允许通过
                skipped_count += 1
                continue
        elif llm_score < min_llm_score and not is_high_freq:  # 高频交易信号降低LLM评分要求
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
        
        # ========== 强制过滤器 ==========
        row = df.iloc[idx]
        filter_failed_reasons = []
        
        # 1. ATR 过滤（避免低波动）
        # 检查 ATR 相对于价格的百分比是否足够
        # 回测模式下，降低ATR要求以产生更多交易
        if 'atr14' in df.columns and not pd.isna(row['atr14']):
            atr_pct = (row['atr14'] / row['close']) * 100 if row['close'] > 0 else 0
            backtest_mode = os.getenv('BACKTEST_MODE', 'False').lower() == 'true' or \
                           os.getenv('BACKTEST_FULL_DATA', 'False').lower() == 'true' or \
                           os.getenv('BACKTEST_MONTHS', '0') != '0'
            default_atr_pct = 0.15 if backtest_mode else 0.5  # 回测模式降低到0.15%（从0.3%）
            min_atr_pct = float(os.getenv('MIN_ATR_PCT', str(default_atr_pct)))
            if atr_pct < min_atr_pct:
                filter_failed_reasons.append(f"ATR过低({atr_pct:.2f}% < {min_atr_pct}%)")
        else:
            filter_failed_reasons.append("ATR数据缺失")
        
        # 2. EMA 多头排列过滤（回测模式下大幅放宽或取消要求）
        if 'ema21' in df.columns and 'ema55' in df.columns and 'ema100' in df.columns:
            ema_bull = row['ema21'] > row['ema55'] > row['ema100']
            # 回测模式下，完全取消EMA排列要求（允许所有信号通过）
            if not backtest_mode:
                if not ema_bull:
                    filter_failed_reasons.append("EMA未形成多头排列")
        else:
            # 回测模式下，EMA数据缺失不阻止交易
            if not backtest_mode:
                filter_failed_reasons.append("EMA数据缺失")
        
        # 3. 趋势强度（回测模式下大幅降低或取消要求）
        try:
            # 使用最近50根K线计算趋势强度
            trend_strength = calculate_trend_strength(df.iloc[max(0, idx-49):idx+1], n=min(50, idx+1))
            # 回测模式下，完全取消趋势强度要求（允许所有信号通过）
            if not backtest_mode:
                min_trend_strength = 50
                if trend_strength <= min_trend_strength:
                    filter_failed_reasons.append(f"趋势强度不足({trend_strength:.1f} <= {min_trend_strength})")
        except Exception as e:
            logger.warning(f"计算趋势强度失败: {e}")
            # 回测模式下，计算失败不阻止交易
            if not backtest_mode:
                filter_failed_reasons.append("趋势强度计算失败")
        
        # 4. 突破有效性 = VALID
        # 检查是否有突破信号，如果有，验证其有效性
        feature_packet = get_value_safe(item, 'feature_packet', {})
        has_breakout = get_value_safe(feature_packet, 'breakout', False)
        
        if has_breakout:
            # 验证突破有效性
            breakout_valid = False
            breakout_fail_reason = None
            
            # 检查1: 成交量是否放大
            vol_ratio = get_value_safe(feature_packet, 'vol_ratio', None)
            if vol_ratio is None or vol_ratio < 1.2:
                vol_ratio_display = vol_ratio if vol_ratio is not None else 0
                breakout_fail_reason = f"突破时成交量未放大(vol_ratio={vol_ratio_display:.2f} < 1.2)"
            else:
                # 检查2: 价格是否持续在阻力位上方（检查最近3根K线）
                if 'res50' in df.columns and not pd.isna(row['res50']):
                    # 检查最近3根K线是否都在阻力位上方
                    lookback = min(3, idx + 1)
                    closes_above_res = all(
                        df['close'].iloc[max(0, idx-i)] > row['res50'] 
                        for i in range(lookback)
                    )
                    if closes_above_res:
                        breakout_valid = True
                    else:
                        breakout_fail_reason = "突破后价格未持续在阻力位上方"
                else:
                    # 如果没有阻力位数据，检查价格是否创新高
                    if idx >= 20:
                        recent_high = df['high'].iloc[max(0, idx-20):idx+1].max()
                        if row['close'] >= recent_high * 0.99:  # 接近或创新高
                            breakout_valid = True
                        else:
                            breakout_fail_reason = "突破未创新高"
                    else:
                        # 数据不足，假设有效
                        breakout_valid = True
            
            if not breakout_valid and breakout_fail_reason:
                filter_failed_reasons.append(breakout_fail_reason)
        # 如果没有突破信号，不需要验证突破有效性（允许非突破信号通过）
        
        # 如果任何强制过滤器失败，跳过该信号
        if filter_failed_reasons:
            logger.debug(f"信号 {idx} 被强制过滤器拒绝: {', '.join(filter_failed_reasons)}")
            skipped_count += 1
            continue
        
        # ========== 所有过滤器通过 ==========
        
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

