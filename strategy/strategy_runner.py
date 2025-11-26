# strategy/strategy_runner.py
import pandas as pd
from signal_rules import detect_rules
from ai_agent.signal_interpret import interpret_with_llm
from config import USE_LLM, OPENAI_MODEL, DEEPSEEK_MODEL, LLM_PROVIDER, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, MARKET_INTERVAL, MARKET_TIMEFRAME
from utils.logger import logger
import math
from typing import List, Dict

def build_feature_packet(df, idx):
    row = df.iloc[idx]
    packet = {
        "trend": "up" if row['ema21'] > row['ema55'] else "down",
        "ema_alignment": bool(row['ema21'] > row['ema55'] > row['ema100']),
        "higher_highs": False,
        "higher_lows": False,
        "volume_spike": bool(row['volume'] > row['vol_ma50'] * 1.3) if not pd.isna(row['vol_ma50']) else False,
        "breakout": bool(row['close'] > row['res50']) if 'res50' in row.index else False,
        "rsi_divergence": None,
        "atr": float(row['atr14']) if not pd.isna(row['atr14']) else None,
        "vol_ratio": float(row['volume'] / row['vol_ma50']) if row.get('vol_ma50', None) and row['vol_ma50']>0 else None,
        "close": float(row['close'])
    }
    window = df['close'].iloc[max(0, idx-19):idx+1]
    if len(window)>1:
        packet['higher_highs'] = (window.iloc[-1] == window.max())
        packet['higher_lows'] = (window.min() > df['close'].iloc[max(0, idx-40):idx+1].mean())
    return packet

def run_strategy(df, use_llm=USE_LLM, use_advanced_ta=True, use_eric_indicators=False, lookback_days=7):
    """
    运行完整的策略流程
    
    Args:
        df: 价格数据 DataFrame
        use_llm: 是否使用 LLM 分析
        use_advanced_ta: 是否使用高级技术指标
        use_eric_indicators: 是否使用 Eric 指标
        lookback_days: 倒推天数（默认7天，即1周）
    
    Returns:
        (df, enhanced_signals): 增强后的 DataFrame 和信号列表
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 如果数据有时间索引，计算倒推时间点
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        # 计算倒推时间点（从最新数据往前推）
        latest_time = df.index[-1]
        lookback_time = latest_time - timedelta(days=lookback_days)
        
        # 找到倒推时间点的索引位置
        lookback_idx = df.index.get_indexer([lookback_time], method='nearest')[0]
        if lookback_idx < 0:
            lookback_idx = 0
        
        logger.info(f"倒推 {lookback_days} 天内的交易信号（从 {lookback_time} 到 {latest_time}）")
        logger.info(f"数据范围: {len(df)} 条，将分析最近 {len(df) - lookback_idx} 条数据")
        
        # 只分析最近 lookback_days 天的数据
        df_analysis = df.iloc[lookback_idx:].copy()
    else:
        # 如果没有时间索引，使用数据条数估算
        # 假设小时级数据，1周 = 7 * 24 = 168 条
        if '1h' in str(MARKET_INTERVAL) or '1h' in str(MARKET_TIMEFRAME):
            lookback_bars = lookback_days * 24
        else:
            lookback_bars = lookback_days * 24  # 默认按小时估算
        
        lookback_idx = max(0, len(df) - lookback_bars)
        df_analysis = df.iloc[lookback_idx:].copy()
        logger.info(f"倒推 {lookback_days} 天内的交易信号（最近 {len(df_analysis)} 条数据）")
    
    logger.info("检测交易信号...")
    df_analysis_with_ta, signals = detect_rules(df_analysis, use_advanced_ta=use_advanced_ta, use_eric_indicators=use_eric_indicators)
    logger.info(f"发现 {len(signals)} 个原始信号（在最近 {lookback_days} 天内）")
    
    if not signals:
        logger.warning("No signals detected. Try adjusting signal detection parameters or using more data.")
        return df, []
    
    # 将分析数据的索引映射回原始数据的索引
    # 因为 df_analysis 是 df 的子集，需要调整信号索引
    if lookback_idx > 0:
        for signal in signals:
            signal['idx'] = signal['idx'] + lookback_idx
    
    # 确保完整数据也包含技术指标（用于后续分析）
    if len(df_analysis_with_ta) < len(df):
        # 如果只分析了部分数据，需要重新计算完整数据的技术指标
        logger.info("重新计算完整数据的技术指标...")
        df, _ = detect_rules(df, use_advanced_ta=use_advanced_ta, use_eric_indicators=use_eric_indicators)
    else:
        df = df_analysis_with_ta
    
    enhanced_signals = []
    
    # 批量处理信号（减少日志噪音）
    batch_size = 10
    total_batches = (len(signals) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(signals))
        batch_signals = signals[start_idx:end_idx]
        
        for i, s in enumerate(batch_signals):
            global_idx = start_idx + i
            idx = s['idx']
            # 确保索引在有效范围内
            if idx >= len(df):
                continue
            packet = build_feature_packet(df, idx)
            # 根据 provider 选择对应的模型
            model = DEEPSEEK_MODEL if LLM_PROVIDER == 'deepseek' else OPENAI_MODEL
            try:
                llm_out = interpret_with_llm(
                    packet, 
                    provider=LLM_PROVIDER, 
                    model=model, 
                    use_llm=use_llm,
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=OPENAI_MAX_TOKENS
                )
            except Exception as e:
                logger.warning(f"LLM interpretation failed for signal {global_idx+1}/{len(signals)}: {e}")
                # 使用 fallback
                llm_out = interpret_with_llm(packet, provider=LLM_PROVIDER, model=model, use_llm=False)
            
            enhanced_signals.append({'rule': s, 'feature_packet': packet, 'llm': llm_out})
        
        if batch_idx < total_batches - 1:
            logger.debug(f"Processed {end_idx}/{len(signals)} signals...")
    
    logger.info(f"已增强 {len(enhanced_signals)} 个信号")
    return df, enhanced_signals
