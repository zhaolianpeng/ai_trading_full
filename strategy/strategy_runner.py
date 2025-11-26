# strategy/strategy_runner.py
import pandas as pd
from signal_rules import detect_rules
from ai_agent.signal_interpret import interpret_with_llm
from config import USE_LLM, OPENAI_MODEL, LLM_PROVIDER, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
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

def run_strategy(df, use_llm=USE_LLM, use_advanced_ta=True, use_eric_indicators=False):
    """
    运行完整的策略流程
    
    Args:
        df: 价格数据 DataFrame
        use_llm: 是否使用 LLM 分析
        use_advanced_ta: 是否使用高级技术指标
    
    Returns:
        (df, enhanced_signals): 增强后的 DataFrame 和信号列表
    """
    logger.info("检测交易信号...")
    df, signals = detect_rules(df, use_advanced_ta=use_advanced_ta, use_eric_indicators=use_eric_indicators)
    logger.info(f"发现 {len(signals)} 个原始信号")
    
    if not signals:
        logger.warning("No signals detected. Try adjusting signal detection parameters or using more data.")
        return df, []
    
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
            packet = build_feature_packet(df, idx)
            try:
                llm_out = interpret_with_llm(
                    packet, 
                    provider=LLM_PROVIDER, 
                    model=OPENAI_MODEL, 
                    use_llm=use_llm,
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=OPENAI_MAX_TOKENS
                )
            except Exception as e:
                logger.warning(f"LLM interpretation failed for signal {global_idx+1}/{len(signals)}: {e}")
                # 使用 fallback
                llm_out = interpret_with_llm(packet, provider=LLM_PROVIDER, model=OPENAI_MODEL, use_llm=False)
            
            enhanced_signals.append({'rule': s, 'feature_packet': packet, 'llm': llm_out})
        
        if batch_idx < total_batches - 1:
            logger.debug(f"Processed {end_idx}/{len(signals)} signals...")
    
    logger.info(f"已增强 {len(enhanced_signals)} 个信号")
    return df, enhanced_signals
