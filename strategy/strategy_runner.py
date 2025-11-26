# strategy/strategy_runner.py
import pandas as pd
from signal_rules import detect_rules
from ai_agent.signal_interpret import interpret_with_llm
from config import USE_LLM, OPENAI_MODEL, LLM_PROVIDER
import math

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

def run_strategy(df, use_llm=USE_LLM):
    """
    返回：增强后的 df 和 enhanced_signals（包含规则、feature_packet、llm_output）
    """
    df, signals = detect_rules(df)
    enhanced_signals = []
    for s in signals:
        idx = s['idx']
        packet = build_feature_packet(df, idx)
        llm_out = interpret_with_llm(packet, provider=LLM_PROVIDER, model=OPENAI_MODEL, use_llm=use_llm)
        enhanced_signals.append({'rule': s, 'feature_packet': packet, 'llm': llm_out})
    return df, enhanced_signals
