# signal_rules.py
from features.ta_basic import add_basic_ta
from features.divergence import detect_rsi_divergence
import numpy as np

def detect_rules(df):
    """
    基于规则的信号检测：long_structure / breakout / rsi divergence
    返回：增强后的 df（含指标列）与 signals 列表（每项包含 type、score、confidence、idx）
    """
    df = add_basic_ta(df)
    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        found = []
        # long_structure
        try:
            cond1 = row['ema21'] > row['ema55'] > row['ema100']
        except Exception:
            cond1 = False
        cond2 = False
        if i >= 20:
            window = df['close'].iloc[i-19:i+1]
            cond2 = (window.iloc[-1] == window.max()) and (window.min() > df['close'].iloc[max(0, i-40):i+1].mean())
        cond3 = row['volume'] > row['vol_ma50'] * 1.3 if not np.isnan(row['vol_ma50']) else False
        if cond1 and cond2 and cond3:
            found.append({'type':'long_structure','score':4,'confidence':'high','idx':i})
        # breakout_long
        if not np.isnan(df['res50'].iloc[i]) and row['close'] > df['res50'].iloc[i]:
            found.append({'type':'breakout_long','score':3,'confidence':'medium','idx':i})
        # divergence
        df_at_idx = df.iloc[:i+1]
        divs = detect_rsi_divergence(df_at_idx, order=3)
        for d in divs:
            if d[0] == i:
                tp = 'rsi_positive_divergence' if d[1]=='positive' else 'rsi_negative_divergence'
                found.append({'type':tp,'score':2,'confidence':'medium','idx':i})
        if found:
            signals.extend(found)
    return df, signals
