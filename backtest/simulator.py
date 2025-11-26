# backtest/simulator.py
import pandas as pd
import numpy as np

def simple_backtest(df, enhanced_signals, max_hold=20, atr_mult_stop=1.0, atr_mult_target=2.0):
    trades = []
    used_idxs = set()
    for item in enhanced_signals:
        s = item['rule']
        idx = s['idx']
        # 默认只跟随 LLM 的 Long 建议
        llm = item.get('llm', {})
        signal = llm.get('signal','Neutral') if isinstance(llm, dict) else 'Neutral'
        # 允许 llm score 是 str（解析），所以做容错
        raw_score = llm.get('score', 0)
        try:
            score = int(raw_score)
        except Exception:
            try:
                score = int(float(raw_score))
            except:
                score = 0
        if signal != 'Long' or score < 40:
            continue
        if idx in used_idxs or idx+1>=len(df):
            continue
        entry_price = df['close'].iloc[idx+1]
        atr = df['atr14'].iloc[idx+1] if not pd.isna(df['atr14'].iloc[idx+1]) else entry_price*0.01
        stop = entry_price - atr * atr_mult_stop
        target = entry_price + atr * atr_mult_target
        entry_idx = idx+1
        exit_idx = None
        exit_price = None
        for j in range(entry_idx, min(len(df), entry_idx+max_hold)):
            low = df['low'].iloc[j]
            high = df['high'].iloc[j]
            close = df['close'].iloc[j]
            if low <= stop:
                exit_idx = j; exit_price = stop; break
            if high >= target:
                exit_idx = j; exit_price = target; break
            if j == min(len(df)-1, entry_idx+max_hold-1):
                exit_idx = j; exit_price = close
        if exit_idx is None:
            continue
        ret = (exit_price - entry_price) / entry_price
        trades.append({'entry_idx': entry_idx, 'exit_idx': exit_idx, 'entry_price': entry_price, 'exit_price': exit_price, 'return': ret, 'rule_type': s['type'], 'llm_score': score})
        for k in range(entry_idx, exit_idx+1):
            used_idxs.add(k)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {'total_trades':0,'win_rate':0,'avg_return':0,'profit_factor':0,'max_drawdown':0}
        return trades_df, metrics
    total = len(trades_df)
    win_rate = (trades_df['return']>0).sum()/total
    avg_ret = trades_df['return'].mean()
    gross_profit = trades_df.loc[trades_df['return']>0,'return'].sum()
    gross_loss = -trades_df.loc[trades_df['return']<=0,'return'].sum()
    profit_factor = (gross_profit/(gross_loss+1e-9)) if gross_loss>0 else float('inf')
    equity = (1+trades_df['return']).cumprod()
    peak = equity.cummax()
    drawdown = (equity-peak)/peak
    max_dd = drawdown.min()
    metrics = {'total_trades':total,'win_rate':win_rate,'avg_return':avg_ret,'profit_factor':profit_factor,'max_drawdown':max_dd}
    return trades_df, metrics
