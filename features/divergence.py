# features/divergence.py
import numpy as np

def local_peaks(series, order=3):
    """
    简单本地峰谷检测（无外部依赖）
    order 控制噪声抑制，order 越大越平滑
    """
    arr = series.values
    peaks = np.zeros(len(arr), dtype=bool)
    troughs = np.zeros(len(arr), dtype=bool)
    L = len(arr)
    for i in range(order, L-order):
        window = arr[i-order:i+order+1]
        if arr[i] == window.max() and np.sum(window==arr[i])==1:
            peaks[i] = True
        if arr[i] == window.min() and np.sum(window==arr[i])==1:
            troughs[i] = True
    return peaks, troughs

def detect_rsi_divergence(df, rsi_col='rsi14', price_col='close', order=3):
    price_peaks, price_troughs = local_peaks(df[price_col].ffill(), order=order)
    rsi_peaks, rsi_troughs = local_peaks(df[rsi_col].fillna(50), order=order)
    divs = []
    for i in range(len(df)):
        if price_peaks[i]:
            prev = np.where(price_peaks[:i])[0]
            if prev.size>0:
                last = prev[-1]
                if df[price_col].iloc[i] > df[price_col].iloc[last] and df[rsi_col].iloc[i] < df[rsi_col].iloc[last]:
                    divs.append((i, 'negative'))
        if price_troughs[i]:
            prev = np.where(price_troughs[:i])[0]
            if prev.size>0:
                last = prev[-1]
                if df[price_col].iloc[i] < df[price_col].iloc[last] and df[rsi_col].iloc[i] > df[rsi_col].iloc[last]:
                    divs.append((i, 'positive'))
    return divs
