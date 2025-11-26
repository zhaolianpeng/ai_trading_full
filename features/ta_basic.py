# features/ta_basic.py
import pandas as pd

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def add_basic_ta(df):
    df = df.copy()
    df['ema21'] = ema(df['close'], 21)
    df['ema55'] = ema(df['close'], 55)
    df['ema100'] = ema(df['close'], 100)
    df['ema200'] = ema(df['close'], 200)
    df['rsi14'] = rsi(df['close'], 14)
    df['atr14'] = atr(df, 14)
    df['vol_ma50'] = df['volume'].rolling(50, min_periods=1).mean()
    df['res50'] = df['close'].rolling(50).max()
    return df

def add_vegas_ema(df):
    """
    添加维加斯通道 EMA（144/169）
    """
    df = df.copy()
    df['ema144'] = ema(df['close'], 144)
    df['ema169'] = ema(df['close'], 169)
    return df
