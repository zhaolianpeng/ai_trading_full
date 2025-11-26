# data/loader.py
import pandas as pd
import numpy as np

def load_csv(path):
    df = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime')
    return df

def gen_synthetic(n=1500, seed=42):
    """
    生成合成 BTC-like 数据，便于 demo 测试
    """
    rng = np.random.default_rng(seed)
    dt = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=n, freq='h')
    price = 30000 + np.cumsum(rng.normal(0, 50, size=n).astype(float))
    # 注入多段趋势以产生信号
    for t in range(200, n, 250):
        price[t:t+30] += np.linspace(0, 1500, min(30, n-t))
    high = price + np.abs(rng.normal(0, 20, size=n))
    low = price - np.abs(rng.normal(0, 20, size=n))
    close = price
    volume = np.abs(rng.normal(200, 50, size=n)) * (1 + np.sin(np.linspace(0, 10, n)))
    df = pd.DataFrame({'open':close, 'high':high, 'low':low, 'close':close, 'volume':volume}, index=dt)
    df.index.name = 'datetime'
    return df
