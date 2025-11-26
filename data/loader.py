# data/loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import logger
from utils.validators import validate_price_data, fix_price_data

def load_csv(path: str) -> pd.DataFrame:
    """
    从 CSV 文件加载交易数据
    
    Args:
        path: CSV 文件路径，必须包含 datetime, open, high, low, close, volume 列
    
    Returns:
        包含交易数据的 DataFrame，以 datetime 为索引
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 缺少必需的列
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading CSV from {path}")
    try:
        df = pd.read_csv(path, parse_dates=['datetime']).set_index('datetime')
    except KeyError as e:
        raise ValueError(f"CSV file must contain 'datetime' column. Error: {e}")
    
    # 验证必需的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")
    
    # 数据验证和修复
    is_valid, errors = validate_price_data(df)
    if not is_valid:
        logger.warning(f"Data validation found issues: {errors}")
        logger.info("Attempting to fix data issues...")
        df = fix_price_data(df)
        # 再次验证
        is_valid, errors = validate_price_data(df)
        if not is_valid:
            logger.error(f"Data still has issues after fixing: {errors}")
            raise ValueError(f"Data validation failed: {errors}")
    
    logger.info(f"Loaded {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
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
    # 生成更真实的 OHLC
    open_prices = close.copy()
    for i in range(1, n):
        open_prices[i] = close[i-1] * (1 + rng.normal(0, 0.001))
    
    high_prices = np.maximum(open_prices, close) + np.abs(rng.normal(0, 20, size=n))
    low_prices = np.minimum(open_prices, close) - np.abs(rng.normal(0, 20, size=n))
    
    # 确保价格逻辑正确
    for i in range(n):
        high_prices[i] = max(open_prices[i], close[i], high_prices[i])
        low_prices[i] = min(open_prices[i], close[i], low_prices[i])
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close,
        'volume': volume
    }, index=dt)
    df.index.name = 'datetime'
    
    logger.info(f"Generated synthetic data: {len(df)} rows, price range ${df['close'].min():.2f}-${df['close'].max():.2f}")
    return df
