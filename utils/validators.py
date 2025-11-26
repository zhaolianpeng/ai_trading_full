# utils/validators.py
"""
数据验证工具
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from utils.logger import logger

def validate_price_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    验证价格数据的有效性
    
    Args:
        df: 价格数据 DataFrame
    
    Returns:
        (is_valid, error_messages): 是否有效和错误消息列表
    """
    errors = []
    
    # 检查必需的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors
    
    # 检查数据是否为空
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    # 检查价格逻辑
    if (df['high'] < df['low']).any():
        errors.append("Found rows where high < low")
    
    if (df['close'] > df['high']).any():
        errors.append("Found rows where close > high")
    
    if (df['close'] < df['low']).any():
        errors.append("Found rows where close < low")
    
    if (df['open'] > df['high']).any():
        errors.append("Found rows where open > high")
    
    if (df['open'] < df['low']).any():
        errors.append("Found rows where open < low")
    
    # 检查负值
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (df[col] <= 0).any():
            errors.append(f"Found non-positive values in {col}")
    
    if (df['volume'] < 0).any():
        errors.append("Found negative values in volume")
    
    # 检查缺失值
    for col in required_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            errors.append(f"Found {missing_count} missing values in {col}")
    
    # 检查索引
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be DatetimeIndex")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def fix_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复价格数据中的常见问题
    
    Args:
        df: 价格数据 DataFrame
    
    Returns:
        修复后的 DataFrame
    """
    df = df.copy()
    fixed_count = 0
    
    # 修复 high < low
    mask = df['high'] < df['low']
    if mask.any():
        df.loc[mask, 'high'] = df.loc[mask, 'low']
        fixed_count += mask.sum()
        logger.warning(f"Fixed {mask.sum()} rows where high < low")
    
    # 修复 close 超出范围
    mask = df['close'] > df['high']
    if mask.any():
        df.loc[mask, 'close'] = df.loc[mask, 'high']
        fixed_count += mask.sum()
        logger.warning(f"Fixed {mask.sum()} rows where close > high")
    
    mask = df['close'] < df['low']
    if mask.any():
        df.loc[mask, 'close'] = df.loc[mask, 'low']
        fixed_count += mask.sum()
        logger.warning(f"Fixed {mask.sum()} rows where close < low")
    
    # 修复 open 超出范围
    mask = df['open'] > df['high']
    if mask.any():
        df.loc[mask, 'open'] = df.loc[mask, 'high']
        fixed_count += mask.sum()
        logger.warning(f"Fixed {mask.sum()} rows where open > high")
    
    mask = df['open'] < df['low']
    if mask.any():
        df.loc[mask, 'open'] = df.loc[mask, 'low']
        fixed_count += mask.sum()
        logger.warning(f"Fixed {mask.sum()} rows where open < low")
    
    # 处理缺失值（前向填充）
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            missing_before = df[col].isna().sum()
            df[col] = df[col].ffill().bfill()
            missing_after = df[col].isna().sum()
            if missing_before > 0:
                logger.warning(f"Filled {missing_before - missing_after} missing values in {col}")
    
    # 处理负值
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            mask = df[col] <= 0
            if mask.any():
                df.loc[mask, col] = df[col].shift(1).loc[mask]
                fixed_count += mask.sum()
                logger.warning(f"Fixed {mask.sum()} non-positive values in {col}")
    
    if df['volume'] is not None:
        mask = df['volume'] < 0
        if mask.any():
            df.loc[mask, 'volume'] = 0
            fixed_count += mask.sum()
            logger.warning(f"Fixed {mask.sum()} negative values in volume")
    
    if fixed_count > 0:
        logger.info(f"Total fixes applied: {fixed_count}")
    
    return df
