# utils/time_utils.py
"""
时间工具函数
"""
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import pytz

# 北京时区
BEIJING_TZ = pytz.timezone('Asia/Shanghai')
UTC_TZ = pytz.UTC

def to_beijing_time(dt: Union[datetime, str, pd.Timestamp, None]) -> Optional[str]:
    """
    将时间转换为北京时间（UTC+8）
    
    Args:
        dt: 时间对象（可以是 datetime, str, pd.Timestamp 或 None）
    
    Returns:
        ISO格式的北京时间字符串，如果输入为None则返回None
    """
    if dt is None:
        return None
    
    # 如果是字符串，先转换为datetime
    if isinstance(dt, str):
        try:
            dt = pd.to_datetime(dt)
        except:
            return None
    
    # 如果是 pandas Timestamp，转换为 datetime
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    
    # 如果是 naive datetime，假设为 UTC
    if dt.tzinfo is None:
        dt = UTC_TZ.localize(dt)
    
    # 转换为北京时间
    beijing_dt = dt.astimezone(BEIJING_TZ)
    
    # 返回不带时区的格式：YYYY-MM-DDTHH:MM:SS
    return beijing_dt.strftime('%Y-%m-%dT%H:%M:%S')

def convert_dict_times_to_beijing(data: dict) -> dict:
    """
    递归地将字典中的所有时间字段转换为北京时间（格式：YYYY-MM-DDTHH:MM:SS，不带时区）
    
    Args:
        data: 字典数据
    
    Returns:
        转换后的字典
    """
    if not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        # 如果键名包含"时间"或"time"，尝试转换
        if isinstance(value, str) and ('时间' in key or 'time' in key.lower()):
            # 如果已经是目标格式（YYYY-MM-DDTHH:MM:SS），跳过
            if value and len(value) >= 19 and 'T' in value and '+' not in value and 'Z' not in value:
                result[key] = value
            else:
                converted = to_beijing_time(value)
                result[key] = converted if converted is not None else value
        elif isinstance(value, dict):
            result[key] = convert_dict_times_to_beijing(value)
        elif isinstance(value, list):
            result[key] = [convert_dict_times_to_beijing(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    
    return result

