# utils/entry_finder.py
"""
入场点查找工具
在3分钟周期中找到最佳入场点
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from data.market_data import fetch_market_data
from config import DATA_SOURCE, MARKET_SYMBOL
from utils.logger import logger

def find_best_entry_point_3m(
    signal_time: datetime,
    signal_price: float,
    signal_direction: str = 'Long',
    lookforward_minutes: int = 60
) -> Optional[Dict]:
    """
    在3分钟周期中找到最佳入场点
    在小时级指标命中之后，观察多个3分钟周期的K线值，找到最适合的入场时机
    
    Args:
        signal_time: 信号产生的时间（小时级）
        signal_price: 信号产生时的价格
        signal_direction: 信号方向 ('Long' 或 'Short')
        lookforward_minutes: 向前查找的分钟数（默认60分钟，即20个3分钟K线）
    
    Returns:
        包含最佳入场点信息的字典，如果找不到则返回None
        {
            'entry_time': 入场时间,
            'entry_price': 入场价格,
            'entry_idx': 入场索引,
            'entry_reason': 入场原因
        }
    """
    try:
        # 计算时间范围：从信号时间开始，向前查找（未来）
        start_time = signal_time
        end_time = signal_time + timedelta(minutes=lookforward_minutes)
        
        logger.info(f"查找3分钟周期入场点: 信号时间={signal_time}, 方向={signal_direction}, 向前查找{lookforward_minutes}分钟")
        
        # 根据数据源获取3分钟数据
        if DATA_SOURCE == 'binance':
            # Binance 支持 3m 时间框架
            from data.market_data import fetch_binance_data
            df_3m = fetch_binance_data(
                symbol=MARKET_SYMBOL,
                timeframe='3m',
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
        elif DATA_SOURCE == 'yahoo':
            # Yahoo Finance 最小支持 5m，使用 5m 作为替代
            logger.warning("Yahoo Finance 不支持3分钟数据，使用5分钟数据作为替代")
            from data.market_data import fetch_yahoo_data
            try:
                df_5m = fetch_yahoo_data(
                    symbol=MARKET_SYMBOL,
                    interval='5m',
                    start=start_time.strftime('%Y-%m-%d'),
                    end=end_time.strftime('%Y-%m-%d')
                )
                # 如果获取失败，返回None
                if df_5m.empty:
                    logger.warning("无法获取5分钟数据，跳过入场点查找")
                    return None
                df_3m = df_5m
            except Exception as e:
                logger.warning(f"获取Yahoo Finance数据失败: {e}，跳过入场点查找")
                return None
        else:
            # 其他数据源可能不支持3分钟，返回None
            logger.warning(f"数据源 {DATA_SOURCE} 可能不支持3分钟数据，跳过入场点查找")
            return None
        
        if df_3m.empty:
            logger.warning("3分钟周期数据为空，跳过入场点查找")
            return None
        
        # 找到信号时间对应的K线
        signal_idx = None
        if isinstance(df_3m.index, pd.DatetimeIndex):
            # 找到最接近信号时间的K线
            try:
                time_diffs = abs(df_3m.index - signal_time)
                closest_time = time_diffs.idxmin()
                signal_idx = df_3m.index.get_loc(closest_time)
                # 如果 get_loc 返回的是切片，取第一个
                if isinstance(signal_idx, slice):
                    signal_idx = signal_idx.start if signal_idx.start is not None else 0
            except Exception as e:
                logger.warning(f"查找信号时间对应的K线时出错: {e}")
                # 使用最后一个索引作为备选
                signal_idx = len(df_3m) - 1
        
        if signal_idx is None or signal_idx >= len(df_3m) or signal_idx < 0:
            logger.warning(f"无法找到信号时间对应的K线 (signal_idx={signal_idx}, df_len={len(df_3m)})")
            return None
        
        # 从信号时间之后（未来）查找最佳入场点
        best_entry = None
        best_score = -float('inf')
        
        # 查找范围：从信号时间开始，向后查找最多20个K线（约1小时）
        search_range = min(len(df_3m) - signal_idx, 20)
        
        for i in range(signal_idx, min(len(df_3m), signal_idx + search_range)):
            if i >= len(df_3m):
                continue
            
            row = df_3m.iloc[i]
            entry_time = df_3m.index[i]
            
            # 计算入场点评分
            score = 0
            reasons = []
            
            if signal_direction == 'Long':
                # 做多：寻找相对低点或回调点
                # 使用收盘价或最低价作为入场价（选择更低的）
                entry_price = min(row['close'], row['low'])
                
                # 评分因素：
                # 1. 价格相对信号价格的回调（适度回调更好，但不能太深）
                price_change = (entry_price - signal_price) / signal_price
                if -0.01 < price_change <= 0:  # 0-1%的回调，最佳
                    score += 35
                    reasons.append(f"价格回调{abs(price_change)*100:.2f}%")
                elif -0.02 < price_change <= -0.01:  # 1-2%的回调
                    score += 25
                    reasons.append(f"价格回调{abs(price_change)*100:.2f}%")
                elif -0.03 < price_change <= -0.02:  # 2-3%的回调
                    score += 15
                    reasons.append(f"价格回调{abs(price_change)*100:.2f}%")
                elif price_change > 0:  # 价格上涨，可能错过最佳时机
                    score += 5
                    reasons.append(f"价格上涨{price_change*100:.2f}%")
                
                # 2. 成交量（成交量放大更好）
                if 'volume' in row:
                    vol_ratio = row['volume'] / df_3m['volume'].iloc[max(0, i-10):i+1].mean() if i > 0 else 1
                    if vol_ratio > 1.5:
                        score += 25
                        reasons.append(f"成交量放大{vol_ratio:.2f}x")
                    elif vol_ratio > 1.2:
                        score += 15
                        reasons.append(f"成交量放大{vol_ratio:.2f}x")
                    elif vol_ratio < 0.8:
                        score -= 10  # 成交量萎缩，扣分
                        reasons.append(f"成交量萎缩{vol_ratio:.2f}x")
                
                # 3. 距离信号时间（越近越好，但不要太远）
                time_diff = (entry_time - signal_time).total_seconds() / 60  # 分钟
                if 0 <= time_diff <= 15:  # 0-15分钟，最佳
                    score += 20
                    reasons.append(f"距离信号{time_diff:.0f}分钟")
                elif 15 < time_diff <= 30:  # 15-30分钟
                    score += 15
                    reasons.append(f"距离信号{time_diff:.0f}分钟")
                elif 30 < time_diff <= 45:  # 30-45分钟
                    score += 10
                    reasons.append(f"距离信号{time_diff:.0f}分钟")
                else:  # 超过45分钟，扣分
                    score -= 5
                    reasons.append(f"距离信号{time_diff:.0f}分钟")
                
                # 4. K线形态（下影线长表示有支撑）
                if row['close'] > row['open']:  # 阳线
                    score += 10
                    reasons.append("阳线")
                lower_shadow = min(row['open'], row['close']) - row['low']
                body = abs(row['close'] - row['open'])
                if body > 0 and lower_shadow / body > 1.5:  # 下影线是实体的1.5倍以上
                    score += 15
                    reasons.append("长下影线支撑")
                
                # 5. 价格稳定性（波动适中）
                price_range = (row['high'] - row['low']) / row['close']
                if 0.003 < price_range < 0.01:  # 0.3%-1%的波动，适中
                    score += 10
                    reasons.append("波动适中")
                elif price_range > 0.02:  # 波动太大，扣分
                    score -= 10
                    reasons.append("波动较大")
                
            else:  # Short
                # 做空：寻找相对高点或反弹点
                entry_price = max(row['close'], row['high'])
                
                # 评分因素（类似做多，但方向相反）
                price_change = (entry_price - signal_price) / signal_price
                if 0 <= price_change < 0.01:  # 0-1%的反弹
                    score += 35
                    reasons.append(f"价格反弹{price_change*100:.2f}%")
                elif 0.01 <= price_change < 0.02:  # 1-2%的反弹
                    score += 25
                    reasons.append(f"价格反弹{price_change*100:.2f}%")
                elif 0.02 <= price_change < 0.03:  # 2-3%的反弹
                    score += 15
                    reasons.append(f"价格反弹{price_change*100:.2f}%")
                elif price_change < 0:  # 价格下跌
                    score += 5
                    reasons.append(f"价格下跌{abs(price_change)*100:.2f}%")
                
                if 'volume' in row:
                    vol_ratio = row['volume'] / df_3m['volume'].iloc[max(0, i-10):i+1].mean() if i > 0 else 1
                    if vol_ratio > 1.5:
                        score += 25
                        reasons.append(f"成交量放大{vol_ratio:.2f}x")
                    elif vol_ratio > 1.2:
                        score += 15
                        reasons.append(f"成交量放大{vol_ratio:.2f}x")
                    elif vol_ratio < 0.8:
                        score -= 10
                        reasons.append(f"成交量萎缩{vol_ratio:.2f}x")
                
                time_diff = (entry_time - signal_time).total_seconds() / 60
                if 0 <= time_diff <= 15:
                    score += 20
                    reasons.append(f"距离信号{time_diff:.0f}分钟")
                elif 15 < time_diff <= 30:
                    score += 15
                    reasons.append(f"距离信号{time_diff:.0f}分钟")
                elif 30 < time_diff <= 45:
                    score += 10
                    reasons.append(f"距离信号{time_diff:.0f}分钟")
                else:
                    score -= 5
                    reasons.append(f"距离信号{time_diff:.0f}分钟")
                
                if row['close'] < row['open']:  # 阴线
                    score += 10
                    reasons.append("阴线")
                upper_shadow = row['high'] - max(row['open'], row['close'])
                body = abs(row['close'] - row['open'])
                if body > 0 and upper_shadow / body > 1.5:  # 上影线是实体的1.5倍以上
                    score += 15
                    reasons.append("长上影线压力")
                
                price_range = (row['high'] - row['low']) / row['close']
                if 0.003 < price_range < 0.01:
                    score += 10
                    reasons.append("波动适中")
                elif price_range > 0.02:
                    score -= 10
                    reasons.append("波动较大")
            
            # 更新最佳入场点
            if score > best_score:
                best_score = score
                best_entry = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'entry_idx': i,
                    'entry_reason': ', '.join(reasons) if reasons else '默认入场点',
                    'entry_score': score
                }
        
        if best_entry:
            logger.info(f"找到最佳入场点: 时间={best_entry['entry_time']}, "
                       f"价格={best_entry['entry_price']:.2f}, "
                       f"评分={best_entry['entry_score']}, "
                       f"原因={best_entry['entry_reason']}")
            return best_entry
        else:
            logger.warning("未找到合适的入场点")
            return None
            
    except Exception as e:
        logger.error(f"查找入场点时出错: {e}", exc_info=True)
        return None

