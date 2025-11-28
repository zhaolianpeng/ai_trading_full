# utils/entry_finder.py
"""
入场点查找工具
在短周期（5分钟或3分钟）中找到最佳入场点
注意：虽然函数名是3m，但实际优先使用5分钟数据（更稳定）
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
    lookforward_minutes: int = 60,
    max_time_window_hours: int = 8
) -> Optional[Dict]:
    """
    在短周期（5分钟或3分钟）中找到最佳入场点
    在小时级指标命中之后，观察多个短周期的K线值，找到最适合的入场时机
    注意：优先使用5分钟数据（更稳定），如果不可用则尝试3分钟
    
    Args:
        signal_time: 信号产生的时间（小时级）
        signal_price: 信号产生时的价格
        signal_direction: 信号方向 ('Long' 或 'Short')
        lookforward_minutes: 向前查找的分钟数（默认60分钟，已废弃，使用max_time_window_hours）
        max_time_window_hours: 最大时间窗口（小时），在信号时间前后查找（默认8小时）
    
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
        # 计算时间范围：在信号时间前后max_time_window_hours小时内查找
        # 例如：信号时间是10:00，则在02:00到18:00之间查找（前后8小时）
        time_window_minutes = max_time_window_hours * 60
        start_time = signal_time - timedelta(hours=max_time_window_hours)
        end_time = signal_time + timedelta(hours=max_time_window_hours)
        
        logger.info(f"📊 大趋势信号已确认（小时级）: 信号时间={signal_time}, 方向={signal_direction}")
        logger.info(f"🔍 重新查询5分钟级数据，查找具体开单时间: 时间窗口=±{max_time_window_hours}小时（{start_time} 到 {end_time}）")
        
        # 根据数据源获取短周期数据（优先使用5分钟，因为更通用）
        # 注意：虽然函数名叫3m，但实际使用5m数据，因为Binance的3m在某些市场可能不稳定
        # 这是在大趋势信号确认后，重新查询5分钟级数据来找具体开单时间
        if DATA_SOURCE == 'binance':
            from data.market_data import fetch_binance_data
            # 优先尝试5m，如果失败则尝试3m
            try:
                df_3m = fetch_binance_data(
                    symbol=MARKET_SYMBOL,
                    timeframe='5m',  # 使用5分钟，更稳定
                    start_time=start_time,
                    end_time=end_time,
                    limit=100
                )
            except ValueError as e:
                # 如果5m失败，尝试3m（某些市场可能支持）
                if 'Unsupported timeframe' in str(e):
                    try:
                        logger.info("5分钟数据获取失败，尝试使用3分钟数据")
                        df_3m = fetch_binance_data(
                            symbol=MARKET_SYMBOL,
                            timeframe='3m',
                            start_time=start_time,
                            end_time=end_time,
                            limit=100
                        )
                    except Exception as e2:
                        logger.warning(f"无法获取3分钟或5分钟数据: {e2}，跳过入场点查找")
                        return None
                else:
                    raise
        elif DATA_SOURCE == 'yahoo':
            # Yahoo Finance 最小支持 5m，使用 5m 作为替代
            logger.info("Yahoo Finance 使用5分钟数据作为入场点查找")
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
            # 其他数据源可能不支持短周期，返回None
            logger.warning(f"数据源 {DATA_SOURCE} 可能不支持短周期数据，跳过入场点查找")
            return None
        
        if df_3m.empty:
            logger.warning("短周期数据为空，跳过入场点查找")
            return None
        
        # 确保索引是时间类型
        if not isinstance(df_3m.index, pd.DatetimeIndex):
            df_3m.index = pd.to_datetime(df_3m.index)
        
        # 过滤数据：只保留在信号时间前后8小时内的数据
        time_window = timedelta(hours=max_time_window_hours)
        mask = (df_3m.index >= signal_time - time_window) & (df_3m.index <= signal_time + time_window)
        df_3m_filtered = df_3m[mask].copy()
        
        if df_3m_filtered.empty:
            logger.warning(f"在信号时间前后{max_time_window_hours}小时内没有找到短周期数据")
            logger.warning(f"信号时间: {signal_time}, 查找范围: {signal_time - time_window} 到 {signal_time + time_window}")
            logger.warning(f"数据时间范围: {df_3m.index.min()} 到 {df_3m.index.max()}")
            return None
        
        logger.debug(f"过滤后数据: {len(df_3m_filtered)} 根K线（原始: {len(df_3m)} 根）")
        logger.debug(f"过滤后时间范围: {df_3m_filtered.index.min()} 到 {df_3m_filtered.index.max()}")
        
        # 找到信号时间对应的K线（在过滤后的数据中）
        signal_idx = None
        try:
            # 计算时间差（转换为总秒数，便于比较）
            time_diffs = abs(df_3m_filtered.index - signal_time)
            
            # 将时间差转换为数值（总秒数），然后使用 numpy 的 argmin
            if hasattr(time_diffs, 'total_seconds'):
                # 如果是 TimedeltaIndex，转换为秒数
                time_diffs_seconds = np.array([td.total_seconds() for td in time_diffs])
            elif hasattr(time_diffs, 'values'):
                # 如果有 values 属性，直接使用
                time_diffs_seconds = np.array(time_diffs.values)
            else:
                # 转换为列表再转数组
                time_diffs_seconds = np.array([float(td) for td in time_diffs])
            
            # 使用 numpy 的 argmin 获取最小值的索引位置
            min_idx = np.argmin(time_diffs_seconds)
            
            # 获取对应的索引位置（在过滤后的DataFrame中）
            if 0 <= min_idx < len(df_3m_filtered):
                signal_idx = int(min_idx)
            else:
                signal_idx = len(df_3m_filtered) - 1
                
            # 验证找到的K线是否在8小时窗口内
            if signal_idx < len(df_3m_filtered):
                found_time = df_3m_filtered.index[signal_idx]
                time_diff_hours = abs((found_time - signal_time).total_seconds() / 3600)
                if time_diff_hours > max_time_window_hours:
                    logger.warning(f"找到的K线距离信号时间{time_diff_hours:.1f}小时，超过{max_time_window_hours}小时窗口")
                    return None
                
        except Exception as e:
            logger.warning(f"查找信号时间对应的K线时出错: {e}")
            return None
        
        if signal_idx is None or signal_idx >= len(df_3m_filtered) or signal_idx < 0:
            logger.warning(f"无法找到信号时间对应的K线 (signal_idx={signal_idx}, df_len={len(df_3m_filtered)})")
            return None
        
        # 使用过滤后的数据
        df_3m = df_3m_filtered
        
        # 在信号时间前后max_time_window_hours小时内查找最佳入场点
        best_entry = None
        best_score = -float('inf')
        
        # 计算查找范围：在信号时间前后max_time_window_hours小时内
        # 对于5分钟K线，8小时 = 96个K线，前后各96个，总共最多192个K线
        # 但为了效率，限制在合理范围内
        max_k_bars = max_time_window_hours * 12  # 5分钟K线，8小时=96个
        search_start_idx = max(0, signal_idx - max_k_bars)
        search_end_idx = min(len(df_3m), signal_idx + max_k_bars)
        
        logger.debug(f"查找范围: 索引 {search_start_idx} 到 {search_end_idx} "
                   f"(信号索引: {signal_idx}, 时间窗口: ±{max_time_window_hours}小时)")
        
        for i in range(search_start_idx, search_end_idx):
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
                
                # 3. 距离信号时间（越近越好，在8小时窗口内）
                time_diff_minutes = (entry_time - signal_time).total_seconds() / 60  # 分钟
                time_diff_hours = abs(time_diff_minutes) / 60  # 小时
                
                # 如果超过8小时，跳过这个K线
                if time_diff_hours > max_time_window_hours:
                    continue
                
                # 距离信号时间越近，评分越高
                if abs(time_diff_minutes) <= 15:  # 0-15分钟，最佳
                    score += 30
                    reasons.append(f"距离信号{abs(time_diff_minutes):.0f}分钟")
                elif abs(time_diff_minutes) <= 30:  # 15-30分钟
                    score += 25
                    reasons.append(f"距离信号{abs(time_diff_minutes):.0f}分钟")
                elif abs(time_diff_minutes) <= 60:  # 30-60分钟
                    score += 20
                    reasons.append(f"距离信号{abs(time_diff_minutes):.0f}分钟")
                elif abs(time_diff_minutes) <= 120:  # 1-2小时
                    score += 15
                    reasons.append(f"距离信号{time_diff_hours:.1f}小时")
                elif abs(time_diff_minutes) <= 240:  # 2-4小时
                    score += 10
                    reasons.append(f"距离信号{time_diff_hours:.1f}小时")
                elif abs(time_diff_minutes) <= 480:  # 4-8小时
                    score += 5
                    reasons.append(f"距离信号{time_diff_hours:.1f}小时")
                else:  # 超过8小时（不应该到这里，但保险起见）
                    continue
                
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
                
                time_diff_minutes = (entry_time - signal_time).total_seconds() / 60  # 分钟
                time_diff_hours = abs(time_diff_minutes) / 60  # 小时
                
                # 如果超过8小时，跳过这个K线
                if time_diff_hours > max_time_window_hours:
                    continue
                
                # 距离信号时间越近，评分越高（做空逻辑相同）
                if abs(time_diff_minutes) <= 15:  # 0-15分钟，最佳
                    score += 30
                    reasons.append(f"距离信号{abs(time_diff_minutes):.0f}分钟")
                elif abs(time_diff_minutes) <= 30:  # 15-30分钟
                    score += 25
                    reasons.append(f"距离信号{abs(time_diff_minutes):.0f}分钟")
                elif abs(time_diff_minutes) <= 60:  # 30-60分钟
                    score += 20
                    reasons.append(f"距离信号{abs(time_diff_minutes):.0f}分钟")
                elif abs(time_diff_minutes) <= 120:  # 1-2小时
                    score += 15
                    reasons.append(f"距离信号{time_diff_hours:.1f}小时")
                elif abs(time_diff_minutes) <= 240:  # 2-4小时
                    score += 10
                    reasons.append(f"距离信号{time_diff_hours:.1f}小时")
                elif abs(time_diff_minutes) <= 480:  # 4-8小时
                    score += 5
                    reasons.append(f"距离信号{time_diff_hours:.1f}小时")
                else:  # 超过8小时（不应该到这里，但保险起见）
                    continue
                
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

