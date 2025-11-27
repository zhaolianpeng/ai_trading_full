# strategy/multi_timeframe_analyzer.py
"""
多时间周期综合分析模块
查询过往7天的行情数据，分别计算1小时、4小时、天级的K线，综合判断是否能命中交易信号
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from data.market_data import fetch_market_data
from signal_rules import detect_rules
from config import DATA_SOURCE, MARKET_SYMBOL, USE_ADVANCED_TA, USE_ERIC_INDICATORS
from utils.logger import logger

def fetch_multi_timeframe_data(
    symbol: str,
    data_source: str,
    lookback_days: int = 7
) -> Dict[str, pd.DataFrame]:
    """
    获取多个时间周期的数据（1小时、4小时、天级）
    
    Args:
        symbol: 交易对符号
        data_source: 数据源 ('yahoo', 'binance')
        lookback_days: 倒推天数（默认7天）
    
    Returns:
        包含不同时间周期数据的字典: {'1h': df_1h, '4h': df_4h, '1d': df_1d}
    """
    logger.info(f"获取多时间周期数据（过去 {lookback_days} 天，包含最新价格）...")
    
    # 计算时间范围（只指定开始时间，不限制结束时间，确保获取最新数据）
    start_time = datetime.now() - timedelta(days=lookback_days)
    
    dataframes = {}
    
    if data_source == 'binance':
        # Binance 支持多个时间框架
        timeframes = {
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        for tf_key, tf_value in timeframes.items():
            try:
                logger.info(f"获取 {tf_key} 时间周期数据...")
                # 只指定start_time，不指定end_time，确保获取到最新数据
                df = fetch_market_data(
                    symbol=symbol,
                    data_source='binance',
                    timeframe=tf_value,
                    start_time=start_time,
                    limit=1000
                )
                
                if not df.empty:
                    # 不过滤数据，保留所有获取到的数据（包括最新数据）
                    # 只记录时间范围，不实际过滤
                    dataframes[tf_key] = df
                    latest_price = df['close'].iloc[-1] if len(df) > 0 else None
                    latest_time = df.index[-1] if len(df) > 0 else None
                    oldest_time = df.index[0] if len(df) > 0 else None
                    logger.info(f"  {tf_key}: 获取了 {len(df)} 条数据")
                    logger.info(f"    时间范围: {oldest_time} 到 {latest_time}")
                    logger.info(f"    最新价格: ${latest_price:.2f} (时间: {latest_time})")
                    
                    # 验证价格是否合理（BTC价格应该在合理范围内）
                    if latest_price and (latest_price < 10000 or latest_price > 200000):
                        logger.warning(f"    ⚠️ 价格异常: ${latest_price:.2f}，可能获取了错误的市场类型或历史数据")
                else:
                    logger.warning(f"  {tf_key}: 未获取到数据")
            except Exception as e:
                logger.error(f"获取 {tf_key} 数据失败: {e}")
        
    elif data_source == 'yahoo':
        # Yahoo Finance 的时间间隔映射
        intervals = {
            '1h': '1h',
            '4h': '4h',  # Yahoo Finance 可能不支持4h，使用1h替代
            '1d': '1d'
        }
        
        for tf_key, interval in intervals.items():
            try:
                logger.info(f"获取 {tf_key} 时间周期数据...")
                if tf_key == '4h':
                    # Yahoo Finance 不支持4h，使用1h数据然后重采样
                    df = fetch_market_data(
                        symbol=symbol,
                        data_source='yahoo',
                        interval='1h',
                        period=f'{lookback_days}d'
                    )
                    if not df.empty:
                        # 重采样为4小时
                        df = df.resample('4H').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                else:
                    df = fetch_market_data(
                        symbol=symbol,
                        data_source='yahoo',
                        interval=interval,
                        period=f'{lookback_days}d'
                    )
                
                if not df.empty:
                    dataframes[tf_key] = df
                    logger.info(f"  {tf_key}: 获取了 {len(df)} 条数据")
                else:
                    logger.warning(f"  {tf_key}: 未获取到数据")
            except Exception as e:
                logger.error(f"获取 {tf_key} 数据失败: {e}")
    
    else:
        logger.warning(f"数据源 {data_source} 可能不支持多时间周期，跳过")
    
    return dataframes

def analyze_timeframe_signals(
    df: pd.DataFrame,
    timeframe: str,
    use_advanced_ta: bool = True,
    use_eric_indicators: bool = False
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    分析单个时间周期的信号
    
    Args:
        df: 价格数据 DataFrame
        timeframe: 时间周期标识 ('1h', '4h', '1d')
        use_advanced_ta: 是否使用高级技术指标
        use_eric_indicators: 是否使用 Eric 指标
    
    Returns:
        (df_with_ta, signals): 带技术指标的 DataFrame 和信号列表
    """
    logger.info(f"分析 {timeframe} 时间周期的信号...")
    
    # 检测信号
    df_with_ta, signals = detect_rules(
        df,
        use_advanced_ta=use_advanced_ta,
        use_eric_indicators=use_eric_indicators
    )
    
    # 为每个信号添加时间周期标识
    for signal in signals:
        signal['timeframe'] = timeframe
    
    logger.info(f"  {timeframe}: 检测到 {len(signals)} 个信号")
    return df_with_ta, signals

def combine_multi_timeframe_signals(
    all_signals: Dict[str, List[Dict]],
    dataframes: Dict[str, pd.DataFrame],
    timeframes: List[str] = ['1h', '4h', '1d'],
    min_confirmations: int = 2
) -> List[Dict]:
    """
    综合多个时间周期的信号，判断是否能命中交易信号
    
    Args:
        all_signals: 各时间周期的信号字典 {'1h': [signals], '4h': [signals], '1d': [signals]}
        dataframes: 各时间周期的数据框字典 {'1h': df, '4h': df, '1d': df}
        timeframes: 时间周期列表
        min_confirmations: 最少需要多少个时间周期确认（默认2个）
    
    Returns:
        综合后的信号列表（只有多个时间周期都确认的信号才会被返回）
    """
    logger.info(f"综合多时间周期信号（需要至少 {min_confirmations} 个时间周期确认）...")
    
    if not all_signals:
        return []
    
    combined_signals = []
    
    # 以1h信号为基准（因为1h是最细粒度的时间周期）
    if '1h' in all_signals and all_signals['1h'] and '1h' in dataframes:
        df_1h = dataframes['1h']
        
        for signal_1h in all_signals['1h']:
            signal_idx_1h = signal_1h.get('idx', -1)
            if signal_idx_1h < 0 or signal_idx_1h >= len(df_1h):
                continue
            
            # 获取1h信号的时间
            signal_time_1h = None
            if isinstance(df_1h.index, pd.DatetimeIndex):
                signal_time_1h = df_1h.index[signal_idx_1h]
            
            if signal_time_1h is None:
                continue
            
            confirmations = [signal_1h]
            confirmed_timeframes = ['1h']
            
            # 查找4h和1d的对应信号（在相同或接近的时间点）
            for tf in ['4h', '1d']:
                if tf in all_signals and all_signals[tf] and tf in dataframes:
                    df_tf = dataframes[tf]
                    
                    # 找到时间上最接近的信号（允许一定的时间偏差）
                    best_match = None
                    min_time_diff = timedelta(hours=24)  # 最大允许24小时偏差
                    
                    for signal in all_signals[tf]:
                        signal_idx = signal.get('idx', -1)
                        if signal_idx < 0 or signal_idx >= len(df_tf):
                            continue
                        
                        # 获取信号时间
                        signal_time = None
                        if isinstance(df_tf.index, pd.DatetimeIndex):
                            signal_time = df_tf.index[signal_idx]
                        
                        if signal_time is None:
                            continue
                        
                        # 计算时间差
                        time_diff = abs(signal_time - signal_time_1h)
                        
                        # 检查信号类型是否一致
                        signal_type_1h = signal_1h.get('type', '')
                        signal_type_other = signal.get('type', '')
                        
                        # 如果信号类型相似且时间接近
                        is_same_type = (
                            ('long' in signal_type_1h.lower() and 'long' in signal_type_other.lower()) or
                            ('short' in signal_type_1h.lower() and 'short' in signal_type_other.lower())
                        )
                        
                        # 对于4h，允许6小时内偏差；对于1d，允许12小时内偏差
                        max_diff = timedelta(hours=6) if tf == '4h' else timedelta(hours=12)
                        
                        if is_same_type and time_diff <= max_diff and time_diff < min_time_diff:
                            best_match = signal
                            min_time_diff = time_diff
                    
                    if best_match:
                        confirmations.append(best_match)
                        confirmed_timeframes.append(tf)
                        logger.debug(f"  {tf} 时间周期确认: 时间差 {min_time_diff}")
            
            # 如果确认的时间周期数量达到要求
            if len(confirmations) >= min_confirmations:
                combined_signal = {
                    'base_signal': signal_1h,
                    'confirmations': confirmations,
                    'confirmed_timeframes': confirmed_timeframes,
                    'confirmation_count': len(confirmations),
                    'signal_time': signal_time_1h,
                    'signal_type': signal_1h.get('type', 'unknown')
                }
                combined_signals.append(combined_signal)
                logger.info(f"  找到多时间周期确认信号: {confirmed_timeframes}, 类型: {signal_1h.get('type', 'unknown')}")
    
    logger.info(f"综合后共有 {len(combined_signals)} 个多时间周期确认的信号")
    return combined_signals

def run_multi_timeframe_strategy(
    symbol: str,
    data_source: str,
    lookback_days: int = 7,
    min_confirmations: int = 2,
    use_advanced_ta: bool = True,
    use_eric_indicators: bool = False
) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """
    运行多时间周期策略分析
    
    Args:
        symbol: 交易对符号
        data_source: 数据源
        lookback_days: 倒推天数
        min_confirmations: 最少确认时间周期数
        use_advanced_ta: 是否使用高级技术指标
        use_eric_indicators: 是否使用 Eric 指标
    
    Returns:
        (dataframes, combined_signals): 各时间周期的数据框和综合后的信号列表
    """
    logger.info("=" * 60)
    logger.info("开始多时间周期策略分析")
    logger.info("=" * 60)
    
    # 1. 获取多时间周期数据
    dataframes = fetch_multi_timeframe_data(symbol, data_source, lookback_days)
    
    if not dataframes:
        logger.error("未能获取任何时间周期的数据")
        return {}, []
    
    # 2. 分析各时间周期的信号
    all_signals = {}
    for timeframe, df in dataframes.items():
        if not df.empty:
            df_with_ta, signals = analyze_timeframe_signals(
                df, timeframe, use_advanced_ta, use_eric_indicators
            )
            dataframes[timeframe] = df_with_ta
            all_signals[timeframe] = signals
    
    # 3. 综合多时间周期信号
    combined_signals = combine_multi_timeframe_signals(
        all_signals,
        dataframes=dataframes,
        timeframes=list(dataframes.keys()),
        min_confirmations=min_confirmations
    )
    
    logger.info("=" * 60)
    logger.info(f"多时间周期分析完成: 找到 {len(combined_signals)} 个确认信号")
    logger.info("=" * 60)
    
    return dataframes, combined_signals

