#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 交易系统主程序

功能：
1. 数据加载：支持多种数据源（CSV、Yahoo Finance、Binance、合成数据）
2. 策略执行：单时间周期和多时间周期分析
3. 信号过滤：质量评分、风险收益比、LLM评分等
4. 合约交易增强：杠杆、仓位管理、强制平仓保护
5. 高频交易：多时间周期超买/超卖判断
6. 回测系统：支持做多/做空、杠杆、强制平仓
7. 结果可视化：价格图表、回测结果、性能报告

作者: AI Trading System
版本: 4.2
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# 数据层导入
from data.loader import gen_synthetic, load_csv
from data.market_data import fetch_market_data, get_popular_symbols

# 策略层导入
from strategy.strategy_runner import run_strategy

# 回测层导入
from backtest.simulator import simple_backtest

# 配置导入
from config import (
    DATA_SOURCE, DATA_PATH, MARKET_SYMBOL, MARKET_PERIOD, MARKET_INTERVAL,
    MARKET_TIMEFRAME, MARKET_LIMIT, USE_LLM, SYNTHETIC_DATA_SIZE, OUTPUT_DIR,
    BACKTEST_MAX_HOLD, BACKTEST_ATR_STOP_MULT, BACKTEST_ATR_TARGET_MULT, MIN_LLM_SCORE,
    USE_ADVANCED_TA, USE_ERIC_INDICATORS, MIN_RISK_REWARD, MIN_QUALITY_SCORE,
    MIN_CONFIRMATIONS, USE_SIGNAL_FILTER, BACKTEST_PARTIAL_TP_RATIO, BACKTEST_PARTIAL_TP_MULT,
    TRADING_MODE, SIGNAL_LOOKBACK_DAYS, MARKET_TYPE, USE_CACHED_DATA
)

# 工具层导入
from utils.logger import logger
from utils.visualization import plot_price_with_signals, plot_backtest_results, generate_report
from utils.config_validator import validate_config, print_config_summary

def main() -> int:
    """
    主函数：运行完整的交易策略流程
    
    流程：
    1. 配置验证和交易模式应用
    2. 数据加载（CSV/Yahoo/Binance/合成数据）
    3. 策略执行（单时间周期或多时间周期）
    4. 合约交易策略增强（如果启用）
    5. 高频交易策略（如果启用）
    6. 信号过滤（质量评分、风险收益比等）
    7. 回测执行
    8. 结果可视化和报告生成
    
    Returns:
        int: 退出码（0表示成功，非0表示失败）
    """
    try:
        logger.info("=" * 60)
        logger.info("AI 交易系统 - 启动中")
        logger.info("=" * 60)
        
        # 应用交易模式配置
        from utils.trading_mode import apply_trading_mode_config
        trading_config = apply_trading_mode_config()
        logger.info(f"交易模式: {TRADING_MODE}")
        logger.info(f"自动调整参数: 质量评分>={trading_config['min_quality_score']}, "
                   f"确认数>={trading_config['min_confirmations']}, "
                   f"LLM评分>={trading_config['min_llm_score']}, "
                   f"盈亏比>={trading_config['min_risk_reward']}, "
                   f"最大持仓={trading_config['max_hold']}")
        
        # 验证配置
        is_valid, errors = validate_config()
        if not is_valid:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            logger.error("Please fix the configuration errors and try again.")
            return 1
        
        print_config_summary()
        
        # 1. 加载数据
        logger.info(f"数据源: {DATA_SOURCE}")
        
        # 检查是否存在缓存数据
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        cached_data_file = output_dir / 'sample_data.csv'
        use_cache = USE_CACHED_DATA and cached_data_file.exists()
        
        if use_cache:
            logger.info(f"发现缓存数据文件: {cached_data_file}")
            logger.info("尝试从缓存加载数据...")
            try:
                df = load_csv(str(cached_data_file))
                logger.info(f"✅ 成功从缓存加载 {len(df)} 行数据")
                logger.info(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
                # 验证数据完整性
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_columns):
                    logger.info("缓存数据验证通过，使用缓存数据进行回放")
                else:
                    logger.warning(f"缓存数据缺少必要列，将重新获取数据")
                    use_cache = False
            except Exception as e:
                logger.warning(f"加载缓存数据失败: {e}，将重新获取数据")
                use_cache = False
        
        if not use_cache:
            # 从数据源获取数据
            if DATA_SOURCE == 'csv':
            if not DATA_PATH or not Path(DATA_PATH).exists():
                raise FileNotFoundError(f"DATA_SOURCE is 'csv' but DATA_PATH does not exist: {DATA_PATH}")
            logger.info(f"从 CSV 文件加载数据: {DATA_PATH}...")
            df = load_csv(DATA_PATH)
            logger.info(f"已加载 {len(df)} 行数据从 {DATA_PATH}")
            
        elif DATA_SOURCE == 'yahoo':
            logger.info(f"从 Yahoo Finance 获取 {MARKET_SYMBOL} 的数据...")
            try:
                df = fetch_market_data(
                    symbol=MARKET_SYMBOL,
                    data_source='yahoo',
                    period=MARKET_PERIOD,
                    interval=MARKET_INTERVAL
                )
                logger.info(f"已从 Yahoo Finance 获取 {len(df)} 行数据")
            except (ValueError, Exception) as e:
                error_msg = str(e)
                logger.error(f"从 Yahoo Finance 获取数据失败: {error_msg}")
                # 如果是加密货币，提供自动降级到 Binance 的建议
                is_crypto = any(x in MARKET_SYMBOL.upper() for x in ['BTC', 'ETH', 'USD', 'USDT'])
                if is_crypto:
                    logger.error("=" * 60)
                    logger.error("Yahoo Finance 对加密货币支持不稳定！")
                    logger.error("=" * 60)
                    logger.info("推荐使用 Binance 获取加密货币数据：")
                    binance_symbol = MARKET_SYMBOL.replace('-USD', '/USDT').replace('-', '/')
                    logger.info(f"  DATA_SOURCE=binance MARKET_SYMBOL={binance_symbol} MARKET_TIMEFRAME=1h")
                    logger.info("")
                    logger.info("或者使用股票数据测试：")
                    logger.info(f"  DATA_SOURCE=yahoo MARKET_SYMBOL=AAPL MARKET_PERIOD=3mo MARKET_INTERVAL=1d")
                    logger.info("")
                    logger.info("或者使用合成数据：")
                    logger.info(f"  DATA_SOURCE=synthetic")
                    logger.error("=" * 60)
                raise
            
        elif DATA_SOURCE == 'binance':
            logger.info(f"从 Binance 获取 {MARKET_SYMBOL} 的数据...")
            # 检查是否需要获取6个月数据（用于回测）
            backtest_months = int(os.getenv('BACKTEST_MONTHS', '0'))  # 0表示使用默认limit
            if backtest_months > 0:
                logger.info(f"回测模式：获取 {backtest_months} 个月的数据...")
                from data.market_data import fetch_binance_data
                df = fetch_binance_data(
                    symbol=MARKET_SYMBOL,
                    timeframe=MARKET_TIMEFRAME,
                    months=backtest_months
                )
            else:
                df = fetch_market_data(
                    symbol=MARKET_SYMBOL,
                    data_source='binance',
                    timeframe=MARKET_TIMEFRAME,
                    limit=MARKET_LIMIT
                )
            logger.info(f"已从 Binance 获取 {len(df)} 行数据")
            
            # 验证数据时间范围
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                time_span = (df.index[-1] - df.index[0]).days
                logger.info(f"数据时间跨度: {time_span} 天（约 {time_span/30:.1f} 个月）")
            
        else:  # synthetic
            logger.info(f"生成合成数据 (大小={SYNTHETIC_DATA_SIZE})...")
            df = gen_synthetic(SYNTHETIC_DATA_SIZE)
            logger.info(f"已生成 {len(df)} 行合成数据")
        
        # 2. 多时间周期综合分析（查询过往7天的行情数据，分别计算1小时、4小时、天级的K线）
        logger.info("=" * 60)
        logger.info("开始多时间周期综合分析")
        logger.info("=" * 60)
        
        # 检查是否为回测模式（需要至少6个月数据和200+交易）
        backtest_mode = os.getenv('BACKTEST_MODE', 'False').lower() == 'true'
        backtest_months = int(os.getenv('BACKTEST_MONTHS', '0'))
        
        # 如果设置了BACKTEST_MONTHS，自动启用回测模式
        if backtest_months > 0:
            backtest_mode = True
            logger.info(f"回测模式：目标 {backtest_months} 个月数据，至少 200+ 笔交易")
            # 回测模式下禁用多时间周期分析（会产生更多信号）
            use_multi_timeframe = False
            logger.info("回测模式：禁用多时间周期分析，使用单时间周期以产生更多交易信号")
        else:
            use_multi_timeframe = os.getenv('USE_MULTI_TIMEFRAME', 'True').lower() == 'true'
        
        min_timeframe_confirmations = int(os.getenv('MIN_TIMEFRAME_CONFIRMATIONS', '2'))
        
        if use_multi_timeframe and DATA_SOURCE in ['binance', 'yahoo']:
            from strategy.multi_timeframe_analyzer import run_multi_timeframe_strategy
            
            # 运行多时间周期分析
            multi_timeframe_data, combined_signals = run_multi_timeframe_strategy(
                symbol=MARKET_SYMBOL,
                data_source=DATA_SOURCE,
                lookback_days=SIGNAL_LOOKBACK_DAYS,
                min_confirmations=min_timeframe_confirmations,
                use_advanced_ta=USE_ADVANCED_TA,
                use_eric_indicators=USE_ERIC_INDICATORS
            )
            
            if combined_signals:
                logger.info(f"多时间周期分析找到 {len(combined_signals)} 个确认信号")
                # 使用1h数据作为主数据（用于后续分析和回测）
                if '1h' in multi_timeframe_data:
                    df = multi_timeframe_data['1h']
                else:
                    # 如果没有1h数据，使用第一个可用的时间周期
                    df = list(multi_timeframe_data.values())[0] if multi_timeframe_data else df
                
                # 将多时间周期确认的信号转换为标准格式，并进行LLM分析（并发处理）
                enhanced = []
                from strategy.strategy_runner import build_feature_packet
                from ai_agent.signal_interpret import interpret_with_llm
                from config import LLM_PROVIDER, DEEPSEEK_MODEL, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                # 准备所有信号的数据
                signal_data_list = []
                for i, combined_signal in enumerate(combined_signals):
                    base_signal = combined_signal['base_signal']
                    signal_idx = base_signal.get('idx', -1)
                    
                    if signal_idx >= 0 and signal_idx < len(df):
                        packet = build_feature_packet(df, signal_idx)
                        signal_time = None
                        if isinstance(df.index, pd.DatetimeIndex) and signal_idx < len(df.index):
                            signal_time = df.index[signal_idx]
                        
                        signal_data_list.append({
                            'index': i,
                            'combined_signal': combined_signal,
                            'base_signal': base_signal,
                            'packet': packet,
                            'signal_time': signal_time
                        })
                
                # 并发处理LLM调用
                model = DEEPSEEK_MODEL if LLM_PROVIDER == 'deepseek' else OPENAI_MODEL
                from config import LLM_CONCURRENT_WORKERS
                max_workers = LLM_CONCURRENT_WORKERS
                
                def process_multi_timeframe_signal(signal_data):
                    """处理单个多时间周期信号的LLM调用"""
                    i = signal_data['index']
                    combined_signal = signal_data['combined_signal']
                    base_signal = signal_data['base_signal']
                    packet = signal_data['packet']
                    signal_time = signal_data['signal_time']
                    
                    try:
                        llm_out = interpret_with_llm(
                            packet,
                            provider=LLM_PROVIDER,
                            model=model,
                            use_llm=USE_LLM,
                            temperature=OPENAI_TEMPERATURE,
                            max_tokens=OPENAI_MAX_TOKENS
                        )
                    except Exception as e:
                        error_msg = str(e)
                        error_type = type(e).__name__
                        logger.warning(f"LLM分析失败 (信号 {i+1}/{len(combined_signals)}): {error_type}: {error_msg}，使用fallback")
                        try:
                            llm_out = interpret_with_llm(packet, provider=LLM_PROVIDER, model=model, use_llm=False)
                        except Exception as fallback_error:
                            logger.error(f"Fallback也失败: {fallback_error}")
                            llm_out = {
                                'trend_structure': 'Neutral',
                                'signal': 'Neutral',
                                'score': 0,
                                'confidence': 'Low',
                                'explanation': 'Error in LLM interpretation',
                                'risk': 'Unknown'
                            }
                    
                    return {
                        'index': i,
                        'rule': base_signal,
                        'feature_packet': packet,
                        'llm': llm_out,
                        'signal_time': signal_time.isoformat() if signal_time else None,
                        'multi_timeframe': {
                            'confirmations': combined_signal['confirmed_timeframes'],
                            'confirmation_count': combined_signal['confirmation_count']
                        }
                    }
                
                # 使用线程池并发处理
                if len(signal_data_list) > 0:
                    logger.info(f"使用 {max_workers} 个并发线程处理 {len(signal_data_list)} 个多时间周期信号的LLM分析...")
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_signal = {executor.submit(process_multi_timeframe_signal, signal_data): signal_data 
                                           for signal_data in signal_data_list}
                        
                        completed = 0
                        results = {}
                        for future in as_completed(future_to_signal):
                            completed += 1
                            try:
                                result = future.result()
                                results[result['index']] = result
                            except Exception as e:
                                signal_data = future_to_signal[future]
                                logger.error(f"处理信号 {signal_data['index']+1} 时发生异常: {e}")
                                results[signal_data['index']] = {
                                    'index': signal_data['index'],
                                    'rule': signal_data['base_signal'],
                                    'feature_packet': signal_data['packet'],
                                    'llm': {
                                        'trend_structure': 'Neutral',
                                        'signal': 'Neutral',
                                        'score': 0,
                                        'confidence': 'Low',
                                        'explanation': 'Error in processing',
                                        'risk': 'Unknown'
                                    },
                                    'signal_time': signal_data['signal_time'].isoformat() if signal_data['signal_time'] else None,
                                    'multi_timeframe': {
                                        'confirmations': signal_data['combined_signal']['confirmed_timeframes'],
                                        'confirmation_count': signal_data['combined_signal']['confirmation_count']
                                    }
                                }
                            
                            if completed % 10 == 0 or completed == len(signal_data_list):
                                logger.info(f"已处理 {completed}/{len(signal_data_list)} 个多时间周期信号...")
                        
                        # 按原始顺序排序结果
                        enhanced = [results[i] for i in sorted(results.keys())]
                
                logger.info(f"转换后共有 {len(enhanced)} 个多时间周期确认的信号（已进行LLM分析）")
            else:
                logger.warning("多时间周期分析未找到确认信号，使用单时间周期分析")
                # 回退到单时间周期分析
                lookback_days_for_strategy = None if backtest_mode else SIGNAL_LOOKBACK_DAYS
                df, enhanced = run_strategy(df, use_llm=USE_LLM, use_advanced_ta=USE_ADVANCED_TA, 
                                           use_eric_indicators=USE_ERIC_INDICATORS, lookback_days=lookback_days_for_strategy)
        else:
            # 单时间周期分析（原有逻辑）
            logger.info(f"运行单时间周期策略 (使用LLM={USE_LLM}, 使用高级指标={USE_ADVANCED_TA}, 使用Eric指标={USE_ERIC_INDICATORS})...")
            # 回测模式：分析全部数据；正常模式：只分析最近N天
            lookback_days_for_strategy = None if backtest_mode else SIGNAL_LOOKBACK_DAYS
            df, enhanced = run_strategy(df, use_llm=USE_LLM, use_advanced_ta=USE_ADVANCED_TA, 
                                       use_eric_indicators=USE_ERIC_INDICATORS, lookback_days=lookback_days_for_strategy)
        
        # 2.5. 高频交易策略（如果启用）
        use_high_frequency = os.getenv('USE_HIGH_FREQUENCY', 'True').lower() == 'true'
        if use_high_frequency and DATA_SOURCE in ['binance', 'yahoo'] and not backtest_mode:
            logger.info("=" * 60)
            logger.info("开始高频交易策略分析")
            logger.info("=" * 60)
            
            from strategy.high_frequency_strategy import detect_high_frequency_signals, enhance_with_5m_entry
            from data.market_data import fetch_market_data, fetch_binance_data
            
            # 获取多时间周期数据
            df_daily = None
            df_4h = None
            df_1h = df.copy()  # 使用当前1小时数据
            df_5m = None
            
            try:
                # 获取日线数据
                if DATA_SOURCE == 'binance':
                    df_daily = fetch_binance_data(
                        symbol=MARKET_SYMBOL,
                        timeframe='1d',
                        limit=100,
                        market_type=MARKET_TYPE
                    )
                    # 获取4小时数据
                    df_4h = fetch_binance_data(
                        symbol=MARKET_SYMBOL,
                        timeframe='4h',
                        limit=200,
                        market_type=MARKET_TYPE
                    )
                    # 获取5分钟数据（用于优化入场点）
                    df_5m = fetch_binance_data(
                        symbol=MARKET_SYMBOL,
                        timeframe='5m',
                        limit=500,
                        market_type=MARKET_TYPE
                    )
                elif DATA_SOURCE == 'yahoo':
                    # Yahoo Finance 数据
                    df_daily = fetch_market_data(
                        symbol=MARKET_SYMBOL,
                        data_source='yahoo',
                        period='6mo',
                        interval='1d'
                    )
                    df_4h = fetch_market_data(
                        symbol=MARKET_SYMBOL,
                        data_source='yahoo',
                        period='3mo',
                        interval='4h'
                    )
                    # Yahoo Finance 最小支持5分钟
                    df_5m = fetch_market_data(
                        symbol=MARKET_SYMBOL,
                        data_source='yahoo',
                        period='5d',
                        interval='5m'
                    )
                
                # 确保数据有必要的指标
                from features.ta_basic import add_basic_ta
                if df_daily is not None and len(df_daily) > 0:
                    df_daily = add_basic_ta(df_daily)
                if df_4h is not None and len(df_4h) > 0:
                    df_4h = add_basic_ta(df_4h)
                if df_1h is not None and len(df_1h) > 0:
                    df_1h = add_basic_ta(df_1h)
                if df_5m is not None and len(df_5m) > 0:
                    df_5m = add_basic_ta(df_5m)
                
                # 检测高频交易信号
                min_consecutive_overbought = int(os.getenv('HF_MIN_CONSECUTIVE_OVERBOUGHT', '3'))
                min_consecutive_oversold = int(os.getenv('HF_MIN_CONSECUTIVE_OVERSOLD', '3'))
                
                hf_signals = detect_high_frequency_signals(
                    df_daily=df_daily,
                    df_4h=df_4h,
                    df_1h=df_1h,
                    min_consecutive_overbought=min_consecutive_overbought,
                    min_consecutive_oversold=min_consecutive_oversold
                )
                
                if hf_signals:
                    logger.info(f"高频交易策略找到 {len(hf_signals)} 个信号")
                    
                    # 使用5分钟线优化入场点
                    enhanced_hf_signals = []
                    for signal in hf_signals:
                        enhanced_signal = enhance_with_5m_entry(signal, df_5m)
                        
                        # 转换为标准格式
                        from strategy.strategy_runner import build_feature_packet
                        from ai_agent.signal_interpret import interpret_with_llm
                        
                        entry_idx = enhanced_signal.get('entry_idx', -1)
                        if entry_idx >= 0 and entry_idx < len(df_1h):
                            packet = build_feature_packet(df_1h, entry_idx)
                            
                            # LLM分析（可选）
                            if USE_LLM:
                                try:
                                    llm_out = interpret_with_llm(
                                        packet,
                                        provider=LLM_PROVIDER,
                                        model=DEEPSEEK_MODEL if LLM_PROVIDER == 'deepseek' else OPENAI_MODEL,
                                        use_llm=USE_LLM,
                                        temperature=OPENAI_TEMPERATURE,
                                        max_tokens=OPENAI_MAX_TOKENS
                                    )
                                except:
                                    llm_out = {
                                        'trend_structure': 'Neutral',
                                        'signal': enhanced_signal['direction'],
                                        'score': enhanced_signal.get('score', 50),
                                        'confidence': 'Medium',
                                        'explanation': f"高频交易信号: {', '.join(enhanced_signal.get('reasons', []))}",
                                        'risk': []
                                    }
                            else:
                                llm_out = {
                                    'trend_structure': 'Neutral',
                                    'signal': enhanced_signal['direction'],
                                    'score': enhanced_signal.get('score', 50),
                                    'confidence': 'Medium',
                                    'explanation': f"高频交易信号: {', '.join(enhanced_signal.get('reasons', []))}",
                                    'risk': []
                                }
                            
                            # 构建标准信号格式
                            standard_signal = {
                                'rule': {
                                    'type': 'high_frequency',
                                    'idx': entry_idx,
                                    'score': enhanced_signal.get('score', 50),
                                    'confidence': 'high'
                                },
                                'feature_packet': packet,
                                'llm': llm_out,
                                'signal_time': enhanced_signal.get('entry_time').isoformat() if enhanced_signal.get('entry_time') else None,
                                'structure_label': enhanced_signal.get('higher_timeframe_signal', {}).get('signal_type', 'HIGH_FREQ'),
                                'hf_signal': enhanced_signal  # 保留高频信号原始信息
                            }
                            
                            enhanced_hf_signals.append(standard_signal)
                    
                    # 合并到主信号列表
                    if enhanced:
                        enhanced.extend(enhanced_hf_signals)
                    else:
                        enhanced = enhanced_hf_signals
                    
                    logger.info(f"已添加 {len(enhanced_hf_signals)} 个高频交易信号，总信号数: {len(enhanced)}")
                else:
                    logger.info("高频交易策略未找到信号")
                    
            except Exception as e:
                logger.warning(f"高频交易策略执行失败: {e}，继续使用原有信号")
                import traceback
                logger.debug(traceback.format_exc())
        
        if backtest_mode:
            logger.info(f"检测到 {len(enhanced)} 个信号（全量数据回测）")
        else:
            logger.info(f"检测到 {len(enhanced)} 个信号（最近 {SIGNAL_LOOKBACK_DAYS} 天内）")
        
        # 2.4. 合约交易策略增强（如果启用）
        if os.getenv('FUTURES_USE_ENHANCED_STRATEGY', 'True').lower() == 'true' and MARKET_TYPE == 'future':
            logger.info("=" * 60)
            logger.info("应用合约交易策略增强")
            logger.info("=" * 60)
            
            from strategy.futures_strategy import enhance_long_signal_for_futures, enhance_short_signal_for_futures
            from config import FUTURES_LEVERAGE, FUTURES_RISK_PER_TRADE
            
            enhanced_with_futures = []
            for signal in enhanced:
                rule = signal.get('rule', {})
                signal_idx = rule.get('idx', -1)
                llm = signal.get('llm', {})
                signal_direction = llm.get('signal', 'Neutral')
                
                if signal_idx >= 0 and signal_idx < len(df):
                    if signal_direction == 'Long':
                        enhanced_signal = enhance_long_signal_for_futures(
                            signal, df, signal_idx,
                            leverage=FUTURES_LEVERAGE,
                            risk_per_trade=FUTURES_RISK_PER_TRADE
                        )
                    elif signal_direction == 'Short':
                        enhanced_signal = enhance_short_signal_for_futures(
                            signal, df, signal_idx,
                            leverage=FUTURES_LEVERAGE,
                            risk_per_trade=FUTURES_RISK_PER_TRADE
                        )
                    else:
                        enhanced_signal = signal
                    
                    enhanced_with_futures.append(enhanced_signal)
                else:
                    enhanced_with_futures.append(signal)
            
            enhanced = enhanced_with_futures
            logger.info(f"已为 {len(enhanced)} 个信号应用合约交易策略增强")
        
        # 2.5. 应用信号过滤器（提升胜率）
        # 回测模式下，降低阈值以产生更多交易
        if USE_SIGNAL_FILTER:
            from strategy.signal_filter import apply_signal_filters
            # 使用交易模式配置的参数（如果已应用）
            from utils.trading_mode import get_trading_mode_config
            data_interval = MARKET_INTERVAL if DATA_SOURCE in ['yahoo', 'csv'] else MARKET_TIMEFRAME
            mode_config = get_trading_mode_config(TRADING_MODE, data_interval)
            
            # 回测模式下，降低阈值以产生更多交易
            if backtest_mode:
                logger.info("回测模式：大幅降低过滤阈值以产生更多交易信号（目标：200+交易）")
                # 使用非常宽松的阈值，以产生更多交易
                min_quality = int(os.getenv('MIN_QUALITY_SCORE', '10'))  # 降低到10（从20）
                min_conf = int(os.getenv('MIN_CONFIRMATIONS', '1'))  # 保持1
                min_rr = float(os.getenv('MIN_RISK_REWARD', '1.5'))  # 保持1.5（用户要求）
                min_llm = int(os.getenv('MIN_LLM_SCORE', '10'))  # 降低到10（从20）
                logger.info(f"回测模式过滤阈值: 质量评分>={min_quality}, 确认数>={min_conf}, 盈亏比>={min_rr}, LLM评分>={min_llm}")
            else:
                # 使用交易模式配置的参数，但允许环境变量覆盖
                min_quality = int(os.getenv('MIN_QUALITY_SCORE', mode_config['min_quality_score']))
                min_conf = int(os.getenv('MIN_CONFIRMATIONS', mode_config['min_confirmations']))
                min_rr = float(os.getenv('MIN_RISK_REWARD', mode_config['min_risk_reward']))
                min_llm = int(os.getenv('MIN_LLM_SCORE', mode_config['min_llm_score']))
            
            enhanced = apply_signal_filters(
                df, enhanced,
                min_quality_score=min_quality,
                min_confirmations=min_conf,
                min_risk_reward=min_rr,
                min_llm_score=min_llm
            )
            logger.info(f"信号过滤后剩余 {len(enhanced)} 个高质量信号")
            
            # 回测模式：检查是否达到目标交易数量
            if backtest_mode and len(enhanced) < 200:
                logger.warning(f"⚠️ 当前只有 {len(enhanced)} 个信号，未达到 200+ 的目标")
                logger.warning("建议进一步降低阈值：")
                logger.warning(f"  MIN_QUALITY_SCORE={max(20, min_quality-10)}")
                logger.warning(f"  MIN_LLM_SCORE={max(20, min_llm-10)}")
                logger.warning(f"  MIN_RISK_REWARD={max(1.0, min_rr-0.1):.1f}")
        
        # 3. 大趋势信号确认后，在小周期查找具体开单时间
        # 流程：小时级数据找到大趋势信号 -> 重新查询信号时间范围内的5分钟级数据 -> 找具体开单时间
        logger.info("=" * 60)
        logger.info("📈 大趋势信号已确认（小时级），开始在小周期查找具体开单时间...")
        logger.info("=" * 60)
        from utils.entry_finder import find_best_entry_point_3m
        from datetime import datetime
        
        for signal in enhanced:
            # 获取信号信息
            rule = signal.get('rule', {})
            llm = signal.get('llm', {})
            signal_idx = rule.get('idx', -1)
            signal_direction = llm.get('signal', 'Neutral')
            
            # 只处理 Long 信号
            if signal_direction != 'Long':
                continue
            
            # 获取信号时间和价格
            signal_time = None
            signal_price = None
            
            if 'signal_time' in signal and signal['signal_time']:
                try:
                    signal_time = pd.to_datetime(signal['signal_time'])
                except:
                    pass
            
            if signal_time is None and signal_idx >= 0 and signal_idx < len(df):
                if isinstance(df.index, pd.DatetimeIndex):
                    signal_time = df.index[signal_idx]
                else:
                    continue
            
            if signal_time is None:
                continue
            
            signal_price = df['close'].iloc[signal_idx] if signal_idx < len(df) else None
            if signal_price is None:
                continue
            
            # 大趋势信号确认后，重新查询5分钟级数据，找具体开单时间
            # 查询范围：信号时间前后8小时内
            logger.info(f"🔍 信号 {signal_idx}: 重新查询5分钟级数据，查找开单时间（信号时间: {signal_time}）")
            entry_point = find_best_entry_point_3m(
                signal_time=signal_time,
                signal_price=signal_price,
                signal_direction=signal_direction
            )
            
            if entry_point:
                # 将入场时间转换为北京时间并格式化
                from utils.time_utils import to_beijing_time
                entry_time_str = to_beijing_time(entry_point['entry_time'])
                
                # 直接使用中文关键字，避免翻译歧义
                signal['best_entry_3m'] = {
                    '入场时间': entry_time_str,
                    '入场价格': entry_point['entry_price'],
                    '入场原因': entry_point['entry_reason'],
                    '入场评分': entry_point['entry_score']
                }
                logger.info(f"信号 {signal_idx}: 找到短周期入场点，价格={entry_point['entry_price']:.2f}, 时间={entry_time_str}")
        
        # 3.5. 保存信号日志（使用中文关键字，并转换为北京时间）
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        signals_file = output_dir / 'signals_log.json'
        logger.info(f"保存信号到 {signals_file}（中文关键字，北京时间）")
        from utils.json_i18n import translate_keys_to_chinese
        from utils.time_utils import convert_dict_times_to_beijing
        
        enhanced_cn = translate_keys_to_chinese(enhanced)
        # 转换所有时间为北京时间
        enhanced_cn = [convert_dict_times_to_beijing(signal) for signal in enhanced_cn]
        
        with open(signals_file, 'w', encoding='utf8') as f:
            json.dump(enhanced_cn, f, ensure_ascii=False, indent=2, default=str)
        
        # 4. 运行回测
        logger.info("运行回测...")
        # 使用交易模式配置的参数
        from utils.trading_mode import get_trading_mode_config
        data_interval = MARKET_INTERVAL if DATA_SOURCE in ['yahoo', 'csv'] else MARKET_TIMEFRAME
        mode_config = get_trading_mode_config(TRADING_MODE, data_interval)
        
        max_hold = int(os.getenv('BACKTEST_MAX_HOLD', mode_config['max_hold']))
        atr_stop = float(os.getenv('BACKTEST_ATR_STOP_MULT', mode_config['atr_stop_mult']))
        atr_target = float(os.getenv('BACKTEST_ATR_TARGET_MULT', mode_config['atr_target_mult']))
        min_rr = float(os.getenv('MIN_RISK_REWARD', mode_config['min_risk_reward']))
        min_llm = int(os.getenv('MIN_LLM_SCORE', mode_config['min_llm_score']))
        partial_tp_mult = float(os.getenv('BACKTEST_PARTIAL_TP_MULT', mode_config['partial_tp_mult']))
        
        # 检查是否启用高频交易模式
        allow_multiple_trades_per_day = os.getenv('ALLOW_MULTIPLE_TRADES_PER_DAY', 'True').lower() == 'true'
        
        trades_df, metrics = simple_backtest(
            df, enhanced,
            max_hold=max_hold,
            atr_mult_stop=atr_stop,
            atr_mult_target=atr_target,
            min_llm_score=min_llm,
            min_risk_reward=min_rr,
            partial_tp_ratio=BACKTEST_PARTIAL_TP_RATIO,
            partial_tp_mult=partial_tp_mult,
            allow_multiple_trades_per_day=allow_multiple_trades_per_day
        )
        
        # 3.5. 更新信号日志，添加止盈止损时间信息
        if not trades_df.empty and len(enhanced_cn) > 0:
            logger.info("更新信号日志，添加止盈止损时间信息...")
            # 建立信号索引到交易记录的映射
            signal_to_trade = {}
            for trade_idx, trade in trades_df.iterrows():
                entry_idx = int(trade['entry_idx'])
                # 找到对应的信号（通过 entry_idx 匹配）
                for signal_idx, signal in enumerate(enhanced):
                    rule = signal.get('rule', {})
                    signal_entry_idx = rule.get('idx', -1)
                    # entry_idx 是 signal_entry_idx + 1（因为开单在信号后一个周期）
                    if signal_entry_idx + 1 == entry_idx:
                        signal_to_trade[signal_idx] = trade
                        break
            
            # 更新信号日志
            for signal_idx, trade in signal_to_trade.items():
                if signal_idx < len(enhanced_cn):
                    signal = enhanced_cn[signal_idx]
                    # 添加交易时间信息
                    stop_loss_val = trade.get('stop_loss', None)
                    full_tp_val = trade.get('full_take_profit', None)
                    partial_tp_val = trade.get('partial_take_profit', None)
                    
                    # 安全地转换为 float（处理 None 和 NaN 值）
                    def safe_float(val):
                        if val is None:
                            return None
                        try:
                            # 如果是 pandas Series 或 numpy 类型，先转换为 Python 类型
                            if hasattr(val, 'item'):
                                val = val.item()
                            # 检查是否为 NaN（float('nan') 或 numpy.nan）
                            if isinstance(val, float) and val != val:  # NaN 检查
                                return None
                            return float(val)
                        except (ValueError, TypeError):
                            return None
                    
                    signal['交易时间'] = {
                        '开单时间': trade.get('entry_time', None),
                        '平仓时间': trade.get('exit_time', None),
                        '部分止盈时间': trade.get('partial_exit_time', None),
                        '止损价': safe_float(stop_loss_val),
                        '全部止盈价': safe_float(full_tp_val),
                        '部分止盈价': safe_float(partial_tp_val)
                    }
            
            # 重新保存更新后的信号日志（转换为北京时间）
            from utils.time_utils import convert_dict_times_to_beijing
            enhanced_cn = [convert_dict_times_to_beijing(signal) for signal in enhanced_cn]
            
            with open(signals_file, 'w', encoding='utf8') as f:
                json.dump(enhanced_cn, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"已更新信号日志，添加了 {len(signal_to_trade)} 个交易的止盈止损时间信息（北京时间）")
        
        # 5. 输出结果
        from utils.i18n import format_metric_value
        logger.info("=" * 60)
        logger.info("回测结果汇总:")
        for k, v in metrics.items():
            logger.info(format_metric_value(k, v))
        logger.info("=" * 60)
        
        # 回测模式：验证是否达到目标
        if backtest_mode:
            total_trades = metrics.get('total_trades', 0)
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                time_span_days = (df.index[-1] - df.index[0]).days
                time_span_months = time_span_days / 30
            else:
                time_span_months = 0
            
            logger.info("=" * 60)
            logger.info("回测目标验证:")
            logger.info(f"  数据时间跨度: {time_span_months:.1f} 个月 {'✅' if time_span_months >= 6 else '❌'}")
            logger.info(f"  交易数量: {total_trades} 笔 {'✅' if total_trades >= 200 else '❌'}")
            if time_span_months < 6:
                logger.warning(f"  ⚠️ 数据时间跨度不足6个月，建议设置 BACKTEST_MONTHS=6")
            if total_trades < 200:
                logger.warning(f"  ⚠️ 交易数量不足200笔，建议进一步降低过滤阈值")
            logger.info("=" * 60)
        
        # 输出每笔交易的详细信息
        if not trades_df.empty:
            logger.info("\n交易明细:")
            logger.info("-" * 80)
            for idx, trade in trades_df.iterrows():
                logger.info(f"交易 #{idx+1}:")
                logger.info(f"  开单价: {trade['entry_price']:.4f}")
                logger.info(f"  止损价: {trade['stop_loss']:.4f}")
                if 'partial_take_profit' in trade and pd.notna(trade['partial_take_profit']):
                    logger.info(f"  部分止盈价: {trade['partial_take_profit']:.4f}")
                logger.info(f"  全部止盈价: {trade['full_take_profit']:.4f}")
                logger.info(f"  平仓价: {trade['exit_price']:.4f}")
                logger.info(f"  收益率: {trade['return']:.2%}")
                if 'partial_exited' in trade and pd.notna(trade.get('partial_exited')) and trade['partial_exited']:
                    logger.info(f"  部分止盈: 是 (在索引 {int(trade['partial_exit_idx'])} 以 {trade['partial_exit_price']:.4f} 平仓)")
                logger.info(f"  信号类型: {trade['rule_type']}, LLM评分: {int(trade['llm_score'])}")
                logger.info("-" * 80)
        
        # 6. 保存文件
        trades_file = output_dir / 'trades.csv'
        
        logger.info(f"保存交易记录到 {trades_file}")
        trades_df.to_csv(trades_file, index=False)
        
        # 注意：sample_data.csv 已在数据获取时保存到缓存文件
        # 这里不再重复保存，因为原始K线数据已经在获取时保存
        # 如果需要在处理后的数据（包含技术指标）也保存，可以取消下面的注释
        # data_file = output_dir / 'sample_data.csv'
        # logger.info(f"保存处理后的数据到 {data_file}")
        # df.to_csv(data_file)
        
        # 7. 生成可视化图表和报告
        try:
            logger.info("生成可视化图表和报告...")
            chart_file = output_dir / 'trading_chart.png'
            plot_price_with_signals(df, enhanced, output_path=str(chart_file))
            
            if not trades_df.empty:
                backtest_chart_file = output_dir / 'backtest_results.png'
                plot_backtest_results(trades_df, output_path=str(backtest_chart_file))
            
            report_file = output_dir / 'analysis_report.txt'
            generate_report(df, enhanced, trades_df, metrics, output_path=str(report_file))
        except Exception as e:
            logger.warning(f"生成可视化失败: {e}")
        
        logger.info("=" * 60)
        logger.info("所有文件保存成功！")
        logger.info(f"输出目录: {output_dir.absolute()}")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 1
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
