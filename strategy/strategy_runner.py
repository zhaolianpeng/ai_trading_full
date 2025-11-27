# strategy/strategy_runner.py
import pandas as pd
import numpy as np
from signal_rules import detect_rules
from ai_agent.signal_interpret import interpret_with_llm
from config import USE_LLM, OPENAI_MODEL, DEEPSEEK_MODEL, LLM_PROVIDER, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS, MARKET_INTERVAL, MARKET_TIMEFRAME
from utils.logger import logger
import math
from typing import List, Dict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from strategy.market_structure_analyzer import analyze_market_structure

def build_feature_packet(df, idx, window: int = 50):
    """
    构建特征包，包含所有技术指标用于决策判断
    参考 ai_quant_strategy.py 的实现，添加价格序列
    
    Args:
        df: 价格数据 DataFrame
        idx: 当前索引
        window: 价格序列窗口大小（默认50）
    """
    row = df.iloc[idx]
    
    # 获取最近 window 根K线的收盘价序列（用于LLM分析）
    start = max(0, idx - window + 1)
    subset = df.iloc[start:idx+1]
    close_tail = list(subset['close'].round(6).tolist()[-min(len(subset), window):])
    
    packet = {
        # 价格序列（用于LLM分析市场结构）
        "close_tail": close_tail,
        # 基础趋势指标
        "trend": "up" if row['ema21'] > row['ema55'] else "down",
        "ema_alignment": bool(row['ema21'] > row['ema55'] > row['ema100']) if all(col in df.columns for col in ['ema21', 'ema55', 'ema100']) else False,
        "higher_highs": False,
        "higher_lows": False,
        
        # 成交量指标
        "volume_spike": bool(row['volume'] > row['vol_ma50'] * 1.3) if 'vol_ma50' in df.columns and not pd.isna(row.get('vol_ma50', np.nan)) else False,
        "vol_ratio": float(row['volume'] / row['vol_ma50']) if 'vol_ma50' in df.columns and row.get('vol_ma50', 0) > 0 else None,
        
        # 突破指标
        "breakout": bool(row['close'] > row['res50']) if 'res50' in df.columns and not pd.isna(row.get('res50', np.nan)) else False,
        
        # 波动率指标
        "atr": float(row['atr14']) if 'atr14' in df.columns and not pd.isna(row.get('atr14', np.nan)) else None,
        "atr_pct": float((row['atr14'] / row['close']) * 100) if 'atr14' in df.columns and not pd.isna(row.get('atr14', np.nan)) and row['close'] > 0 else None,
        
        # RSI指标
        "rsi14": float(row['rsi14']) if 'rsi14' in df.columns and not pd.isna(row.get('rsi14', np.nan)) else None,
        "rsi_divergence": None,
        
        # MACD指标（如果可用）
        "macd": float(row['macd']) if 'macd' in df.columns and not pd.isna(row.get('macd', np.nan)) else None,
        "macd_signal": float(row['macd_signal']) if 'macd_signal' in df.columns and not pd.isna(row.get('macd_signal', np.nan)) else None,
        "macd_hist": float(row['macd_hist']) if 'macd_hist' in df.columns and not pd.isna(row.get('macd_hist', np.nan)) else None,
        "macd_bullish": bool(row['macd'] > row['macd_signal']) if 'macd' in df.columns and 'macd_signal' in df.columns and not pd.isna(row.get('macd', np.nan)) and not pd.isna(row.get('macd_signal', np.nan)) else False,
        
        # 布林带指标（如果可用）
        "bb_upper": float(row['bb_upper']) if 'bb_upper' in df.columns and not pd.isna(row.get('bb_upper', np.nan)) else None,
        "bb_middle": float(row['bb_middle']) if 'bb_middle' in df.columns and not pd.isna(row.get('bb_middle', np.nan)) else None,
        "bb_lower": float(row['bb_lower']) if 'bb_lower' in df.columns and not pd.isna(row.get('bb_lower', np.nan)) else None,
        "bb_width": float(row['bb_width']) if 'bb_width' in df.columns and not pd.isna(row.get('bb_width', np.nan)) else None,
        "price_above_bb_mid": bool(row['close'] > row['bb_middle']) if 'bb_middle' in df.columns and not pd.isna(row.get('bb_middle', np.nan)) else False,
        
        # 随机指标（如果可用）
        "stoch_k": float(row['stoch_k']) if 'stoch_k' in df.columns and not pd.isna(row.get('stoch_k', np.nan)) else None,
        "stoch_d": float(row['stoch_d']) if 'stoch_d' in df.columns and not pd.isna(row.get('stoch_d', np.nan)) else None,
        
        # 威廉指标（如果可用）
        "williams_r": float(row['williams_r']) if 'williams_r' in df.columns and not pd.isna(row.get('williams_r', np.nan)) else None,
        
        # CCI指标（如果可用）
        "cci": float(row['cci']) if 'cci' in df.columns and not pd.isna(row.get('cci', np.nan)) else None,
        
        # ADX指标（如果可用）
        "adx": float(row['adx']) if 'adx' in df.columns and not pd.isna(row.get('adx', np.nan)) else None,
        "plus_di": float(row['plus_di']) if 'plus_di' in df.columns and not pd.isna(row.get('plus_di', np.nan)) else None,
        "minus_di": float(row['minus_di']) if 'minus_di' in df.columns and not pd.isna(row.get('minus_di', np.nan)) else None,
        
        # Eric Score指标（如果可用）
        "eric_score": float(row['eric_score']) if 'eric_score' in df.columns and not pd.isna(row.get('eric_score', np.nan)) else None,
        "eric_score_smoothed": float(row['eric_score_smoothed']) if 'eric_score_smoothed' in df.columns and not pd.isna(row.get('eric_score_smoothed', np.nan)) else None,
        
        # Donchian通道（如果可用）
        "donchian_upper": float(row['donchian_upper']) if 'donchian_upper' in df.columns and not pd.isna(row.get('donchian_upper', np.nan)) else None,
        "donchian_lower": float(row['donchian_lower']) if 'donchian_lower' in df.columns and not pd.isna(row.get('donchian_lower', np.nan)) else None,
        "donchian_trend": str(row['donchian_trend']) if 'donchian_trend' in df.columns and not pd.isna(row.get('donchian_trend', np.nan)) else None,
        
        # EMA眼（如果可用）
        "ema_eye": float(row['ema_eye']) if 'ema_eye' in df.columns and not pd.isna(row.get('ema_eye', np.nan)) else None,
        
        # 价格动量
        "price_momentum_5": None,
        "price_momentum_20": None,
        
        # 价格位置
        "price_position": None,
        
        # 价格信息
        "close": float(row['close']),
        "open": float(row['open']) if 'open' in df.columns else None,
        "high": float(row['high']) if 'high' in df.columns else None,
        "low": float(row['low']) if 'low' in df.columns else None,
        
        # 支撑阻力位（用于突破判断）
        "res50": float(row['res50']) if 'res50' in df.columns and not pd.isna(row.get('res50', np.nan)) else None,
        "sup50": float(row['sup50']) if 'sup50' in df.columns and not pd.isna(row.get('sup50', np.nan)) else None,
    }
    
    # 计算higher_highs和higher_lows
    window = df['close'].iloc[max(0, idx-19):idx+1]
    if len(window) > 1:
        packet['higher_highs'] = (window.iloc[-1] == window.max())
        packet['higher_lows'] = (window.min() > df['close'].iloc[max(0, idx-40):idx+1].mean())
    
    # 计算价格动量
    if idx >= 5:
        packet['price_momentum_5'] = float((row['close'] - df['close'].iloc[idx-5]) / df['close'].iloc[idx-5])
    if idx >= 20:
        packet['price_momentum_20'] = float((row['close'] - df['close'].iloc[idx-20]) / df['close'].iloc[idx-20])
    
    # 计算价格位置（在最近50根K线中的位置）
    if idx >= 50:
        high_50 = df['high'].iloc[max(0, idx-49):idx+1].max()
        low_50 = df['low'].iloc[max(0, idx-49):idx+1].min()
        if high_50 > low_50:
            packet['price_position'] = float((row['close'] - low_50) / (high_50 - low_50))
    
    return packet

def rule_structure_classifier(df: pd.DataFrame, idx: int) -> str:
    """
    规则基础的结构分类器（作为LLM的fallback）
    参考 ai_quant_strategy.py 的实现
    
    Returns:
        市场结构标签：TREND_UP, TREND_DOWN, RANGE, BREAKOUT_UP, BREAKOUT_DOWN, 
                     REVERSAL_UP, REVERSAL_DOWN
    """
    if idx < 0 or idx >= len(df):
        return "RANGE"
    
    row = df.iloc[idx]
    n = 20
    start = max(0, idx - n + 1)
    window = df['close'].iloc[start:idx+1]
    
    if len(window) < 5:
        return "RANGE"
    
    # EMA 排列判断
    is_ema_up = False
    is_ema_down = False
    if all(col in df.columns for col in ['ema21', 'ema55', 'ema100']):
        is_ema_up = row['ema21'] > row['ema55'] > row['ema100']
        is_ema_down = row['ema21'] < row['ema55'] < row['ema100']
    
    # 更高高点/更高低点判断
    hh = window.iloc[-1] == window.max()
    hl = False
    if start > 0:
        prev_window = df['close'].iloc[max(0, start - n):idx+1]
        if len(prev_window) > 0:
            hl = window.min() > prev_window.mean()
    
    # 更低低点/更低高点判断
    ll = window.iloc[-1] == window.min()
    lh = False
    if start > 0:
        prev_window = df['close'].iloc[max(0, start - n):idx+1]
        if len(prev_window) > 0:
            lh = window.max() < prev_window.mean()
    
    # 突破判断（使用res50/sup50或rolling max/min）
    if 'res50' in df.columns and not pd.isna(row.get('res50', np.nan)):
        if row['close'] > row['res50']:
            return "BREAKOUT_UP"
    elif len(window) >= 20:
        res50 = window.max()
        if row['close'] > res50 * 0.995:
            return "BREAKOUT_UP"
    
    if 'sup50' in df.columns and not pd.isna(row.get('sup50', np.nan)):
        if row['close'] < row['sup50']:
            return "BREAKOUT_DOWN"
    elif len(window) >= 20:
        sup50 = window.min()
        if row['close'] < sup50 * 1.005:
            return "BREAKOUT_DOWN"
    
    # 趋势判断
    if is_ema_up and hh and hl:
        return "TREND_UP"
    if is_ema_down and ll and lh:
        return "TREND_DOWN"
    
    # 反转判断（检查最近5根K线）
    if idx >= 5:
        last5 = df.iloc[max(0, idx-5):idx+1]
        if len(last5) >= 3:
            # 看涨反转：最近下跌后出现大阳线
            if (last5['close'].iloc[-1] > last5['open'].iloc[-1] and 
                last5['close'].iloc[-2] < last5['open'].iloc[-2] and 
                last5['close'].iloc[-3] < last5['open'].iloc[-3]):
                return "REVERSAL_UP"
            # 看跌反转：最近上涨后出现大阴线
            if (last5['close'].iloc[-1] < last5['open'].iloc[-1] and 
                last5['close'].iloc[-2] > last5['open'].iloc[-2] and 
                last5['close'].iloc[-3] > last5['open'].iloc[-3]):
                return "REVERSAL_DOWN"
    
    return "RANGE"

def run_strategy(df, use_llm=USE_LLM, use_advanced_ta=True, use_eric_indicators=False, lookback_days=None):
    """
    运行完整的策略流程
    
    Args:
        df: 价格数据 DataFrame
        use_llm: 是否使用 LLM 分析
        use_advanced_ta: 是否使用高级技术指标
        use_eric_indicators: 是否使用 Eric 指标
        lookback_days: 倒推天数（None表示分析全部数据，用于回测）
    
    Returns:
        (df, enhanced_signals): 增强后的 DataFrame 和信号列表
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 如果lookback_days为None，分析全部数据（用于回测）
    if lookback_days is None:
        logger.info(f"分析全部数据（共 {len(df)} 条），用于回测")
        df_analysis = df.copy()
        lookback_idx = 0
    elif isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        # 计算倒推时间点（从最新数据往前推）
        latest_time = df.index[-1]
        lookback_time = latest_time - timedelta(days=lookback_days)
        
        # 找到倒推时间点的索引位置
        lookback_idx = df.index.get_indexer([lookback_time], method='nearest')[0]
        if lookback_idx < 0:
            lookback_idx = 0
        
        logger.info(f"倒推 {lookback_days} 天内的交易信号（从 {lookback_time} 到 {latest_time}）")
        logger.info(f"数据范围: {len(df)} 条，将分析最近 {len(df) - lookback_idx} 条数据")
        
        # 只分析最近 lookback_days 天的数据
        df_analysis = df.iloc[lookback_idx:].copy()
    else:
        # 如果没有时间索引，使用数据条数估算
        # 假设小时级数据，1周 = 7 * 24 = 168 条
        if '1h' in str(MARKET_INTERVAL) or '1h' in str(MARKET_TIMEFRAME):
            lookback_bars = lookback_days * 24
        else:
            lookback_bars = lookback_days * 24  # 默认按小时估算
        
        lookback_idx = max(0, len(df) - lookback_bars)
        df_analysis = df.iloc[lookback_idx:].copy()
        logger.info(f"倒推 {lookback_days} 天内的交易信号（最近 {len(df_analysis)} 条数据）")
    
    logger.info("检测交易信号...")
    df_analysis_with_ta, signals = detect_rules(df_analysis, use_advanced_ta=use_advanced_ta, use_eric_indicators=use_eric_indicators)
    logger.info(f"发现 {len(signals)} 个原始信号（在最近 {lookback_days} 天内）")
    
    if not signals:
        logger.warning("No signals detected. Try adjusting signal detection parameters or using more data.")
        return df, []
    
    # 将分析数据的索引映射回原始数据的索引
    # 因为 df_analysis 是 df 的子集，需要调整信号索引
    if lookback_idx > 0:
        for signal in signals:
            signal['idx'] = signal['idx'] + lookback_idx
    
    # 确保完整数据也包含技术指标（用于后续分析）
    if len(df_analysis_with_ta) < len(df):
        # 如果只分析了部分数据，需要重新计算完整数据的技术指标
        logger.info("重新计算完整数据的技术指标...")
        df, _ = detect_rules(df, use_advanced_ta=use_advanced_ta, use_eric_indicators=use_eric_indicators)
    else:
        df = df_analysis_with_ta
    
    enhanced_signals = []
    
    # 获取并发数配置（默认5，可以通过环境变量调整）
    max_workers = int(os.getenv('LLM_CONCURRENT_WORKERS', '5'))
    
    # 准备所有信号的数据
    signal_data_list = []
    for i, s in enumerate(signals):
        idx = s['idx']
        if idx >= len(df):
            continue
        packet = build_feature_packet(df, idx)
        signal_time = None
        if isinstance(df.index, pd.DatetimeIndex) and idx < len(df.index):
            signal_time = df.index[idx]
        elif hasattr(df.index, '__getitem__') and idx < len(df):
            try:
                signal_time = df.index[idx]
            except:
                pass
        
        signal_data_list.append({
            'index': i,
            'signal': s,
            'packet': packet,
            'signal_time': signal_time
        })
    
    # 并发处理LLM调用
    model = DEEPSEEK_MODEL if LLM_PROVIDER == 'deepseek' else OPENAI_MODEL
    
    def process_signal(signal_data):
        """处理单个信号的LLM调用"""
        i = signal_data['index']
        s = signal_data['signal']
        packet = signal_data['packet']
        signal_time = signal_data['signal_time']
        idx = s['idx']
        
        # 首先使用规则分类器判断市场结构（作为fallback和过滤）
        structure_label = rule_structure_classifier(df, idx)
        
        # 如果使用LLM，尝试获取更精确的结构判断
        llm_structure_label = None
        llm_score = 50  # 默认分数
        
        if use_llm:
            try:
                llm_out = interpret_with_llm(
                    packet, 
                    provider=LLM_PROVIDER, 
                    model=model, 
                    use_llm=use_llm,
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=OPENAI_MAX_TOKENS
                )
                # 尝试从LLM输出中提取结构标签
                llm_trend = llm_out.get('trend_structure', '')
                # 将LLM的趋势结构映射到结构标签
                if 'Bull' in llm_trend or 'Strong Bull' in llm_trend:
                    llm_structure_label = "TREND_UP"
                elif 'Bear' in llm_trend or 'Strong Bear' in llm_trend:
                    llm_structure_label = "TREND_DOWN"
                else:
                    llm_structure_label = structure_label  # 使用规则分类器的结果
                
                llm_score = llm_out.get('score', 50)
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.warning(f"LLM interpretation failed for signal {i+1}/{len(signals)}: {error_type}: {error_msg}")
                try:
                    llm_out = interpret_with_llm(packet, provider=LLM_PROVIDER, model=model, use_llm=False)
                    llm_structure_label = structure_label
                    llm_score = 50
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for signal {i+1}: {fallback_error}")
                    llm_out = {
                        'trend_structure': 'Neutral',
                        'signal': 'Neutral',
                        'score': 0,
                        'confidence': 'Low',
                        'explanation': 'Error in LLM interpretation',
                        'risk': 'Unknown'
                    }
                    llm_structure_label = structure_label
                    llm_score = 0
        else:
            # 不使用LLM时，使用规则分类器的结果
            llm_out = {
                'trend_structure': structure_label,
                'signal': 'Long' if structure_label in ("TREND_UP", "BREAKOUT_UP", "REVERSAL_UP") else 'Neutral',
                'score': 50,
                'confidence': 'Medium',
                'explanation': f'Rule-based structure: {structure_label}',
                'risk': []
            }
            llm_structure_label = structure_label
            llm_score = 50
        
        # 添加结构标签到LLM输出
        llm_out['structure_label'] = llm_structure_label or structure_label
        llm_out['rule_structure_label'] = structure_label
        
        return {
            'index': i,
            'rule': s, 
            'feature_packet': packet, 
            'llm': llm_out,
            'signal_time': signal_time.isoformat() if signal_time else None,
            'structure_label': llm_structure_label or structure_label
        }
    
    # 使用线程池并发处理
    if len(signal_data_list) > 0:
        logger.info(f"使用 {max_workers} 个并发线程处理 {len(signal_data_list)} 个信号的LLM分析...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_signal = {executor.submit(process_signal, signal_data): signal_data 
                               for signal_data in signal_data_list}
            
            # 收集结果（按完成顺序）
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
                    # 创建默认结果
                    results[signal_data['index']] = {
                        'index': signal_data['index'],
                        'rule': signal_data['signal'],
                        'feature_packet': signal_data['packet'],
                        'llm': {
                            'trend_structure': 'Neutral',
                            'signal': 'Neutral',
                            'score': 0,
                            'confidence': 'Low',
                            'explanation': 'Error in processing',
                            'risk': 'Unknown'
                        },
                        'signal_time': signal_data['signal_time'].isoformat() if signal_data['signal_time'] else None
                    }
                
                # 每处理10个信号输出一次进度
                if completed % 10 == 0 or completed == len(signal_data_list):
                    logger.info(f"已处理 {completed}/{len(signal_data_list)} 个信号...")
        
        # 按原始顺序排序结果
        enhanced_signals = [results[i] for i in sorted(results.keys())]
    
    logger.info(f"已增强 {len(enhanced_signals)} 个信号")
    return df, enhanced_signals
