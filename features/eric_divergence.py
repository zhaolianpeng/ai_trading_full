# features/eric_divergence.py
"""
价格与 Eric Score 的背离检测
参考 TradingView "Eric 全面策略"
"""
import pandas as pd
import numpy as np

def eric_divergence(df, lookback=10):
    """
    检测价格与 Eric Score 的背离（优化版本，使用向量化）
    
    Args:
        df: 价格数据 DataFrame（需包含 high, low, close, eric_score_smoothed）
        lookback: 背离回溯周期（默认10）
    
    Returns:
        DataFrame 添加以下列：
        - bull_divergence: 牛背离（价格新低但score更高）
        - bear_divergence: 空背离（价格新高但score更低）
        - divergence_strength: 背离强度（0-4）
    """
    df = df.copy()
    
    if 'eric_score_smoothed' not in df.columns:
        df['bull_divergence'] = False
        df['bear_divergence'] = False
        df['divergence_strength'] = 0
        return df
    
    df['bull_divergence'] = False
    df['bear_divergence'] = False
    df['divergence_strength'] = 0
    
    # 优化的背离检测：只检查最近的数据点以减少计算量
    check_range = min(200, len(df))  # 只检查最近200个点
    start_idx = max(lookback, len(df) - check_range)
    
    # 遍历每个点，检查过去 lookback 周期内的背离
    for i in range(start_idx, len(df)):
        # 牛背离：价格新低但score更高
        window_low = df['low'].iloc[i-lookback:i+1]
        window_score = df['eric_score_smoothed'].iloc[i-lookback:i+1]
        
        # 找到最低价的位置
        min_low_idx = window_low.idxmin()
        min_low_pos = window_low.index.get_loc(min_low_idx)
        
        # 检查在最低价之前是否有更高的score
        if min_low_pos > 0:
            before_min = window_score.iloc[:min_low_pos]
            if len(before_min) > 0 and window_score.iloc[min_low_pos] > before_min.max():
                df.loc[df.index[i], 'bull_divergence'] = True
                df.loc[df.index[i], 'divergence_strength'] = 1
        
        # 空背离：价格新高但score更低
        window_high = df['high'].iloc[i-lookback:i+1]
        max_high_idx = window_high.idxmax()
        max_high_pos = window_high.index.get_loc(max_high_idx)
        
        # 检查在最高价之前是否有更低的score
        if max_high_pos > 0:
            before_max = window_score.iloc[:max_high_pos]
            if len(before_max) > 0 and window_score.iloc[max_high_pos] < before_max.min():
                df.loc[df.index[i], 'bear_divergence'] = True
                df.loc[df.index[i], 'divergence_strength'] = max(
                    df.loc[df.index[i], 'divergence_strength'], 1
                )
    
    return df

