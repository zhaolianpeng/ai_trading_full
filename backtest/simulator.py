#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测模拟器模块

支持功能：
1. 做多/做空交易
2. 杠杆交易（合约）
3. 强制平仓检查
4. 部分止盈
5. 一天多次交易（高频模式）
6. 手续费和滑点模拟

作者: AI Trading System
版本: 4.2
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import logger

def simple_backtest(df, enhanced_signals, max_hold=20, atr_mult_stop=1.0, atr_mult_target=2.0, 
                    min_llm_score=40, min_risk_reward=1.5, partial_tp_ratio=0.5, partial_tp_mult=1.0,
                    max_positions: int = 5, max_daily_loss: float = 0.05, 
                    fee_rate: float = 0.0005, slippage: float = 0.0005,
                    allow_multiple_trades_per_day: bool = True):
    """
    简单回测系统
    支持高频交易：允许一天多次交易（如果趋势允许）
    
    Args:
        df: 价格数据 DataFrame
        enhanced_signals: 增强信号列表
        max_hold: 最大持仓周期
        atr_mult_stop: 止损 ATR 倍数
        atr_mult_target: 止盈 ATR 倍数
        min_llm_score: LLM 评分最低阈值
        allow_multiple_trades_per_day: 是否允许一天多次交易（高频模式）
    
    Returns:
        (trades_df, metrics): 交易记录 DataFrame 和回测指标字典
    """
    logger.info(f"开始回测，共有 {len(enhanced_signals)} 个信号（最小盈亏比={min_risk_reward}）")
    if allow_multiple_trades_per_day:
        logger.info("高频交易模式：允许一天多次交易")
    trades = []
    used_idxs = set()
    daily_trades = {}  # 记录每天的交易次数
    from utils.json_i18n import get_value_safe
    
    # 统计回测阶段的过滤原因
    backtest_skip_reasons = {
        '索引超出范围': 0,
        '盈亏比不足（合约）': 0,
        '强制平仓价格不合理': 0,
        '盈亏比不足（过滤后）': 0,
        '信号不是Long/Short': 0,
        'LLM评分不足': 0,
        '索引超出范围（传统方式）': 0,
        '风险为0': 0,
        '重叠持仓': 0,
        '其他原因': 0
    }
    
    for item in enhanced_signals:
        s = get_value_safe(item, 'rule', {})
        idx = get_value_safe(s, 'idx', 0)
        
        # 获取 LLM 评分（用于记录）
        llm = get_value_safe(item, 'llm', {})
        raw_score = get_value_safe(llm, 'score', 0)
        try:
            score = int(float(raw_score))
        except:
            score = 0
        
        # 获取信号方向
        signal_direction = get_value_safe(llm, 'signal', 'Long') if isinstance(llm, dict) else 'Long'
        
        # 检查是否有合约交易信息（优先使用）
        futures_info = get_value_safe(item, 'futures_info', None)
        leverage = 1
        liquidation_price = None
        
        if futures_info:
            # 使用合约交易的止损止盈
            stop = futures_info.get('stop_loss')
            target = futures_info.get('take_profit')
            risk_reward_ratio = futures_info.get('risk_reward_ratio', 0)
            entry_price = futures_info.get('entry_price', df['close'].iloc[idx+1])
            leverage = futures_info.get('leverage', 1)
            liquidation_price = futures_info.get('liquidation_price')
            
            # 再次检查盈亏比
            if risk_reward_ratio < min_risk_reward:
                continue
            
            # 检查强制平仓价格是否合理
            if liquidation_price:
                if signal_direction == 'Long' and liquidation_price >= stop:
                    logger.warning(f"信号 {idx}: 强制平仓价格 {liquidation_price:.2f} 高于止损 {stop:.2f}，跳过")
                    backtest_skip_reasons['强制平仓价格不合理'] += 1
                    continue
                elif signal_direction == 'Short' and liquidation_price <= stop:
                    logger.warning(f"信号 {idx}: 强制平仓价格 {liquidation_price:.2f} 低于止损 {stop:.2f}，跳过")
                    backtest_skip_reasons['强制平仓价格不合理'] += 1
                    continue
        # 如果信号已经包含过滤后的信息（quality_score, risk_reward_ratio等），直接使用
        elif 'risk_reward_ratio' in item and 'stop_loss' in item and 'take_profit' in item:
            # 使用过滤后的止损止盈
            stop = item['stop_loss']
            target = item['take_profit']
            risk_reward_ratio = item['risk_reward_ratio']
            leverage = 1  # 默认无杠杆
            liquidation_price = None
            
            # 再次检查盈亏比
            if risk_reward_ratio < min_risk_reward:
                backtest_skip_reasons['盈亏比不足（过滤后）'] += 1
                continue
            
            entry_price = df['close'].iloc[idx+1]
        else:
            # 传统方式：计算止损止盈并检查盈亏比
            signal = get_value_safe(llm, 'signal', 'Neutral') if isinstance(llm, dict) else 'Neutral'
            
            # 检查是否为高频交易信号
            is_high_freq = get_value_safe(item, 'hf_signal', None) is not None or \
                          get_value_safe(item, 'structure_label', '') == 'HIGH_FREQ'
            
            # 支持Long和Short信号（高频交易可能产生Short信号）
            if signal not in ['Long', 'Short']:
                if not is_high_freq or signal == 'Neutral':
                    backtest_skip_reasons['信号不是Long/Short'] += 1
                    continue
            
            if not is_high_freq and score < min_llm_score:
                backtest_skip_reasons['LLM评分不足'] += 1
                continue
            
            if idx+1 >= len(df):
                backtest_skip_reasons['索引超出范围（传统方式）'] += 1
                continue
            
            entry_price = df['close'].iloc[idx+1]
            atr = df['atr14'].iloc[idx+1] if not pd.isna(df['atr14'].iloc[idx+1]) else entry_price*0.01
            
            # 根据信号方向计算止损止盈
            if signal == 'Short':
                # 做空：止损在上方，止盈在下方
                stop = entry_price + atr * atr_mult_stop
                target = entry_price - atr * atr_mult_target
            else:
                # 做多：止损在下方，止盈在上方
                stop = entry_price - atr * atr_mult_stop
                target = entry_price + atr * atr_mult_target
            
            risk = abs(entry_price - stop)
            reward = abs(target - entry_price)
            
            if risk <= 0:
                backtest_skip_reasons['风险为0'] += 1
                continue
            
            risk_reward_ratio = reward / risk
            
            # 盈亏比检查
            if risk_reward_ratio < min_risk_reward:
                # 调整止盈以满足最小盈亏比
                required_reward = risk * min_risk_reward
                if signal == 'Short':
                    target = entry_price - required_reward
                else:
                    target = entry_price + required_reward
                risk_reward_ratio = required_reward / risk
        
        # 检查是否可以使用该信号
        if idx+1 >= len(df):
            backtest_skip_reasons['索引超出范围'] += 1
            continue
        
        # 高频模式：允许一天多次交易，但避免重叠持仓
        if allow_multiple_trades_per_day:
            # 检查是否有重叠持仓
            has_overlap = False
            for used_idx in used_idxs:
                if abs(used_idx - idx) < max_hold:  # 如果距离太近，可能有重叠
                    has_overlap = True
                    break
            
            if has_overlap:
                backtest_skip_reasons['重叠持仓'] += 1
                continue
        else:
            # 传统模式：完全避免重叠
            if idx in used_idxs:
                backtest_skip_reasons['重叠持仓'] += 1
                continue
        
        # 获取信号方向（用于计算部分止盈和交易逻辑）
        signal_direction = get_value_safe(llm, 'signal', 'Long') if isinstance(llm, dict) else 'Long'
        is_short = signal_direction == 'Short'
        
        # 计算部分止盈价（如果启用）
        partial_tp = None
        if partial_tp_ratio > 0 and partial_tp_mult > 0:
            atr = df['atr14'].iloc[idx+1] if 'atr14' in df.columns and not pd.isna(df['atr14'].iloc[idx+1]) else entry_price*0.01
            if is_short:
                # 做空：部分止盈在下方
                partial_tp = entry_price - atr * partial_tp_mult
            else:
                # 做多：部分止盈在上方
                partial_tp = entry_price + atr * partial_tp_mult
        
        # 记录核心数据
        stop_loss = stop
        full_take_profit = target
        partial_take_profit = partial_tp
        
        entry_idx = idx+1
        exit_idx = None
        exit_price = None
        partial_exited = False
        partial_exit_price = None
        partial_exit_idx = None
        
        # 执行交易逻辑（支持部分止盈、做空和强制平仓检查）
        for j in range(entry_idx, min(len(df), entry_idx+max_hold)):
            low = df['low'].iloc[j]
            high = df['high'].iloc[j]
            close = df['close'].iloc[j]
            
            # 检查强制平仓（如果启用合约交易）
            if futures_info and 'liquidation_price' in futures_info:
                liquidation_price = futures_info['liquidation_price']
                if is_short:
                    # 做空：价格上涨到强制平仓价
                    if high >= liquidation_price:
                        exit_idx = j
                        exit_price = liquidation_price
                        logger.warning(f"强制平仓（做空）: 在索引 {j} 以 {exit_price:.4f} 平仓")
                        break
                else:
                    # 做多：价格下跌到强制平仓价
                    if low <= liquidation_price:
                        exit_idx = j
                        exit_price = liquidation_price
                        logger.warning(f"强制平仓（做多）: 在索引 {j} 以 {exit_price:.4f} 平仓")
                        break
            
            if is_short:
                # 做空：止损在上方，止盈在下方
                # 止损检查（价格向上突破止损）
                if high >= stop_loss:
                    exit_idx = j
                    exit_price = stop_loss
                    break
                
                # 部分止盈检查（价格向下达到部分止盈）
                if partial_take_profit and not partial_exited and low <= partial_take_profit:
                    partial_exited = True
                    partial_exit_price = partial_take_profit
                    partial_exit_idx = j
                    # 继续持仓剩余部分
                
                # 全部止盈检查（价格向下达到全部止盈）
                if low <= full_take_profit:
                    exit_idx = j
                    exit_price = full_take_profit
                    break
            else:
                # 做多：止损在下方，止盈在上方
                # 止损检查
                if low <= stop_loss:
                    exit_idx = j
                    exit_price = stop_loss
                    break
                
                # 部分止盈检查（如果启用且尚未部分止盈）
                if partial_take_profit and not partial_exited and high >= partial_take_profit:
                    partial_exited = True
                    partial_exit_price = partial_take_profit
                    partial_exit_idx = j
                    # 继续持仓剩余部分
                
                # 全部止盈检查
                if high >= full_take_profit:
                    exit_idx = j
                    exit_price = full_take_profit
                    break
            
            # 达到最大持仓周期
            if j == min(len(df)-1, entry_idx+max_hold-1):
                exit_idx = j
                exit_price = close
                break
        
        if exit_idx is None:
            continue
        
        # 计算收益率（考虑部分止盈和做空）
        if is_short:
            # 做空：价格下跌为盈利
            if partial_exited:
                # 部分止盈 + 剩余部分平仓
                partial_return = (entry_price - partial_exit_price) / entry_price * partial_tp_ratio
                remaining_return = (entry_price - exit_price) / entry_price * (1 - partial_tp_ratio)
                total_return = partial_return + remaining_return
            else:
                # 全部平仓
                total_return = (entry_price - exit_price) / entry_price
        else:
            # 做多：价格上涨为盈利
            if partial_exited:
                # 部分止盈 + 剩余部分平仓
                partial_return = (partial_exit_price - entry_price) / entry_price * partial_tp_ratio
                remaining_return = (exit_price - entry_price) / entry_price * (1 - partial_tp_ratio)
                total_return = partial_return + remaining_return
            else:
                # 全部平仓
                total_return = (exit_price - entry_price) / entry_price
        
        rule_type = get_value_safe(s, 'type', 'unknown')
        
        # 获取时间信息
        entry_time = None
        exit_time = None
        partial_exit_time = None
        if isinstance(df.index, pd.DatetimeIndex):
            if entry_idx < len(df.index):
                entry_time = df.index[entry_idx]
            if exit_idx is not None and exit_idx < len(df.index):
                exit_time = df.index[exit_idx]
            if partial_exit_idx is not None and partial_exit_idx < len(df.index):
                partial_exit_time = df.index[partial_exit_idx]
        
        # 记录详细交易信息
        trade_record = {
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time.isoformat() if entry_time else None,
            'exit_time': exit_time.isoformat() if exit_time else None,
            'stop_loss': stop_loss,
            'full_take_profit': full_take_profit,
            'partial_take_profit': partial_take_profit if partial_take_profit else None,
            'partial_exited': partial_exited,
            'partial_exit_price': partial_exit_price if partial_exited else None,
            'partial_exit_idx': partial_exit_idx if partial_exited else None,
            'partial_exit_time': partial_exit_time.isoformat() if partial_exit_time else None,
            'return': total_return,
            'rule_type': rule_type,
            'llm_score': score,
            'risk_reward_ratio': risk_reward_ratio if 'risk_reward_ratio' in locals() else None
        }
        
        trades.append(trade_record)
        
        # 记录每次开单的详细信息到日志
        logger.info(f"开单 #{len(trades)}: 类型={rule_type}, LLM评分={score}")
        logger.info(f"  开单价: {entry_price:.4f}")
        logger.info(f"  止损价: {stop_loss:.4f} (风险: {abs(entry_price - stop_loss):.4f})")
        if partial_take_profit:
            logger.info(f"  部分止盈价: {partial_take_profit:.4f} ({partial_tp_ratio*100:.0f}%仓位)")
        logger.info(f"  全部止盈价: {full_take_profit:.4f} (收益: {abs(full_take_profit - entry_price):.4f})")
        logger.info(f"  盈亏比: {risk_reward_ratio:.2f}" if 'risk_reward_ratio' in locals() else f"  盈亏比: {abs(full_take_profit - entry_price) / abs(entry_price - stop_loss):.2f}")
        if partial_exited:
            logger.info(f"  部分止盈: 在索引 {partial_exit_idx} 以 {partial_exit_price:.4f} 平仓 {partial_tp_ratio*100:.0f}%")
        logger.info(f"  平仓: 在索引 {exit_idx} 以 {exit_price:.4f} 平仓, 收益率: {total_return:.2%}")
        
        # 记录已使用的索引（避免重叠持仓）
        for k in range(entry_idx, exit_idx+1):
            used_idxs.add(k)
        
        # 记录每日交易次数（用于高频交易统计）
        if isinstance(df.index, pd.DatetimeIndex) and entry_idx < len(df.index):
            trade_date = df.index[entry_idx].strftime('%Y-%m-%d')
            daily_trades[trade_date] = daily_trades.get(trade_date, 0) + 1
    # 输出回测阶段的过滤统计
    total_skipped_in_backtest = sum(backtest_skip_reasons.values())
    if total_skipped_in_backtest > 0 or len(trades) < len(enhanced_signals):
        logger.info("=" * 60)
        logger.info("📊 回测阶段过滤统计详情：")
        logger.info("=" * 60)
        logger.info(f"  输入信号数: {len(enhanced_signals)}")
        logger.info(f"  实际交易数: {len(trades)}")
        logger.info(f"  回测阶段跳过: {total_skipped_in_backtest} 个")
        for reason, count in backtest_skip_reasons.items():
            if count > 0:
                percentage = (count / len(enhanced_signals)) * 100 if len(enhanced_signals) > 0 else 0
                logger.info(f"  {reason}: {count} 个 ({percentage:.1f}%)")
        logger.info("=" * 60)
    
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        logger.warning("回测中未执行任何交易")
        metrics = {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'max_consecutive_losses': 0,
            'avg_hold_period': 0,
            'gross_profit': 0,
            'gross_loss': 0
        }
        return trades_df, metrics
    
    logger.info(f"执行了 {len(trades_df)} 笔交易")
    total = len(trades_df)
    win_rate = (trades_df['return']>0).sum()/total
    avg_ret = trades_df['return'].mean()
    gross_profit = trades_df.loc[trades_df['return']>0,'return'].sum()
    gross_loss = -trades_df.loc[trades_df['return']<=0,'return'].sum()
    profit_factor = (gross_profit/(gross_loss+1e-9)) if gross_loss>0 else float('inf')
    equity = (1+trades_df['return']).cumprod()
    peak = equity.cummax()
    drawdown = (equity-peak)/peak
    max_dd = drawdown.min()
    
    # 计算更多指标
    total_return = equity.iloc[-1] - 1 if len(equity) > 0 else 0
    sharpe_ratio = (avg_ret / (trades_df['return'].std() + 1e-9)) * np.sqrt(252) if len(trades_df) > 1 else 0
    
    # 计算最大连续亏损
    consecutive_losses = 0
    max_consecutive_losses = 0
    for ret in trades_df['return']:
        if ret <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    # 计算平均持仓时间
    if 'entry_idx' in trades_df.columns and 'exit_idx' in trades_df.columns:
        avg_hold_period = (trades_df['exit_idx'] - trades_df['entry_idx']).mean()
    else:
        avg_hold_period = 0
    
    metrics = {
        'total_trades': total,
        'win_rate': win_rate,
        'avg_return': avg_ret,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe_ratio,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_hold_period': avg_hold_period,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }
    
    from utils.i18n import get_metric_name_cn
    logger.info(f"回测完成: {total} 笔交易, {get_metric_name_cn('win_rate')}={win_rate:.2%}, {get_metric_name_cn('total_return')}={total_return:.2%}")
    return trades_df, metrics
