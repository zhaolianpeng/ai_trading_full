# backtest/simulator.py
import pandas as pd
import numpy as np
from utils.logger import logger

def simple_backtest(df, enhanced_signals, max_hold=20, atr_mult_stop=1.0, atr_mult_target=2.0, 
                    min_llm_score=40, min_risk_reward=1.5, partial_tp_ratio=0.5, partial_tp_mult=1.0):
    """
    简单回测系统
    
    Args:
        df: 价格数据 DataFrame
        enhanced_signals: 增强信号列表
        max_hold: 最大持仓周期
        atr_mult_stop: 止损 ATR 倍数
        atr_mult_target: 止盈 ATR 倍数
        min_llm_score: LLM 评分最低阈值
    
    Returns:
        (trades_df, metrics): 交易记录 DataFrame 和回测指标字典
    """
    logger.info(f"开始回测，共有 {len(enhanced_signals)} 个信号（最小盈亏比={min_risk_reward}）")
    trades = []
    used_idxs = set()
    from utils.json_i18n import get_value_safe
    
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
        
        # 如果信号已经包含过滤后的信息（quality_score, risk_reward_ratio等），直接使用
        if 'risk_reward_ratio' in item and 'stop_loss' in item and 'take_profit' in item:
            # 使用过滤后的止损止盈
            stop = item['stop_loss']
            target = item['take_profit']
            risk_reward_ratio = item['risk_reward_ratio']
            
            # 再次检查盈亏比
            if risk_reward_ratio < min_risk_reward:
                continue
            
            entry_price = df['close'].iloc[idx+1]
        else:
            # 传统方式：计算止损止盈并检查盈亏比
            signal = get_value_safe(llm, 'signal', 'Neutral') if isinstance(llm, dict) else 'Neutral'
            if signal != 'Long' or score < min_llm_score:
                continue
            if idx in used_idxs or idx+1>=len(df):
                continue
            
            entry_price = df['close'].iloc[idx+1]
            atr = df['atr14'].iloc[idx+1] if not pd.isna(df['atr14'].iloc[idx+1]) else entry_price*0.01
            
            # 计算盈亏比
            stop = entry_price - atr * atr_mult_stop
            target = entry_price + atr * atr_mult_target
            risk = abs(entry_price - stop)
            reward = abs(target - entry_price)
            
            if risk <= 0:
                continue
            
            risk_reward_ratio = reward / risk
            
            # 盈亏比检查
            if risk_reward_ratio < min_risk_reward:
                # 调整止盈以满足最小盈亏比
                required_reward = risk * min_risk_reward
                target = entry_price + required_reward
                risk_reward_ratio = required_reward / risk
        
        if idx in used_idxs or idx+1>=len(df):
            continue
        
        # 计算部分止盈价（如果启用）
        partial_tp = None
        if partial_tp_ratio > 0 and partial_tp_mult > 0:
            atr = df['atr14'].iloc[idx+1] if 'atr14' in df.columns and not pd.isna(df['atr14'].iloc[idx+1]) else entry_price*0.01
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
        
        # 执行交易逻辑（支持部分止盈）
        for j in range(entry_idx, min(len(df), entry_idx+max_hold)):
            low = df['low'].iloc[j]
            high = df['high'].iloc[j]
            close = df['close'].iloc[j]
            
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
        
        # 计算收益率（考虑部分止盈）
        if partial_exited:
            # 部分止盈 + 剩余部分平仓
            partial_return = (partial_exit_price - entry_price) / entry_price * partial_tp_ratio
            remaining_return = (exit_price - entry_price) / entry_price * (1 - partial_tp_ratio)
            total_return = partial_return + remaining_return
        else:
            # 全部平仓
            total_return = (exit_price - entry_price) / entry_price
        
        rule_type = get_value_safe(s, 'type', 'unknown')
        
        # 记录详细交易信息
        trade_record = {
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'full_take_profit': full_take_profit,
            'partial_take_profit': partial_take_profit if partial_take_profit else None,
            'partial_exited': partial_exited,
            'partial_exit_price': partial_exit_price if partial_exited else None,
            'partial_exit_idx': partial_exit_idx if partial_exited else None,
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
        
        for k in range(entry_idx, exit_idx+1):
            used_idxs.add(k)
        for k in range(entry_idx, exit_idx+1):
            used_idxs.add(k)
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
