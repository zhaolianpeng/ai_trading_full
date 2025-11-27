# backtest/risk_manager.py
"""
风险管理模块
包括：持仓限制、每日最大亏损、手续费/滑点计算等
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.logger import logger
from datetime import datetime, timedelta

class RiskManager:
    """
    风险管理器
    """
    
    def __init__(self, 
                 max_positions: int = 5,
                 max_daily_loss: float = 0.05,
                 fee_rate: float = 0.0005,
                 slippage: float = 0.0005,
                 initial_capital: float = 100000.0):
        """
        初始化风险管理器
        
        Args:
            max_positions: 最大同时持仓数
            max_daily_loss: 每日最大亏损比例（5%）
            fee_rate: 手续费率（0.05%）
            slippage: 滑点率（0.05%）
            initial_capital: 初始资金
        """
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # 持仓管理
        self.open_positions: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}  # 每日盈亏记录
        
    def can_open_position(self, entry_price: float, stop_loss: float, 
                         position_size: float = None) -> Tuple[bool, str]:
        """
        检查是否可以开仓
        
        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
            position_size: 仓位大小（如果为None，使用固定风险比例）
        
        Returns:
            (can_open, reason): 是否可以开仓及原因
        """
        # 1. 检查持仓数量
        if len(self.open_positions) >= self.max_positions:
            return False, f"已达到最大持仓数 {self.max_positions}"
        
        # 2. 检查每日亏损限制
        today = datetime.now().strftime('%Y-%m-%d')
        daily_loss = self.daily_pnl.get(today, 0.0)
        if daily_loss <= -self.max_daily_loss * self.current_capital:
            return False, f"今日已超过最大亏损限制 {self.max_daily_loss*100:.1f}%"
        
        # 3. 计算风险
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return False, "止损价格无效"
        
        # 4. 计算仓位大小（如果未指定）
        if position_size is None:
            # 使用固定风险比例（例如，每笔交易风险为总资金的1%）
            risk_per_trade = self.current_capital * 0.01
            position_size = risk_per_trade / risk_per_unit
        
        # 5. 检查资金是否足够
        required_capital = entry_price * position_size * (1 + self.fee_rate + self.slippage)
        if required_capital > self.current_capital * 0.9:  # 保留10%缓冲
            return False, f"资金不足，需要 {required_capital:.2f}，可用 {self.current_capital * 0.9:.2f}"
        
        return True, "可以开仓"
    
    def open_position(self, entry_idx: int, entry_price: float, stop_loss: float,
                     take_profit: float, position_size: float = None, 
                     signal_data: Dict = None) -> Optional[Dict]:
        """
        开仓
        
        Args:
            entry_idx: 入场索引
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 止盈价格
            position_size: 仓位大小
            signal_data: 信号数据（用于记录）
        
        Returns:
            持仓记录，如果无法开仓则返回None
        """
        can_open, reason = self.can_open_position(entry_price, stop_loss, position_size)
        if not can_open:
            logger.warning(f"无法开仓: {reason}")
            return None
        
        # 计算仓位大小
        risk_per_unit = abs(entry_price - stop_loss)
        if position_size is None:
            risk_per_trade = self.current_capital * 0.01
            position_size = risk_per_trade / risk_per_unit
        
        # 计算实际入场价格（考虑滑点）
        actual_entry_price = entry_price * (1 + self.slippage)
        
        # 计算手续费
        entry_fee = actual_entry_price * position_size * self.fee_rate
        
        # 创建持仓记录
        position = {
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'actual_entry_price': actual_entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_fee': entry_fee,
            'signal_data': signal_data,
            'entry_time': datetime.now()
        }
        
        self.open_positions.append(position)
        
        # 扣除资金
        self.current_capital -= (actual_entry_price * position_size + entry_fee)
        
        logger.info(f"开仓: 价格={actual_entry_price:.4f}, 仓位={position_size:.4f}, 手续费={entry_fee:.2f}")
        
        return position
    
    def close_position(self, position: Dict, exit_idx: int, exit_price: float,
                      exit_reason: str = 'target') -> Dict:
        """
        平仓
        
        Args:
            position: 持仓记录
            exit_idx: 出场索引
            exit_price: 出场价格
            exit_reason: 出场原因（'target', 'stop', 'timeout'）
        
        Returns:
            交易记录
        """
        # 计算实际出场价格（考虑滑点）
        actual_exit_price = exit_price * (1 - self.slippage)
        
        # 计算出场手续费
        exit_fee = actual_exit_price * position['position_size'] * self.fee_rate
        
        # 计算盈亏
        pnl = (actual_exit_price - position['actual_entry_price']) * position['position_size']
        net_pnl = pnl - position['entry_fee'] - exit_fee
        return_pct = net_pnl / (position['actual_entry_price'] * position['position_size'])
        
        # 更新资金
        self.current_capital += (actual_exit_price * position['position_size'] - exit_fee)
        
        # 更新每日盈亏
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_pnl[today] = self.daily_pnl.get(today, 0.0) + net_pnl
        
        # 创建交易记录
        trade_record = {
            'entry_idx': position['entry_idx'],
            'exit_idx': exit_idx,
            'entry_price': position['entry_price'],
            'actual_entry_price': position['actual_entry_price'],
            'exit_price': exit_price,
            'actual_exit_price': actual_exit_price,
            'position_size': position['position_size'],
            'pnl': pnl,
            'net_pnl': net_pnl,
            'return': return_pct,
            'entry_fee': position['entry_fee'],
            'exit_fee': exit_fee,
            'total_fee': position['entry_fee'] + exit_fee,
            'exit_reason': exit_reason,
            'signal_data': position.get('signal_data', {})
        }
        
        # 从持仓列表中移除
        if position in self.open_positions:
            self.open_positions.remove(position)
        
        logger.info(f"平仓: 价格={actual_exit_price:.4f}, 盈亏={net_pnl:.2f}, 收益率={return_pct:.2%}, 原因={exit_reason}")
        
        return trade_record
    
    def check_positions(self, df: pd.DataFrame, current_idx: int) -> List[Dict]:
        """
        检查所有持仓，返回需要平仓的持仓
        
        Args:
            df: 价格数据DataFrame
            current_idx: 当前索引
        
        Returns:
            需要平仓的交易记录列表
        """
        closed_trades = []
        positions_to_remove = []
        
        for position in self.open_positions:
            entry_idx = position['entry_idx']
            
            # 检查是否超出最大持仓周期
            if current_idx - entry_idx > 48:  # 假设最大持仓48根K线
                trade = self.close_position(position, current_idx, 
                                          df['close'].iloc[current_idx], 'timeout')
                closed_trades.append(trade)
                positions_to_remove.append(position)
                continue
            
            # 检查止损
            current_low = df['low'].iloc[current_idx]
            if current_low <= position['stop_loss']:
                trade = self.close_position(position, current_idx, 
                                          position['stop_loss'], 'stop')
                closed_trades.append(trade)
                positions_to_remove.append(position)
                continue
            
            # 检查止盈
            current_high = df['high'].iloc[current_idx]
            if current_high >= position['take_profit']:
                trade = self.close_position(position, current_idx, 
                                          position['take_profit'], 'target')
                closed_trades.append(trade)
                positions_to_remove.append(position)
                continue
        
        return closed_trades
    
    def get_current_capital(self) -> float:
        """获取当前资金"""
        return self.current_capital
    
    def get_daily_pnl(self, date: str = None) -> float:
        """
        获取指定日期的盈亏
        
        Args:
            date: 日期字符串（YYYY-MM-DD），如果为None则返回今日
        
        Returns:
            当日盈亏
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        return self.daily_pnl.get(date, 0.0)
    
    def reset_daily_pnl(self):
        """重置每日盈亏（用于新的一天）"""
        self.daily_pnl = {}

