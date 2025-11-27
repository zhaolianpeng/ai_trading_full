# monitoring/signal_logger.py
"""
信号监控和日志系统
记录每条信号的特征、ML得分、LLM输出、最终结果及后验PnL
用于持续学习与参数调优
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from utils.logger import logger

class SignalLogger:
    """
    信号日志记录器
    """
    
    def __init__(self, log_dir: str = 'logs'):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 日志文件路径
        self.signals_log_path = self.log_dir / 'signals_detailed.json'
        self.ml_scores_log_path = self.log_dir / 'ml_scores.csv'
        self.pnl_log_path = self.log_dir / 'posterior_pnl.csv'
        
        # 内存中的日志数据
        self.signals_log: List[Dict] = []
        self.ml_scores_log: List[Dict] = []
        self.pnl_log: List[Dict] = []
    
    def log_signal(self, signal_data: Dict, df: pd.DataFrame, idx: int,
                   ml_score: Optional[float] = None, ml_prediction: Optional[int] = None,
                   structure_label: Optional[str] = None):
        """
        记录信号详情
        
        Args:
            signal_data: 信号数据字典
            df: 价格数据DataFrame
            idx: 信号索引
            ml_score: ML模型评分
            ml_prediction: ML模型预测（0/1）
            structure_label: 市场结构标签
        """
        if idx >= len(df):
            return
        
        row = df.iloc[idx]
        packet = signal_data.get('feature_packet', {})
        llm = signal_data.get('llm', {})
        rule = signal_data.get('rule', {})
        
        # 构建详细日志记录
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal_idx': idx,
            'signal_time': signal_data.get('signal_time'),
            
            # 规则信号
            'rule_type': rule.get('type', 'unknown'),
            'rule_score': rule.get('score', 0),
            'rule_confidence': rule.get('confidence', 'unknown'),
            
            # 市场结构
            'structure_label': structure_label or signal_data.get('structure_label'),
            
            # 特征包（关键特征）
            'features': {
                'trend': packet.get('trend'),
                'ema_alignment': packet.get('ema_alignment'),
                'vol_ratio': packet.get('vol_ratio'),
                'rsi14': packet.get('rsi14'),
                'atr_pct': packet.get('atr_pct'),
                'price_momentum_5': packet.get('price_momentum_5'),
                'price_momentum_20': packet.get('price_momentum_20'),
                'price_position': packet.get('price_position'),
            },
            
            # LLM输出
            'llm': {
                'trend_structure': llm.get('trend_structure'),
                'signal': llm.get('signal'),
                'score': llm.get('score'),
                'confidence': llm.get('confidence'),
                'explanation': llm.get('explanation'),
                'risk': llm.get('risk'),
            },
            
            # ML评分
            'ml_score': ml_score,
            'ml_prediction': ml_prediction,
            
            # 信号评分
            'signal_scores': signal_data.get('signal_scores', {}),
            'composite_score': signal_data.get('composite_score'),
            'quality_score': signal_data.get('quality_score'),
            
            # 价格信息
            'price': {
                'open': float(row['open']) if 'open' in df.columns else None,
                'high': float(row['high']) if 'high' in df.columns else None,
                'low': float(row['low']) if 'low' in df.columns else None,
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in df.columns else None,
            },
            
            # 止损止盈（如果已计算）
            'risk_management': {
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit': signal_data.get('take_profit'),
                'risk_reward_ratio': signal_data.get('risk_reward_ratio'),
            }
        }
        
        self.signals_log.append(log_entry)
    
    def log_trade_result(self, signal_idx: int, trade_record: Dict):
        """
        记录交易结果（后验PnL）
        
        Args:
            signal_idx: 信号索引
            trade_record: 交易记录
        """
        pnl_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal_idx': signal_idx,
            'entry_idx': trade_record.get('entry_idx'),
            'exit_idx': trade_record.get('exit_idx'),
            'entry_price': trade_record.get('entry_price'),
            'exit_price': trade_record.get('exit_price'),
            'return': trade_record.get('return'),
            'pnl': trade_record.get('net_pnl'),
            'exit_reason': trade_record.get('exit_reason'),
            'hold_period': trade_record.get('exit_idx', 0) - trade_record.get('entry_idx', 0),
        }
        
        self.pnl_log.append(pnl_entry)
        
        # 更新对应信号的PnL
        for signal in self.signals_log:
            if signal.get('signal_idx') == signal_idx:
                signal['posterior_pnl'] = {
                    'return': trade_record.get('return'),
                    'pnl': trade_record.get('net_pnl'),
                    'exit_reason': trade_record.get('exit_reason'),
                }
                break
    
    def save_logs(self):
        """保存所有日志到文件"""
        # 保存信号详细日志
        if self.signals_log:
            with open(self.signals_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.signals_log, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"已保存 {len(self.signals_log)} 条信号日志到 {self.signals_log_path}")
        
        # 保存ML评分日志
        if self.ml_scores_log:
            ml_df = pd.DataFrame(self.ml_scores_log)
            ml_df.to_csv(self.ml_scores_log_path, index=False, encoding='utf-8')
            logger.info(f"已保存 {len(self.ml_scores_log)} 条ML评分日志到 {self.ml_scores_log_path}")
        
        # 保存后验PnL日志
        if self.pnl_log:
            pnl_df = pd.DataFrame(self.pnl_log)
            pnl_df.to_csv(self.pnl_log_path, index=False, encoding='utf-8')
            logger.info(f"已保存 {len(self.pnl_log)} 条PnL日志到 {self.pnl_log_path}")
    
    def load_logs(self):
        """从文件加载日志"""
        # 加载信号日志
        if self.signals_log_path.exists():
            try:
                with open(self.signals_log_path, 'r', encoding='utf-8') as f:
                    self.signals_log = json.load(f)
                logger.info(f"已加载 {len(self.signals_log)} 条信号日志")
            except Exception as e:
                logger.warning(f"加载信号日志失败: {e}")
        
        # 加载ML评分日志
        if self.ml_scores_log_path.exists():
            try:
                ml_df = pd.read_csv(self.ml_scores_log_path)
                self.ml_scores_log = ml_df.to_dict('records')
                logger.info(f"已加载 {len(self.ml_scores_log)} 条ML评分日志")
            except Exception as e:
                logger.warning(f"加载ML评分日志失败: {e}")
        
        # 加载PnL日志
        if self.pnl_log_path.exists():
            try:
                pnl_df = pd.read_csv(self.pnl_log_path)
                self.pnl_log = pnl_df.to_dict('records')
                logger.info(f"已加载 {len(self.pnl_log)} 条PnL日志")
            except Exception as e:
                logger.warning(f"加载PnL日志失败: {e}")
    
    def get_training_data(self) -> pd.DataFrame:
        """
        获取训练数据（用于ML模型训练）
        
        Returns:
            包含特征和标签的DataFrame
        """
        training_data = []
        
        for signal in self.signals_log:
            if 'posterior_pnl' not in signal:
                continue
            
            # 提取特征
            features = signal.get('features', {})
            ml_score = signal.get('ml_score', 0.5)
            composite_score = signal.get('composite_score', 0.0)
            quality_score = signal.get('quality_score', 0.0)
            llm_score = signal.get('llm', {}).get('score', 0)
            
            # 标签（基于后验PnL）
            label = 1 if signal['posterior_pnl'].get('return', 0) > 0 else 0
            return_pct = signal['posterior_pnl'].get('return', 0)
            
            sample = {
                'signal_idx': signal.get('signal_idx'),
                'ml_score': ml_score,
                'composite_score': composite_score,
                'quality_score': quality_score,
                'llm_score': llm_score,
                'vol_ratio': features.get('vol_ratio', 1.0),
                'rsi14': features.get('rsi14', 50.0),
                'atr_pct': features.get('atr_pct', 0.0),
                'price_momentum_5': features.get('price_momentum_5', 0.0),
                'price_momentum_20': features.get('price_momentum_20', 0.0),
                'price_position': features.get('price_position', 0.5),
                'label': label,
                'return_pct': return_pct,
            }
            
            training_data.append(sample)
        
        if len(training_data) == 0:
            logger.warning("没有足够的训练数据")
            return pd.DataFrame()
        
        training_df = pd.DataFrame(training_data)
        logger.info(f"准备了 {len(training_df)} 个训练样本")
        
        return training_df

