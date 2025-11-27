# strategy/ml_filter.py
"""
监督学习ML过滤器
使用RandomForest/LogisticRegression对候选信号进行筛选
基于历史信号的后验收益作为标签训练模型
"""
import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from utils.logger import logger

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. ML filter will be disabled. Install with: pip install scikit-learn")

class MLSignalFilter:
    """
    ML信号过滤器
    使用监督学习模型对候选信号进行评分和筛选
    """
    
    def __init__(self, model_type: str = 'random_forest', retrain: bool = False):
        """
        初始化ML过滤器
        
        Args:
            model_type: 模型类型 ('random_forest' 或 'logistic_regression')
            retrain: 是否重新训练模型
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn not available. Please install: pip install scikit-learn")
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = Path('models') / f'ml_filter_{model_type}.pkl'
        self.scaler_path = Path('models') / f'ml_scaler_{model_type}.pkl'
        
        # 创建模型目录
        Path('models').mkdir(exist_ok=True)
        
        # 加载或训练模型
        if not retrain and self.model_path.exists():
            self.load_model()
        else:
            logger.info(f"ML过滤器: 将使用新训练的{model_type}模型")
    
    def extract_features(self, signal_data: Dict, df: pd.DataFrame, idx: int) -> np.ndarray:
        """
        从信号数据中提取特征向量
        
        Args:
            signal_data: 信号数据字典
            df: 价格数据DataFrame
            idx: 信号索引
        
        Returns:
            特征向量
        """
        if idx >= len(df):
            return None
        
        row = df.iloc[idx]
        packet = signal_data.get('feature_packet', {})
        
        features = []
        feature_names = []
        
        # 1. 趋势特征
        if 'ema21' in df.columns and 'ema55' in df.columns and 'ema100' in df.columns:
            features.extend([
                float(row['ema21']),
                float(row['ema55']),
                float(row['ema100']),
                float(row['ema21'] > row['ema55']),
                float(row['ema55'] > row['ema100']),
                float(row['ema21'] > row['ema55'] > row['ema100']),
            ])
            feature_names.extend(['ema21', 'ema55', 'ema100', 'ema21_gt_55', 'ema55_gt_100', 'ema_alignment'])
        
        # 2. 动量特征
        if 'rsi14' in df.columns:
            features.append(float(row['rsi14']))
            feature_names.append('rsi14')
        
        if 'price_momentum_5' in packet:
            features.append(float(packet.get('price_momentum_5', 0)))
            feature_names.append('momentum_5')
        
        if 'price_momentum_20' in packet:
            features.append(float(packet.get('price_momentum_20', 0)))
            feature_names.append('momentum_20')
        
        # 3. 波动率特征
        if 'atr14' in df.columns:
            atr_val = float(row['atr14']) if not pd.isna(row['atr14']) else 0
            atr_pct = (atr_val / row['close']) * 100 if row['close'] > 0 else 0
            features.extend([atr_val, atr_pct])
            feature_names.extend(['atr14', 'atr_pct'])
        
        # 4. 成交量特征
        if 'vol_ratio' in packet:
            features.append(float(packet.get('vol_ratio', 1.0)))
            feature_names.append('vol_ratio')
        
        if 'volume_spike' in packet:
            features.append(float(packet.get('volume_spike', False)))
            feature_names.append('volume_spike')
        
        # 5. 价格位置特征
        if 'price_position' in packet:
            features.append(float(packet.get('price_position', 0.5)))
            feature_names.append('price_position')
        
        # 6. MACD特征（如果可用）
        if 'macd' in packet and packet.get('macd') is not None:
            features.extend([
                float(packet.get('macd', 0)),
                float(packet.get('macd_hist', 0)),
                float(packet.get('macd_bullish', False)),
            ])
            feature_names.extend(['macd', 'macd_hist', 'macd_bullish'])
        
        # 7. 布林带特征（如果可用）
        if 'bb_width' in packet and packet.get('bb_width') is not None:
            features.extend([
                float(packet.get('bb_width', 0)),
                float(packet.get('price_above_bb_mid', False)),
            ])
            feature_names.extend(['bb_width', 'price_above_bb_mid'])
        
        # 8. ADX特征（如果可用）
        if 'adx' in packet and packet.get('adx') is not None:
            features.append(float(packet.get('adx', 0)))
            feature_names.append('adx')
        
        # 9. Eric Score特征（如果可用）
        if 'eric_score_smoothed' in packet and packet.get('eric_score_smoothed') is not None:
            features.append(float(packet.get('eric_score_smoothed', 0)))
            feature_names.append('eric_score')
        
        # 10. 质量评分
        if 'quality_score' in signal_data:
            features.append(float(signal_data.get('quality_score', 0)))
            feature_names.append('quality_score')
        
        # 11. LLM评分
        llm = signal_data.get('llm', {})
        if isinstance(llm, dict):
            llm_score = llm.get('score', 0)
            try:
                features.append(float(llm_score))
            except:
                features.append(0.0)
            feature_names.append('llm_score')
        
        self.feature_names = feature_names
        
        return np.array(features)
    
    def train_model(self, signals_df: pd.DataFrame, target_col: str = 'label'):
        """
        训练ML模型
        
        Args:
            signals_df: 包含特征和标签的DataFrame
            target_col: 目标列名（标签列）
        """
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn not available, cannot train model")
            return
        
        if target_col not in signals_df.columns:
            logger.error(f"Target column '{target_col}' not found in signals_df")
            return
        
        # 准备特征和标签
        feature_cols = [col for col in signals_df.columns if col != target_col]
        X = signals_df[feature_cols].fillna(0).values
        y = signals_df[target_col].values
        
        if len(np.unique(y)) < 2:
            logger.warning("Not enough classes for training. Need at least 2 classes.")
            return
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        logger.info(f"训练{self.model_type}模型...")
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        logger.info(f"训练集准确率: {train_score:.4f}")
        logger.info(f"测试集准确率: {test_score:.4f}")
        
        # 预测概率
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_proba)
            logger.info(f"测试集AUC: {auc:.4f}")
        
        # 保存模型
        self.save_model()
        
        logger.info(f"模型已保存到 {self.model_path}")
    
    def predict(self, signal_data: Dict, df: pd.DataFrame, idx: int) -> Tuple[float, float]:
        """
        预测信号的质量（概率和类别）
        
        Args:
            signal_data: 信号数据字典
            df: 价格数据DataFrame
            idx: 信号索引
        
        Returns:
            (probability, prediction): 预测概率和预测类别
        """
        if self.model is None:
            logger.warning("ML模型未训练，返回默认值")
            return 0.5, 0
        
        # 提取特征
        features = self.extract_features(signal_data, df, idx)
        if features is None or len(features) == 0:
            return 0.5, 0
        
        # 标准化
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 预测
        proba = self.model.predict_proba(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        
        # 返回正类概率和预测
        positive_prob = proba[1] if len(proba) > 1 else proba[0]
        
        return float(positive_prob), int(prediction)
    
    def save_model(self):
        """保存模型和标准化器"""
        if self.model is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"模型已保存: {self.model_path}")
    
    def load_model(self):
        """加载模型和标准化器"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"模型已加载: {self.model_path}")
        except Exception as e:
            logger.warning(f"加载模型失败: {e}，将使用新模型")
            self.model = None

def prepare_training_data(enhanced_signals: List[Dict], trades_df: pd.DataFrame, 
                         df: pd.DataFrame) -> pd.DataFrame:
    """
    准备训练数据：将信号特征与后验收益（标签）结合
    
    Args:
        enhanced_signals: 增强信号列表
        trades_df: 交易记录DataFrame（包含entry_idx, return_pct等）
        df: 价格数据DataFrame
    
    Returns:
        包含特征和标签的DataFrame
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available, cannot prepare training data")
        return pd.DataFrame()
    
    ml_filter = MLSignalFilter(model_type='random_forest')
    
    training_data = []
    
    for signal in enhanced_signals:
        rule = signal.get('rule', {})
        idx = rule.get('idx', -1)
        
        if idx < 0 or idx >= len(df):
            continue
        
        # 提取特征
        features = ml_filter.extract_features(signal, df, idx)
        if features is None:
            continue
        
        # 查找对应的交易结果
        trade = trades_df[trades_df['entry_idx'] == idx]
        if len(trade) > 0:
            # 使用后验收益作为标签（收益>0为1，否则为0）
            label = 1 if trade.iloc[0]['return_pct'] > 0 else 0
            return_pct = trade.iloc[0]['return_pct']
        else:
            # 如果没有交易记录，跳过
            continue
        
        # 构建训练样本
        sample = {}
        for i, name in enumerate(ml_filter.feature_names):
            if i < len(features):
                sample[name] = features[i]
        sample['label'] = label
        sample['return_pct'] = return_pct
        sample['entry_idx'] = idx
        
        training_data.append(sample)
    
    if len(training_data) == 0:
        logger.warning("没有足够的训练数据")
        return pd.DataFrame()
    
    training_df = pd.DataFrame(training_data)
    logger.info(f"准备了 {len(training_df)} 个训练样本")
    
    return training_df

