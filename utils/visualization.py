# utils/visualization.py
"""
可视化工具：生成交易图表和分析报告
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from utils.logger import logger
from utils.i18n import get_metric_name_cn, get_signal_type_cn, get_llm_signal_cn, format_metric_value

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.font_manager as fm
    MATPLOTLIB_AVAILABLE = True
    
    # 配置中文字体
    def setup_chinese_font():
        """设置 matplotlib 使用中文字体"""
        # macOS 常见中文字体（按优先级排序）
        chinese_fonts = [
            'PingFang SC',           # 苹方（macOS 默认）
            'PingFang TC',           # 苹方繁体
            'STHeiti',               # 华文黑体
            'STHeiti Light',         # 华文黑体 Light
            'Heiti SC',              # 黑体-简
            'Heiti TC',              # 黑体-繁
            'Arial Unicode MS',      # Arial Unicode MS
            'STSong',                # 华文宋体
            'SimHei',                # 黑体（Windows）
            'Microsoft YaHei',       # 微软雅黑（Windows）
            'WenQuanYi Micro Hei',   # 文泉驿微米黑（Linux）
            'Noto Sans CJK SC',      # Noto Sans（Linux）
        ]
        
        # 查找可用的中文字体
        try:
            # 刷新字体缓存（如果需要）
            try:
                fm._rebuild()
            except:
                pass
            
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            for font_name in chinese_fonts:
                if font_name in available_fonts:
                    # 将中文字体设置为第一优先级
                    current_fonts = plt.rcParams['font.sans-serif']
                    plt.rcParams['font.sans-serif'] = [font_name] + [f for f in current_fonts if f != font_name]
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    logger.info(f"已配置中文字体: {font_name}")
                    return font_name
            
            # 如果列表中没有，尝试通过路径查找
            system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            chinese_font_files = [f for f in system_fonts if any(name in f for name in ['PingFang', 'STHeiti', 'Heiti', 'STSong'])]
            if chinese_font_files:
                # 使用第一个找到的字体文件
                font_prop = fm.FontProperties(fname=chinese_font_files[0])
                font_name = font_prop.get_name()
                current_fonts = plt.rcParams['font.sans-serif']
                plt.rcParams['font.sans-serif'] = [font_name] + [f for f in current_fonts if f != font_name]
                plt.rcParams['axes.unicode_minus'] = False
                logger.info(f"通过路径找到中文字体: {font_name}")
                return font_name
        except Exception as e:
            logger.warning(f"设置中文字体时出错: {e}")
        
        # 如果都没有找到，设置基本配置
        plt.rcParams['axes.unicode_minus'] = False
        logger.warning("未找到中文字体，图表中的中文可能无法正确显示")
        return None
    
    # 初始化时设置字体
    setup_chinese_font()
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Visualization features disabled.")

def plot_price_with_signals(df: pd.DataFrame, signals: List[Dict], 
                           output_path: Optional[str] = None,
                           figsize: tuple = (15, 10)) -> None:
    """
    绘制价格图表，标注交易信号
    
    Args:
        df: 价格数据 DataFrame
        signals: 信号列表
        output_path: 输出文件路径
        figsize: 图表大小
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # 1. 价格和 EMA
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='收盘价', linewidth=1.5, color='black')
    if 'ema21' in df.columns:
        ax1.plot(df.index, df['ema21'], label='EMA21', alpha=0.7, linewidth=1)
    if 'ema55' in df.columns:
        ax1.plot(df.index, df['ema55'], label='EMA55', alpha=0.7, linewidth=1)
    if 'ema100' in df.columns:
        ax1.plot(df.index, df['ema100'], label='EMA100', alpha=0.7, linewidth=1)
    
    # 标注信号
    from utils.json_i18n import get_value_safe
    for sig in signals:
        rule = get_value_safe(sig, 'rule', {})
        idx = get_value_safe(rule, 'idx', 0)
        if idx < len(df):
            llm = get_value_safe(sig, 'llm', {})
            signal_type = get_value_safe(llm, 'signal', 'Neutral') if isinstance(llm, dict) else 'Neutral'
            color = 'green' if signal_type == 'Long' else 'red' if signal_type == 'Short' else 'gray'
            ax1.scatter(df.index[idx], df['close'].iloc[idx], 
                       color=color, s=100, alpha=0.6, marker='^' if signal_type == 'Long' else 'v')
    
    ax1.set_ylabel('价格')
    ax1.set_title('价格图表与交易信号')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RSI
    ax2 = axes[1]
    if 'rsi14' in df.columns:
        ax2.plot(df.index, df['rsi14'], label='RSI14', color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(['超买线 (70)', '超卖线 (30)'])
    ax2.grid(True, alpha=0.3)
    
    # 3. Volume
    ax3 = axes[2]
    ax3.bar(df.index, df['volume'], alpha=0.6, color='blue', width=0.8)
    if 'vol_ma50' in df.columns:
        ax3.plot(df.index, df['vol_ma50'], label='成交量均线50', color='orange', linewidth=1.5)
    ax3.set_ylabel('成交量')
    ax3.set_xlabel('时间')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 格式化 x 轴日期
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved chart to {output_path}")
    else:
        # 使用 OUTPUT_DIR 配置
        from config import OUTPUT_DIR
        from pathlib import Path
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / 'trading_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved chart to {chart_path}")
    
    plt.close()

def plot_backtest_results(trades_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """
    绘制回测结果图表
    
    Args:
        trades_df: 交易记录 DataFrame
        output_path: 输出文件路径
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    if trades_df.empty:
        logger.warning("No trades to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 累计收益曲线
    ax1 = axes[0, 0]
    equity = (1 + trades_df['return']).cumprod()
    ax1.plot(range(len(equity)), equity, linewidth=2, color='blue')
    ax1.set_title('累计收益曲线')
    ax1.set_xlabel('交易次数')
    ax1.set_ylabel('累计收益率')
    ax1.grid(True, alpha=0.3)
    
    # 2. 收益分布
    ax2 = axes[0, 1]
    ax2.hist(trades_df['return'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('收益分布')
    ax2.set_xlabel('收益率')
    ax2.set_ylabel('频次')
    ax2.grid(True, alpha=0.3)
    
    # 3. 每笔交易收益
    ax3 = axes[1, 0]
    colors = ['green' if r > 0 else 'red' for r in trades_df['return']]
    ax3.bar(range(len(trades_df)), trades_df['return'], color=colors, alpha=0.6)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('单笔交易收益')
    ax3.set_xlabel('交易次数')
    ax3.set_ylabel('收益率')
    ax3.grid(True, alpha=0.3)
    
    # 4. 信号类型统计
    ax4 = axes[1, 1]
    if 'rule_type' in trades_df.columns:
        rule_counts = trades_df['rule_type'].value_counts()
        # 使用中文标签
        labels_cn = [get_signal_type_cn(label) for label in rule_counts.index]
        ax4.pie(rule_counts.values, labels=labels_cn, autopct='%1.1f%%', startangle=90)
        ax4.set_title('信号类型分布')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved backtest chart to {output_path}")
    else:
        # 使用 OUTPUT_DIR 配置
        from config import OUTPUT_DIR
        from pathlib import Path
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / 'backtest_results.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved backtest chart to {chart_path}")
    
    plt.close()

def generate_report(df: pd.DataFrame, signals: List[Dict], 
                   trades_df: pd.DataFrame, metrics: Dict,
                   output_path: Optional[str] = None) -> str:
    """
    生成文本格式的分析报告
    
    Args:
        df: 价格数据
        signals: 信号列表
        trades_df: 交易记录
        metrics: 回测指标
        output_path: 输出文件路径
    
    Returns:
        报告文本
    """
    report = []
    report.append("=" * 80)
    report.append("AI 交易系统 - 分析报告")
    report.append("=" * 80)
    report.append("")
    
    # 数据概览
    report.append("## 数据概览")
    report.append(f"数据点总数: {len(df)}")
    report.append(f"日期范围: {df.index.min()} 至 {df.index.max()}")
    report.append(f"价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    report.append("")
    
    # 信号统计
    report.append("## 信号统计")
    report.append(f"检测到的信号总数: {len(signals)}")
    if signals:
        signal_types = {}
        llm_signals = {}
        from utils.json_i18n import get_value_safe
        for sig in signals:
            rule = get_value_safe(sig, 'rule', {})
            rule_type = get_value_safe(rule, 'type', 'unknown')
            signal_types[rule_type] = signal_types.get(rule_type, 0) + 1
            
            llm = get_value_safe(sig, 'llm', {})
            if isinstance(llm, dict):
                llm_signal = get_value_safe(llm, 'signal', 'Neutral')
                llm_signals[llm_signal] = llm_signals.get(llm_signal, 0) + 1
        
        report.append("\n信号类型分布:")
        for stype, count in signal_types.items():
            report.append(f"  - {get_signal_type_cn(stype)}: {count}")
        
        report.append("\nLLM 推荐分布:")
        for lsig, count in llm_signals.items():
            report.append(f"  - {get_llm_signal_cn(lsig)}: {count}")
    report.append("")
    
    # 回测结果
    report.append("## 回测结果")
    for key, value in metrics.items():
        report.append(format_metric_value(key, value))
    report.append("")
    
    # 交易统计
    if not trades_df.empty:
        report.append("## 交易统计")
        report.append(f"总交易次数: {len(trades_df)}")
        report.append(f"盈利交易: {(trades_df['return'] > 0).sum()}")
        report.append(f"亏损交易: {(trades_df['return'] <= 0).sum()}")
        if (trades_df['return'] > 0).any():
            avg_win = trades_df[trades_df['return'] > 0]['return'].mean()
            report.append(f"平均盈利: {avg_win:.4f} ({avg_win:.2%})")
        else:
            report.append("平均盈利: 无")
        if (trades_df['return'] <= 0).any():
            avg_loss = trades_df[trades_df['return'] <= 0]['return'].mean()
            report.append(f"平均亏损: {avg_loss:.4f} ({avg_loss:.2%})")
        else:
            report.append("平均亏损: 无")
        report.append(f"最佳交易: {trades_df['return'].max():.4f} ({trades_df['return'].max():.2%})")
        report.append(f"最差交易: {trades_df['return'].min():.4f} ({trades_df['return'].min():.2%})")
    report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Saved report to {output_path}")
    else:
        # 使用 OUTPUT_DIR 配置
        from config import OUTPUT_DIR
        from pathlib import Path
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / 'analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Saved report to {report_path}")
    
    return report_text
