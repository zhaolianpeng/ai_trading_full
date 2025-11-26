# main.py
import json
import sys
import os
from pathlib import Path
from data.loader import gen_synthetic, load_csv
from data.market_data import fetch_market_data, get_popular_symbols
from strategy.strategy_runner import run_strategy
from backtest.simulator import simple_backtest
from config import (
    DATA_SOURCE, DATA_PATH, MARKET_SYMBOL, MARKET_PERIOD, MARKET_INTERVAL,
    MARKET_TIMEFRAME, MARKET_LIMIT, USE_LLM, SYNTHETIC_DATA_SIZE, OUTPUT_DIR,
    BACKTEST_MAX_HOLD, BACKTEST_ATR_STOP_MULT, BACKTEST_ATR_TARGET_MULT, MIN_LLM_SCORE,
    USE_ADVANCED_TA, USE_ERIC_INDICATORS, MIN_RISK_REWARD, MIN_QUALITY_SCORE,
    MIN_CONFIRMATIONS, USE_SIGNAL_FILTER, BACKTEST_PARTIAL_TP_RATIO, BACKTEST_PARTIAL_TP_MULT,
    TRADING_MODE, SIGNAL_LOOKBACK_DAYS
)
from utils.logger import logger
from utils.visualization import plot_price_with_signals, plot_backtest_results, generate_report
from utils.config_validator import validate_config, print_config_summary

def main():
    """主函数：运行完整的交易策略流程"""
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
            df = fetch_market_data(
                symbol=MARKET_SYMBOL,
                data_source='binance',
                timeframe=MARKET_TIMEFRAME,
                limit=MARKET_LIMIT
            )
            logger.info(f"已从 Binance 获取 {len(df)} 行数据")
            
        else:  # synthetic
            logger.info(f"生成合成数据 (大小={SYNTHETIC_DATA_SIZE})...")
            df = gen_synthetic(SYNTHETIC_DATA_SIZE)
            logger.info(f"已生成 {len(df)} 行合成数据")
        
        # 2. 运行策略（倒推指定天数内的信号）
        logger.info(f"运行策略 (使用LLM={USE_LLM}, 使用高级指标={USE_ADVANCED_TA}, 使用Eric指标={USE_ERIC_INDICATORS})...")
        df, enhanced = run_strategy(df, use_llm=USE_LLM, use_advanced_ta=USE_ADVANCED_TA, 
                                   use_eric_indicators=USE_ERIC_INDICATORS, lookback_days=SIGNAL_LOOKBACK_DAYS)
        logger.info(f"检测到 {len(enhanced)} 个信号（最近 {SIGNAL_LOOKBACK_DAYS} 天内）")
        
        # 2.5. 应用信号过滤器（提升胜率）
        if USE_SIGNAL_FILTER:
            from strategy.signal_filter import apply_signal_filters
            # 使用交易模式配置的参数（如果已应用）
            from utils.trading_mode import get_trading_mode_config
            data_interval = MARKET_INTERVAL if DATA_SOURCE in ['yahoo', 'csv'] else MARKET_TIMEFRAME
            mode_config = get_trading_mode_config(TRADING_MODE, data_interval)
            
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
        
        # 3. 保存信号日志（使用中文关键字）
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        signals_file = output_dir / 'signals_log.json'
        logger.info(f"保存信号到 {signals_file}（中文关键字）")
        from utils.json_i18n import translate_keys_to_chinese
        enhanced_cn = translate_keys_to_chinese(enhanced)
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
        
        trades_df, metrics = simple_backtest(
            df, enhanced,
            max_hold=max_hold,
            atr_mult_stop=atr_stop,
            atr_mult_target=atr_target,
            min_llm_score=min_llm,
            min_risk_reward=min_rr,
            partial_tp_ratio=BACKTEST_PARTIAL_TP_RATIO,
            partial_tp_mult=partial_tp_mult
        )
        
        # 5. 输出结果
        from utils.i18n import format_metric_value
        logger.info("=" * 60)
        logger.info("回测结果汇总:")
        for k, v in metrics.items():
            logger.info(format_metric_value(k, v))
        logger.info("=" * 60)
        
        # 输出每笔交易的详细信息
        import pandas as pd
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
        data_file = output_dir / 'sample_data.csv'
        
        logger.info(f"保存交易记录到 {trades_file}")
        trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"保存数据到 {data_file}")
        df.to_csv(data_file)
        
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
