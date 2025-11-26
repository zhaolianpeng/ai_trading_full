# main.py
import json
import sys
from pathlib import Path
from data.loader import gen_synthetic, load_csv
from data.market_data import fetch_market_data, get_popular_symbols
from strategy.strategy_runner import run_strategy
from backtest.simulator import simple_backtest
from config import (
    DATA_SOURCE, DATA_PATH, MARKET_SYMBOL, MARKET_PERIOD, MARKET_INTERVAL,
    MARKET_TIMEFRAME, MARKET_LIMIT, USE_LLM, SYNTHETIC_DATA_SIZE, OUTPUT_DIR,
    BACKTEST_MAX_HOLD, BACKTEST_ATR_STOP_MULT, BACKTEST_ATR_TARGET_MULT, MIN_LLM_SCORE,
    USE_ADVANCED_TA, USE_ERIC_INDICATORS
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
            df = fetch_market_data(
                symbol=MARKET_SYMBOL,
                data_source='yahoo',
                period=MARKET_PERIOD,
                interval=MARKET_INTERVAL
            )
            logger.info(f"已从 Yahoo Finance 获取 {len(df)} 行数据")
            
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
        
        # 2. 运行策略
        logger.info(f"运行策略 (使用LLM={USE_LLM}, 使用高级指标={USE_ADVANCED_TA}, 使用Eric指标={USE_ERIC_INDICATORS})...")
        df, enhanced = run_strategy(df, use_llm=USE_LLM, use_advanced_ta=USE_ADVANCED_TA, use_eric_indicators=USE_ERIC_INDICATORS)
        logger.info(f"检测到 {len(enhanced)} 个信号")
        
        # 3. 保存信号日志
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        signals_file = output_dir / 'signals_log.json'
        logger.info(f"保存信号到 {signals_file}")
        with open(signals_file, 'w', encoding='utf8') as f:
            json.dump(enhanced, f, ensure_ascii=False, indent=2, default=str)
        
        # 4. 运行回测
        logger.info("运行回测...")
        trades_df, metrics = simple_backtest(
            df, enhanced,
            max_hold=BACKTEST_MAX_HOLD,
            atr_mult_stop=BACKTEST_ATR_STOP_MULT,
            atr_mult_target=BACKTEST_ATR_TARGET_MULT,
            min_llm_score=MIN_LLM_SCORE
        )
        
        # 5. 输出结果
        from utils.i18n import format_metric_value
        logger.info("=" * 60)
        logger.info("回测指标")
        logger.info("=" * 60)
        for k, v in metrics.items():
            logger.info(format_metric_value(k, v))
        
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
