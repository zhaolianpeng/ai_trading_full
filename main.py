# main.py
from data.loader import gen_synthetic, load_csv
from strategy.strategy_runner import run_strategy
from backtest.simulator import simple_backtest
from config import DATA_PATH, USE_LLM
import json

def main():
    if DATA_PATH:
        df = load_csv(DATA_PATH)
    else:
        df = gen_synthetic(1500)
    df, enhanced = run_strategy(df, use_llm=USE_LLM)
    # save signals
    with open('signals_log.json','w',encoding='utf8') as f:
        json.dump(enhanced, f, ensure_ascii=False, indent=2)
    trades_df, metrics = simple_backtest(df, enhanced)
    print('--- Backtest Metrics ---')
    for k,v in metrics.items():
        print(f'{k}: {v}')
    trades_df.to_csv('trades.csv', index=False)
    df.to_csv('sample_data.csv')
    print('Saved sample_data.csv, trades.csv, signals_log.json')

if __name__ == '__main__':
    main()
