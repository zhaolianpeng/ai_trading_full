# ai_agent/llm_prompt.py
# 多个 Prompt 模板，直接用于 ask_llm(prompt)

MARKET_STRUCTURE_PROMPT = '''你是一名专业的数字货币量化交易分析师。
我将给出一组结构化特征（JSON），请完成：
1) 判断总体市场结构（Strong Bull / Bull / Neutral / Bear / Strong Bear）
2) 给出最终建议（Long / Short / Neutral）
3) 给出 0-100 的评分 score，以及置信度（Low/Medium/High）
4) 用 2-3 句说明理由，必须引用输入特征中的字段
5) 给出 2 条可操作的风险提示（短句）
只输出合法的 JSON，例如：
{"trend_structure":"Strong Bull","signal":"Long","score":87,"confidence":"High","explanation":"...","risk":"..."}'''

SIMPLE_SIGNAL_PROMPT = '''你是一名量化策略工程师。
给定以下信号包（JSON），请整合它们并返回最终决策（Long/Short/Neutral）、置信度(0-100)、理由（1-3条）以及建议的 entry/stop/targets（基于传入 atr 值）。
输出 JSON。'''

MULTI_TIMEFRAME_PROMPT = '''你是多周期交易分析师。
输入：短中长期（例如 1h,4h,1d）的关键特征数组（每个周期都有ema_alignment, trend, rsi, vol_ratio）。
输出：是否多周期一致（Yes/No）、如果不一致说明冲突点并建议优先级。输出 JSON。'''

VOLUME_ANALYSIS_PROMPT = '''你是量能分析师。输入：vol_ratio, large_trades_fraction, vwap_gap。输出：是否量能支持当前方向（Yes/No），并给出证据与警告。JSON 输出。'''

BREAKOUT_VALIDATION_PROMPT = '''你是突破验证器。输入：breakout_flag, vol_ratio, close_above_level_bars（持续几根收在阻力上方），higher_timeframe_trend。输出：是否为有效突破（True/False）、置信度、说明、建议。JSON。'''

FAKE_BREAKOUT_DETECT_PROMPT = '''请判断是否是假突破。输入：volume_trend, wick_size, market_liquidity, orderbook_imbalance。输出 True/False + 理由 + 推荐操作（例如等待回踩、设置小仓位）。JSON。'''

RISK_MANAGEMENT_PROMPT = '''基于当前信号与账户参数（position_size, max_drawdown, leverage），给出仓位和止损建议（百分比或基于ATR）。输出 JSON。'''

NEWS_SENTIMENT_PROMPT = '''（可选）输入一段市场新闻或社媒文本，输出对币价短期可能影响（Bullish/Neutral/Bearish），并给出原因与置信度。'''

FEATURE_IMPORTANCE_PROMPT = '''你是模型解释师。输入：signal_features（字典），模型预测（0-1）。输出：每个特征对预测的贡献排序，并给出可视化建议。'''

TRADE_SUMMARY_PROMPT = '''输入：一笔交易的开平仓记录（entry, exit, timestamps, PnL），输出一段用于日报的自然语言总结（中文），包含绩效、原因、教训与改进点。'''

AUTOTUNE_PROMPT = '''你是超参优化器。输入：策略名称、当前参数、历史回测结果，输出下一组建议参数（包括搜索空间）、评价指标目标与理由。JSON。'''

ALERT_PROMPT = '''输入：突发市场事件（价格暴跌/暴涨/大型成交），返回紧急动作建议（例如暂停交易、减仓、人工复核），按优先级排序。JSON。'''
