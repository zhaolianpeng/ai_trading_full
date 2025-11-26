AI Trading System — LLM + Quant (Demo)
======================================

功能：
- 计算技术指标（EMA、RSI、ATR）
- 基于规则检测信号（long_structure、breakout、RSI 背离）
- 可选调用 LLM（OpenAI 兼容）解释信号并生成最终决策
- 运行简单回测并输出指标
- 包含 10+ 条专业 Prompt 模板，便于扩展

运行说明：
1. 将代码保存到文件夹 ai_trading_full/
2. 创建并激活虚拟环境：
   python -m venv venv
   source venv/bin/activate  # mac/linux
   venv\\Scripts\\activate   # windows
3. 安装依赖：
   pip install -r requirements.txt
4. （可选）设置 OpenAI API Key：
   export OPENAI_API_KEY="sk-..."  # mac/linux
   set OPENAI_API_KEY="sk-..."     # windows
5. 运行 demo（合成数据）：
   python main.py

输出：
- sample_data.csv
- signals_log.json
- trades.csv

替换真实数据：
- 把 config.py 中 DATA_PATH 指向包含 `datetime,open,high,low,close,volume` 的 CSV 文件，或在 data/loader.py 中添加 exchange API 拉取函数。
