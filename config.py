# config.py — 全局配置
DATA_PATH = None       # 指定历史CSV路径（含datetime列）以使用真实数据，None = 使用合成数据
USE_LLM = True         # 是否启用 LLM 分析（True/False）
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-4o-mini"   # 根据你账户可用模型修改（例如 gpt-4o-mini/gpt-4o）
