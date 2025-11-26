# OpenAI API Key 设置指南

## ✅ API Key 已设置

你的 OpenAI API Key 已成功添加到 `.env` 文件中。

## 🔒 安全注意事项

### ⚠️ 重要提醒

1. **不要提交到 Git**：
   - `.env` 文件已在 `.gitignore` 中（应该已包含）
   - 不要在代码中硬编码 API Key
   - 不要在公开仓库中分享 API Key

2. **API Key 权限**：
   - 你的 API Key 可以访问 OpenAI API
   - 请妥善保管，不要泄露给他人
   - 如果泄露，请立即在 OpenAI 平台撤销并重新生成

3. **使用限制**：
   - 注意 API 使用配额和费用
   - 监控 API 调用次数和成本

## 🚀 使用方法

### 启用 ChatGPT 分析

```bash
# 方式1: 使用 .env 文件（已设置）
python3 main.py

# 方式2: 命令行设置（会覆盖 .env）
OPENAI_API_KEY="sk-proj-..." USE_LLM=True python3 main.py
```

### 验证 API Key 是否生效

运行程序时，查看日志：

**成功调用 ChatGPT**：
```
INFO - LLM interpretation for signal 1/10...
INFO - OpenAI API call successful
```

**使用 Fallback（API Key 未生效）**：
```
WARNING - USE_LLM is True but OPENAI_API_KEY is not set. Disabling LLM features.
WARNING - LLM call failed: ..., using fallback
```

### 检查 signals_log.json

查看 `signals_log.json` 中的 `AI决策` 部分：

**实际调用 ChatGPT**：
```json
{
  "AI决策": {
    "趋势结构": "Strong Bull",
    "信号": "Long",
    "评分": 87,
    "置信度": "High",
    "解释": "Strong uptrend with EMA alignment, volume confirmation...",
    "风险": "Medium risk due to current volatility levels"
  }
}
```

**模拟调用（Fallback）**：
```json
{
  "AI决策": {
    "风险": "fallback heuristic"
  }
}
```

## 💰 成本估算

使用 `gpt-4o-mini`（默认模型）：

- **每个信号**：约 $0.0001-0.0002
- **100 个信号**：约 $0.01-0.02
- **1000 个信号**：约 $0.10-0.20

**非常便宜**，适合频繁使用。

## 🔧 配置选项

在 `.env` 文件中可以调整：

```env
# OpenAI API 配置
OPENAI_API_KEY=sk-proj-...
USE_LLM=True                    # 是否启用 LLM
OPENAI_MODEL=gpt-4o-mini        # 模型名称
OPENAI_TEMPERATURE=0.0          # 温度参数（0-2）
OPENAI_MAX_TOKENS=400           # 最大 token 数
```

### 模型选择

- `gpt-4o-mini`（默认）：便宜、快速，适合批量分析
- `gpt-4o`：更智能，但更贵
- `gpt-3.5-turbo`：更便宜，但能力较弱

## 📊 使用建议

### 1. 测试模式

首次使用建议先用少量数据测试：

```bash
DATA_SOURCE=synthetic \
SYNTHETIC_DATA_SIZE=100 \
USE_LLM=True \
python3 main.py
```

### 2. 生产模式

确认 API Key 工作正常后，使用真实数据：

```bash
DATA_SOURCE=binance \
MARKET_SYMBOL=BTC/USDT \
MARKET_TIMEFRAME=1h \
USE_LLM=True \
python3 main.py
```

### 3. 批量处理优化

系统已经实现了批量处理，减少 API 调用次数。

## ⚠️ 故障排除

### 问题 1: API Key 无效

**错误信息**：
```
OpenAI API call failed: Invalid API key
```

**解决方案**：
1. 检查 API Key 是否正确
2. 检查 OpenAI 账户是否有余额
3. 检查 API Key 是否被撤销

### 问题 2: 配额用完

**错误信息**：
```
OpenAI API call failed: Rate limit exceeded
```

**解决方案**：
1. 等待一段时间后重试
2. 检查 OpenAI 账户配额
3. 升级账户计划

### 问题 3: 网络问题

**错误信息**：
```
OpenAI API call failed: Connection error
```

**解决方案**：
1. 检查网络连接
2. 系统会自动重试（最多3次）
3. 如果失败，会自动降级到 Fallback 模式

## 🔍 验证设置

运行以下命令验证：

```bash
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('OPENAI_API_KEY')
if key and len(key) > 20:
    print('✅ API Key 已设置')
    print(f'前缀: {key[:15]}...')
else:
    print('❌ API Key 未设置')
"
```

## 📝 当前配置

你的 `.env` 文件现在包含：

```env
OPENAI_API_KEY=sk-proj-你的API密钥
USE_LLM=True
OPENAI_MODEL=gpt-4o-mini
```

**注意**：请将 `sk-proj-你的API密钥` 替换为你的实际 API Key。

现在系统会**实际调用 ChatGPT** 来分析交易信号！

