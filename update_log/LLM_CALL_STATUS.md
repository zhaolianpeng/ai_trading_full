# LLM 调用状态说明

## 🔍 当前状态

你的项目**支持实际调用 ChatGPT**，但根据配置可能处于不同状态：

### 1. 实际调用 ChatGPT（真实 API）

**条件**：
- ✅ `USE_LLM=True`（默认）
- ✅ `OPENAI_API_KEY` 已设置（格式：`sk-...`）
- ✅ `openai` 包已安装

**代码位置**：
- `ai_agent/llm_client.py` - 实际调用 OpenAI API
- `ai_agent/signal_interpret.py` - 调用 LLM 分析信号

**调用流程**：
```python
# 1. 获取 OpenAI 客户端
client = OpenAI(api_key=OPENAI_API_KEY)

# 2. 实际调用 API
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    max_tokens=400
)

# 3. 返回真实的分析结果
result = response.choices[0].message.content
```

### 2. 模拟调用（Fallback 启发式方法）

**触发条件**（满足任一）：
- ❌ `USE_LLM=False`
- ❌ `OPENAI_API_KEY` 未设置
- ❌ `openai` 包未安装
- ❌ API 调用失败（网络问题、API 错误等）

**代码位置**：
- `ai_agent/signal_interpret.py` 第 15-30 行

**Fallback 逻辑**：
```python
# 简单的启发式评分
if ema_alignment: score += 30
if higher_highs: score += 25
if volume_spike: score += 20
if breakout: score += 15
signal = 'Long' if score >= 50 else 'Neutral'
```

## 📊 如何判断当前状态

### 方法 1: 检查日志

运行程序时查看日志：

**实际调用 ChatGPT**：
```
INFO - LLM interpretation for signal 1/41...
INFO - OpenAI API call successful
```

**模拟调用（Fallback）**：
```
WARNING - USE_LLM is True but OPENAI_API_KEY is not set. Disabling LLM features.
WARNING - LLM call failed: ..., using fallback
```

### 方法 2: 检查环境变量

```bash
# 检查 API Key 是否设置
echo $OPENAI_API_KEY

# 或在 Python 中检查
python3 -c "import os; print('API Key:', '已设置' if os.getenv('OPENAI_API_KEY') else '未设置')"
```

### 方法 3: 检查 signals_log.json

查看 `signals_log.json` 中的 `AI决策` 部分：

**实际调用 ChatGPT**：
```json
{
  "AI决策": {
    "趋势结构": "Strong Bull",
    "信号": "Long",
    "评分": 87,
    "置信度": "High",
    "解释": "Strong uptrend with EMA alignment, volume confirmation, and positive momentum...",
    "风险": "Medium risk due to current volatility levels"
  }
}
```

**模拟调用（Fallback）**：
```json
{
  "AI决策": {
    "趋势结构": "Bull",
    "信号": "Long",
    "评分": 50,
    "置信度": "Medium",
    "解释": "EMA alignment, Higher highs, Volume spike",
    "风险": "fallback heuristic"
  }
}
```

**关键标识**：如果 `risk` 字段是 `"fallback heuristic"`，说明是模拟调用。

## 🚀 启用实际 ChatGPT 调用

### 步骤 1: 获取 OpenAI API Key

1. 访问 https://platform.openai.com/
2. 注册/登录账号
3. 进入 API Keys 页面
4. 创建新的 API Key（格式：`sk-...`）

### 步骤 2: 设置环境变量

**方式 1: 命令行设置**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
python3 main.py
```

**方式 2: .env 文件**
创建 `.env` 文件：
```env
OPENAI_API_KEY=sk-your-api-key-here
USE_LLM=True
OPENAI_MODEL=gpt-4o-mini
```

**方式 3: 运行时设置**
```bash
OPENAI_API_KEY="sk-your-api-key-here" python3 main.py
```

### 步骤 3: 确保包已安装

```bash
pip install openai
```

### 步骤 4: 验证

运行程序，查看日志：
```bash
python3 main.py 2>&1 | grep -E "(LLM|OpenAI|API)"
```

应该看到：
```
INFO - LLM interpretation for signal 1/41...
INFO - OpenAI API call successful
```

## 💰 成本说明

### 实际调用 ChatGPT 的成本

- **模型**: `gpt-4o-mini`（默认）
- **输入**: 每个信号约 200-300 tokens
- **输出**: 约 100-200 tokens
- **总成本**: 每个信号约 $0.0001-0.0002（非常便宜）

**示例**：
- 100 个信号 ≈ $0.01-0.02
- 1000 个信号 ≈ $0.10-0.20

### 节省成本的方法

1. **使用 Fallback 模式**：
   ```bash
   USE_LLM=False python3 main.py
   ```

2. **批量处理**：系统已经实现了批量处理，减少 API 调用

3. **使用更便宜的模型**：
   ```bash
   OPENAI_MODEL=gpt-3.5-turbo python3 main.py
   ```

## 🔧 故障排除

### 问题 1: API Key 未设置

**错误信息**：
```
RuntimeError: OPENAI_API_KEY not set in environment
```

**解决方案**：
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 问题 2: API 调用失败

**错误信息**：
```
OpenAI API call failed: ...
WARNING - LLM call failed: ..., using fallback
```

**可能原因**：
- 网络连接问题
- API Key 无效
- API 配额用完
- 模型不可用

**解决方案**：
- 检查网络连接
- 验证 API Key
- 检查 OpenAI 账户余额
- 系统会自动降级到 Fallback 模式

### 问题 3: 包未安装

**错误信息**：
```
RuntimeError: openai package not installed
```

**解决方案**：
```bash
pip install openai
```

## 📝 总结

| 状态 | 条件 | 结果 |
|------|------|------|
| **实际调用** | `USE_LLM=True` + `OPENAI_API_KEY` 已设置 | 真实 ChatGPT 分析 |
| **模拟调用** | `USE_LLM=False` 或 `OPENAI_API_KEY` 未设置 | Fallback 启发式方法 |

**当前你的状态**：根据检查，`OPENAI_API_KEY` 未设置，所以当前是**模拟调用（Fallback）**模式。

**要启用实际调用**：设置 `OPENAI_API_KEY` 环境变量即可。

