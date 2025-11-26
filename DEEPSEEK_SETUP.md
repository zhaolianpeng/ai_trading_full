# DeepSeek API 配置指南

## 🚀 快速开始

DeepSeek 是一个高性能的 AI 模型提供商，提供与 OpenAI 兼容的 API 接口。现在系统已支持使用 DeepSeek 作为 LLM 提供商。

## 📋 配置步骤

### 1. 获取 DeepSeek API Key

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册/登录账户
3. 在"API 密钥管理"页面创建新的 API Key
4. 复制 API Key（格式类似：`sk-...`）

### 2. 配置环境变量

在 `.env` 文件中添加：

```env
# DeepSeek API 配置
DEEPSEEK_API_KEY=sk-你的API密钥
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-chat
```

### 3. 可用的 DeepSeek 模型

- `deepseek-chat`：通用对话模型（推荐，默认）
- `deepseek-reasoner`：推理模型，适合复杂分析

### 4. 运行系统

```bash
# 使用 DeepSeek
LLM_PROVIDER=deepseek \
DEEPSEEK_API_KEY=sk-... \
DEEPSEEK_MODEL=deepseek-chat \
USE_LLM=True \
python3 main.py
```

或者直接在 `.env` 文件中配置后运行：

```bash
python3 main.py
```

## 🔄 切换提供商

### 使用 OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

### 使用 DeepSeek

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
```

## 💰 成本对比

### DeepSeek 优势

- **更便宜**：DeepSeek 的价格通常比 OpenAI 更优惠
- **高性能**：提供与 GPT-4 相当的性能
- **兼容性**：使用 OpenAI 兼容的 API 接口

### 价格参考（以实际 DeepSeek 官网为准）

- `deepseek-chat`：通常比 `gpt-4o-mini` 更便宜
- `deepseek-reasoner`：适合需要复杂推理的场景

## 📊 完整配置示例

### .env 文件配置

```env
# 数据源配置
DATA_SOURCE=binance
MARKET_SYMBOL=BTC/USDT
MARKET_TIMEFRAME=1h
MARKET_TYPE=future

# LLM 配置 - 使用 DeepSeek
USE_LLM=True
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-你的API密钥
DEEPSEEK_MODEL=deepseek-chat
OPENAI_TEMPERATURE=0.0
OPENAI_MAX_TOKENS=400

# 交易模式
TRADING_MODE=scalping
SIGNAL_LOOKBACK_DAYS=7
```

## 🔍 验证配置

运行以下命令验证 DeepSeek API Key：

```bash
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
from ai_agent.llm_client import get_deepseek_client

try:
    client = get_deepseek_client()
    print('✅ DeepSeek API Key 配置成功')
    print('✅ 可以开始使用 DeepSeek 进行 AI 分析')
except Exception as e:
    print(f'❌ 配置失败: {e}')
"
```

## 🧪 测试 DeepSeek API 调用

```bash
python3 << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()
from ai_agent.signal_interpret import interpret_with_llm

test_packet = {
    "trend": "up",
    "ema_alignment": True,
    "higher_highs": True,
    "volume_spike": True,
    "breakout": True,
    "close": 50000.0
}

print("测试 DeepSeek API 调用...")
try:
    result = interpret_with_llm(
        test_packet, 
        provider='deepseek',
        model='deepseek-chat',
        use_llm=True
    )
    print("✅ DeepSeek API 调用成功！")
    print(f"   信号: {result.get('signal', 'N/A')}")
    print(f"   评分: {result.get('score', 'N/A')}")
    print(f"   解释: {result.get('explanation', 'N/A')[:50]}...")
except Exception as e:
    print(f"❌ API 调用失败: {e}")
EOF
```

## ⚙️ 高级配置

### 调整模型参数

```env
# 温度参数（0-2，控制随机性）
OPENAI_TEMPERATURE=0.0

# 最大 token 数
OPENAI_MAX_TOKENS=400
```

### 使用推理模型

对于需要复杂分析的场景，可以使用 `deepseek-reasoner`：

```env
DEEPSEEK_MODEL=deepseek-reasoner
```

## 🔧 故障排除

### 问题 1: API Key 无效

**错误信息**：
```
RuntimeError: DEEPSEEK_API_KEY not set in environment
```

**解决方案**：
1. 检查 `.env` 文件中是否设置了 `DEEPSEEK_API_KEY`
2. 确认 API Key 格式正确（以 `sk-` 开头）
3. 检查 API Key 是否在 DeepSeek 平台有效

### 问题 2: 配额不足

**错误信息**：
```
Error code: 429 - insufficient_quota
```

**解决方案**：
1. 检查 DeepSeek 账户余额
2. 在 DeepSeek 平台充值
3. 检查 API 使用配额限制

### 问题 3: 模型不存在

**错误信息**：
```
Model not found: deepseek-xxx
```

**解决方案**：
1. 确认模型名称正确（`deepseek-chat` 或 `deepseek-reasoner`）
2. 检查账户是否有权限使用该模型
3. 查看 DeepSeek 文档确认可用模型列表

## 📝 注意事项

1. **API Key 安全**：
   - 不要将 API Key 提交到 Git
   - `.env` 文件已在 `.gitignore` 中
   - 定期轮换 API Key

2. **成本控制**：
   - 监控 API 调用次数
   - 设置合理的 `max_tokens` 限制
   - 使用 `USE_LLM=False` 进行测试时不会产生费用

3. **性能优化**：
   - `deepseek-chat` 适合大多数场景
   - `deepseek-reasoner` 适合需要复杂推理的场景，但可能更慢更贵

## 🎯 使用建议

1. **开发/测试阶段**：
   - 使用 `USE_LLM=False` 进行快速测试
   - 或使用 DeepSeek（更便宜）进行测试

2. **生产环境**：
   - 根据需求选择 OpenAI 或 DeepSeek
   - 监控 API 调用成本和性能

3. **成本优化**：
   - DeepSeek 通常更便宜，适合高频调用
   - OpenAI 在某些场景下可能更稳定

## 🔗 相关文档

- [DeepSeek 开放平台](https://platform.deepseek.com/)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [OpenAI API 文档](https://platform.openai.com/docs/)（兼容接口参考）

