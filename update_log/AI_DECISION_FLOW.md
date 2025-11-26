# AI 决策流程说明

## 📍 AI 决策在项目中的使用位置

### 1. 信号增强阶段（`strategy/strategy_runner.py`）

**位置**：`strategy/strategy_runner.py` 第 64-77 行

**功能**：对每个检测到的交易信号，调用 AI 进行分析和评分

```python
# 对每个信号构建特征包
packet = build_feature_packet(df, idx)

# 调用 AI 进行信号解释
llm_out = interpret_with_llm(
    packet, 
    provider=LLM_PROVIDER, 
    model=OPENAI_MODEL, 
    use_llm=use_llm,
    temperature=OPENAI_TEMPERATURE,
    max_tokens=OPENAI_MAX_TOKENS
)

# 将 AI 决策附加到信号中
enhanced_signals.append({
    'rule': s,                    # 原始规则信号
    'feature_packet': packet,     # 特征数据
    'llm': llm_out               # AI 决策结果 ⭐
})
```

**AI 输入**：特征包（feature_packet），包含：
- 趋势方向（trend）
- EMA 排列（ema_alignment）
- 更高高点/更高低点（higher_highs/higher_lows）
- 量能爆发（volume_spike）
- 突破（breakout）
- RSI 背离（rsi_divergence）
- ATR 波动率
- 成交量比率
- 当前价格

**AI 输出**：JSON 格式的决策结果，包含：
- `signal`: 交易信号（'Long', 'Short', 'Neutral', 'Hold'）
- `score`: 评分（0-100）
- `confidence`: 置信度（'High', 'Medium', 'Low'）
- `trend_structure`: 趋势结构
- `explanation`: 解释说明
- `risk`: 风险评估

### 2. AI 决策核心函数（`ai_agent/signal_interpret.py`）

**位置**：`ai_agent/signal_interpret.py` 第 9-49 行

**功能**：调用 LLM API，将技术指标特征转换为交易决策

```python
def interpret_with_llm(feature_packet, provider='openai', model='gpt-4o-mini', 
                       use_llm=True, temperature=0.0, max_tokens=400):
    """
    把结构化特征传给 LLM，解析返回的 JSON。
    若无法调用 LLM 或解析失败，返回简单启发式聚合。
    """
    if not use_llm:
        # 降级：使用启发式规则
        return fallback_heuristic(feature_packet)
    
    # 构建 Prompt
    prompt = MARKET_STRUCTURE_PROMPT + "\n\n特征数据：\n" + json.dumps(feature_packet, ensure_ascii=False)
    
    # 调用 LLM API ⭐
    txt = ask_llm(prompt, provider=provider, model=model)
    
    # 解析 LLM 返回的 JSON
    parsed = json.loads(txt.strip())
    return parsed
```

**Prompt 模板**：`ai_agent/llm_prompt.py`
- 包含市场结构分析的 Prompt
- 指导 LLM 如何分析技术指标
- 要求返回结构化的 JSON 决策

### 3. 回测执行阶段（`backtest/simulator.py`）

**位置**：`backtest/simulator.py` 第 24-40 行

**功能**：使用 AI 决策来过滤和执行交易

```python
for item in enhanced_signals:
    s = item['rule']           # 原始规则信号
    llm = item.get('llm', {})  # AI 决策结果 ⭐
    
    # 从 AI 决策中提取信号和评分
    signal = llm.get('signal', 'Neutral') if isinstance(llm, dict) else 'Neutral'
    raw_score = llm.get('score', 0)
    score = int(raw_score)
    
    # ⭐ AI 决策过滤：只执行 AI 推荐的 Long 信号，且评分 >= 40
    if signal != 'Long' or score < min_llm_score:
        continue  # 跳过不符合 AI 决策的信号
    
    # 执行交易...
```

**AI 决策的作用**：
1. **信号过滤**：只执行 AI 推荐为 'Long' 的信号
2. **评分阈值**：只执行评分 >= `MIN_LLM_SCORE`（默认40）的信号
3. **决策依据**：AI 的 `signal` 和 `score` 是回测执行的关键判断条件

### 4. LLM 客户端（`ai_agent/llm_client.py`）

**位置**：`ai_agent/llm_client.py` 第 9-35 行

**功能**：实际调用 OpenAI API

```python
def call_openai_chat(prompt, model='gpt-4o-mini', temperature=0.0, max_tokens=400):
    """
    调用 OpenAI ChatCompletion API
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    txt = response.choices[0].message.content
    return txt
```

## 🔄 完整决策流程

```
1. 检测规则信号
   └─> signal_rules.py: detect_rules()
       └─> 输出：原始信号列表

2. 构建特征包
   └─> strategy_runner.py: build_feature_packet()
       └─> 输出：特征字典（包含 EMA、RSI、成交量等）

3. AI 分析决策 ⭐
   └─> signal_interpret.py: interpret_with_llm()
       └─> llm_client.py: call_openai_chat()
           └─> 输出：AI 决策（signal, score, confidence, explanation）

4. 信号增强
   └─> strategy_runner.py: run_strategy()
       └─> 输出：增强信号列表（包含 AI 决策）

5. 回测执行（使用 AI 决策过滤）⭐
   └─> simulator.py: simple_backtest()
       └─> 根据 AI 的 signal 和 score 决定是否执行交易
           └─> 输出：交易记录和回测指标
```

## 🎯 AI 决策的关键作用

### 1. 信号质量评估
- AI 对每个技术指标信号进行评分（0-100）
- 评估信号的可靠性和置信度

### 2. 交易方向判断
- AI 决定是 'Long'（做多）、'Short'（做空）还是 'Neutral'（中性）
- 当前回测系统只执行 'Long' 信号

### 3. 风险分析
- AI 提供风险评估和解释
- 帮助理解为什么做出这个决策

### 4. 信号过滤
- 在回测中，只有 AI 推荐且评分 >= 40 的信号才会被执行
- 这大大减少了假信号的影响

## 📊 AI 决策示例

**输入特征包**：
```json
{
  "trend": "up",
  "ema_alignment": true,
  "higher_highs": true,
  "volume_spike": true,
  "breakout": false,
  "rsi_divergence": null,
  "atr": 50.5,
  "vol_ratio": 2.1,
  "close": 30000
}
```

**AI 输出决策**：
```json
{
  "signal": "Long",
  "score": 75,
  "confidence": "High",
  "trend_structure": "Bull",
  "explanation": "Strong uptrend with EMA alignment, volume confirmation",
  "risk": "Medium"
}
```

**回测执行**：
- ✅ 信号 = 'Long'，评分 = 75 >= 40
- ✅ 执行交易

## ⚙️ 配置控制

### 启用/禁用 AI 决策

```bash
# 启用 AI 决策（默认）
USE_LLM=True python3 main.py

# 禁用 AI 决策（使用启发式规则）
USE_LLM=False python3 main.py
```

### 调整 AI 决策阈值

```bash
# 只执行评分 >= 50 的信号（默认40）
MIN_LLM_SCORE=50 python3 main.py
```

### AI 模型配置

```bash
# 使用不同的模型
OPENAI_MODEL=gpt-4o python3 main.py

# 调整温度参数（影响随机性）
OPENAI_TEMPERATURE=0.2 python3 main.py
```

## 🔍 查看 AI 决策结果

AI 决策结果保存在 `signals_log.json` 文件中：

```json
[
  {
    "rule": {
      "type": "long_structure",
      "score": 4,
      "confidence": "high",
      "idx": 123
    },
    "feature_packet": {
      "trend": "up",
      "ema_alignment": true,
      ...
    },
    "llm": {                    // ⭐ AI 决策结果
      "signal": "Long",
      "score": 75,
      "confidence": "High",
      "explanation": "...",
      "risk": "Medium"
    }
  }
]
```

## 💡 总结

**AI 决策在项目中的核心位置**：

1. **`strategy/strategy_runner.py`** - 调用 AI 分析每个信号
2. **`ai_agent/signal_interpret.py`** - AI 决策的核心函数
3. **`ai_agent/llm_client.py`** - 实际调用 OpenAI API
4. **`backtest/simulator.py`** - 使用 AI 决策过滤和执行交易

**AI 决策的作用**：
- ✅ 评估信号质量
- ✅ 决定交易方向
- ✅ 提供风险分析
- ✅ 过滤低质量信号

**关键配置**：
- `USE_LLM`: 是否启用 AI 决策
- `MIN_LLM_SCORE`: AI 评分最低阈值（默认40）
- `OPENAI_MODEL`: 使用的 AI 模型
- `OPENAI_TEMPERATURE`: AI 温度参数

