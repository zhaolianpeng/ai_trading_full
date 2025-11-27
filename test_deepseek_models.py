#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 DeepSeek API 可用模型
用于诊断 deepseek-chat 和 deepseek-reasoner 是否可用
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("❌ 请先安装 openai 包: pip install openai")
    exit(1)

# 获取 API Key
api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    print("❌ DEEPSEEK_API_KEY 未设置")
    print("   请在 .env 文件中设置: DEEPSEEK_API_KEY=sk-...")
    exit(1)

print(f"✅ API Key 已设置: {api_key[:15]}...")
print()

# 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

# 测试的模型列表
models_to_test = [
    'deepseek-chat',
    'deepseek-reasoner',
    'deepseek-chat-v2',
    'deepseek-chat-v2.5',
]

print("🔍 测试 DeepSeek 模型可用性...")
print("=" * 60)

for model in models_to_test:
    print(f"\n测试模型: {model}")
    try:
        # 对于推理模型，使用更大的 max_tokens
        # 对于普通模型，也使用合理的值（50）以避免截断
        if 'reasoner' in model:
            test_max_tokens = 800
        else:
            test_max_tokens = 50  # 足够返回完整问候语
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=test_max_tokens,
            stream=False
        )
        
        # 检查响应结构
        if not response or not response.choices:
            print(f"  ❌ {model} 响应结构无效")
            continue
            
        txt = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
        
        if txt:
            status_icon = "✅"
            if finish_reason == 'length':
                status_icon = "⚠️"  # 虽然可用，但被截断了
                print(f"  {status_icon} {model} 可用（但响应被截断）- 响应: {txt[:50]}...")
            else:
                print(f"  {status_icon} {model} 可用 - 响应: {txt[:50]}")
            
            if finish_reason:
                reason_text = {
                    'stop': '正常完成',
                    'length': '达到最大token限制（被截断）',
                    'content_filter': '内容被过滤',
                    'function_call': '函数调用',
                    'tool_calls': '工具调用'
                }.get(finish_reason, finish_reason)
                print(f"     完成原因: {finish_reason} ({reason_text})")
            
            # 显示使用情况
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"     Token使用: 输入={usage.prompt_tokens}, 输出={usage.completion_tokens}, 总计={usage.total_tokens}")
        else:
            print(f"  ⚠️  {model} 返回空响应")
            if finish_reason:
                print(f"     完成原因: {finish_reason}")
            # 尝试获取更多信息
            if hasattr(response, 'usage'):
                print(f"     使用情况: {response.usage}")
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        # 检查错误类型
        if '404' in error_msg or 'not found' in error_msg.lower():
            print(f"  ❌ {model} 不存在 (404)")
        elif '401' in error_msg or 'unauthorized' in error_msg.lower():
            print(f"  ❌ {model} 认证失败 (401) - 请检查API Key")
        elif '429' in error_msg or 'rate limit' in error_msg.lower():
            print(f"  ⚠️  {model} 限流 (429)")
        elif 'quota' in error_msg.lower() or 'insufficient balance' in error_msg.lower():
            print(f"  ❌ {model} 余额不足")
        else:
            print(f"  ❌ {model} 失败: {error_type}: {error_msg[:100]}")

print("\n" + "=" * 60)
print("\n💡 建议：")
print("   1. 如果所有模型都返回 404，可能是模型名称错误")
print("   2. 如果返回 401，请检查 API Key 是否正确")
print("   3. 如果返回 429，可能是限流，稍后重试")
print("   4. 如果返回余额不足，请充值")
print("   5. 查看 DeepSeek 官方文档确认当前可用模型: https://platform.deepseek.com/")

