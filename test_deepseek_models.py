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
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        txt = response.choices[0].message.content
        if txt:
            print(f"  ✅ {model} 可用 - 响应: {txt[:50]}")
        else:
            print(f"  ⚠️  {model} 返回空响应")
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

