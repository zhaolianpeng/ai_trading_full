# ai_agent/llm_client.py
import os
import json
try:
    import openai
except Exception:
    openai = None

def call_openai_chat(prompt, model='gpt-4o-mini', temperature=0.0, max_tokens=400):
    """
    简单封装 OpenAI ChatCompletion 调用。
    需要在环境变量中设置 OPENAI_API_KEY。
    """
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set in environment.')
    if openai is None:
        raise RuntimeError('openai package not installed.')
    openai.api_key = key
    # ChatCompletion（gpt-3.5/gpt-4 风格）
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    txt = resp['choices'][0]['message']['content']
    return txt

def ask_llm(prompt, provider='openai', model='gpt-4o-mini'):
    if provider == 'openai':
        return call_openai_chat(prompt, model=model)
    else:
        raise NotImplementedError('Only openai provider implemented in demo.')
