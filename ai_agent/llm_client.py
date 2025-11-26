# ai_agent/llm_client.py
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. LLM features will be disabled.")

# 全局客户端实例（延迟初始化）
_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    """获取或创建 OpenAI 客户端实例"""
    global _client
    if _client is None:
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY not set in environment. '
                             'Please set it using: export OPENAI_API_KEY="sk-..."')
        if not OPENAI_AVAILABLE:
            raise RuntimeError('openai package not installed. '
                             'Please install it using: pip install openai')
        _client = OpenAI(api_key=key)
    return _client

def call_openai_chat(prompt: str, model: str = 'gpt-4o-mini', 
                     temperature: float = 0.0, max_tokens: int = 400,
                     retry_count: int = 3) -> str:
    """
    调用 OpenAI ChatCompletion API（使用新版本 SDK）
    
    Args:
        prompt: 提示词
        model: 模型名称
        temperature: 温度参数（0-2）
        max_tokens: 最大token数
    
    Returns:
        LLM 返回的文本内容
    
    Raises:
        RuntimeError: API key 未设置或包未安装
        Exception: API 调用失败
    """
    import time
    for attempt in range(retry_count):
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            txt = response.choices[0].message.content
            if txt is None:
                raise ValueError("Empty response from OpenAI API")
            return txt
        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt  # 指数退避
                logger.warning(f"OpenAI API call failed (attempt {attempt+1}/{retry_count}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"OpenAI API call failed after {retry_count} attempts: {e}")
                raise

def ask_llm(prompt: str, provider: str = 'openai', model: str = 'gpt-4o-mini',
            temperature: float = 0.0, max_tokens: int = 400) -> str:
    """
    统一的 LLM 调用接口
    
    Args:
        prompt: 提示词
        provider: 提供商（目前仅支持 'openai'）
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大token数
    
    Returns:
        LLM 返回的文本内容
    """
    if provider == 'openai':
        return call_openai_chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise NotImplementedError(f'Provider "{provider}" not implemented. Only "openai" is supported.')
