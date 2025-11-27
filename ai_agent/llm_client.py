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
_deepseek_client: Optional[OpenAI] = None

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

def get_deepseek_client() -> OpenAI:
    """获取或创建 DeepSeek 客户端实例（使用 OpenAI 兼容接口）"""
    global _deepseek_client
    if _deepseek_client is None:
        key = os.environ.get('DEEPSEEK_API_KEY')
        if not key:
            raise RuntimeError('DEEPSEEK_API_KEY not set in environment. '
                             'Please set it using: export DEEPSEEK_API_KEY="sk-..."')
        if not OPENAI_AVAILABLE:
            raise RuntimeError('openai package not installed. '
                             'Please install it using: pip install openai')
        # DeepSeek 使用 OpenAI 兼容的 API，只需要改变 base_url
        _deepseek_client = OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com/v1"
        )
    return _deepseek_client

def _rate_limit_wait(provider: str):
    """简单的速率限制：确保请求间隔"""
    global _last_request_time
    current_time = time.time()
    if provider in _last_request_time:
        elapsed = current_time - _last_request_time[provider]
        if elapsed < _min_request_interval:
            time.sleep(_min_request_interval - elapsed)
    _last_request_time[provider] = time.time()

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
    for attempt in range(retry_count):
        try:
            # 速率限制
            _rate_limit_wait('openai')
            
            client = get_openai_client()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            txt = response.choices[0].message.content
            if txt is None or txt.strip() == '':
                raise ValueError(f"Empty response from OpenAI API (model={model})")
            return txt.strip()
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # 检查是否是限流错误（429）
            is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower() or 'quota' in error_msg.lower()
            
            if attempt < retry_count - 1:
                # 限流错误使用更长的等待时间
                if is_rate_limit:
                    wait_time = 10 + (2 ** attempt)  # 限流时等待更长时间
                    logger.warning(f"OpenAI API限流 (attempt {attempt+1}/{retry_count}): {error_msg}. 等待 {wait_time}s 后重试...")
                else:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"OpenAI API调用失败 (attempt {attempt+1}/{retry_count}): {error_type}: {error_msg}. 等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"OpenAI API调用失败，已重试 {retry_count} 次: {error_type}: {error_msg}")
                raise

def call_deepseek_chat(prompt: str, model: str = 'deepseek-reasoner', 
                      temperature: float = 0.0, max_tokens: int = 400,
                      retry_count: int = 3) -> str:
    """
    调用 DeepSeek ChatCompletion API（使用 OpenAI 兼容接口）
    
    默认使用 deepseek-reasoner 推理模型，适合复杂分析场景。
    
    Args:
        prompt: 提示词
        model: 模型名称（默认 'deepseek-reasoner'，推理模型；也可使用 'deepseek-chat'）
        temperature: 温度参数（0-2）
        max_tokens: 最大token数
        retry_count: 重试次数（默认3次）
    
    Returns:
        LLM 返回的文本内容
    
    Raises:
        RuntimeError: API key 未设置或包未安装
        Exception: API 调用失败
    """
    for attempt in range(retry_count):
        try:
            # 速率限制
            _rate_limit_wait('deepseek')
            
            client = get_deepseek_client()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            txt = response.choices[0].message.content
            if txt is None or txt.strip() == '':
                raise ValueError(f"Empty response from DeepSeek API (model={model})")
            return txt.strip()
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # 检查是否是限流错误（429）
            is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower() or 'quota' in error_msg.lower()
            
            if attempt < retry_count - 1:
                # 限流错误使用更长的等待时间
                if is_rate_limit:
                    wait_time = 10 + (2 ** attempt)  # 限流时等待更长时间
                    logger.warning(f"DeepSeek API限流 (attempt {attempt+1}/{retry_count}): {error_msg}. 等待 {wait_time}s 后重试...")
                else:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"DeepSeek API调用失败 (attempt {attempt+1}/{retry_count}): {error_type}: {error_msg}. 等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"DeepSeek API调用失败，已重试 {retry_count} 次: {error_type}: {error_msg}")
                raise

def ask_llm(prompt: str, provider: str = 'openai', model: str = 'gpt-4o-mini',
            temperature: float = 0.0, max_tokens: int = 400) -> str:
    """
    统一的 LLM 调用接口
    
    Args:
        prompt: 提示词
        provider: 提供商（支持 'openai' 或 'deepseek'）
        model: 模型名称
            - OpenAI: 'gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo' 等
            - DeepSeek: 'deepseek-reasoner'（默认，推理模型）、'deepseek-chat' 等
        temperature: 温度参数（0-2，默认0.0）
        max_tokens: 最大token数（默认400）
    
    Returns:
        LLM 返回的文本内容
    
    Note:
        DeepSeek 默认使用 'deepseek-reasoner' 推理模型，适合复杂分析场景。
        可通过 config.py 中的 DEEPSEEK_MODEL 配置项修改。
    """
    if provider == 'openai':
        return call_openai_chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == 'deepseek':
        return call_deepseek_chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise NotImplementedError(f'Provider "{provider}" not implemented. Supported providers: "openai", "deepseek".')
