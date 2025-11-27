# ai_agent/llm_client.py
import os
import json
import time
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

# 全局请求计数器（用于限流控制）
_last_request_time: dict = {}
# 最小请求间隔（秒），避免过快请求
# DeepSeek 没有限速，可以使用更小的间隔；OpenAI 有限速，使用较大的间隔
_min_request_interval = 0.05  # 降低到 0.05 秒，提高并发效率

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
            
            # 检查是否是空响应错误
            is_empty_response = 'empty response' in error_msg.lower() or 'Empty response' in error_msg
            
            # 检查是否是限流错误（429）
            is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower() or 'quota' in error_msg.lower() or 'insufficient balance' in error_msg.lower()
            
            # 空响应或余额不足时，不重试，直接抛出异常（让上层使用fallback）
            if is_empty_response or is_rate_limit:
                if attempt == 0:  # 第一次尝试就失败
                    logger.warning(f"OpenAI API返回空响应或余额不足 (model={model}): {error_msg}，跳过重试，使用fallback")
                else:
                    logger.warning(f"OpenAI API返回空响应或余额不足 (attempt {attempt+1}/{retry_count}): {error_msg}，跳过重试，使用fallback")
                raise  # 直接抛出，让上层使用fallback
            
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt  # 指数退避
                logger.warning(f"OpenAI API调用失败 (attempt {attempt+1}/{retry_count}): {error_type}: {error_msg}. 等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"OpenAI API调用失败，已重试 {retry_count} 次: {error_type}: {error_msg}")
                raise

def call_deepseek_chat(prompt: str, model: str = 'deepseek-reasoner', 
                      temperature: float = 0.0, max_tokens: int = 400,
                      retry_count: int = 3, fallback_model: str = 'deepseek-chat') -> str:
    """
    调用 DeepSeek ChatCompletion API（使用 OpenAI 兼容接口）
    
    默认使用 deepseek-reasoner 推理模型，适合复杂分析场景。
    如果 deepseek-reasoner 不可用，会自动尝试 fallback_model（默认 deepseek-chat）。
    
    Args:
        prompt: 提示词
        model: 模型名称（默认 'deepseek-reasoner'，推理模型；也可使用 'deepseek-chat'）
        temperature: 温度参数（0-2）
        max_tokens: 最大token数
        retry_count: 重试次数（默认3次）
        fallback_model: 如果主模型失败，尝试的备用模型（默认 'deepseek-chat'）
    
    Returns:
        LLM 返回的文本内容
    
    Raises:
        RuntimeError: API key 未设置或包未安装
        Exception: API 调用失败
    """
    # 尝试使用主模型
    for attempt in range(retry_count):
        try:
            # 速率限制
            _rate_limit_wait('deepseek')
            
            client = get_deepseek_client()
            
            # 对于 deepseek-reasoner，可能需要更长的等待时间和更大的 max_tokens
            # 因为推理模型需要更多时间处理，且输出通常更长
            adjusted_max_tokens = max_tokens
            if model == 'deepseek-reasoner':
                # 推理模型可能需要更多token来返回完整结果
                # 默认至少 1600，因为推理模型的输出通常很长
                adjusted_max_tokens = max(max_tokens, 1600)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=adjusted_max_tokens,
                stream=False  # 明确指定非流式响应
            )
            
            # 检查响应结构
            if not response or not response.choices:
                raise ValueError(f"Invalid response structure from DeepSeek API (model={model})")
            
            txt = response.choices[0].message.content
            
            # 对于 deepseek-reasoner，如果返回空响应，可能是还在处理中
            # 检查是否有 finish_reason
            finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
            
            # 获取使用情况（用于调试）
            usage_info = ""
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                usage_info = f", 使用tokens: 输入={usage.prompt_tokens}, 输出={usage.completion_tokens}, 总计={usage.total_tokens}"
            
            if txt is None or txt.strip() == '':
                # 如果 finish_reason 是 'length'，说明响应被截断了
                # 但内容为空，这可能是 API 的问题，或者响应确实被完全截断了
                if finish_reason == 'length':
                    logger.error(f"DeepSeek API响应被截断且内容为空 (model={model}, finish_reason=length, max_tokens={adjusted_max_tokens}{usage_info})")
                    logger.error(f"   这可能是 API 问题，或者 max_tokens 太小导致响应被完全截断")
                    logger.error(f"   建议：1) 增加 OPENAI_MAX_TOKENS 到至少 {adjusted_max_tokens * 2}")
                    logger.error(f"         2) 或者使用 deepseek-chat 作为替代")
                    # 对于这种情况，如果是第一次尝试且有 fallback 模型，尝试 fallback
                    if attempt == 0 and fallback_model and fallback_model != model:
                        logger.warning(f"尝试使用备用模型 {fallback_model} 替代 {model}")
                        try:
                            return call_deepseek_chat(prompt, model=fallback_model, temperature=temperature, 
                                                    max_tokens=max_tokens, retry_count=retry_count, 
                                                    fallback_model=None)
                        except Exception as fallback_error:
                            logger.error(f"备用模型 {fallback_model} 也失败: {fallback_error}")
                    raise ValueError(f"Response truncated from DeepSeek API (model={model}), but content is empty")
                else:
                    logger.warning(f"DeepSeek API返回空响应 (model={model}, finish_reason={finish_reason}{usage_info})")
                    raise ValueError(f"Empty response from DeepSeek API (model={model}, finish_reason={finish_reason})")
            
            # 如果 finish_reason 是 'length'，说明响应被截断了，但至少有一些内容
            # 对于这种情况，我们仍然返回已返回的内容（即使被截断），而不是抛出异常
            # 因为部分内容总比没有内容好，上层可以处理不完整的响应
            if finish_reason == 'length':
                logger.warning(f"DeepSeek API响应被截断 (model={model}, finish_reason=length, max_tokens={adjusted_max_tokens}{usage_info})，返回部分内容")
                logger.warning(f"   建议：如果响应不完整，可以增加 OPENAI_MAX_TOKENS 配置（当前: {max_tokens} -> 建议: {adjusted_max_tokens * 2}）")
                # 仍然返回已返回的内容，让上层决定如何处理
                return txt.strip()
            
            return txt.strip()
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # 记录详细的错误信息（用于诊断）
            if attempt == 0:  # 只在第一次尝试时记录详细信息
                logger.debug(f"DeepSeek API错误详情 (model={model}): 类型={error_type}, 消息={error_msg}")
                # 如果是API错误，尝试获取更多信息
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_body = e.response.json() if hasattr(e.response, 'json') else str(e.response)
                        logger.debug(f"DeepSeek API错误响应体: {error_body}")
                    except:
                        pass
            
            # 检查是否是模型不存在错误（404, model not found等）
            is_model_not_found = ('404' in error_msg or 
                                 'model' in error_msg.lower() and 'not found' in error_msg.lower() or
                                 'invalid model' in error_msg.lower() or
                                 'unknown model' in error_msg.lower() or
                                 'model_not_found' in error_msg.lower() or
                                 'does not exist' in error_msg.lower())
            
            # 检查是否是空响应错误
            is_empty_response = 'empty response' in error_msg.lower() or 'Empty response' in error_msg
            
            # 检查是否是限流错误（429）
            is_rate_limit = ('429' in error_msg or 
                           'rate limit' in error_msg.lower() or 
                           'quota' in error_msg.lower() or 
                           'insufficient balance' in error_msg.lower())
            
            # 检查是否是认证错误（401）
            is_auth_error = ('401' in error_msg or 
                           'unauthorized' in error_msg.lower() or
                           'invalid api key' in error_msg.lower() or
                           'authentication' in error_msg.lower())
            
            # 如果是认证错误，不重试，直接抛出
            if is_auth_error:
                logger.error(f"DeepSeek API认证失败 (model={model}): {error_msg}，请检查API Key是否正确")
                raise
            
            # 如果是模型不存在错误，且是第一次尝试，尝试使用fallback模型
            if is_model_not_found and attempt == 0 and fallback_model and fallback_model != model:
                logger.warning(f"DeepSeek模型 {model} 不可用: {error_msg}，尝试使用备用模型 {fallback_model}")
                try:
                    return call_deepseek_chat(prompt, model=fallback_model, temperature=temperature, 
                                            max_tokens=max_tokens, retry_count=retry_count, 
                                            fallback_model=None)  # 避免无限递归
                except Exception as fallback_error:
                    fallback_error_msg = str(fallback_error)
                    logger.error(f"备用模型 {fallback_model} 也失败: {fallback_error_msg}")
                    # 如果fallback也失败，记录可能的原因
                    if 'model' in fallback_error_msg.lower() and 'not found' in fallback_error_msg.lower():
                        logger.error(f"⚠️  DeepSeek模型 {fallback_model} 也不可用，可能的原因：")
                        logger.error(f"   1. 模型名称错误（当前尝试: {model}, {fallback_model}）")
                        logger.error(f"   2. API Key权限不足（无法访问这些模型）")
                        logger.error(f"   3. DeepSeek服务暂时不可用")
                        logger.error(f"   建议：检查DeepSeek官方文档确认可用模型名称，或检查API Key权限")
                    # 继续使用原模型的重试逻辑
            
            # 空响应或余额不足时，不重试，直接抛出异常（让上层使用fallback）
            if is_empty_response or is_rate_limit:
                if attempt == 0:  # 第一次尝试就失败
                    logger.warning(f"DeepSeek API返回空响应或余额不足 (model={model}): {error_msg}，跳过重试，使用fallback")
                else:
                    logger.warning(f"DeepSeek API返回空响应或余额不足 (attempt {attempt+1}/{retry_count}): {error_msg}，跳过重试，使用fallback")
                raise  # 直接抛出，让上层使用fallback
            
            if attempt < retry_count - 1:
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
