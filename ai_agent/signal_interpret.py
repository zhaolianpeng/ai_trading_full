# ai_agent/signal_interpret.py
import json
import logging
from .llm_client import ask_llm
from .llm_prompt import MARKET_STRUCTURE_PROMPT

logger = logging.getLogger(__name__)

def interpret_with_llm(feature_packet, provider='openai', model='gpt-4o-mini', 
                       use_llm=True, temperature=0.0, max_tokens=400):
    """
    把结构化特征传给 LLM，解析返回的 JSON。
    若无法调用 LLM 或解析失败，返回简单启发式聚合。
    """
    if not use_llm:
        # fallback heuristic
        score = 0
        reasons = []
        if feature_packet.get('ema_alignment'):
            score += 30; reasons.append('EMA alignment')
        if feature_packet.get('higher_highs'):
            score += 25; reasons.append('Higher highs')
        if feature_packet.get('volume_spike'):
            score += 20; reasons.append('Volume spike')
        if feature_packet.get('breakout'):
            score += 15; reasons.append('Breakout')
        signal = 'Long' if score >= 50 else 'Neutral'
        return {'trend_structure':'Bull' if signal=='Long' else 'Neutral', 'signal':signal, 'score':score,
                'confidence':'High' if score>75 else 'Medium' if score>50 else 'Low',
                'explanation':', '.join(reasons), 'risk':'fallback heuristic'}

    # 确保所有值都可以被 JSON 序列化（Python 的 json.dumps 支持布尔值，但需要确保没有其他问题）
    try:
        prompt = MARKET_STRUCTURE_PROMPT + "\n\n特征数据：\n" + json.dumps(feature_packet, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization failed: {e}, converting values...")
        # 如果序列化失败，手动转换
        safe_packet = {}
        for k, v in feature_packet.items():
            if isinstance(v, (bool, int, float, str)) or v is None:
                safe_packet[k] = v
            else:
                safe_packet[k] = str(v)
        prompt = MARKET_STRUCTURE_PROMPT + "\n\n特征数据：\n" + json.dumps(safe_packet, ensure_ascii=False)
    try:
        txt = ask_llm(prompt, provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)
        # 检查返回文本是否为空
        if not txt or txt.strip() == '':
            logger.warning(f"LLM返回空响应 (provider={provider}, model={model}), 使用fallback")
            return interpret_with_llm(feature_packet, provider=provider, model=model, use_llm=False)
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"LLM调用失败 (provider={provider}, model={model}): {error_msg}, 使用fallback")
        return interpret_with_llm(feature_packet, provider=provider, model=model, use_llm=False)
    # 尝试把 LLM 返回解析为 JSON
    parsed = None
    json_error = None
    
    # 方法1: 直接解析
    try:
        parsed = json.loads(txt.strip())
    except json.JSONDecodeError as e:
        json_error = str(e)
        logger.debug(f"直接JSON解析失败: {e}")
        
        # 方法2: 清理文本后解析（移除可能的markdown代码块标记）
        try:
            cleaned = txt.strip()
            # 移除可能的 ```json 和 ``` 标记
            if cleaned.startswith('```'):
                lines = cleaned.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                cleaned = '\n'.join(lines)
            
            # 尝试解析清理后的文本
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e2:
            logger.debug(f"清理后JSON解析失败: {e2}")
            
            # 方法3: 使用正则表达式提取JSON对象
            try:
                import re
                # 匹配最外层的JSON对象（支持嵌套）
                pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(pattern, txt, re.DOTALL)
                if matches:
                    # 尝试解析最长的匹配
                    longest_match = max(matches, key=len)
                    try:
                        parsed = json.loads(longest_match)
                    except json.JSONDecodeError:
                        # 如果最长匹配失败，尝试所有匹配
                        for match in sorted(matches, key=len, reverse=True):
                            try:
                                parsed = json.loads(match)
                                break
                            except json.JSONDecodeError:
                                continue
                else:
                    # 尝试匹配更宽松的模式
                    pattern2 = r'\{.*\}'
                    m = re.search(pattern2, txt, re.DOTALL)
                    if m:
                        try:
                            # 尝试修复常见的JSON问题
                            json_str = m.group(0)
                            # 移除尾随逗号
                            json_str = re.sub(r',\s*}', '}', json_str)
                            json_str = re.sub(r',\s*]', ']', json_str)
                            # 将单引号替换为双引号（仅在字符串值中，但要小心处理转义）
                            # 先处理没有转义的单引号字符串
                            json_str = re.sub(r"(?<!\\)'([^']*)'(?=\s*:)", r'"\1"', json_str)
                            json_str = re.sub(r"(?<!\\)'([^']*)'(?=\s*[,}])", r'"\1"', json_str)
                            parsed = json.loads(json_str)
                        except json.JSONDecodeError as e4:
                            logger.debug(f"JSON修复后仍解析失败: {e4}")
                            json_error = str(e4)
            except Exception as e3:
                logger.debug(f"正则提取JSON失败: {e3}")
                if not json_error:
                    json_error = str(e3)
    
    # 如果所有方法都失败，使用fallback
    if parsed is None:
        # 记录更详细的错误信息
        txt_preview = txt[:200] if txt else "(空响应)"
        logger.warning(f"LLM返回的JSON解析失败: {json_error}, 原始文本前200字符: {txt_preview}")
        
        # 如果响应为空，直接使用fallback
        if not txt or txt.strip() == '':
            logger.warning("LLM返回空响应，使用fallback启发式方法")
            return interpret_with_llm(feature_packet, provider=provider, model=model, use_llm=False)
        logger.warning(f"使用fallback启发式方法")
        parsed = interpret_with_llm(feature_packet, provider=provider, model=model, use_llm=False)
    
    return parsed
