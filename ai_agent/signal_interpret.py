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
        txt = ask_llm(prompt, provider=provider, model=model)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}, using fallback")
        return interpret_with_llm(feature_packet, provider=provider, model=model, use_llm=False)
    # 尝试把 LLM 返回解析为 JSON
    try:
        parsed = json.loads(txt.strip())
    except Exception:
        import re
        m = re.search(r'\{.*\}', txt, re.S)
        if m:
            parsed = json.loads(m.group(0))
        else:
            # fallback
            parsed = interpret_with_llm(feature_packet, provider=provider, model=model, use_llm=False)
    return parsed
