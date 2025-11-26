# ai_agent/signal_interpret.py
import json
from .llm_client import ask_llm
from .llm_prompt import MARKET_STRUCTURE_PROMPT

def interpret_with_llm(feature_packet, provider='openai', model='gpt-4o-mini', use_llm=True):
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

    prompt = MARKET_STRUCTURE_PROMPT + "\n\n特征数据：\n" + json.dumps(feature_packet, ensure_ascii=False)
    txt = ask_llm(prompt, provider=provider, model=model)
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
