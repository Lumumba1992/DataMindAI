from __future__ import annotations
import json
from config import AI_API_KEY, AI_BASE_URL, AI_MODEL

def ask_ai(question: str, context: dict) -> str:
    if not AI_API_KEY:
        return "AI provider is not configured. Core analytics remain available. Add OPENAI_API_KEY to .env or the environment to enable dataset-aware assistance.\n\nSafe local context available:\n" + json.dumps(context, indent=2, default=str)[:4000]
    try:
        from openai import OpenAI
        client=OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
        system=("You are DataMind AI, a careful statistical assistant. Distinguish suggestions from computed results. "
                "Never invent numbers; use only the supplied context. Explain assumptions and avoid causal claims. Context:\n" + json.dumps(context,default=str)[:12000])
        r=client.chat.completions.create(model=AI_MODEL,messages=[{"role":"system","content":system},{"role":"user","content":question}],temperature=.2)
        return r.choices[0].message.content
    except Exception as exc:
        return f"AI provider error: {exc}. Check the API key, base URL, model, and network settings."
