from __future__ import annotations
import json
import os
from config import AI_API_KEY, AI_BASE_URL, AI_MODEL

def _settings():
    """Read local environment settings, with Streamlit secrets as a cloud fallback."""
    # Read environment variables at request time. This matters because Streamlit
    # apps may load modules before dotenv configuration is initialized.
    key = os.getenv("OPENAI_API_KEY", "") or AI_API_KEY
    base_url = os.getenv("OPENAI_API_BASE", "") or AI_BASE_URL
    model = os.getenv("DATAMIND_AI_MODEL", "") or AI_MODEL
    try:
        import streamlit as st
        key = key or st.secrets.get("OPENAI_API_KEY", "")
        base_url = st.secrets.get("OPENAI_API_BASE", base_url)
        model = st.secrets.get("DATAMIND_AI_MODEL", model)
    except Exception:
        pass
    return key, base_url, model

def ask_ai(question: str, context: dict) -> str:
    api_key, base_url, model = _settings()
    if not api_key:
        return "AI provider is not configured. Core analytics remain available. Add OPENAI_API_KEY to .env or the environment to enable dataset-aware assistance.\n\nSafe local context available:\n" + json.dumps(context, indent=2, default=str)[:4000]
    try:
        from openai import OpenAI
        client=OpenAI(api_key=api_key, base_url=base_url)
        system=("You are DataMind AI, a careful statistical assistant. Distinguish suggestions from computed results. "
                "Never invent numbers; use only the supplied context. Explain assumptions and avoid causal claims. Context:\n" + json.dumps(context,default=str)[:12000])
        r=client.chat.completions.create(model=model,messages=[{"role":"system","content":system},{"role":"user","content":question}],temperature=.2)
        return r.choices[0].message.content
    except Exception as exc:
        return f"AI provider error: {exc}. Check the API key, base URL, model, and network settings."
