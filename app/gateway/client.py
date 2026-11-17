"""
Single entry point for every LLM the app uses.

- If PORTKEY_API_KEY + PORTKEY_CONFIG_ID are set, calls go through the Portkey
  gateway (fallbacks, caching, retries, analytics configured in the saved config).
- Otherwise calls go straight to Groq, trying GROQ_API_KEY then GROQ_FALLBACK_API_KEY.
"""

from typing import Sequence

import logfire
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

from app.config import settings

if settings.portkey_enabled:
    logfire.info("LLM route: Portkey gateway.")
elif settings.PORTKEY_API_KEY:
    logfire.warning("PORTKEY_API_KEY set but PORTKEY_CONFIG_ID missing — calling Groq directly.")
else:
    logfire.info("LLM route: Groq direct.")


def _model_for(feature: str) -> str:
    # Cheap/fast model for classification-style work, big model for the agent.
    return settings.GROQ_FAST_MODEL if feature in ("planner", "guard") else settings.GROQ_MODEL


def _base_models(feature: str, temperature: float) -> list[BaseChatModel]:
    """Chat models in the order they should be tried."""
    model = _model_for(feature)

    if settings.portkey_enabled:
        from langchain_openai import ChatOpenAI
        from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

        # Portkey exposes an OpenAI-compatible endpoint; "@slug/model" picks the provider
        # configured under that slug. Fallbacks/retries live in the Portkey config.
        return [
            ChatOpenAI(
                api_key=settings.PORTKEY_API_KEY,
                base_url=PORTKEY_GATEWAY_URL,
                model=f"@{settings.GROQ_SLUG}/{model}",
                temperature=temperature,
                default_headers=createHeaders(
                    api_key=settings.PORTKEY_API_KEY,
                    config=settings.PORTKEY_CONFIG_ID,
                    metadata={"feature": feature, "_user": "rag-system"},
                ),
            )
        ]

    keys = [k for k in (settings.GROQ_API_KEY, settings.GROQ_FALLBACK_API_KEY) if k]
    if not keys:
        raise RuntimeError("No LLM configured: set GROQ_API_KEY (or Portkey keys).")
    return [ChatGroq(api_key=k, model=model, temperature=temperature) for k in keys]


def get_langchain_llm(feature: str = "agent", temperature: float = 0.0, key_fallback: bool = True):
    """
    A chat model with the usual .invoke() interface.
    key_fallback=False returns a plain chat model (NeMo inspects the LLM type).
    """
    models = _base_models(feature, temperature)
    if len(models) == 1 or not key_fallback:
        return models[0]
    return models[0].with_fallbacks(models[1:])


def get_tool_llm(tools: Sequence[BaseTool], feature: str = "agent", temperature: float = 0.0) -> Runnable:
    """A chat model with `tools` bound, falling back across keys/providers."""
    bound = [m.bind_tools(list(tools)) for m in _base_models(feature, temperature)]
    return bound[0] if len(bound) == 1 else bound[0].with_fallbacks(bound[1:])
