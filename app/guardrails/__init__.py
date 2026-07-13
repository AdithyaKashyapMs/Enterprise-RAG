from typing import Any

from langchain_groq import ChatGroq
from portkey_ai import Portkey

from app.config import settings

from .rails import guard, initialize_rails


portkey_client = Portkey(api_key=settings.PORTKEY_API_KEY)


def get_langchain_llm(feature: str = "planner") -> ChatGroq:
    """
    Build the LangChain LLM used by agent nodes.
    Planner uses the smaller Groq model for quick routing decisions.
    """
    model = "llama-3.1-8b-instant" if feature == "planner" else settings.GROQ_MODEL
    return ChatGroq(
        api_key=settings.GROQ_API_KEY or settings.GROQ_FALLBACK_API_KEY,
        model=model,
        temperature=0,
    )


def extract_cache_status(response: Any) -> str:
    """
    Read the Portkey cache status from a completion response.
    Falls back to MISS when the header is absent or unavailable.
    """
    header_sources = [
        getattr(response, "headers", None),
        getattr(getattr(response, "response", None), "headers", None),
        getattr(getattr(response, "_response", None), "headers", None),
    ]

    for headers in header_sources:
        if not headers:
            continue
        for key in ("x-portkey-cache-status", "X-Portkey-Cache-Status"):
            value = headers.get(key)
            if value:
                return str(value).upper()

    if isinstance(response, dict):
        for key in ("x-portkey-cache-status", "X-Portkey-Cache-Status"):
            value = response.get(key)
            if value:
                return str(value).upper()

    return "MISS"


__all__ = [
    "guard",
    "initialize_rails",
    "get_langchain_llm",
    "portkey_client",
    "extract_cache_status",
]
