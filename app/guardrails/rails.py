"""
Input gate that runs before the LangGraph pipeline.

Decision order (cheapest first):
  1. Jailbreak phrase (whole-phrase match)      -> block, no LLM call
  2. Clearly technical (whole-word keyword hit) -> pass, no LLM call
  3. Everything else                            -> ask NeMo Guardrails; a rail
     has fired only if NeMo answered with one of our canned bot messages.
     Otherwise pass through — the agent decides whether it needs the docs
     (e.g. "what did I just ask?") or needs retrieval.

The gate fails OPEN (passes) if NeMo is unavailable, because the downstream
agent is itself grounded and scoped.
"""

import logfire
from nemoguardrails import LLMRails, RailsConfig

from app.core.intent import is_jailbreak_attempt, looks_technical
from app.gateway.client import get_langchain_llm
from app.guardrails.colang_rules import COLANG_CONTENT, RAIL_INDICATORS, YAML_CONTENT

JAILBREAK_RESPONSE = (
    "I maintain consistent guidelines regardless of how I am prompted. "
    "I am here to help with Kubernetes, Intel, and networking. What can I help you with?"
)

_rails: LLMRails | None = None


def initialize_rails() -> None:
    """Build the NeMo LLMRails singleton at app startup (fast 8B model)."""
    global _rails
    try:
        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT
        )
        _rails = LLMRails(config, llm=get_langchain_llm(feature="guard", key_fallback=False))
        logfire.info("NeMo Guardrails initialised.")
    except Exception as e:
        _rails = None
        logfire.error(f"NeMo Guardrails failed to initialise, gate will fail open: {e}")


def _rail_fired(response: str) -> bool:
    return any(indicator in response for indicator in RAIL_INDICATORS)


def guard(message: str) -> tuple[bool, str | None]:
    """
    Returns:
        (True,  response) — a rail fired; reply with `response` and skip RAG.
        (False, None)     — continue to the LangGraph pipeline.
    """
    with logfire.span("Guardrails Check"):
        if is_jailbreak_attempt(message):
            logfire.info("Guardrails: jailbreak pattern matched.")
            return True, JAILBREAK_RESPONSE

        if looks_technical(message):
            logfire.info("Guardrails: technical query, passing without LLM check.")
            return False, None

        if _rails is None:
            logfire.warning("Guardrails not initialised — passing.")
            return False, None

        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
        except Exception as e:
            logfire.warning(f"Guardrails LLM call failed — passing: {e}")
            return False, None

        content = result.get("content", "") if isinstance(result, dict) else str(result)
        if _rail_fired(content):
            logfire.info(f"Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("Guardrails passed.")
        return False, None
