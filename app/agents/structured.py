"""
Structured final answers.

The agent finishes by *calling a tool*, `submit_answer`, whose arguments follow the
StructuredAnswer schema. Tool calling is how the LLM API enforces structured output here:
the provider returns JSON arguments matching the schema instead of free text, in the same
single LLM call that would otherwise produce the answer (no extra "now format it" call).

The server never trusts the structure blindly. `validate()` checks it against what actually
happened in the turn:
  - every citation must be a passage that a tool returned in this turn; anything else
    (a hallucinated or stale id) is dropped, and removed from the answer text
  - an answer that claims to come from the docs but has no valid citation is downgraded
    to low confidence

If the model replies with plain text instead (or the budget / degraded paths produce text),
`from_text()` builds the same structure from the text, so callers always get one shape.
`source` records which path produced it.
"""

import json
import re
from typing import Literal, TypedDict

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, ValidationError

SUBMIT_ANSWER = "submit_answer"

_CITATION_RE = re.compile(r"\[([^\[\]]+#\d+)\]")
_NOT_FOUND = "couldn't find this in the documentation"


class StructuredAnswer(BaseModel):
    """Submit your final answer to the user. Call this exactly once, when you are done
    searching. Do not reply with plain text."""

    answer: str = Field(
        description="The answer in Markdown. Cite every fact inline with the passage id in "
        "square brackets, e.g. [cronjobs.docx#12]. Use code blocks for YAML and commands."
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Ids of every passage the answer relies on, e.g. ['cronjobs.docx#12'].",
    )
    answered_from_docs: bool = Field(
        description="True if the documentation passages answer the question; false if they "
        "don't cover it, or the message needs no documentation (greeting, memory question)."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="high: the passages state the answer directly; medium: it needs combining "
        "or light inference; low: only partly covered."
    )
    # No max_length here on purpose: a 4th suggestion shouldn't invalidate the whole answer
    # (and its verified citations). The limit is asked for in the description and enforced
    # by trimming in validate().
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Up to 3 short follow-up questions the user might ask next, answerable "
        "from the documentation.",
    )


# The schema as an OpenAI-format tool definition, bound to the LLM next to the search tools.
SUBMIT_ANSWER_TOOL = convert_to_openai_tool(StructuredAnswer)
SUBMIT_ANSWER_TOOL["function"]["name"] = SUBMIT_ANSWER


class AnswerRecord(TypedDict):
    answer: str
    citations: list[str]
    answered_from_docs: bool
    confidence: str
    follow_up_questions: list[str]
    invalid_citations: list[str]   # cited by the model but never retrieved this turn
    source: str                    # "structured" | "structured_text" (schema as JSON text) | "text"


def _strip_invalid(text: str, invalid: set[str]) -> str:
    if not invalid:
        return text
    text = _CITATION_RE.sub(lambda m: "" if m.group(1) in invalid else m.group(0), text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()


def validate(args: dict, retrieved_ids: list[str]) -> AnswerRecord:
    """Parse submit_answer arguments and check them against the passages retrieved this turn."""
    try:
        parsed = StructuredAnswer.model_validate(args)
    except ValidationError:
        return from_text(str(args.get("answer", "")), retrieved_ids)

    known = set(retrieved_ids)
    inline = _CITATION_RE.findall(parsed.answer)
    claimed = list(dict.fromkeys([*parsed.citations, *inline]))
    valid = [c for c in claimed if c in known]
    invalid = [c for c in claimed if c not in known]

    confidence = parsed.confidence
    if parsed.answered_from_docs and not valid:
        confidence = "low"

    return AnswerRecord(
        answer=_strip_invalid(parsed.answer, set(invalid)),
        citations=valid,
        answered_from_docs=parsed.answered_from_docs and bool(valid),
        confidence=confidence,
        follow_up_questions=[q.strip() for q in parsed.follow_up_questions if q.strip()][:3],
        invalid_citations=invalid,
        source="structured",
    )


def _json_object(text: str) -> dict | None:
    """The answer schema written out as JSON text (some models do this instead of calling the tool)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").removeprefix("json").strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        obj = json.loads(stripped)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and isinstance(obj.get("answer"), str) else None


def from_text(text: str, retrieved_ids: list[str]) -> AnswerRecord:
    """
    Structure for a plain-text answer. If the text is the answer schema as JSON (the model wrote
    the tool arguments as text), validate it like a tool call; otherwise parse citations from it.
    """
    obj = _json_object(text)
    if obj is not None:
        record = validate(obj, retrieved_ids)
        if record["source"] == "structured":
            return {**record, "source": "structured_text"}
        text = obj["answer"]
    known = set(retrieved_ids)
    claimed = list(dict.fromkeys(_CITATION_RE.findall(text)))
    valid = [c for c in claimed if c in known]
    invalid = [c for c in claimed if c not in known]
    from_docs = bool(valid) and _NOT_FOUND not in text.lower()
    return AnswerRecord(
        answer=_strip_invalid(text, set(invalid)),
        citations=valid,
        answered_from_docs=from_docs,
        confidence="medium" if from_docs else "low",
        follow_up_questions=[],
        invalid_citations=invalid,
        source="text",
    )
