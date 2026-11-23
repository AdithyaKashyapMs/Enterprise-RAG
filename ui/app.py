import os
import re
import uuid

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env from the project root regardless of the working directory.
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT_S = 300  # multi-step answers on a rate-limited free LLM tier can take a few minutes

try:
    logfire.configure(
        service_name="enterprise-rag-ui",
        token=os.getenv("LOGFIRE_TOKEN"),
        send_to_logfire="if-token-present",
    )
    LOGFIRE_STATUS = "Tracing" if os.getenv("LOGFIRE_TOKEN") else "Local only (no LOGFIRE_TOKEN)"
except Exception as e:
    LOGFIRE_STATUS = f"Disabled ({e})"

st.set_page_config(page_title="Enterprise Agentic RAG", page_icon="🤖", layout="wide")

AI_AVATAR, USER_AVATAR = "🤖", "👤"


def new_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


if "session_id" not in st.session_state:
    new_session()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**[{src.get('id', src['source'])}]** · relevance {src['score']:.2f}")
            st.caption(src["content"][:1200] + ("…" if len(src["content"]) > 1200 else ""))


CITATION_RE = re.compile(r"\[([^\[\]]+#\d+)\]")


def voted_chunk_ids(message: dict) -> list[str]:
    """The passages a vote applies to: the ones the answer cites, else every source shown."""
    source_ids = [s["id"] for s in message.get("sources", []) if s.get("id")]
    verified = (message.get("structured") or {}).get("citations") or []
    cited = verified or [cid for cid in CITATION_RE.findall(message["content"]) if cid in source_ids]
    return list(dict.fromkeys(cited)) or source_ids


def send_feedback(key: str, message: dict) -> None:
    choice = st.session_state.get(key)  # st.feedback("thumbs"): 1 = 👍, 0 = 👎, None = cleared
    if choice is None:
        return
    try:
        requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "thread_id": st.session_state.session_id,
                "question": message["question"],
                "chunk_ids": voted_chunk_ids(message),
                "vote": 1 if choice == 1 else -1,
            },
            timeout=10,
        ).raise_for_status()
        st.toast("Thanks — feedback saved.")
    except Exception as e:
        st.toast(f"Couldn't save feedback: {e}")


def render_feedback(index: int, message: dict) -> None:
    if message.get("sources") and message.get("question"):
        key = f"feedback_{st.session_state.session_id}_{index}"
        st.feedback("thumbs", key=key, on_change=send_feedback, args=(key, message))


def queue_question(question: str) -> None:
    st.session_state.queued_prompt = question


CONFIDENCE_BADGE = {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🔴 Low confidence"}


def render_structured(index: int, message: dict) -> None:
    info = message.get("structured") or {}
    if not info:
        return
    badge = CONFIDENCE_BADGE.get(info.get("confidence"), "")
    note = "" if info.get("answered_from_docs") else " · not found in the documentation"
    if info.get("invalid_citations"):
        note += f" · {len(info['invalid_citations'])} unverified citation(s) removed"
    st.caption(badge + note)
    for j, q in enumerate(info.get("follow_up_questions", [])):
        st.button(f"↪ {q}", key=f"followup_{st.session_state.session_id}_{index}_{j}", on_click=queue_question, args=(q,))


def render_related(index: int, message: dict) -> None:
    docs, questions = message.get("related_documents", []), message.get("related_questions", [])
    if docs:
        st.markdown("**📚 Related documents**")
        for d in docs:
            st.caption(f"[{d['id']}] · {d['reason']}" + (f" — {d['snippet'][:120]}…" if d.get("snippet") else ""))
    if questions:
        st.markdown("**💬 People who asked this also asked**")
        for j, q in enumerate(questions):
            st.button(q, key=f"related_{st.session_state.session_id}_{index}_{j}", on_click=queue_question, args=(q,))


def render_starters() -> None:
    if "starters" not in st.session_state:
        try:
            st.session_state.starters = requests.get(f"{BACKEND_URL}/starters", timeout=5).json()
        except Exception:
            st.session_state.starters = []
    if st.session_state.starters:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for j, s in enumerate(st.session_state.starters):
            cols[j % 2].button(s["question"], key=f"starter_{j}", help=s["reason"], on_click=queue_question,
                               args=(s["question"],), width="stretch")


def ask_backend(prompt: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/query",
        json={"q": prompt, "thread_id": st.session_state.session_id},
        timeout=REQUEST_TIMEOUT_S,
    )
    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.reason
        raise RuntimeError(f"Backend returned {response.status_code}: {detail}")
    return response.json()


# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.caption(f"Backend: {BACKEND_URL}")
    st.caption(f"Logfire: {LOGFIRE_STATUS}")
    st.caption(f"Conversation: {st.session_state.session_id[:8]}")
    if st.button("🗑️ New conversation", width="stretch", type="primary"):
        new_session()
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")

for i, message in enumerate(st.session_state.messages):
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))
        render_structured(i, message)
        render_feedback(i, message)
        render_related(i, message)

if not st.session_state.messages:
    render_starters()

# A typed question, or one queued by clicking a starter / related question.
typed = st.chat_input("Ask about your documentation...")
prompt = typed or st.session_state.pop("queued_prompt", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        with logfire.span("User Chat Interaction", session_id=st.session_state.session_id):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    data = ask_backend(prompt)
                except requests.exceptions.Timeout:
                    status.update(label="⏱️ Timed out", state="error")
                    st.error(f"No answer after {REQUEST_TIMEOUT_S}s. The LLM provider may be rate-limiting; try again in a minute.")
                    st.stop()
                except requests.exceptions.ConnectionError:
                    status.update(label="❌ Backend offline", state="error")
                    st.error(f"Could not reach the backend at {BACKEND_URL}. Is uvicorn running?")
                    st.stop()
                except Exception as e:
                    logfire.error(f"UI-backend call failed: {e}")
                    status.update(label="❌ Request failed", state="error")
                    st.error(str(e))
                    st.stop()

                for step in data.get("thought_process", []):
                    st.write(f"⚙️ {step}")
                status.update(label="✅ Answer ready", state="complete", expanded=False)

        answer = data.get("answer") or "No response."
        sources = data.get("sources", [])
        st.markdown(answer)
        render_sources(sources)
        message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "question": prompt,
            "related_documents": data.get("related_documents", []),
            "related_questions": data.get("related_questions", []),
            "structured": data.get("structured") or {},
        }
        st.session_state.messages.append(message)
        render_structured(len(st.session_state.messages) - 1, message)
        render_feedback(len(st.session_state.messages) - 1, message)
        render_related(len(st.session_state.messages) - 1, message)
