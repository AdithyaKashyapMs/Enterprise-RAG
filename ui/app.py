import os
import uuid

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env from the project root regardless of the working directory.
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT_S = 90

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

for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if prompt := st.chat_input("Ask about your documentation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        with logfire.span("User Chat Interaction", session_id=st.session_state.session_id):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    data = ask_backend(prompt)
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
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
