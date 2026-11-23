"""Faculty: upload a subject, see the most-asked topics, approve generated questions."""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Daily 10 · Faculty", page_icon="🧑‍🏫", layout="wide")
st.title("🧑‍🏫 Daily 10 · Faculty")

admin_key = st.sidebar.text_input("Admin key", type="password", help="STUDY_ADMIN_KEY on the server, if set")
HEADERS = {"X-Admin-Key": admin_key} if admin_key else {}


def api(method: str, path: str, **kwargs):
    r = requests.request(method, f"{BACKEND_URL}/study{path}", headers=HEADERS, timeout=600, **kwargs)
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.content else r.reason
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


subjects = api("GET", "/subjects")

with st.expander("➕ New subject", expanded=not subjects):
    c1, c2, c3, c4 = st.columns([1, 3, 1, 1])
    code = c1.text_input("Code", placeholder="21CS53")
    name = c2.text_input("Name", placeholder="Database Management Systems")
    dept = c3.text_input("Dept", value="CSE")
    sem = c4.number_input("Sem", 1, 8, 5)
    if st.button("Create subject") and code and name:
        try:
            api("POST", "/subjects", json={"code": code, "name": name, "department": dept, "semester": int(sem)})
            st.rerun()
        except Exception as e:
            st.error(str(e))

if not subjects:
    st.stop()

options = {f"{s['code']} · {s['name']} ({s['topics']} topics)": s["code"] for s in subjects}
code = options[st.sidebar.selectbox("Subject", list(options))]

tab_up, tab_topics, tab_review = st.tabs(["1 · Upload", "2 · Topics", "3 · Review questions"])

with tab_up:
    notes = st.file_uploader("Lecture notes (in syllabus order)", type=["pdf", "docx", "pptx", "txt", "html"],
                             accept_multiple_files=True)
    if st.button("Add notes", disabled=not notes):
        res = api("POST", f"/subjects/{code}/notes", files=[("files", (f.name, f.getvalue())) for f in notes])
        st.success(f"Added {res['topics_added']} daily topics.")
    paper = st.file_uploader("Past exam paper (one at a time; year read from the file name)", type=["pdf", "docx", "txt"])
    year = st.number_input("Year (optional)", 0, 2100, 0)
    if st.button("Add past paper", disabled=paper is None):
        res = api("POST", f"/subjects/{code}/past-papers", files={"file": (paper.name, paper.getvalue())},
                  data={"year": int(year)} if year else {})
        st.success(f"Found {res['questions_found']} questions and matched them to topics.")
    st.divider()
    limit = st.number_input("Generate for how many topics now", 1, 200, 5,
                            help="Uses the LLM once per topic. Free tiers have daily limits, so do it in batches.")
    if st.button("✨ Generate summaries and questions", type="primary"):
        with st.spinner("Generating... (about 10-30 s per topic)"):
            st.write(api("POST", f"/subjects/{code}/generate", params={"limit": int(limit)}))

with tab_topics:
    topics = api("GET", f"/subjects/{code}/topics")
    st.caption("Exam weight = how many past-paper questions matched the topic.")
    st.dataframe(
        [{"#": t["position"], "Topic": t["title"], "Asked in exams": t["exam_weight"],
          "Approved": t["questions"].get("approved", 0), "Drafts": t["questions"].get("draft", 0),
          "Auto-rejected": t["questions"].get("auto_rejected", 0), "Source": t["source"]} for t in topics],
        hide_index=True, width="stretch",
    )

with tab_review:
    drafts = api("GET", f"/subjects/{code}/questions", params={"status": "draft"})
    st.caption(f"{len(drafts)} question(s) waiting. Students only ever see approved questions.")
    for q in drafts:
        with st.container(border=True):
            st.caption(q["topic"] + (f" · ⚠️ {q['status_reason']}" if q["status_reason"] else ""))
            st.markdown(f"**{q['prompt']}**")
            for i, opt in enumerate(q["options"]):
                st.markdown(f"{'✅' if i == q['answer_index'] else '▫️'} {opt}")
            st.caption(f"Why: {q['explanation']}  \nFrom the notes: “{q['evidence']}”")
            a, r, _ = st.columns([1, 1, 6])
            if a.button("Approve", key=f"a{q['id']}", type="primary"):
                api("POST", f"/questions/{q['id']}/status", json={"status": "approved"})
                st.rerun()
            if r.button("Reject", key=f"r{q['id']}"):
                api("POST", f"/questions/{q['id']}/status", json={"status": "rejected"})
                st.rerun()
    with st.expander("Auto-rejected (evidence not found in the notes)"):
        for q in api("GET", f"/subjects/{code}/questions", params={"status": "auto_rejected"}):
            st.markdown(f"- {q['prompt']}  \n  _claimed evidence:_ “{q['evidence']}”")
