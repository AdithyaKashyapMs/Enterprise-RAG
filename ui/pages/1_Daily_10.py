"""Daily 10: the student's 10-minute session (mobile-friendly)."""

import os
from datetime import date, timedelta

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BOT_NAME = os.getenv("TELEGRAM_BOT_USERNAME", "")

st.set_page_config(page_title="Daily 10", page_icon="📚", layout="centered")


def api(method: str, path: str, **kwargs):
    r = requests.request(method, f"{BACKEND_URL}/study{path}", timeout=30, **kwargs)
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.content else r.reason
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


# --- Sign in -------------------------------------------------------------------------------
code = st.session_state.get("student_code") or st.query_params.get("code")
if code:
    st.session_state.student_code = code.upper()

st.title("📚 Daily 10")
st.caption("Ten minutes a day, all semester. No panic before exams.")

if not st.session_state.get("student_code"):
    tab_in, tab_new = st.tabs(["I have a code", "I'm new"])
    with tab_in:
        entered = st.text_input("Your student code", max_chars=6, key="code_input")
        if st.button("Open my session", type="primary") and entered:
            st.session_state.student_code = entered.strip().upper()
            st.rerun()
    with tab_new:
        name = st.text_input("Your name", key="new_name")
        when = st.time_input("Remind me every day at", value=None, step=900, key="new_time")
        if st.button("Create my account", type="primary") and name:
            try:
                res = api("POST", "/students", json={"name": name, "notify_time": (when.strftime("%H:%M") if when else "19:00")})
                st.session_state.student_code = res["code"]
                st.session_state.just_registered = True
                st.rerun()
            except Exception as e:
                st.error(str(e))
    st.stop()

code = st.session_state.student_code
if st.session_state.pop("just_registered", False):
    st.success(f"Your code is **{code}**. Keep it to sign in.")
    if BOT_NAME:
        st.info(f"Get daily reminders on Telegram: open @{BOT_NAME} and send `/start {code}`")

# --- Subjects ------------------------------------------------------------------------------
try:
    prog = api("GET", f"/students/{code}/progress")
except Exception as e:
    st.error(f"Couldn't load your account ({e}).")
    if st.button("Sign out"):
        st.session_state.pop("student_code", None)
        st.rerun()
    st.stop()

with st.expander("➕ Add a subject" if prog["subjects"] else "➕ Add your subjects to start", expanded=not prog["subjects"]):
    try:
        subjects = api("GET", "/subjects")
    except Exception:
        subjects = []
    if subjects:
        options = {f"{s['code']} · {s['name']} (Sem {s['semester']})": s["code"] for s in subjects}
        choice = st.selectbox("Subject", list(options))
        exam = st.date_input("Exam date", value=date.today() + timedelta(days=90), min_value=date.today())
        if st.button("Add subject"):
            api("POST", f"/students/{code}/enroll", json={"subject_code": options[choice], "exam_date": str(exam)})
            st.rerun()
    else:
        st.write("No subjects yet: ask your faculty to upload one.")

if not prog["subjects"]:
    st.stop()

# --- Today's session -----------------------------------------------------------------------
session = api("GET", f"/students/{code}/today")
questions = session["questions"]
done = sum(q["answered"] for q in questions)  # the server is the source of truth
total = len(questions)

st.subheader(f"Today · {date.fromisoformat(session['day']).strftime('%a %d %b')}")
if total:
    st.progress(min(done, total) / total, text=f"{min(done, total)} of {total} questions")

for note in session["notes"]:
    st.info(note)

card = session.get("card")
if card:
    with st.container(border=True):
        st.caption(f"{card['subject']} · {card['source']}"
                   + (f" · 📝 asked {card['exam_weight']}× in past exams" if card.get("exam_weight") else ""))
        st.markdown(f"### {card['title']}")
        if card.get("summary"):
            st.write(card["summary"])
        for point in card.get("key_points") or []:
            st.markdown(f"- {point}")

for i, q in enumerate(questions, 1):
    key = f"ans_{session['day']}_{q['id']}"
    with st.container(border=True):
        st.caption(("🆕 New" if q["kind"] == "new" else "🔁 Review") + f" · {q['topic']}")
        st.markdown(f"**{i}. {q['prompt']}**")
        if q["answered"] and key not in st.session_state:
            st.write("✅ Answered today.")
            continue
        picked = st.radio("Answer", q["options"], index=None, key=f"radio_{key}", label_visibility="collapsed",
                          disabled=key in st.session_state)
        if key not in st.session_state:
            if st.button("Check", key=f"check_{key}", disabled=picked is None):
                st.session_state[key] = api("POST", f"/students/{code}/answers",
                                            json={"question_id": q["id"], "chosen_index": q["options"].index(picked)})
                st.rerun()
        else:
            res = st.session_state[key]
            right = q["options"][res["answer_index"]]
            (st.success if res["correct"] else st.error)(
                ("Correct! " if res["correct"] else f"Not quite. The answer is **{right}**. ") + res["explanation"])
            st.caption(f"From your notes: “{res['evidence']}” · next review {res['next_review']}")
            if st.button("⚠️ This question is wrong", key=f"report_{key}"):
                api("POST", f"/questions/{q['id']}/report")
                st.toast("Thanks, sent to your faculty for review.")

past = session.get("past_question")
if past:
    with st.expander("📝 Past exam question: outline your answer (2 min)"):
        meta = " · ".join(x for x in [str(past["year"]) if past.get("year") else "", f"{past['marks']} marks" if past.get("marks") else ""] if x)
        st.markdown(f"**{past['text']}**")
        if meta:
            st.caption(meta)
        st.text_area("Your outline (just for you)", key=f"outline_{session['day']}")

if total and min(done, total) >= total:
    st.balloons()
    st.success("🎉 Done for today! See you tomorrow.")

# --- Progress ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"**{prog['student']}** · code `{code}`")
    st.metric("🔥 Streak", f"{prog['streak_days']} day(s)")
    if prog["accuracy"] is not None:
        st.metric("Accuracy", f"{prog['accuracy'] * 100:.0f}%", help=f"{prog['answers']} answers so far")
    for s in prog["subjects"]:
        st.progress(s["coverage"], text=f"{s['code']}: {s['covered']}/{s['topics']} topics · exam in {s['days_to_exam']}d")
    if prog["weak_topics"]:
        st.markdown("**Needs practice**")
        for w in prog["weak_topics"]:
            st.caption(f"• {w['topic']} ({w['misses']} missed)")
    if BOT_NAME:
        st.caption(f"Reminders: send `/start {code}` to @{BOT_NAME} on Telegram")
    if st.button("Sign out"):
        st.session_state.pop("student_code", None)
        st.rerun()
