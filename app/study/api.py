"""
Daily 10 REST API, mounted under /study.

Faculty endpoints (create subjects, upload notes and papers, generate and approve questions)
require the X-Admin-Key header when STUDY_ADMIN_KEY is set. Student endpoints use the
student's code; real sign-in (college email) is the next step before a wider rollout.
"""

import os
import tempfile
from datetime import date

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.study import content, service

router = APIRouter(prefix="/study", tags=["Daily 10"])


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("STUDY_ADMIN_KEY")
    if expected and x_admin_key != expected:
        raise HTTPException(status_code=401, detail="Faculty endpoint: send a valid X-Admin-Key header.")


def _not_found(e: Exception):
    raise HTTPException(status_code=404, detail=str(e))


# --- Faculty ---------------------------------------------------------------------------------

class SubjectIn(BaseModel):
    code: str = Field(..., min_length=2, max_length=32)
    name: str = Field(..., min_length=2, max_length=200)
    department: str = ""
    semester: int = 0


@router.post("/subjects", dependencies=[Depends(require_admin)])
def create_subject(body: SubjectIn):
    s = service.create_subject(body.code, body.name, body.department, body.semester)
    return {"code": s.code, "name": s.name}


@router.get("/subjects")
def subjects():
    return service.list_subjects()


async def _texts(files: list[UploadFile]) -> list[tuple[str, str]]:
    out = []
    for f in files:
        suffix = os.path.splitext(f.filename or "")[1]
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(await f.read())
            tmp.flush()
            try:
                out.append((f.filename, content.read_file(tmp.name)))
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
    return out


@router.post("/subjects/{code}/notes", dependencies=[Depends(require_admin)])
async def upload_notes(code: str, files: list[UploadFile] = File(...)):
    try:
        return {"topics_added": service.add_notes(code, await _texts(files))}
    except service.NotFound as e:
        _not_found(e)


@router.post("/subjects/{code}/past-papers", dependencies=[Depends(require_admin)])
async def upload_past_paper(code: str, file: UploadFile = File(...), year: int | None = Form(default=None)):
    (name, text), = await _texts([file])
    try:
        n = service.add_past_paper(code, text, year or content.year_from_name(name))
    except service.NotFound as e:
        _not_found(e)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"questions_found": n}


@router.post("/subjects/{code}/generate", dependencies=[Depends(require_admin)])
def generate(code: str, limit: int | None = None):
    from app.gateway.client import get_json_llm

    try:
        return service.generate_questions(code, get_json_llm(feature="agent"), limit=limit)
    except service.NotFound as e:
        _not_found(e)


@router.get("/subjects/{code}/topics")
def topics(code: str):
    try:
        return service.topic_overview(code)
    except service.NotFound as e:
        _not_found(e)


@router.get("/subjects/{code}/questions", dependencies=[Depends(require_admin)])
def questions(code: str, status: str | None = None):
    try:
        return service.list_questions(code, status)
    except service.NotFound as e:
        _not_found(e)


class StatusIn(BaseModel):
    status: str = Field(..., pattern="^(approved|rejected|draft)$")
    reason: str = ""


@router.post("/questions/{question_id}/status", dependencies=[Depends(require_admin)])
def set_status(question_id: int, body: StatusIn):
    try:
        service.set_question_status(question_id, body.status, body.reason)
    except service.NotFound as e:
        _not_found(e)
    return {"ok": True}


@router.post("/questions/{question_id}/report")
def report(question_id: int):
    try:
        service.report_question(question_id)
    except service.NotFound as e:
        _not_found(e)
    return {"ok": True}


# --- Students --------------------------------------------------------------------------------

class StudentIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    notify_time: str = Field("19:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


@router.post("/students")
def register(body: StudentIn):
    s = service.register_student(body.name, body.notify_time)
    return {"code": s.code, "name": s.name, "notify_time": s.notify_time}


class EnrollIn(BaseModel):
    subject_code: str
    exam_date: date


@router.post("/students/{code}/enroll")
def enroll(code: str, body: EnrollIn):
    try:
        service.enroll(code, body.subject_code, body.exam_date)
    except service.NotFound as e:
        _not_found(e)
    return {"ok": True}


class PrefsIn(BaseModel):
    notify_time: str | None = Field(None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


@router.patch("/students/{code}")
def preferences(code: str, body: PrefsIn):
    try:
        service.set_preferences(code, notify_time=body.notify_time)
    except service.NotFound as e:
        _not_found(e)
    return {"ok": True}


@router.get("/students/{code}/today")
def today(code: str):
    try:
        return service.today_session(code)
    except service.NotFound as e:
        _not_found(e)


class AnswerIn(BaseModel):
    question_id: int
    chosen_index: int = Field(..., ge=0, le=3)


@router.post("/students/{code}/answers")
def answer(code: str, body: AnswerIn):
    try:
        return service.submit_answer(code, body.question_id, body.chosen_index)
    except service.NotFound as e:
        _not_found(e)


@router.get("/students/{code}/progress")
def progress(code: str):
    try:
        return service.progress(code)
    except service.NotFound as e:
        _not_found(e)
