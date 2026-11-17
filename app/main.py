# Logfire must be configured before the app modules are imported so that
# spans emitted at import time (model/client initialisation) are captured.
import logfire

from app.config import settings

logfire.configure(
    service_name="enterprise-rag-api",
    token=settings.LOGFIRE_TOKEN,
    send_to_logfire="if-token-present",
)

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.agents import service  # noqa: E402
from app.agents.graph import rag_agent  # noqa: E402
from app.guardrails import initialize_rails  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_rails()
    yield


app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)
logfire.instrument_fastapi(app)


class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=4000)
    # Omit to start a new conversation; reuse the returned thread_id to continue it.
    thread_id: str | None = None


class Source(BaseModel):
    id: str
    source: str
    chunk_index: int | None = None
    content: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    thread_id: str
    thought_process: list[str]
    status: str
    sources: list[Source]


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graph")
def get_graph_image():
    """Mermaid PNG of the agent workflow."""
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not render graph: {e}")
    return Response(content=png_bytes, media_type="image/png")


# Sync endpoint on purpose: FastAPI runs it in a threadpool, and the graph is synchronous.
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        return service.ask(request.q, request.thread_id)
    except Exception as e:
        logfire.exception(f"Backend execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error while processing the request. Please try again later.",
        )
