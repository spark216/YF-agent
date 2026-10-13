"""Local demonstration UI; all displayed results come from the booking service."""
from __future__ import annotations

import argparse
import os
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent_factory import create_agent

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app):
    cache = None
    if os.getenv("DEMO_REDIS_SIMULATED") == "1":
        import fakeredis.aioredis
        cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    async with create_agent(demo=True, work_dir=ROOT / "data/qa_runtime", redis_client=cache) as agent:
        app.state.agent = agent
        yield


app = FastAPI(title="YF-Agent local showcase", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


class ChatRequest(BaseModel):
    chat_id: str = Field(min_length=1, max_length=128)
    message_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=2000)


class CardCommit(BaseModel):
    chat_id: str = Field(min_length=1, max_length=128)
    message_id: str = Field(min_length=1, max_length=128)
    expected_revision: int = Field(ge=0)
    changes: dict


class Decision(BaseModel):
    chat_id: str = Field(min_length=1, max_length=128)
    message_id: str = Field(min_length=1, max_length=128)
    id: str = Field(min_length=1, max_length=128)


class ApprovalRequest(BaseModel):
    chat_id: str = Field(min_length=1, max_length=128)
    draft_id: str = Field(min_length=1, max_length=128)
    actor: str = Field(min_length=1, max_length=100)


def meta():
    agent = app.state.agent
    return {"catalog_count": agent.repository.count(), "encoder": agent.repository.encoder.name,
            "cache": agent.store.cache_status, "redis_simulated": os.getenv("DEMO_REDIS_SIMULATED") == "1",
            "start": (date.today() + timedelta(days=1)).isoformat(),
            "end": (date.today() + timedelta(days=3)).isoformat()}


@app.get("/")
async def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/state")
async def state(chat_id: str):
    if len(chat_id) > 128:
        raise HTTPException(400, "Invalid chat ID")
    agent = app.state.agent
    session = await agent.store.load(chat_id)
    return {"session": session, "meta": meta(), "traces": agent.store.traces(chat_id)[-50:]}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    agent = app.state.agent
    try:
        result = await agent.handle(request.chat_id, request.message_id, request.text)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except Exception as exc:
        # A retry with the same message ID resumes the persisted node.
        raise HTTPException(503, "Task interrupted: " + type(exc).__name__)
    return {**result, "meta": meta(), "traces": agent.store.traces(request.chat_id, request.message_id)}


@app.post("/api/reset")
async def reset(request: ChatRequest):
    agent = app.state.agent
    async with agent.serial(request.chat_id):
        await agent.store.reset(request.chat_id)
    return {"ok": True}


async def result_payload(result, chat_id):
    return {**result, "meta": meta(), "traces": app.state.agent.store.traces(chat_id)[-40:]}


@app.post("/api/requirements/commit")
async def commit_card(request: CardCommit):
    try:
        result = await app.state.agent.commit_card(request.chat_id, request.message_id, request.expected_revision, request.changes)
        return await result_payload(result, request.chat_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/requirements/accept")
async def accept_proposal(request: Decision):
    try:
        result = await app.state.agent.accept_pending(request.chat_id, request.message_id, request.id)
        return await result_payload(result, request.chat_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/relaxations/accept")
async def accept_relaxation(request: Decision):
    try:
        result = await app.state.agent.accept_relaxation(request.chat_id, request.message_id, request.id)
        return await result_payload(result, request.chat_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/drafts/approve")
async def approve(request: ApprovalRequest, x_reviewer_token: str = Header(default="")):
    try:
        receipt = await app.state.agent.approve_draft(request.chat_id, request.draft_id, request.actor, x_reviewer_token)
        session = await app.state.agent.store.load(request.chat_id)
        return {"receipt": receipt, "session": session, "meta": meta(),
                "traces": app.state.agent.store.traces(request.chat_id)[-40:]}
    except PermissionError as exc:
        raise HTTPException(403, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/events")
async def events(request: Request, chat_id: str):
    if not chat_id or len(chat_id) > 128:
        raise HTTPException(400, "Invalid conversation")
    async def stream():
        signature = None
        while not await request.is_disconnected():
            agent = app.state.agent
            session = await agent.store.load(chat_id)
            traces = agent.store.traces(chat_id)[-30:]
            current = json.dumps([session, traces], sort_keys=True, ensure_ascii=False)
            if current != signature:
                signature = current
                payload = {"session": session, "traces": traces, "meta": meta()}
                yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--redis-simulated", action="store_true", help="Use fakeredis API emulation, clearly labelled in UI")
    args = parser.parse_args()
    if args.redis_simulated:
        os.environ["DEMO_REDIS_SIMULATED"] = "1"
    uvicorn.run(app, host=args.host, port=args.port)
