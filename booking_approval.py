"""Durable, authorized draft approval. No supplier orders or payments are executed."""
from __future__ import annotations

import hmac
import json
import time
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.runnables.config import var_child_runnable_config


class ApprovalState(TypedDict, total=False):
    draft_id: str
    actor: str
    receipt: dict


class BookingApproval:
    def __init__(self, store, token, checkpointer, validate_quote=None):
        self.store, self.token = store, token
        self.validate_quote = validate_quote
        with store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS booking_drafts(
                    id TEXT PRIMARY KEY,chat_id TEXT NOT NULL,revision INTEGER NOT NULL,
                    selection_epoch INTEGER NOT NULL,quote_id TEXT NOT NULL,expires REAL NOT NULL,
                    status TEXT NOT NULL,data TEXT NOT NULL,actor TEXT,receipt TEXT);
            """)
        builder = StateGraph(ApprovalState)
        builder.add_node("human_review", self.review)
        builder.add_node("handoff", self.handoff)
        builder.add_edge(START, "human_review")
        builder.add_edge("human_review", "handoff")
        builder.add_edge("handoff", END)
        self.graph = builder.compile(checkpointer=checkpointer)

    def get(self, draft_id):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM booking_drafts WHERE id=?", (draft_id,)).fetchone()
        if not row:
            raise ValueError("Unknown booking draft")
        return {**dict(row), "data": json.loads(row["data"]), "receipt": json.loads(row["receipt"]) if row["receipt"] else None}

    def config(self, draft_id):
        return {"configurable": {"thread_id": "approval:" + draft_id}}

    async def create(self, session, chat_id):
        hotel = session["selected_hotel"]
        if hotel["valid_until"] <= time.time():
            raise ValueError("Quote expired; refresh before selecting")
        # Retrying the same selected quote reuses one draft and one interrupt.
        with self.store.connect() as db:
            row = db.execute("SELECT id FROM booking_drafts WHERE chat_id=? AND revision=? AND selection_epoch=? AND quote_id=? AND status IN ('pending','approved')",
                (chat_id, session["revision"], session.get("selection_epoch", 0), hotel["quote_id"])).fetchone()
            draft_id = row["id"] if row else uuid.uuid4().hex
            if not row:
                data = {"slots": session["slots"], "hotel": hotel, "unknown_preferences": hotel.get("unknown_preferences", [])}
                db.execute("INSERT INTO booking_drafts(id,chat_id,revision,selection_epoch,quote_id,expires,status,data) VALUES(?,?,?,?,?,?,'pending',?)",
                           (draft_id, chat_id, session["revision"], session.get("selection_epoch", 0), hotel["quote_id"], hotel["valid_until"], json.dumps(data, ensure_ascii=False)))
        draft = self.get(draft_id)
        return {key: draft[key] for key in ("id", "revision", "quote_id", "expires", "status", "receipt")}

    async def start(self, draft_id):
        checkpoint = await self.graph.aget_state(self.config(draft_id))
        if not checkpoint.values:
            await self.graph.ainvoke({"draft_id": draft_id}, self.config(draft_id))

    def review(self, state, config):
        # Pass the framework-provided runnable config explicitly for Python <3.11,
        # whose executor tasks do not automatically propagate this context.
        token = var_child_runnable_config.set(config)
        try:
            decision = interrupt({"draft_id": state["draft_id"], "action": "human_review", "scope": "booking_handoff_only"})
        finally:
            var_child_runnable_config.reset(token)
        return {"actor": decision["actor"]}

    async def handoff(self, state):
        draft = self.get(state["draft_id"])
        if self.validate_quote:
            await self.validate_quote(draft)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM booking_drafts WHERE id=?", (state["draft_id"],)).fetchone()
            current = db.execute("SELECT data FROM sessions WHERE chat_id=?", (row["chat_id"],)).fetchone()
            session = json.loads(current["data"]) if current else self.store.empty_session()
            if row["revision"] != session["revision"] or row["selection_epoch"] != session.get("selection_epoch", 0):
                raise ValueError("Approval invalidated by a requirement or selection change")
            if row["expires"] <= time.time():
                raise ValueError("Quote expired; human approval must be renewed")
            if row["receipt"]:
                return {"receipt": json.loads(row["receipt"])}
            receipt = {"id": "handoff-" + row["id"], "draft_id": row["id"], "actor": state["actor"],
                       "revision": row["revision"], "quote_id": row["quote_id"], "created": time.time(),
                       "kind": "approved_handoff", "supplier_order_created": False}
            db.execute("UPDATE booking_drafts SET status='approved',actor=?,receipt=? WHERE id=?",
                       (state["actor"], json.dumps(receipt, ensure_ascii=False), row["id"]))
            session["task_status"] = "approved_handoff"
            session["draft"] = {"id": row["id"], "status": "approved", "receipt": receipt,
                                "revision": row["revision"], "quote_id": row["quote_id"], "expires": row["expires"]}
            db.execute("UPDATE sessions SET data=?,updated=updated+0.000001 WHERE chat_id=?",
                       (json.dumps(session, ensure_ascii=False), row["chat_id"]))
        await self.store.cache(row["chat_id"], session)
        self.store.trace(row["chat_id"], row["id"], "human_approval", "success", detail={"actor": state["actor"], "kind": "handoff_only"})
        return {"receipt": receipt}

    async def approve(self, draft_id, actor, token):
        if not self.token or not isinstance(token, str) or not hmac.compare_digest(token, self.token):
            raise PermissionError("Authorized reviewer token required")
        if not actor or len(actor) > 100:
            raise ValueError("Reviewer identity required")
        draft = self.get(draft_id)
        # Revalidate before every resume and repeat; don't reuse an old approved draft after edits.
        session = await self.store.load(draft["chat_id"])
        if draft["revision"] != session["revision"] or draft["selection_epoch"] != session.get("selection_epoch", 0):
            raise ValueError("Approval invalidated by newer conditions")
        if draft["expires"] <= time.time():
            raise ValueError("Quote expired")
        if self.validate_quote:
            await self.validate_quote(draft)
        if draft["receipt"]:
            return draft["receipt"]
        snapshot = await self.graph.aget_state(self.config(draft_id))
        value = Command(resume={"actor": actor}) if snapshot.next == ("human_review",) else None
        result = await self.graph.ainvoke(value, self.config(draft_id))
        return result["receipt"]
