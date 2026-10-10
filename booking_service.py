"""Stateful booking workflow: LangGraph, Redis cache, SQLite checkpoints and traces."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from booking_slots import LABELS, extract_slots, validate_slots
from hotel_repository import HotelRepository
from session_store import SessionStore


class BookingState(TypedDict, total=False):
    chat_id: str
    message_id: str
    user_msg: str
    session: dict
    intent: str
    changed_slots: list
    missing: list
    errors: list
    retrieval: dict
    response: str
    today: str


class BookingService:
    """Single-process service; per-chat serial execution with bounded global concurrency."""
    def __init__(self, store: SessionStore, repository: HotelRepository, *, max_concurrency=4,
                 timeout=20, retries=2, slot_extractor=None, responder=None, intent_detector=None,
                 quote_provider=None, today=None):
        self.store, self.repository = store, repository
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout, self.retries = timeout, retries
        self.slot_extractor, self.responder, self.intent_detector = slot_extractor, responder, intent_detector
        self.today = today or date.today()
        self.quote_provider = quote_provider
        self.locks = {}
        self.lock_users = {}
        self.graph = None
        self._checkpoint_context = None

    async def __aenter__(self):
        checkpoint_path = str(Path(self.store.db_path).with_name("graph_checkpoints.sqlite"))
        self._checkpoint_context = AsyncSqliteSaver.from_conn_string(checkpoint_path)
        saver = await self._checkpoint_context.__aenter__()
        builder = StateGraph(BookingState)
        builder.add_node("intent_detection", self.traced("intent_detection", self.detect_intent))
        builder.add_node("slot_update", self.traced("slot_update", self.update_slots))
        builder.add_node("slot_validation", self.traced("slot_validation", self.validate))
        builder.add_node("hotel_retrieval", self.traced("hotel_retrieval", self.retrieve))
        if self.quote_provider:
            builder.add_node("live_price_query", self.traced("live_price_query", self.live_quotes))
        builder.add_node("reply_generation", self.traced("reply_generation", self.reply))
        builder.add_edge(START, "intent_detection")
        builder.add_edge("intent_detection", "slot_update")
        builder.add_edge("slot_update", "slot_validation")
        builder.add_conditional_edges("slot_validation", lambda state: "reply" if state["missing"] or state["errors"] or state["session"].get("selected_hotel") else "retrieve",
                                      {"reply": "reply_generation", "retrieve": "live_price_query" if self.quote_provider else "hotel_retrieval"})
        if self.quote_provider:
            builder.add_edge("live_price_query", "hotel_retrieval")
        builder.add_edge("hotel_retrieval", "reply_generation")
        builder.add_edge("reply_generation", END)
        self.graph = builder.compile(checkpointer=saver)
        return self

    async def __aexit__(self, *args):
        await self._checkpoint_context.__aexit__(*args)
        await self.store.close()

    def traced(self, name, handler):
        async def run(state):
            started = time.perf_counter()
            try:
                result = await handler(state)
                detail = {key: result[key] for key in ("intent", "changed_slots", "missing", "errors", "retrieval") if key in result}
                self.store.trace(state["chat_id"], state["message_id"], name, "success",
                                 round((time.perf_counter() - started) * 1000, 2), detail)
                return result
            except Exception as exc:
                # Keep diagnostics without persisting raw API headers or credentials.
                self.store.trace(state["chat_id"], state["message_id"], name, "error",
                                 round((time.perf_counter() - started) * 1000, 2),
                                 {"error_type": type(exc).__name__})
                raise
        return run

    async def detect_intent(self, state):
        text = state["user_msg"]
        if self.intent_detector:
            intent = await self.intent_detector(text)
        elif any(word in text for word in ("改", "换", "调整")):
            intent = "修改条件"
        elif any(word in text for word in ("订", "入住", "住")):
            intent = "下单预订"
        elif any(word in text for word in ("便宜", "优惠")):
            intent = "议价砍价"
        elif any(word in text for word in ("价", "预算", "钱")):
            intent = "查询价格"
        else:
            intent = "常规咨询"
        return {"intent": intent}

    async def update_slots(self, state):
        session = json.loads(json.dumps(state["session"]))
        old = session["slots"]
        patch = extract_slots(state["user_msg"], old, self.repository.cities(), self.today)
        if self.slot_extractor:
            extra = await self.slot_extractor(state["user_msg"], old)
            # Deterministic parsing takes precedence; the provider must return a validated patch.
            patch = {**extra, **patch}
        changed = [key for key, value in patch.items() if old.get(key) != value]
        session["slots"] = {**old, **patch}
        if changed:
            session["revision"] += 1
            session["candidates"] = []
            session["quotes"] = []
            session["selected_hotel"] = None
            session["candidate_revision"] = None
            session["task_status"] = "collecting"
            self.store.trace(state["chat_id"], state["message_id"], "candidate_invalidation", "success",
                             detail={"changed": changed, "revision": session["revision"]})
        elif session.get("candidate_revision") == session["revision"] and session["candidates"]:
            match = re.search(r"(?:选|订|要)?第?([一二三四五1-5])(?:个|家)", state["user_msg"])
            if match:
                ordinal = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
                index = ordinal.get(match[1], int(match[1]) if match[1].isdigit() else 0) - 1
                if 0 <= index < len(session["candidates"]):
                    session["selected_hotel"] = session["candidates"][index]
        await self.store.save(state["chat_id"], session)
        return {"session": session, "changed_slots": changed}

    async def validate(self, state):
        missing, errors = validate_slots(state["session"]["slots"], self.today)
        return {"missing": missing, "errors": errors}

    async def retrieve(self, state):
        session = json.loads(json.dumps(state["session"]))
        slots = session["slots"]
        query = slots["city"] + " " + state["user_msg"]
        for attempt in range(self.retries + 1):
            try:
                # Repository calls are synchronous; cancellation cannot stop a running worker.
                # Repository itself only performs bounded local SQL/vector work (no browser here).
                result = await asyncio.wait_for(asyncio.to_thread(self.repository.search, query, slots), self.timeout)
                break
            except (TimeoutError, asyncio.TimeoutError, OSError) as exc:
                self.store.trace(state["chat_id"], state["message_id"], "tool_retry", "retry",
                                 detail={"attempt": attempt + 1, "error_type": type(exc).__name__})
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(min(0.1 * (2 ** attempt), 1))
        session["candidates"] = result["candidates"]
        session["candidate_revision"] = session["revision"]
        session["task_status"] = "ready" if result["candidates"] else "no_match" if result["offers_known"] else "quote_pending"
        return {"session": session, "retrieval": result}

    async def live_quotes(self, state):
        session = json.loads(json.dumps(state["session"]))
        for attempt in range(self.retries + 1):
            try:
                quotes = await asyncio.wait_for(self.quote_provider(state), timeout=self.timeout)
                session["quotes"] = quotes
                self.store.trace(state["chat_id"], state["message_id"], "live_quote_result", "success",
                                 detail={"quote_count": len(quotes), "inventory_confirmed": False})
                return {"session": session}
            except (OSError, asyncio.TimeoutError) as exc:
                self.store.trace(state["chat_id"], state["message_id"], "tool_retry", "retry",
                                 detail={"attempt": attempt + 1, "error_type": type(exc).__name__})
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(min(0.1 * (2 ** attempt), 1))

    async def reply(self, state):
        session = json.loads(json.dumps(state["session"]))
        if state["errors"]:
            response = "；".join(state["errors"]) + "，请修改后再查询。"
            session["task_status"] = "collecting"
        elif state["missing"]:
            response = "已记录您的需求。请补充：" + "、".join(LABELS[key] for key in state["missing"]) + "。"
            session["task_status"] = "collecting"
        elif session.get("selected_hotel"):
            hotel = session["selected_hotel"]
            slots = session["slots"]
            session["task_status"] = "awaiting_confirmation"
            response = (f"已选择 {hotel['name']}，{slots['check_in']} 入住、{slots['check_out']} 离店，"
                        f"{slots['room_type']}。下一步请确认入住人数与预订联系人，由人工核验房态和最终报价后完成订单。")
            if "demo_fixture" in hotel.get("sources", ""):
                response += "本次是样例库存演示，不会创建真实订单。"
        elif session["candidates"]:
            offers = session["candidates"]
            response = "为您找到以下候选：\n" + "\n".join(
                f"{i}. {offer['name']}｜{offer['room_type']}｜每晚最高 ¥{offer['max_nightly_price']:.0f}｜{offer['nights']} 晚合计 ¥{offer['total_price']:.0f}"
                for i, offer in enumerate(offers, 1))
            if any("demo_fixture" in offer["sources"] for offer in offers):
                response += "\n以上为演示库存，不是实时可成交报价。"
            response += "\n请选择酒店；预订前需人工确认房态和订单信息。"
        elif session["task_status"] == "quote_pending":
            if session.get("quotes"):
                response = "查到以下网页报价，房态尚未核验：\n" + "\n".join(
                    f"{quote['name']}｜{quote['room_type']}｜{quote['nights']} 晚参考价 ¥{quote['total_price']:.0f}"
                    for quote in session["quotes"])
                response += "\n请选择酒店，预订前由人工确认房态与最终价格。"
            else:
                response = "已记录完整需求，但当前没有该日期的报价 / 房态数据，需要查价后再确认，不能仅凭酒店目录保证有房。"
        else:
            response = "当前城市、日期、预算和房型下没有匹配房源。您可以调整预算、日期或房型，我会重新检索。"
        if self.responder and not state["missing"] and not state["errors"]:
            response = await self.responder(state, response)
        session["history"] = (session["history"] + [{"role": "user", "content": state["user_msg"]},
                                                     {"role": "assistant", "content": response}])[-20:]
        return {"response": response, "session": session}

    @asynccontextmanager
    async def serial(self, chat_id):
        lock = self.locks.setdefault(chat_id, asyncio.Lock())
        self.lock_users[chat_id] = self.lock_users.get(chat_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self.lock_users[chat_id] -= 1
            if not self.lock_users[chat_id]:
                del self.lock_users[chat_id]
                self.locks.pop(chat_id, None)

    async def handle(self, chat_id: str, message_id: str, text: str):
        if self.graph is None:
            raise RuntimeError("Use BookingService as an async context manager")
        if not chat_id or not message_id or not text.strip():
            raise ValueError("chat_id, message_id and text are required")
        async with self.serial(chat_id):
            previous = self.store.request(chat_id, message_id)
            if previous and previous["status"] == "superseded":
                raise ValueError("This interrupted task was superseded by a newer message")
            if previous and previous["text"] != text:
                raise ValueError("A message ID cannot be reused for different content")
            if previous and previous["status"] == "completed":
                self.store.trace(chat_id, message_id, "message_dedup", "skipped")
                return {**json.loads(previous["result"]), "session": await self.store.load(chat_id),
                        "changed_slots": [], "duplicate": True}
            async with self.semaphore:
                config = {"configurable": {"thread_id": hashlib.sha256((chat_id + "\0" + message_id).encode()).hexdigest()}}
                checkpoint = await self.graph.aget_state(config)
                self.store.begin(chat_id, message_id, text)
                try:
                    if previous and checkpoint.values:
                        self.store.trace(chat_id, message_id, "checkpoint_resume", "success", detail={"next": list(checkpoint.next)})
                        state = await self.graph.ainvoke(None, config) if checkpoint.next else checkpoint.values
                    else:
                        session = await self.store.load(chat_id)
                        state = await self.graph.ainvoke({"chat_id": chat_id, "message_id": message_id,
                            "user_msg": text, "session": session, "today": self.today.isoformat()}, config)
                    result = {"response": state["response"], "intent": state["intent"],
                              "session": state["session"], "retrieval": state.get("retrieval", {}),
                              "changed_slots": state.get("changed_slots", []), "duplicate": False}
                    await self.store.complete(chat_id, message_id, state["session"], result)
                    return result
                except Exception:
                    self.store.failed(chat_id, message_id)
                    raise
