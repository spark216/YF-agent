"""Stateful booking workflow: LangGraph, Redis cache, SQLite checkpoints and traces."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from functools import partial
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from booking_slots import LABELS, extract_slots, validate_slots
from hotel_repository import HotelRepository
from session_store import SessionStore
from requirement_contracts import propose, apply_proposal, RevisionConflict
from evidence_store import EvidenceStore
from constraint_negotiator import ConstraintNegotiator
from clarification_policy import clarify
from booking_approval import BookingApproval


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
    accepted_proposal: dict


class BookingService:
    """Single-process service; per-chat serial execution with bounded global concurrency."""
    def __init__(self, store: SessionStore, repository: HotelRepository, *, max_concurrency=4,
                 timeout=20, retries=2, slot_extractor=None, responder=None, intent_detector=None,
                 quote_provider=None, today=None, reviewer_token=""):
        self.store, self.repository = store, repository
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.ingress_semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout, self.retries = timeout, retries
        self.slot_extractor, self.responder, self.intent_detector = slot_extractor, responder, intent_detector
        self.today = today or date.today()
        self.quote_provider = quote_provider
        self.locks = {}
        self.lock_users = {}
        self.graph = None
        self._checkpoint_context = None
        self.inflight = {}
        self.evidence = EvidenceStore(repository)
        self.negotiator = ConstraintNegotiator(repository, self.evidence)
        self.reviewer_token = reviewer_token

    async def __aenter__(self):
        checkpoint_path = str(Path(self.store.db_path).with_name("graph_checkpoints.sqlite"))
        self._checkpoint_context = AsyncSqliteSaver.from_conn_string(checkpoint_path)
        saver = await self._checkpoint_context.__aenter__()
        self.approvals = BookingApproval(self.store, self.reviewer_token, saver, self.validate_draft_quote)
        builder = StateGraph(BookingState)
        builder.add_node("slot_validation", self.traced("slot_validation", self.validate))
        builder.add_node("hotel_retrieval", self.traced("hotel_retrieval", self.retrieve))
        if self.quote_provider:
            builder.add_node("live_price_query", self.traced("live_price_query", self.live_quotes))
        builder.add_node("reply_generation", self.traced("reply_generation", self.reply))
        builder.add_edge(START, "slot_validation")
        builder.add_conditional_edges("slot_validation", lambda state: "reply" if state["missing"] or state["errors"] or state["session"].get("selected_hotel") or state["session"].get("pending_proposal") else "retrieve",
                                      {"reply": "reply_generation", "retrieve": "live_price_query" if self.quote_provider else "hotel_retrieval"})
        if self.quote_provider:
            builder.add_edge("live_price_query", "hotel_retrieval")
        builder.add_edge("hotel_retrieval", "reply_generation")
        builder.add_edge("reply_generation", END)
        self.graph = builder.compile(checkpointer=saver)
        return self

    async def __aexit__(self, *args):
        if self.inflight:
            await asyncio.gather(*self.inflight.values(), return_exceptions=True)
        await self._checkpoint_context.__aexit__(*args)
        await self.store.close()

    async def validate_draft_quote(self, draft):
        session = await self.store.load(draft["chat_id"])
        result = await asyncio.wait_for(asyncio.to_thread(self.repository.search,
            draft["data"]["hotel"]["name"], draft["data"]["slots"], self.repository.count()), self.timeout)
        matching = [hotel for hotel in result["candidates"] if hotel["quote_id"] == draft["quote_id"]]
        accepted, _, _ = self.evidence.rank(matching, session.get("preferences", []), session["revision"])
        if not accepted:
            raise ValueError("Quote or required evidence changed; refresh before approval")

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
        session = {**self.store.empty_session(), **json.loads(json.dumps(state["session"]))}
        proposal = state.get("accepted_proposal")
        if not proposal:
            model = await self.slot_extractor(state["user_msg"], session["slots"]) if self.slot_extractor else None
            proposal = propose(state["user_msg"], session, self.repository.cities(), self.today, model)
        if proposal["needs_confirmation"] and not state.get("accepted_proposal"):
            session["pending_proposal"] = proposal
            session["task_status"] = "confirming_requirements"
            changed = []
        elif not proposal["operations"] and session.get("pending_proposal") and state["user_msg"].strip() not in {"取消提案", "拒绝提案"}:
            changed = []
        else:
            session, changed = apply_proposal(session, proposal)
        if changed:
            self.store.trace(state["chat_id"], state["message_id"], "candidate_invalidation", "success",
                             detail={"changed": changed, "revision": session["revision"]})
        elif not session.get("pending_proposal") and session.get("candidate_revision") == session["revision"] and session["candidates"]:
            match = re.search(r"(?:选|订|要)?第?([一二三四五1-5])(?:个|家)", state["user_msg"])
            if match:
                ordinal = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
                index = ordinal.get(match[1], int(match[1]) if match[1].isdigit() else 0) - 1
                if 0 <= index < len(session["candidates"]):
                    chosen = session["candidates"][index]
                    if chosen["valid_until"] <= time.time():
                        raise ValueError("Quote expired; refresh before choosing")
                    session["selected_hotel"] = chosen
                    session["selection_epoch"] += 1
        session["history"] = (session["history"] + [{"role": "user", "content": state["user_msg"]}])[-20:]
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
                search = partial(self.repository.search, limit=self.repository.count()) if session.get("preferences") else self.repository.search
                result = await asyncio.wait_for(asyncio.to_thread(search, query, slots), self.timeout)
                break
            except (TimeoutError, asyncio.TimeoutError, OSError) as exc:
                self.store.trace(state["chat_id"], state["message_id"], "tool_retry", "retry",
                                 detail={"attempt": attempt + 1, "error_type": type(exc).__name__})
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(min(0.1 * (2 ** attempt), 1))
        hotels, uncertain, excluded = self.evidence.rank(result["candidates"], session.get("preferences", []), session["revision"])
        session["candidates"] = hotels[:5]
        session["verification_candidates"] = uncertain[:5]
        session["claims"] = [claim for hotel in hotels[:5] for claim in hotel["claims"]]
        result["candidates"] = hotels[:5]
        result["unknown_hard_preferences"] = len(uncertain)
        result["preference_exclusions"] = excluded
        session["candidate_revision"] = session["revision"]
        session["task_status"] = "ready" if hotels else "quote_pending" if uncertain or not result["offers_known"] else "no_match"
        session["relaxations"] = []
        if session["task_status"] == "no_match":
            diagnosis = await asyncio.to_thread(self.negotiator.diagnose, query, session)
            session["relaxations"] = diagnosis["proposals"]
            result["constraint_report"] = diagnosis
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
        if session.get("pending_proposal"):
            response = "我对本轮需求有推断，请先确认以下变更：" + json.dumps(session["pending_proposal"]["operations"], ensure_ascii=False)
            session["task_status"] = "confirming_requirements"
        elif state["errors"] or state["missing"]:
            question = clarify(state["missing"], state["errors"], session)
            session["clarification"] = question
            response = question["question"]
            session["task_status"] = "collecting"
        elif session.get("selected_hotel"):
            hotel = session["selected_hotel"]
            slots = session["slots"]
            session["task_status"] = "awaiting_confirmation"
            session["draft"] = await self.approvals.create(session, state["chat_id"])
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
            for offer in offers:
                valid = self.evidence.validate_claims(offer, session["revision"])
                if valid:
                    response += "\n" + offer["name"] + "：" + "；".join(claim["text"] for claim in valid)
                if offer.get("unknown_preferences"):
                    response += "\n" + offer["name"] + "：部分偏好缺少已核验资料，不能保证满足。"
        elif session["task_status"] == "quote_pending":
            if session.get("verification_candidates"):
                response = "有基础条件匹配的酒店，但您的必需偏好缺少已核验证据。请交由客服核验，当前不能承诺满足。"
            elif session.get("quotes"):
                response = "查到以下网页报价，房态尚未核验：\n" + "\n".join(
                    f"{quote['name']}｜{quote['room_type']}｜{quote['nights']} 晚参考价 ¥{quote['total_price']:.0f}"
                    for quote in session["quotes"])
                response += "\n请选择酒店，预订前由人工确认房态与最终价格。"
            else:
                response = "已记录完整需求，但当前没有该日期的报价 / 房态数据，需要查价后再确认，不能仅凭酒店目录保证有房。"
        else:
            response = "当前城市、日期、预算和房型下没有匹配房源。您可以调整预算、日期或房型，我会重新检索。"
            for proposal in session.get("relaxations", []):
                response += f"\n可选调整：{proposal['changes']}，可检索到 {proposal['candidate_count']} 家候选。接受后才会修改需求。"
        if self.responder and not state["missing"] and not state["errors"]:
            response = await self.responder(state, response)
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

    async def handle(self, chat_id: str, message_id: str, text: str, *, expected_revision=None, accepted_proposal=None):
        if self.graph is None:
            raise RuntimeError("Use BookingService as an async context manager")
        if not chat_id or not message_id or not text.strip():
            raise ValueError("chat_id, message_id and text are required")
        key = (chat_id, message_id)
        duplicate = False
        async with self.serial(chat_id):
            previous = self.store.request(chat_id, message_id)
            if previous and previous["text"] != text:
                raise ValueError("A message ID cannot be reused for different content")
            if previous and previous["status"] == "superseded":
                raise ValueError("This interrupted task was superseded by a newer message")
            if previous and previous["status"] == "completed":
                self.store.trace(chat_id, message_id, "message_dedup", "skipped")
                return {**json.loads(previous["result"]), "session": await self.store.load(chat_id),
                        "changed_slots": [], "duplicate": True}
            if key in self.inflight:
                task = self.inflight[key]
                duplicate = True
            else:
                self.store.begin(chat_id, message_id, text)
                try:
                    if previous and previous.get("prepared"):
                        prepared = json.loads(previous["prepared"])
                        current = await self.store.load(chat_id)
                        if (current["revision"] != prepared["session"]["revision"] or
                            current.get("turn_epoch", 0) != prepared["session"].get("turn_epoch", 0)):
                            raise RevisionConflict("Interrupted task belongs to older conditions")
                    else:
                        current = await self.store.load(chat_id)
                        if expected_revision is not None and current["revision"] != expected_revision:
                            raise RevisionConflict("Requirement version changed; reload before editing")
                        prepared = {"chat_id": chat_id, "message_id": message_id, "user_msg": text,
                                    "session": current, "today": self.today.isoformat()}
                        if accepted_proposal:
                            prepared["accepted_proposal"] = accepted_proposal
                        async with self.ingress_semaphore:
                            prepared.update(await self.traced("intent_detection", self.detect_intent)(prepared))
                            prepared.update(await self.traced("slot_update", self.update_slots)(prepared))
                        await self.store.prepare(chat_id, message_id, prepared["session"], prepared, current["revision"])
                    task = asyncio.create_task(self.execute(prepared, bool(previous)))
                    self.inflight[key] = task
                    def cleanup(completed):
                        if self.inflight.get(key) is completed:
                            self.inflight.pop(key, None)
                        if not completed.cancelled():
                            completed.exception()
                    task.add_done_callback(cleanup)
                except Exception:
                    self.store.failed(chat_id, message_id)
                    raise
        # Waiting for a tool does not hold the ingress lock. New conditions can commit.
        result = await asyncio.shield(task)
        if duplicate:
            self.store.trace(chat_id, message_id, "message_dedup", "skipped")
            result = {**result, "session": await self.store.load(chat_id), "duplicate": True}
        return result

    async def execute(self, prepared, recover):
        chat_id, message_id = prepared["chat_id"], prepared["message_id"]
        async with self.semaphore:
            config = {"configurable": {"thread_id": hashlib.sha256((chat_id + "\0" + message_id).encode()).hexdigest()}}
            try:
                checkpoint = await self.graph.aget_state(config)
                if recover and checkpoint.values:
                    self.store.trace(chat_id, message_id, "checkpoint_resume", "success", detail={"next": list(checkpoint.next)})
                    state = await self.graph.ainvoke(None, config) if checkpoint.next else checkpoint.values
                else:
                    state = await self.graph.ainvoke(prepared, config)
                if state["session"].get("draft"):
                    await self.approvals.start(state["session"]["draft"]["id"])
                result = {"response": state["response"], "intent": state["intent"], "session": state["session"],
                          "retrieval": state.get("retrieval", {}), "changed_slots": state.get("changed_slots", []), "duplicate": False}
                result = await self.store.publish(chat_id, message_id, result)
                if result["stale"]:
                    self.store.trace(chat_id, message_id, "stale_result_discarded", "skipped",
                                     detail={"worker_revision": state["session"]["revision"], "current_revision": result["session"]["revision"]})
                return result
            except Exception:
                self.store.failed(chat_id, message_id)
                raise

    async def commit_card(self, chat_id, message_id, expected_revision, changes):
        text = "条件卡片修改：" + json.dumps(changes, ensure_ascii=False, sort_keys=True)
        session = await self.store.load(chat_id)
        proposal = {"id": hashlib.sha256(text.encode()).hexdigest(), "base_revision": expected_revision,
                    "text": text, "needs_confirmation": False,
                    "operations": [{"op": "unset" if value is None else "set", "field": field,
                                    "value": value, "origin": "explicit", "evidence": text}
                                   for field, value in changes.items()]}
        return await self.handle(chat_id, message_id, text, expected_revision=expected_revision, accepted_proposal=proposal)

    async def accept_pending(self, chat_id, message_id, proposal_id):
        replay = await self.replay_decision(chat_id, message_id, proposal_id)
        if replay:
            return replay
        session = await self.store.load(chat_id)
        proposal = session.get("pending_proposal")
        if not proposal or proposal["id"] != proposal_id:
            raise ValueError("Unknown or invalidated requirement proposal")
        return await self.handle(chat_id, message_id, "确认需求提案 " + proposal_id,
                                 expected_revision=proposal["base_revision"], accepted_proposal=proposal)

    async def accept_relaxation(self, chat_id, message_id, proposal_id):
        replay = await self.replay_decision(chat_id, message_id, proposal_id)
        if replay:
            return replay
        session = await self.store.load(chat_id)
        proposal = next((item for item in session.get("relaxations", []) if item["id"] == proposal_id), None)
        if not proposal or proposal["expires"] <= time.time() or proposal["base_revision"] != session["revision"]:
            raise ValueError("Relaxation expired or was invalidated")
        text = "接受调整：" + json.dumps(proposal["changes"], ensure_ascii=False, sort_keys=True)
        accepted = {"id": proposal_id, "base_revision": proposal["base_revision"], "text": text,
                    "needs_confirmation": False,
                    "operations": [{"op": "set", "field": key, "value": value, "evidence": text, "origin": "explicit"}
                                   for key, value in proposal["changes"].items()]}
        return await self.handle(chat_id, message_id, text, expected_revision=session["revision"], accepted_proposal=accepted)

    async def replay_decision(self, chat_id, message_id, proposal_id):
        previous = self.store.request(chat_id, message_id)
        if previous and previous.get("prepared"):
            prepared = json.loads(previous["prepared"])
            if prepared.get("accepted_proposal", {}).get("id") != proposal_id:
                raise ValueError("A message ID cannot be reused for a different proposal")
            return await self.handle(chat_id, message_id, previous["text"])
        return None

    async def approve_draft(self, chat_id, draft_id, actor, token):
        async with self.serial(chat_id):
            if self.approvals.get(draft_id)["chat_id"] != chat_id:
                raise PermissionError("Draft belongs to a different conversation")
            return await self.approvals.approve(draft_id, actor, token)
