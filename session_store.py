"""Durable SQLite state with an optional Redis cache for active conversations."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any


class SessionStore:
    def __init__(self, db_path: str, redis_url: str = "", ttl: int = 1800, redis_client=None):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.redis = redis_client
        if redis_url and self.redis is None:
            from redis.asyncio import Redis
            self.redis = Redis.from_url(redis_url, decode_responses=True,
                                        socket_connect_timeout=0.3, socket_timeout=0.3)
        self.cache_status = "disabled" if self.redis is None else "connecting"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    chat_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    chat_id TEXT NOT NULL, message_id TEXT NOT NULL, text TEXT NOT NULL,
                    status TEXT NOT NULL, result TEXT, attempts INTEGER DEFAULT 0,
                    updated REAL NOT NULL, PRIMARY KEY(chat_id,message_id));
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, message_id TEXT,
                    node TEXT, status TEXT, duration_ms REAL, detail TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS deliveries (
                    chat_id TEXT, message_id TEXT, sent REAL,
                    PRIMARY KEY(chat_id,message_id));
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
            if "prepared" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN prepared TEXT")
            if "revision" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN revision INTEGER")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def empty_session():
        return {"slots": {}, "history": [], "revision": 0, "candidates": [],
                "task_status": "collecting", "candidate_revision": None, "selected_hotel": None, "quotes": [],
                "preferences": [], "slot_sources": {}, "claims": [], "relaxations": [],
                "pending_proposal": None, "verification_candidates": [], "selection_epoch": 0, "draft": None}

    def key(self, chat_id):
        # Encode arbitrary platform identifiers without Redis key collisions.
        return "yf:session:" + chat_id.encode("utf-8").hex()

    async def load(self, chat_id: str) -> dict:
        with self.connect() as db:
            durable = db.execute("SELECT updated FROM sessions WHERE chat_id=?", (chat_id,)).fetchone()
        if self.redis is not None:
            try:
                value = await self.redis.get(self.key(chat_id))
                self.cache_status = "redis"
                if value:
                    envelope = json.loads(value)
                    if durable and envelope.get("updated") == durable["updated"]:
                        await self.redis.expire(self.key(chat_id), self.ttl)
                        return envelope["data"]
            except Exception:
                self.cache_status = "sqlite_fallback"
        with self.connect() as db:
            row = db.execute("SELECT data FROM sessions WHERE chat_id=?", (chat_id,)).fetchone()
        session = json.loads(row["data"]) if row else self.empty_session()
        await self.cache(chat_id, session)
        return session

    async def cache(self, chat_id: str, session: dict):
        if self.redis is not None:
            try:
                with self.connect() as db:
                    row = db.execute("SELECT data,updated FROM sessions WHERE chat_id=?", (chat_id,)).fetchone()
                envelope = {"data": json.loads(row["data"]) if row else session, "updated": row["updated"] if row else None}
                await self.redis.set(self.key(chat_id), json.dumps(envelope, ensure_ascii=False), ex=self.ttl)
                self.cache_status = "redis"
            except Exception:
                self.cache_status = "sqlite_fallback"

    async def save(self, chat_id: str, session: dict):
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?) ON CONFLICT(chat_id) DO UPDATE SET data=excluded.data,updated=MAX(sessions.updated+0.000001,excluded.updated)",
                       (chat_id, json.dumps(session, ensure_ascii=False), time.time()))
        await self.cache(chat_id, session)

    def request(self, chat_id: str, message_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM requests WHERE chat_id=? AND message_id=?", (chat_id, message_id)).fetchone()
        return dict(row) if row else None

    def begin(self, chat_id: str, message_id: str, text: str):
        with self.connect() as db:
            db.execute("INSERT INTO requests(chat_id,message_id,text,status,updated) VALUES(?,?,?,'running',?) "
                       "ON CONFLICT(chat_id,message_id) DO UPDATE SET status='running',updated=excluded.updated,attempts=requests.attempts+1",
                       (chat_id, message_id, text, time.time()))

    async def prepare(self, chat_id, message_id, session, state, expected_revision):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM sessions WHERE chat_id=?", (chat_id,)).fetchone()
            current = json.loads(row["data"]) if row else self.empty_session()
            if current["revision"] != expected_revision:
                from requirement_contracts import RevisionConflict
                raise RevisionConflict("Requirement version changed before commit")
            session["turn_epoch"] = current.get("turn_epoch", 0) + 1
            db.execute("INSERT INTO sessions VALUES(?,?,?) ON CONFLICT(chat_id) DO UPDATE SET data=excluded.data,updated=MAX(sessions.updated+0.000001,excluded.updated)",
                       (chat_id, json.dumps(session, ensure_ascii=False), time.time()))
            db.execute("UPDATE requests SET prepared=?,revision=? WHERE chat_id=? AND message_id=?",
                       (json.dumps(state, ensure_ascii=False), session["revision"], chat_id, message_id))
            db.execute("UPDATE requests SET status='superseded' WHERE chat_id=? AND message_id<>? AND status IN ('running','failed')",
                       (chat_id, message_id))
        await self.cache(chat_id, session)

    async def publish(self, chat_id, message_id, result):
        snapshot = result["session"]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM sessions WHERE chat_id=?", (chat_id,)).fetchone()
            current = json.loads(row["data"]) if row else self.empty_session()
            stale = (row is None or current["revision"] != snapshot["revision"] or
                     current.get("turn_epoch", 0) != snapshot.get("turn_epoch", 0) or
                     current.get("selection_epoch", 0) != snapshot.get("selection_epoch", 0))
            if stale:
                db.execute("UPDATE requests SET status='superseded',updated=? WHERE chat_id=? AND message_id=?",
                           (time.time(), chat_id, message_id))
                return {**result, "session": current, "stale": True, "response": "旧条件下的结果已丢弃，请查看最新需求。"}
            for key in ("candidates", "candidate_revision", "quotes", "claims", "verification_candidates", "relaxations", "task_status", "draft", "clarification"):
                if key in snapshot:
                    current[key] = snapshot[key]
            current["history"] = (current["history"] + [{"role": "assistant", "content": result["response"]}])[-20:]
            result = {**result, "session": current, "stale": False}
            db.execute("UPDATE sessions SET data=?,updated=updated+0.000001 WHERE chat_id=?",
                       (json.dumps(current, ensure_ascii=False), chat_id))
            db.execute("UPDATE requests SET status='completed',result=?,updated=? WHERE chat_id=? AND message_id=?",
                       (json.dumps(result, ensure_ascii=False), time.time(), chat_id, message_id))
        await self.cache(chat_id, current)
        return result

    def failed(self, chat_id, message_id):
        with self.connect() as db:
            db.execute("UPDATE requests SET status='failed',updated=? WHERE chat_id=? AND message_id=? AND status<>'superseded'",
                       (time.time(), chat_id, message_id))

    def delivery_sent(self, chat_id, message_id):
        with self.connect() as db:
            return bool(db.execute("SELECT 1 FROM deliveries WHERE chat_id=? AND message_id=?", (chat_id, message_id)).fetchone())

    def mark_delivery(self, chat_id, message_id):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO deliveries VALUES(?,?,?)", (chat_id, message_id, time.time()))

    async def complete(self, chat_id: str, message_id: str, session: dict, result: dict):
        # Session and dedup result commit atomically; Redis is never the source of truth.
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?) ON CONFLICT(chat_id) DO UPDATE SET data=excluded.data,updated=MAX(sessions.updated+0.000001,excluded.updated)",
                       (chat_id, json.dumps(session, ensure_ascii=False), time.time()))
            db.execute("UPDATE requests SET status='completed',result=?,updated=? WHERE chat_id=? AND message_id=?",
                       (json.dumps(result, ensure_ascii=False), time.time(), chat_id, message_id))
        await self.cache(chat_id, session)

    def trace(self, chat_id, message_id, node, status, duration_ms=0, detail=None):
        with self.connect() as db:
            db.execute("INSERT INTO traces(chat_id,message_id,node,status,duration_ms,detail,created) VALUES(?,?,?,?,?,?,?)",
                       (chat_id, message_id, node, status, duration_ms,
                        json.dumps(detail or {}, ensure_ascii=False, default=str), time.time()))

    def traces(self, chat_id, message_id=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM traces WHERE chat_id=?" +
                              (" AND message_id=?" if message_id else "") + " ORDER BY id",
                              (chat_id, message_id) if message_id else (chat_id,)).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail"])} for row in rows]

    async def reset(self, chat_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM sessions WHERE chat_id=?", (chat_id,)).fetchone()
            session = self.empty_session()
            session["revision"] = (json.loads(row["data"])["revision"] if row else 0) + 1
            db.execute("INSERT INTO sessions VALUES(?,?,?) ON CONFLICT(chat_id) DO UPDATE SET data=excluded.data,updated=MAX(sessions.updated+0.000001,excluded.updated)",
                       (chat_id, json.dumps(session, ensure_ascii=False), time.time()))
            db.execute("UPDATE requests SET status='superseded' WHERE chat_id=? AND status IN ('running','failed')", (chat_id,))
        await self.cache(chat_id, session)

    async def close(self):
        if self.redis is not None:
            try:
                await self.redis.aclose()
            except Exception:
                self.cache_status = "sqlite_fallback"
