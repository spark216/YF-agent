"""Run against an actual Redis service in CI or an explicitly configured local instance."""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from session_store import SessionStore


@pytest.mark.skipif(not os.getenv("REDIS_TEST_URL"), reason="Requires explicit REDIS_TEST_URL; CI provides Redis 7.4")
async def test_real_redis_ttl_and_durable_rehydration(tmp_path):
    store = SessionStore(str(tmp_path / "real-redis.sqlite"), redis_url=os.environ["REDIS_TEST_URL"], ttl=1)
    chat_id = "redis-integration-" + uuid.uuid4().hex
    try:
        session = store.empty_session()
        session["slots"] = {"city": "上海", "budget": 300}
        await store.save(chat_id, session)
        assert store.cache_status == "redis", "Redis service must be reachable"
        assert await store.redis.exists(store.key(chat_id))
        await asyncio.sleep(1.1)
        assert not await store.redis.exists(store.key(chat_id))
        restored = await store.load(chat_id)
        assert restored["slots"] == session["slots"]
        assert store.cache_status == "redis"
        assert await store.redis.exists(store.key(chat_id))
    finally:
        await store.redis.delete(store.key(chat_id))
        await store.close()
