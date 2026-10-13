from __future__ import annotations

import asyncio
from datetime import date, timedelta

import fakeredis.aioredis
import pytest

from booking_service import BookingService
from booking_slots import extract_slots, validate_slots
from hotel_repository import HotelRepository
from session_store import SessionStore

TODAY = date(2030, 1, 1)
FULL = "上海，2030-01-02入住，2030-01-04离店，预算400元，大床房"


def repository(tmp_path):
    repo = HotelRepository(str(tmp_path / "hotels.sqlite"))
    with repo.connect() as db:
        db.executemany("INSERT INTO hotels VALUES(?,?,?,?,?)", [
            ("sh1", "上海测试全季", "上海", "1", "全季"),
            ("sh2", "上海测试汉庭", "上海", "1", "汉庭"),
            ("bj1", "北京测试全季", "北京", "2", "全季")])
    for offset in (1, 2, 3, 4):
        night = (TODAY + timedelta(days=offset)).isoformat()
        for hotel_id in ("sh1", "sh2", "bj1"):
            repo.put_offer(hotel_id, night, "大床房", 300 if hotel_id != "sh2" else 350, 2, "demo_fixture")
            repo.put_offer(hotel_id, night, "双床房", 280, 1, "demo_fixture")
    return repo


def service(tmp_path, repo=None, **kwargs):
    return BookingService(SessionStore(str(tmp_path / "sessions.sqlite")), repo or repository(tmp_path), today=TODAY, **kwargs)


async def test_missing_slots_do_not_retrieve(tmp_path):
    async with service(tmp_path) as agent:
        result = await agent.handle("c", "1", "想去上海住酒店")
        assert result["session"]["task_status"] == "collecting"
        assert "入住" in result["response"]
        assert result["session"]["clarification"]["field"] == "check_in"
        assert "hotel_retrieval" not in [trace["node"] for trace in agent.store.traces("c")]


async def test_complete_slots_and_full_night_filter(tmp_path):
    async with service(tmp_path) as agent:
        result = await agent.handle("c", "1", FULL)
        assert len(result["session"]["candidates"]) == 2
        assert all(row["nights"] == 2 for row in result["session"]["candidates"])
        assert result["session"]["candidate_revision"] == result["session"]["revision"]


@pytest.mark.parametrize("change,field,expected", [
    ("改去北京", "city", "北京"),
    ("预算改成320元", "budget", 320),
    ("换双床房", "room_type", "双床房"),
    ("改到2030-01-03入住", "check_in", "2030-01-03")])
async def test_changes_invalidate_candidates(tmp_path, change, field, expected):
    async with service(tmp_path) as agent:
        first = await agent.handle("c", "1", FULL)
        second = await agent.handle("c", "2", change)
        assert second["session"]["slots"][field] == expected
        assert second["session"]["revision"] > first["session"]["revision"]
        assert "candidate_invalidation" in [trace["node"] for trace in agent.store.traces("c", "2")]
        if field == "city":
            assert all(row["city"] == "北京" for row in second["session"]["candidates"])


async def test_no_match_for_small_budget(tmp_path):
    async with service(tmp_path) as agent:
        result = await agent.handle("c", "1", FULL.replace("400", "100"))
        assert result["session"]["task_status"] == "no_match"
        assert not result["session"]["candidates"]


async def test_missing_night_and_sold_out_not_recommended(tmp_path):
    repo = repository(tmp_path)
    with repo.connect() as db:
        db.execute("DELETE FROM offers WHERE hotel_id='sh1' AND night='2030-01-03'")
        db.execute("UPDATE offers SET remaining=0 WHERE hotel_id='sh2'")
    async with service(tmp_path, repo) as agent:
        result = await agent.handle("c", "1", FULL)
        assert result["session"]["candidates"] == []


async def test_duplicate_message_does_not_append_history(tmp_path):
    async with service(tmp_path) as agent:
        first, second = await asyncio.gather(agent.handle("c", "1", FULL), agent.handle("c", "1", FULL))
        assert first["duplicate"] is False and second["duplicate"] is True
        session = await agent.store.load("c")
        assert len(session["history"]) == 2


async def test_message_id_content_collision_rejected(tmp_path):
    async with service(tmp_path) as agent:
        await agent.handle("c", "1", FULL)
        with pytest.raises(ValueError):
            await agent.handle("c", "1", "换双床房")


async def test_consecutive_messages_serialized(tmp_path):
    async with service(tmp_path) as agent:
        await asyncio.gather(agent.handle("c", "1", FULL), agent.handle("c", "2", "换双床房"))
        state = await agent.store.load("c")
        assert state["slots"]["room_type"] == "双床房"
        assert sum(message["role"] == "user" for message in state["history"]) == 2
        assert state["history"][-1]["role"] == "assistant"
        assert not agent.locks


async def test_global_semaphore_limits_active_work(tmp_path):
    active = 0
    peak = 0

    async def detector(text):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return "常规咨询"

    async with service(tmp_path, max_concurrency=2, intent_detector=detector) as agent:
        await asyncio.gather(*(agent.handle(str(i), "1", "上海") for i in range(8)))
        assert peak == 2


async def test_transient_tool_error_retried(tmp_path):
    repo = repository(tmp_path)
    original = repo.search
    attempts = 0

    def flaky(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary failure")
        return original(*args)

    repo.search = flaky
    async with service(tmp_path, repo, retries=1) as agent:
        result = await agent.handle("c", "1", FULL)
        assert result["session"]["task_status"] == "ready"
        assert attempts == 2
        assert any(trace["node"] == "tool_retry" for trace in agent.store.traces("c"))


async def test_restart_resumes_failed_graph_node(tmp_path):
    repo = repository(tmp_path)
    original = repo.search

    def fail(*args):
        raise OSError("tool unavailable")

    repo.search = fail
    async with service(tmp_path, repo, retries=0) as agent:
        with pytest.raises(OSError):
            await agent.handle("c", "1", FULL)
    repo.search = original
    async with service(tmp_path, repo) as restarted:
        result = await restarted.handle("c", "1", FULL)
        assert result["session"]["task_status"] == "ready"
        traces = restarted.store.traces("c", "1")
        assert any(trace["node"] == "checkpoint_resume" for trace in traces)
        assert sum(trace["node"] == "intent_detection" for trace in traces) == 1


async def test_redis_cache_ttl_and_sqlite_rehydration(tmp_path):
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = SessionStore(str(tmp_path / "cache.sqlite"), ttl=60, redis_client=cache)
    state = store.empty_session()
    state["slots"] = {"city": "上海"}
    await store.save("c", state)
    assert 0 < await cache.ttl(store.key("c")) <= 60
    assert (await store.load("c"))["slots"]["city"] == "上海"
    await cache.delete(store.key("c"))
    assert (await store.load("c"))["slots"]["city"] == "上海"
    assert await cache.exists(store.key("c"))
    await store.close()


async def test_redis_outage_falls_back_to_durable_state(tmp_path):
    class BrokenRedis:
        async def get(self, key): raise ConnectionError()
        async def set(self, *args, **kwargs): raise ConnectionError()
        async def aclose(self): pass

    store = SessionStore(str(tmp_path / "fallback.sqlite"), redis_client=BrokenRedis())
    state = store.empty_session()
    state["slots"] = {"budget": 300}
    await store.save("c", state)
    assert (await store.load("c"))["slots"]["budget"] == 300
    assert store.cache_status == "sqlite_fallback"


@pytest.mark.parametrize("message", ["2030-02-30入住", "2030-01-04入住2030-01-02离店", "2020-01-01入住2020-01-02离店"])
def test_invalid_dates(message):
    patch = extract_slots(message, {}, ["上海"], TODAY)
    _, errors = validate_slots(patch, TODAY)
    assert errors


def test_relative_date_change_preserves_stay_length():
    patch = extract_slots("改到后天", {"check_in": "2030-01-02", "check_out": "2030-01-04"}, ["上海"], TODAY)
    assert patch["check_in"] == "2030-01-03"
    assert patch["check_out"] == "2030-01-05"


def test_departure_only_change_preserves_arrival():
    patch = extract_slots("离店改到2030-01-05", {"check_in": "2030-01-02", "check_out": "2030-01-04"}, ["上海"], TODAY)
    assert "check_in" not in patch
    assert patch["check_out"] == "2030-01-05"


async def test_old_failed_task_cannot_overwrite_newer_slots(tmp_path):
    repo = repository(tmp_path)
    original = repo.search
    repo.search = lambda *args: (_ for _ in ()).throw(OSError("failure"))
    async with service(tmp_path, repo, retries=0) as agent:
        with pytest.raises(OSError):
            await agent.handle("c", "1", FULL)
        repo.search = original
        await agent.handle("c", "2", "改去北京")
        with pytest.raises(ValueError, match="superseded"):
            await agent.handle("c", "1", FULL)
        assert (await agent.store.load("c"))["slots"]["city"] == "北京"


async def test_stale_redis_cache_cannot_override_sqlite(tmp_path):
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = SessionStore(str(tmp_path / "cache.sqlite"), redis_client=cache)
    state = store.empty_session()
    state["slots"] = {"city": "上海"}
    await store.save("c", state)
    stale = await cache.get(store.key("c"))
    state["slots"] = {"city": "北京"}
    await store.save("c", state)
    await cache.set(store.key("c"), stale)
    assert (await store.load("c"))["slots"]["city"] == "北京"
    await store.close()


async def test_selection_enters_confirmation_and_change_clears_it(tmp_path):
    async with service(tmp_path) as agent:
        await agent.handle("c", "1", FULL)
        selected = await agent.handle("c", "2", "选第一个")
        assert selected["session"]["task_status"] == "awaiting_confirmation"
        assert selected["session"]["selected_hotel"]
        changed = await agent.handle("c", "3", "改去北京")
        assert changed["session"]["selected_hotel"] is None
        assert all(hotel["city"] == "北京" for hotel in changed["session"]["candidates"])


async def test_duplicate_old_message_returns_current_session(tmp_path):
    async with service(tmp_path) as agent:
        await agent.handle("c", "1", FULL)
        await agent.handle("c", "2", "改去北京")
        duplicate = await agent.handle("c", "1", FULL)
        assert duplicate["duplicate"]
        assert duplicate["session"]["slots"]["city"] == "北京"


def test_date_change_does_not_change_budget():
    patch = extract_slots("入住改成2030-01-03", {"budget": 400}, ["上海"], TODAY)
    assert "budget" not in patch


def test_two_relative_dates_are_extracted_in_order():
    patch = extract_slots("今天入住后天离店", {}, ["上海"], TODAY)
    assert patch["check_in"] == "2030-01-01" and patch["check_out"] == "2030-01-03"


async def test_retrieval_timeout_is_bounded_and_resumable(tmp_path):
    repo = repository(tmp_path)
    original = repo.search
    import time

    def slow(*args):
        time.sleep(0.05)
        return original(*args)

    repo.search = slow
    async with service(tmp_path, repo, timeout=0.005, retries=0) as agent:
        with pytest.raises(asyncio.TimeoutError):
            await agent.handle("c", "1", FULL)
        assert agent.store.request("c", "1")["status"] == "failed"
        # Local CPU/SQL workers cannot be force-cancelled. Let it finish before retry.
        await asyncio.sleep(0.08)
        repo.search = original
        agent.timeout = 2
        recovered = await agent.handle("c", "1", FULL)
        assert recovered["session"]["task_status"] == "ready"


async def test_live_quote_does_not_invent_inventory(tmp_path):
    repo = repository(tmp_path)
    with repo.connect() as db:
        db.execute("DELETE FROM offers")

    async def quote(state):
        return [{"name": "上海测试全季", "room_type": "大床房", "nights": 2, "total_price": 600,
                 "inventory_confirmed": False}]

    async with service(tmp_path, repo, quote_provider=quote) as agent:
        result = await agent.handle("c", "1", FULL)
        assert result["session"]["task_status"] == "quote_pending"
        assert not result["session"]["candidates"]
        assert "房态尚未核验" in result["response"]


async def test_redis_actual_expiration_rehydrates_session(tmp_path):
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = SessionStore(str(tmp_path / "ttl.sqlite"), ttl=1, redis_client=cache)
    state = store.empty_session()
    state["slots"] = {"city": "上海"}
    await store.save("c", state)
    await asyncio.sleep(1.1)
    assert not await cache.exists(store.key("c"))
    assert (await store.load("c"))["slots"]["city"] == "上海"
    await store.close()


def test_persisted_faiss_index_reused_and_catalog_change_invalidates(tmp_path):
    repo = repository(tmp_path)
    repo.build_index()
    reloaded = HotelRepository(repo.db_path)
    reloaded.encoder.encode = lambda texts: (_ for _ in ()).throw(AssertionError("unexpected rebuild"))
    reloaded.build_index()
    assert reloaded.index.ntotal == 3
    with reloaded.connect() as db:
        db.execute("UPDATE hotels SET name='新酒店名' WHERE id='sh1'")
    with pytest.raises(AssertionError, match="unexpected rebuild"):
        reloaded.build_index()
