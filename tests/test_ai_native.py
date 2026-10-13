from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from tests.test_booking_workflow import FULL, TODAY, repository, service
from requirement_contracts import propose, apply_proposal, RevisionConflict


def test_negation_and_preserved_dates():
    state = {"slots": {"room_type": "大床房", "check_in": "2030-01-02", "check_out": "2030-01-04", "budget": 400},
             "revision": 2, "candidates": [], "preferences": []}
    proposal = propose("不要大床，换双床；日期不变，预算可以到450。", state, ["上海"], TODAY)
    updated, changes = apply_proposal(state, proposal)
    assert updated["slots"]["room_type"] == "双床房"
    assert updated["slots"]["budget"] == 450
    assert updated["slots"]["check_in"] == "2030-01-02"
    assert updated["slot_sources"]["room_type"]["origin"] == "explicit"


def test_fabricated_model_evidence_rejected():
    state = {"slots": {}, "revision": 0}
    with pytest.raises(ValueError, match="cite"):
        propose("预算稍微高一点", state, [], TODAY, {"operations": [
            {"op": "set", "field": "budget", "value": 500, "evidence": "我同意500元", "origin": "explicit"}]})


async def test_inferred_change_requires_confirmation(tmp_path):
    async def extractor(text, slots):
        if "稍微" not in text:
            return {"operations": []}
        return {"operations": [{"op": "set", "field": "budget", "value": 450,
                                "origin": "inferred", "evidence": "稍微高一点"}]}
    async with service(tmp_path, slot_extractor=extractor) as agent:
        await agent.handle("c", "1", FULL)
        pending = await agent.handle("c", "2", "预算稍微高一点")
        assert pending["session"]["slots"]["budget"] == 400
        assert pending["session"]["task_status"] == "confirming_requirements"
        accepted = await agent.accept_pending("c", "3", pending["session"]["pending_proposal"]["id"])
        assert accepted["session"]["slots"]["budget"] == 450


async def test_card_and_language_share_revision_contract(tmp_path):
    async with service(tmp_path) as agent:
        first = await agent.handle("c", "1", FULL)
        changed = await agent.commit_card("c", "2", first["session"]["revision"], {"room_type": "双床房"})
        assert changed["session"]["slots"]["room_type"] == "双床房"
        with pytest.raises(RevisionConflict):
            await agent.commit_card("c", "3", first["session"]["revision"], {"budget": 500})


async def test_relaxation_requires_acceptance_and_is_recomputed(tmp_path):
    async with service(tmp_path) as agent:
        result = await agent.handle("c", "1", FULL.replace("400", "100"))
        options = result["session"]["relaxations"]
        assert options and result["session"]["slots"]["budget"] == 100
        assert options[0]["changes"] == {"budget": 300.0}
        accepted = await agent.accept_relaxation("c", "2", options[0]["id"])
        assert accepted["session"]["task_status"] == "ready"
        assert all(hotel["max_nightly_price"] <= 300 for hotel in accepted["session"]["candidates"])
        duplicate = await agent.accept_relaxation("c", "2", options[0]["id"])
        assert duplicate["duplicate"]
        assert duplicate["session"]["revision"] == accepted["session"]["revision"]


async def test_old_relaxation_cannot_change_new_city(tmp_path):
    async with service(tmp_path) as agent:
        first = await agent.handle("c", "1", FULL.replace("400", "100"))
        option = first["session"]["relaxations"][0]
        await agent.handle("c", "2", "改去北京")
        with pytest.raises(ValueError):
            await agent.accept_relaxation("c", "3", option["id"])


async def test_unknown_hard_preference_never_becomes_claim(tmp_path):
    async with service(tmp_path) as agent:
        result = await agent.handle("c", "1", FULL + "，必须安静")
        assert result["session"]["task_status"] == "quote_pending"
        assert not result["session"]["candidates"]
        assert result["session"]["verification_candidates"]
        assert not result["session"]["claims"]


async def test_evidence_ranking_and_expiration(tmp_path):
    async with service(tmp_path) as agent:
        evidence_id = agent.evidence.put("sh1", "quiet", True, "test:verified-record", ttl=10)
        result = await agent.handle("c", "1", FULL + "，希望安静")
        assert result["session"]["candidates"][0]["id"] == "sh1"
        assert any(claim["evidence_id"] == evidence_id for claim in result["session"]["claims"])
        with agent.repository.connect() as db:
            db.execute("UPDATE hotel_facts SET expires=0")
        updated = await agent.handle("c", "2", "重新查询")
        assert not updated["session"]["claims"]
        assert "支持安静偏好" not in updated["response"]


async def test_new_conditions_commit_before_old_tool_returns(tmp_path):
    repo = repository(tmp_path)
    original = repo.search
    entered, release = threading.Event(), threading.Event()
    def delayed(query, slots, *args, **kwargs):
        if slots["city"] == "上海":
            entered.set()
            assert release.wait(3)
        return original(query, slots, *args, **kwargs)
    repo.search = delayed
    async with service(tmp_path, repo) as agent:
        old = asyncio.create_task(agent.handle("c", "1", FULL))
        await asyncio.to_thread(entered.wait, 2)
        new = await asyncio.wait_for(agent.handle("c", "2", "改去北京"), 2)
        assert new["session"]["slots"]["city"] == "北京"
        release.set()
        stale = await old
        assert stale["stale"]
        assert all(hotel["city"] == "北京" for hotel in (await agent.store.load("c"))["candidates"])
        assert any(trace["node"] == "stale_result_discarded" for trace in agent.store.traces("c"))


async def test_approval_authorization_idempotence_and_restart(tmp_path):
    repo = repository(tmp_path)
    async with service(tmp_path, repo, reviewer_token="test-reviewer") as agent:
        await agent.handle("c", "1", FULL)
        selected = await agent.handle("c", "2", "选第一个")
        draft_id = selected["session"]["draft"]["id"]
        snapshot = await agent.approvals.graph.aget_state(agent.approvals.config(draft_id))
        assert snapshot.next == ("human_review",)
    async with service(tmp_path, repo, reviewer_token="test-reviewer") as restarted:
        with pytest.raises(PermissionError):
            await restarted.approve_draft("c", draft_id, "客服", "wrong-token")
        first = await restarted.approve_draft("c", draft_id, "客服", "test-reviewer")
        second = await restarted.approve_draft("c", draft_id, "客服", "test-reviewer")
        assert first == second and not first["supplier_order_created"]
        assert (await restarted.store.load("c"))["task_status"] == "approved_handoff"


@pytest.mark.parametrize("change", ["date", "expiry"])
async def test_approval_invalidated_by_change_or_expiry(tmp_path, change):
    async with service(tmp_path, reviewer_token="test-reviewer") as agent:
        await agent.handle("c", "1", FULL)
        selected = await agent.handle("c", "2", "选第一个")
        draft_id = selected["session"]["draft"]["id"]
        if change == "date":
            await agent.handle("c", "3", "改到2030-01-03入住")
        else:
            with agent.store.connect() as db:
                db.execute("UPDATE booking_drafts SET expires=0 WHERE id=?", (draft_id,))
        with pytest.raises(ValueError):
            await agent.approve_draft("c", draft_id, "客服", "test-reviewer")


async def test_total_budget_is_not_per_night_budget(tmp_path):
    async with service(tmp_path) as agent:
        result = await agent.handle("c", "1", FULL.replace("预算400元", "总预算650元"))
        assert result["session"]["slots"]["budget_scope"] == "stay"
        assert [hotel["id"] for hotel in result["session"]["candidates"]] == ["sh1"]


async def test_expired_inventory_is_not_recommended(tmp_path):
    repo = repository(tmp_path)
    with repo.connect() as db:
        db.execute("UPDATE offers SET valid_until=0")
    async with service(tmp_path, repo) as agent:
        result = await agent.handle("c", "1", FULL)
        assert result["session"]["task_status"] == "quote_pending"
        assert not result["session"]["relaxations"]


@pytest.mark.parametrize("change", ["price", "hard_evidence"])
async def test_approval_revalidates_live_quote_and_required_evidence(tmp_path, change):
    async with service(tmp_path, reviewer_token="test-reviewer") as agent:
        if change == "hard_evidence":
            agent.evidence.put("sh1", "quiet", True, "test:verified")
        await agent.handle("c", "1", FULL + ("，必须安静" if change == "hard_evidence" else ""))
        selected = await agent.handle("c", "2", "选第一个")
        draft_id = selected["session"]["draft"]["id"]
        with agent.repository.connect() as db:
            if change == "price":
                db.execute("UPDATE offers SET price=price+10")
            else:
                db.execute("UPDATE hotel_facts SET expires=0")
        with pytest.raises(ValueError, match="changed"):
            await agent.approve_draft("c", draft_id, "客服", "test-reviewer")


async def test_reset_invalidates_inflight_result(tmp_path):
    async with service(tmp_path) as agent:
        first = await agent.handle("c", "1", FULL)
        await agent.store.reset("c")
        replay = await agent.store.publish("c", "1", first)
        assert replay["stale"]
        assert not (await agent.store.load("c"))["slots"]


def test_model_cannot_self_authorize_an_inferred_budget():
    proposal = propose("预算稍微高一点", {"slots": {"budget": 400}, "revision": 1}, [], TODAY,
        {"operations": [{"op": "set", "field": "budget", "value": 500, "origin": "explicit", "evidence": "预算稍微高一点"}]})
    assert proposal["needs_confirmation"]
    assert proposal["operations"][0]["origin"] == "inferred"


async def test_same_revision_old_result_cannot_overwrite_pending_confirmation(tmp_path):
    async with service(tmp_path) as agent:
        original = await agent.handle("c", "1", FULL)
        async def extractor(text, slots):
            return {"operations": [{"op": "set", "field": "budget", "value": 450,
                "origin": "inferred", "evidence": text}]}
        agent.slot_extractor = extractor
        newer = await agent.handle("c", "2", "预算稍微高一点")
        assert original["session"]["revision"] == newer["session"]["revision"]
        replay = await agent.store.publish("c", "1", original)
        assert replay["stale"]
        assert (await agent.store.load("c"))["task_status"] == "confirming_requirements"


async def test_saturated_tool_semaphore_does_not_block_ingress(tmp_path):
    repo = repository(tmp_path)
    original = repo.search
    entered, release = threading.Event(), threading.Event()
    def delayed(query, slots):
        if slots["city"] == "上海":
            entered.set()
            assert release.wait(3)
        return original(query, slots)
    repo.search = delayed
    async with service(tmp_path, repo, max_concurrency=1) as agent:
        old = asyncio.create_task(agent.handle("c", "1", FULL))
        await asyncio.to_thread(entered.wait, 2)
        new = asyncio.create_task(agent.handle("c", "2", "改去北京"))
        try:
            for _ in range(100):
                if (await agent.store.load("c"))["slots"].get("city") == "北京":
                    break
                await asyncio.sleep(0.01)
            assert (await agent.store.load("c"))["slots"]["city"] == "北京"
        finally:
            release.set()
        assert (await old)["stale"]
        assert (await new)["session"]["task_status"] == "ready"
