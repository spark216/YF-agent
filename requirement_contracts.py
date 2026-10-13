"""Validated language-to-business patches; models never write session state."""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import date

from booking_slots import extract_slots

FIELDS = {"city", "check_in", "check_out", "budget", "room_type", "date_error", "budget_scope", "date_flexible"}
PREFERENCES = {"quiet", "family_friendly", "metro_walk_minutes"}


class RevisionConflict(ValueError):
    pass


def requirement_hash(session):
    body = {"slots": session["slots"], "preferences": session.get("preferences", [])}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_operation(operation, text):
    op = dict(operation)
    if op.get("op") not in {"set", "unset", "add_preference", "remove_preference"}:
        raise ValueError("Unsupported requirement operation")
    preference = op["op"] in {"add_preference", "remove_preference"}
    field = op.get("field")
    if field not in (PREFERENCES if preference else FIELDS):
        raise ValueError("Unknown requirement field")
    evidence = op.get("evidence", "")
    if not evidence or evidence not in text:
        raise ValueError("Requirement changes must cite the current user message")
    if op.get("origin", "explicit") not in {"explicit", "inferred", "derived"}:
        raise ValueError("Invalid requirement origin")
    if op["op"] in {"set", "add_preference"}:
        value = op.get("value")
        if field in {"budget", "metro_walk_minutes"}:
            if isinstance(value, bool):
                raise ValueError("Invalid numeric requirement")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError("Numeric requirement expected") from None
            if not math.isfinite(value) or value <= 0 or value > (240 if preference else 1000000):
                raise ValueError("Numeric requirement out of range")
        elif field in {"check_in", "check_out"}:
            if not isinstance(value, str):
                raise ValueError("ISO date string expected")
            date.fromisoformat(value)
        elif field in {"quiet", "family_friendly", "date_flexible"}:
            if not isinstance(value, bool):
                raise ValueError("Boolean requirement expected")
        elif field == "room_type" and value not in {"大床房", "双床房", "套房", "家庭房"}:
            raise ValueError("Unsupported room type")
        elif field == "budget_scope" and value not in {"night", "stay"}:
            raise ValueError("Invalid budget scope")
        elif not isinstance(value, str) or len(value) > 200:
            raise ValueError("Invalid text requirement")
        op["value"] = value
    op.setdefault("origin", "explicit")
    if preference:
        op["hard"] = bool(op.get("hard", False))
    return op


def propose(text, session, cities, today, model_patch=None):
    patch = extract_slots(text, session["slots"], cities, today)
    amount = re.search(r"预算(?:可以|可)?(?:到|改成|改为|提高到|降到)\s*(\d+(?:\.\d+)?)", text)
    if amount:
        patch["budget"] = float(amount[1])
    # Negated room mentions are not positive assignments; the last positive mention wins.
    mentions = []
    rejected = []
    for match in re.finditer(r"(?:不要|别选|不选|不住|取消)?(?:双床|大床|套房|家庭房)", text):
        word = match[0]
        room = next(room for token, room in (("双床", "双床房"), ("大床", "大床房"), ("套房", "套房"), ("家庭房", "家庭房")) if token in word)
        (rejected if re.match(r"不要|别选|不选|不住|取消", word) else mentions).append(room)
    patch.pop("room_type", None)
    if mentions:
        patch["room_type"] = mentions[-1]
    elif session["slots"].get("room_type") in rejected:
        patch["room_type"] = None
    if "日期不变" in text or "时间不变" in text:
        for key in ("check_in", "check_out", "date_error"):
            patch.pop(key, None)
    if "总预算" in text:
        amount = re.search(r"总预算\s*(\d+(?:\.\d+)?)", text)
        if amount:
            patch["budget"] = float(amount[1])
            patch["budget_scope"] = "stay"
    elif "每晚" in text:
        patch["budget_scope"] = "night"
    if "日期可以调整" in text or "日期可调整" in text:
        patch["date_flexible"] = True
    operations = []
    if model_patch:
        if "operations" in model_patch:
            operations = [validate_operation(op, text) for op in model_patch["operations"]]
        else:
            # Compatibility with the original extractor, while preserving provenance.
            operations = [validate_operation({"op": "set", "field": key, "value": value,
                           "origin": "inferred", "evidence": text}, text)
                          for key, value in model_patch.items() if key in FIELDS and value is not None]
    explicit_fields = set(patch)
    # A model's origin label is a proposal, not proof of user authorization.
    # Independently recognized fields below replace it with explicit operations.
    for operation in operations:
        if operation["origin"] == "explicit":
            operation["origin"] = "inferred"
    operations = [op for op in operations if op["field"] not in explicit_fields]
    for key, value in patch.items():
        operations.append(validate_operation({"op": "unset" if value is None else "set", "field": key,
                           "value": value, "origin": "explicit", "evidence": text}, text))
    for word, field, value in (("安静", "quiet", True), ("带孩子", "family_friendly", True), ("亲子", "family_friendly", True)):
        if word in text:
            remove = any(prefix + word in text for prefix in ("不要求", "不用", "取消"))
            operations = [op for op in operations if op["field"] != field]
            operations.append(validate_operation({"op": "remove_preference" if remove else "add_preference",
                "field": field, "value": value, "origin": "explicit", "evidence": word,
                "hard": ("必须" in text or "一定要" in text) and not remove}, text))
    metro = re.search(r"(?:步行|走路)(\d+)分钟(?:以内|内)?(?:到|到达)?地铁", text)
    if metro:
        operations = [op for op in operations if op["field"] != "metro_walk_minutes"]
        operations.append(validate_operation({"op": "add_preference", "field": "metro_walk_minutes",
            "value": float(metro[1]), "evidence": metro[0], "origin": "explicit", "hard": "必须" in text}, text))
    # Changing an existing hard condition without an explicit span requires confirmation.
    confirmation = any(op["origin"] != "explicit" for op in operations)
    return {"id": uuid.uuid4().hex, "base_revision": session["revision"], "text": text,
            "operations": operations, "needs_confirmation": confirmation}


def apply_proposal(session, proposal):
    if proposal["base_revision"] != session["revision"]:
        raise RevisionConflict("Requirement version changed; regenerate the proposal")
    result = json.loads(json.dumps(session))
    result.setdefault("preferences", [])
    result.setdefault("slot_sources", {})
    changed = []
    for op in proposal["operations"]:
        op = validate_operation(op, proposal["text"])
        field = op["field"]
        if op["op"] in {"add_preference", "remove_preference"}:
            old = [pref for pref in result["preferences"] if pref["field"] == field]
            result["preferences"] = [pref for pref in result["preferences"] if pref["field"] != field]
            new = {key: op[key] for key in ("field", "value", "hard", "evidence") if key in op}
            if op["op"] == "add_preference":
                result["preferences"].append(new)
            if old != ([new] if op["op"] == "add_preference" else []):
                changed.append(field)
        else:
            value = op.get("value") if op["op"] == "set" else None
            if result["slots"].get(field) != value:
                changed.append(field)
                result["slots"][field] = value
            result["slot_sources"][field] = {"origin": op["origin"], "evidence": op["evidence"]}
    if changed:
        result["revision"] += 1
        for field in ("candidates", "quotes", "claims", "relaxations", "verification_candidates"):
            result[field] = []
        result.update(selected_hotel=None, candidate_revision=None, draft=None, task_status="collecting")
    result["requirement_hash"] = requirement_hash(result)
    result["pending_proposal"] = None
    return result, changed
