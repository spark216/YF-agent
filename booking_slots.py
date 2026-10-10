"""Deterministic slot updates; optional model extraction is handled by the service."""
from __future__ import annotations

import re
from datetime import date, timedelta

REQUIRED = ("city", "check_in", "check_out", "budget", "room_type")
LABELS = {"city": "城市", "check_in": "入住日期", "check_out": "离店日期", "budget": "每晚预算", "room_type": "房型"}


def extract_slots(text: str, existing: dict, cities: list[str], today: date | None = None):
    today = today or date.today()
    patch = {}
    # Longest city name wins; use last mention for explicit changes within one message.
    matches = [(text.rfind(city), city) for city in cities if city and city in text]
    if matches:
        patch["city"] = max(matches)[1]
    if re.search(r"不限预算|预算不限|取消预算", text):
        patch["budget"] = None
    else:
        budget = re.search(r"(?:预算(?:每晚)?|每晚|不超过|上限)\s*(?:改为|改成|调整到|为|是|[:：])?\s*[¥￥]?\s*(\d{2,5})(?:\s*元)?", text)
        if not budget:
            budget = re.search(r"(\d{2,5})\s*(?:元以内|元以下|以内|以下)", text)
        if not budget:
            budget = re.search(r"(?:改成|改为)\s*(\d{2,5})\s*元", text)
        if budget:
            patch["budget"] = float(budget.group(1))
    for token, room in (("双床", "双床房"), ("大床", "大床房"), ("套房", "套房"), ("家庭房", "家庭房")):
        if token in text:
            patch["room_type"] = room
    parsed = []
    for match in re.finditer(r"(?:(\d{4})[-年/])?(\d{1,2})[-月/](\d{1,2})日?", text):
        year = int(match[1] or today.year)
        try:
            value = date(year, int(match[2]), int(match[3]))
            if not match[1] and value < today:
                value = value.replace(year=year + 1)
            parsed.append(value)
        except ValueError:
            # Persist the error so the graph asks for correction instead of using old dates.
            patch["date_error"] = "日期不存在，请提供有效的入住和离店日期"
    if not parsed:
        offsets = {"大后天": 3, "后天": 2, "明天": 1, "今天": 0}
        parsed = [today + timedelta(days=offsets[match[0]]) for match in re.finditer(r"大后天|后天|明天|今天", text)]
    if parsed:
        departure_only = len(parsed) == 1 and ("离店" in text or "退房" in text) and "入住" not in text
        patch["check_out" if departure_only else "check_in"] = parsed[0].isoformat()
        patch["date_error"] = None
        if departure_only:
            pass
        elif len(parsed) > 1:
            patch["check_out"] = parsed[1].isoformat()
        else:
            nights = re.search(r"(?:住|入住)?\s*(\d+)\s*(?:晚|天)", text)
            if nights:
                patch["check_out"] = (parsed[0] + timedelta(days=int(nights[1]))).isoformat()
            elif existing.get("check_in") and existing.get("check_out"):
                duration = (date.fromisoformat(existing["check_out"]) - date.fromisoformat(existing["check_in"])).days
                patch["check_out"] = (parsed[0] + timedelta(days=max(duration, 1))).isoformat()
    elif re.search(r"住\s*(\d+)\s*晚", text) and existing.get("check_in"):
        nights = int(re.search(r"住\s*(\d+)\s*晚", text)[1])
        patch["check_out"] = (date.fromisoformat(existing["check_in"]) + timedelta(days=nights)).isoformat()
    return patch


def validate_slots(slots: dict, today: date | None = None):
    today = today or date.today()
    missing = [key for key in REQUIRED if slots.get(key) is None or slots.get(key) == ""]
    errors = []
    if slots.get("date_error"):
        errors.append(slots["date_error"])
    if slots.get("check_in") and slots.get("check_out"):
        try:
            start, end = date.fromisoformat(slots["check_in"]), date.fromisoformat(slots["check_out"])
            if end <= start:
                errors.append("离店日期必须晚于入住日期")
            if start < today:
                errors.append("入住日期不能早于今天")
            if (end - start).days > 30:
                errors.append("目前一次查询最多支持 30 晚")
        except ValueError:
            errors.append("日期格式应为 YYYY-MM-DD")
    if slots.get("budget") is not None and float(slots["budget"]) <= 0:
        errors.append("预算必须大于零")
    return missing, errors
