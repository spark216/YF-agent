"""Ask one primary question, preferring missing hard constraints."""
QUESTIONS = {"city": "您希望入住哪个城市？", "check_in": "您准备哪天入住？",
             "check_out": "准备住几晚，或哪天离店？", "budget": "预算是多少？请注明每晚还是整段住宿总价。",
             "room_type": "希望大床房、双床房还是其他房型？"}


def clarify(missing, errors, session):
    if errors:
        return {"field": "date_error", "question": errors[0] + "，请修改后再查询。", "reason": "invalid_hard_constraint"}
    if missing:
        # Dates determine valid inventory and price scope; don't ask optional preferences first.
        priorities = ("city", "check_in", "check_out", "budget", "room_type")
        field = next(field for field in priorities if field in missing)
        return {"field": field, "question": "已记录您的需求。" + QUESTIONS[field], "reason": "missing_hard_constraint"}
    return None
