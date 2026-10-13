"""Bounded diagnostic queries; hard conditions only change after explicit acceptance."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, timedelta


class ConstraintNegotiator:
    def __init__(self, repository, evidence):
        self.repository, self.evidence = repository, evidence

    def diagnose(self, query, session):
        slots = session["slots"]
        attempts = []
        # Determine the least available budget from actual complete-stay inventory.
        wide = {**slots, "budget": 1000000}
        result = self.repository.search(query, wide, limit=100000)
        accepted, _, _ = self.evidence.rank(result["candidates"], session.get("preferences", []), session["revision"])
        if accepted:
            price_field = "total_price" if slots.get("budget_scope") == "stay" else "max_nightly_price"
            minimum = min(hotel[price_field] for hotel in accepted)
            if minimum > slots["budget"]:
                attempts.append({"slots": {**slots, "budget": minimum}, "changes": {"budget": minimum},
                                 "budget_delta": minimum - slots["budget"], "date_shift": 0})
        if slots.get("date_flexible"):
            for shift in (-1, 1):
                start = date.fromisoformat(slots["check_in"]) + timedelta(days=shift)
                end = date.fromisoformat(slots["check_out"]) + timedelta(days=shift)
                if start < date.today():
                    continue
                changes = {"check_in": start.isoformat(), "check_out": end.isoformat()}
                attempts.append({"slots": {**slots, **changes}, "changes": changes, "budget_delta": 0, "date_shift": abs(shift)})
        proposals = []
        for attempt in attempts[:3]:
            result = self.repository.search(query, attempt["slots"], limit=100000)
            hotels, _, _ = self.evidence.rank(result["candidates"], session.get("preferences", []), session["revision"])
            if not hotels:
                continue
            signature = {"revision": session["revision"], "changes": attempt["changes"],
                         "quotes": sorted(hotel["quote_id"] for hotel in hotels)}
            proposals.append({"id": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24],
                "base_revision": session["revision"], "changes": attempt["changes"],
                "candidate_count": len(hotels), "budget_delta": attempt["budget_delta"],
                "date_shift": attempt["date_shift"], "expires": min(hotel["valid_until"] for hotel in hotels),
                "sources": sorted({hotel["sources"] for hotel in hotels}), "created": time.time()})
        # Remove dominated choices: more costly, larger date change, no more candidates.
        pareto = [p for p in proposals if not any(q is not p and q["budget_delta"] <= p["budget_delta"]
            and q["date_shift"] <= p["date_shift"] and q["candidate_count"] >= p["candidate_count"]
            and (q["budget_delta"] < p["budget_delta"] or q["date_shift"] < p["date_shift"]
                 or q["candidate_count"] > p["candidate_count"]) for q in proposals)]
        return {"proposals": pareto, "offers_known": result["offers_known"],
                "reason": "constraints_exclude_inventory" if result["offers_known"] else "insufficient_inventory_data"}
