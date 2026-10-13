"""Time-bounded hotel facts and claims. Missing facts never become true."""
from __future__ import annotations

import json
import time
import uuid


class EvidenceStore:
    def __init__(self, repository):
        self.repository = repository
        with repository.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS hotel_facts (
                    id TEXT PRIMARY KEY,hotel_id TEXT NOT NULL,field TEXT NOT NULL,
                    value TEXT NOT NULL,source TEXT NOT NULL,status TEXT NOT NULL,
                    observed REAL NOT NULL,expires REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_facts_hotel ON hotel_facts(hotel_id,field,observed);
            """)

    def put(self, hotel_id, field, value, source, *, status="verified", ttl=86400):
        if field not in {"quiet", "family_friendly", "metro_walk_minutes"}:
            raise ValueError("Unknown fact field")
        if status not in {"verified", "reported"} or not source or ttl <= 0:
            raise ValueError("Invalid fact provenance")
        if field == "metro_walk_minutes":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 240:
                raise ValueError("Walking duration must be between 0 and 240 minutes")
        elif not isinstance(value, bool):
            raise ValueError("Boolean fact expected")
        identifier = uuid.uuid4().hex
        with self.repository.connect() as db:
            if not db.execute("SELECT id FROM hotels WHERE id=?", (hotel_id,)).fetchone():
                raise ValueError("Unknown hotel")
            db.execute("INSERT INTO hotel_facts VALUES(?,?,?,?,?,?,?,?)",
                       (identifier, hotel_id, field, json.dumps(value), source, status, time.time(), time.time() + ttl))
        return identifier

    def facts(self, hotel_id):
        with self.repository.connect() as db:
            rows = db.execute("SELECT * FROM hotel_facts WHERE hotel_id=? ORDER BY observed DESC,id DESC", (hotel_id,)).fetchall()
        # The latest observation supersedes older evidence, even if it has expired.
        latest = {}
        for row in rows:
            if row["field"] not in latest:
                latest[row["field"]] = dict(row)
        return {field: {**row, "value": json.loads(row["value"])}
                for field, row in latest.items() if row["expires"] > time.time()}

    def rank(self, candidates, preferences, revision):
        accepted, uncertain, excluded = [], [], []
        for original in candidates:
            hotel = dict(original)
            facts = self.facts(hotel["id"])
            claims, missing, violations, score = [], [], [], 0
            for preference in preferences:
                field = preference["field"]
                fact = facts.get(field)
                if not fact or fact["status"] != "verified":
                    missing.append(field)
                    continue
                matches = fact["value"] <= preference["value"] if field == "metro_walk_minutes" else fact["value"] == preference["value"]
                if not matches:
                    if preference.get("hard"):
                        violations.append(field)
                    continue
                score += 1
                labels = {"quiet": "资料记录支持安静偏好（非绝对保证）", "family_friendly": "资料记录支持亲子偏好", "metro_walk_minutes": f"资料记录步行到地铁约 {fact['value']} 分钟"}
                claims.append({"field": field, "text": labels[field], "evidence_id": fact["id"],
                               "source": fact["source"], "expires": fact["expires"], "revision": revision})
            hotel.update(claims=claims, unknown_preferences=missing, preference_score=score)
            if violations:
                excluded.append({"hotel_id": hotel["id"], "violations": violations})
            elif any(pref.get("hard") and pref["field"] in missing for pref in preferences):
                uncertain.append(hotel)
            else:
                accepted.append(hotel)
        accepted.sort(key=lambda item: (-item["preference_score"], -item["score"], item["total_price"]))
        return accepted, uncertain, excluded

    def validate_claims(self, hotel, revision):
        facts = self.facts(hotel["id"])
        return [claim for claim in hotel.get("claims", []) if claim["revision"] == revision
                and claim["expires"] > time.time() and claim["field"] in facts
                and facts[claim["field"]]["id"] == claim["evidence_id"]]
