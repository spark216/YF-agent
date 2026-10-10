"""Bridge browser quotes into the booking graph without inventing room inventory."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date, timedelta


class LivePriceProvider:
    def __init__(self, repository, limit=2):
        self.repository, self.limit = repository, limit

    async def __call__(self, state):
        from utils.hotel_tools import search_hotel_price
        slots = state["session"]["slots"]
        hotels = await asyncio.to_thread(self.repository.suggest, slots["city"] + " " + state["user_msg"], slots["city"], self.limit)
        start, end = date.fromisoformat(slots["check_in"]), date.fromisoformat(slots["check_out"])
        results = []
        for hotel in hotels:
            nightly = []
            current = start
            while current < end:
                following = current + timedelta(days=1)
                raw = await asyncio.to_thread(search_hotel_price, hotel["name"], current.isoformat(), following.isoformat())
                quote = json.loads(raw)
                if quote.get("error"):
                    raise OSError("Hotel webpage did not return a usable quote")
                if str(quote.get("酒店ID")) != hotel["id"]:
                    raise ValueError("Hotel matcher returned a different property")
                prices = []
                for room in quote.get("房型", []):
                    token = slots["room_type"].removesuffix("房")
                    raw_price = str(room.get("价格", ""))
                    match = re.fullmatch(r"[¥￥]?\s*(\d+(?:\.\d+)?)\s*元?", raw_price)
                    if token in room.get("房型名", "") and match:
                        prices.append(float(match[1]))
                if not prices:
                    break
                nightly.append(min(prices))
                current = following
            if len(nightly) == (end - start).days and max(nightly) <= slots["budget"]:
                results.append({**hotel, "room_type": slots["room_type"], "total_price": sum(nightly),
                                "max_nightly_price": max(nightly), "nights": len(nightly),
                                "inventory_confirmed": False, "source": "web_quote"})
        return results
