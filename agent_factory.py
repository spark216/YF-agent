"""Environment-based construction of demo and model-backed booking agents."""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from booking_service import BookingService
from booking_slots import REQUIRED
from hotel_repository import BGEEncoder, DemoEncoder, HotelRepository
from session_store import SessionStore

ROOT = Path(__file__).resolve().parent


def model_providers():
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["API_KEY"],
                         base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                         timeout=float(os.getenv("MODEL_TIMEOUT", "20")), max_retries=1)

    async def extract(text, slots):
        response = await client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen-max"), temperature=0,
            messages=[{"role": "system", "content":
                "你只提取用户本轮明确表达的酒店需求变更，返回 JSON 对象。字段 city/check_in/check_out/budget/room_type，"
                "budget 为每晚数字，日期 YYYY-MM-DD，房型为大床房/双床房/套房/家庭房。"
                "不推断未表达的需求，不输出未知字段。不清楚时返回 {}。今天是 " + date.today().isoformat()},
                {"role": "user", "content": json.dumps({"existing_slots": slots, "message": text}, ensure_ascii=False)}],
            response_format={"type": "json_object"})
        patch = json.loads(response.choices[0].message.content)
        if not isinstance(patch, dict):
            raise ValueError("Slot extractor must return an object")
        patch = {key: value for key, value in patch.items() if key in REQUIRED and value is not None}
        for key in ("check_in", "check_out"):
            if key in patch:
                date.fromisoformat(patch[key])
        if "budget" in patch:
            patch["budget"] = float(patch["budget"])
        for key in ("city", "room_type"):
            if key in patch and not isinstance(patch[key], str):
                raise ValueError("Invalid slot type")
        return patch

    # Replies retain deterministic factual pricing. LLM extraction enriches language understanding;
    # template replies avoid unsupported availability/price claims.
    extract.aclose = client.close
    return extract


@asynccontextmanager
async def create_agent(*, demo=False, work_dir=None, redis_client=None):
    load_dotenv(ROOT / ".env")
    directory = Path(work_dir or ROOT / "data" / ("demo_runtime" if demo else "runtime"))
    directory.mkdir(parents=True, exist_ok=True)
    redis_url = os.getenv("REDIS_URL", "")
    store = SessionStore(str(directory / "sessions.sqlite"), redis_url=redis_url,
                         ttl=int(os.getenv("SESSION_TTL", "1800")), redis_client=redis_client)
    backend = "demo" if demo else os.getenv("RETRIEVAL_BACKEND", "bge")
    encoder = DemoEncoder() if backend == "demo" else BGEEncoder(os.getenv("BGE_MODEL_PATH", "BAAI/bge-m3"))
    repository = HotelRepository(str(directory / "hotels.sqlite"), encoder)
    if not repository.count():
        repository.import_catalog(str(ROOT / "data/hotel_catalog.xlsx"))
    if demo:
        repository.seed_demo_offers(date.today() + timedelta(days=1))
    extractor = model_providers() if not demo and os.getenv("LLM_SLOT_EXTRACTION", "true").lower() == "true" else None
    quote_provider = None
    if not demo and os.getenv("LIVE_PRICE_ENABLED", "false").lower() == "true":
        from live_price_provider import LivePriceProvider
        quote_provider = LivePriceProvider(repository, int(os.getenv("LIVE_PRICE_HOTELS", "2")))
    detector = None
    if not demo and os.getenv("INTENT_BACKEND", "rules") == "bge":
        import sys
        sys.path.insert(0, str(ROOT / "models"))
        from intent_predictor import predict_intent

        async def detector(text):
            result = await asyncio.to_thread(predict_intent, text)
            return result["intent"]
    service = BookingService(store, repository, max_concurrency=int(os.getenv("MAX_CONCURRENCY", "4")),
                             timeout=float(os.getenv("TOOL_TIMEOUT", "20")),
                             retries=int(os.getenv("TOOL_RETRIES", "2")),
                             slot_extractor=extractor, intent_detector=detector, quote_provider=quote_provider)
    try:
        async with service:
            yield service
    finally:
        # Model client is owned by the extractor closure; close explicitly if enabled.
        if extractor and hasattr(extractor, "aclose"):
            await extractor.aclose()
