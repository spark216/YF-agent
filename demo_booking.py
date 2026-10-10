"""Offline, reproducible booking conversation; no model API or shop messages."""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
import sys
from datetime import date, timedelta
from pathlib import Path

from agent_factory import create_agent


async def run(scripted=False, output=None):
    async with create_agent(demo=True) as agent:
        chat_id = "demo-" + uuid.uuid4().hex[:8]
        start = date.today() + timedelta(days=1)
        end = start + timedelta(days=2)
        print("YF-Agent | LangGraph + Redis/SQLite + FAISS\n离线演示：规则槽位、哈希向量、样例房态，不是线上成交报价。")
        print("酒店目录数量：", agent.repository.count(), "缓存：", agent.store.cache_status)
        samples = ["想去上海住酒店", f"{start}入住，{end}离店，预算每晚400元，大床房",
                   "预算改成300元，换双床房", "改去北京", "预算改成100元"]
        results = []
        while True:
            if scripted:
                if len(results) >= len(samples):
                    break
                text = samples[len(results)]
                print("\n用户:", text)
            else:
                text = await asyncio.to_thread(input, "\n你 (/quit 退出): ")
                if text.strip() in ("/quit", "/exit"):
                    break
            message_id = uuid.uuid4().hex
            result = await agent.handle(chat_id, message_id, text)
            traces = agent.store.traces(chat_id, message_id)
            results.append({"user": text, "message_id": message_id, **result, "traces": traces})
            print("客服:", result["response"])
            print("状态:", result["session"]["task_status"], "槽位版本:", result["session"]["revision"],
                  "变更:", result["changed_slots"])
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"mode": "offline_demo", "catalog_count": agent.repository.count(),
                                       "encoder": agent.repository.encoder.name, "cache": agent.store.cache_status,
                                       "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    asyncio.run(run(args.scripted, args.output))
