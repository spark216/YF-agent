"""Capture real browser interactions against the running local showcase."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "images"
    out.mkdir(exist_ok=True)
    evidence = []
    with sync_playwright() as browser_api:
        browser = browser_api.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto(args.url)
        page.wait_for_function("document.getElementById('catalog').textContent !== '—'")

        def click_and_record(button, expected, image=None):
            with page.expect_response(lambda response: response.url.endswith("/api/chat")) as pending:
                page.locator(button).click()
            response = pending.value
            if response.status != 200:
                raise RuntimeError(response.text())
            payload = response.json()
            assert payload["session"]["task_status"] == expected
            page.wait_for_function("document.getElementById('send').disabled === false")
            evidence.append(payload)
            if image:
                page.screenshot(path=str(out / image), full_page=True)

        click_and_record("#sample-city", "collecting")
        click_and_record("#sample-full", "ready", "booking-conversation.png")
        click_and_record("#sample-change", "ready", "slot-update.png")
        click_and_record("#repeat", "ready", "message-dedup.png")
        assert evidence[-1]["duplicate"] is True
        assert len(evidence[-1]["session"]["history"]) == len(evidence[-2]["session"]["history"])
        click_and_record("#sample-select", "awaiting_confirmation", "booking-guidance.png")
        click_and_record("#sample-none", "no_match", "no-match.png")
        assert not evidence[-1]["session"]["candidates"]
        browser.close()
    target = root / "data/qa_runtime/browser_evidence.json"
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Captured 5 real screenshots; 6 browser workflow assertions passed.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
