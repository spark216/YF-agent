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
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
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
            page.wait_for_function("document.getElementById('send').dataset.active === '0'")
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
        assert evidence[-1]["session"]["relaxations"]
        with page.expect_response(lambda r: r.url.endswith('/api/relaxations/accept')) as pending:
            page.locator('[data-relax]').first.click()
        relaxed = pending.value.json()
        assert relaxed['session']['task_status'] == 'ready'
        page.wait_for_function("document.getElementById('send').dataset.active === '0'")
        evidence.append(relaxed)
        page.screenshot(path=str(out / 'constraint-negotiation.png'), full_page=True)
        page.locator('#field-budget').fill('450')
        with page.expect_response(lambda r: r.url.endswith('/api/requirements/commit')) as pending:
            page.locator('#save-card').click()
        card = pending.value.json()
        assert card['session']['slots']['budget'] == 450
        page.wait_for_function("document.getElementById('send').dataset.active === '0'")
        evidence.append(card)
        page.screenshot(path=str(out / 'requirement-card.png'), full_page=True)
        page.locator('#message').fill('希望安静，带孩子，步行10分钟内到地铁')
        click_and_record('#send', 'ready', 'evidence-recommendation.png')
        assert evidence[-1]['session']['claims']
        click_and_record('#sample-select', 'awaiting_confirmation')
        page.locator('#reviewer-token').fill('demo-reviewer')
        with page.expect_response(lambda r: r.url.endswith('/api/drafts/approve')) as pending:
            page.locator('#approve').click()
        approval = pending.value.json()
        assert approval['receipt']['supplier_order_created'] is False
        assert approval['session']['task_status'] == 'approved_handoff'
        page.wait_for_function("document.getElementById('send').dataset.active === '0'")
        evidence.append(approval)
        page.screenshot(path=str(out / 'human-approval.png'), full_page=True)
        assert not errors, errors
        browser.close()
    target = root / "data/qa_runtime/browser_evidence.json"
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Captured 9 real screenshots; {len(evidence)} browser workflow responses verified; no JavaScript errors.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
