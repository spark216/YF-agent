"""Replay 60 synthetic business trajectories; this is not a model-quality benchmark."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from booking_service import BookingService
from hotel_repository import HotelRepository
from session_store import SessionStore


async def evaluate():
    records = []
    with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temporary:
        directory = Path(temporary)
        repo = HotelRepository(str(directory / 'hotels.sqlite'))
        with repo.connect() as db:
            db.execute("INSERT INTO hotels VALUES('fixture','评测样例酒店','上海','1','样例')")
        today = date(2030, 1, 1)
        for offset in range(1, 20):
            for room in ('大床房', '双床房'):
                repo.put_offer('fixture', (today+timedelta(days=offset)).isoformat(), room, 300, 2, 'evaluation_fixture')
        async with BookingService(SessionStore(str(directory / 'sessions.sqlite')), repo,
                                  today=today, reviewer_token='evaluation-reviewer') as agent:
            for kind in ('negation', 'total_budget', 'card_revision', 'relaxation', 'unknown_evidence', 'human_approval'):
                for i in range(10):
                    chat = f'{kind}-{i}'
                    start, end = (today+timedelta(days=i+1)).isoformat(), (today+timedelta(days=i+3)).isoformat()
                    full = f'上海，{start}入住，{end}离店，预算400元，大床房'
                    steps = []
                    async def send(text):
                        result = await agent.handle(chat, str(len(steps)+1), text)
                        steps.append({'text': text, 'revision': result['session']['revision'], 'status': result['session']['task_status']})
                        return result
                    try:
                        initial = await send(full)
                        assert initial['session']['task_status'] == 'ready'
                        if kind == 'negation':
                            text = ('不要大床，换双床；日期不变，预算可以到450。', '换双床，不要大床；日期不变，预算可以到450。')[i % 2]
                            result = await send(text)
                            assert result['session']['slots']['room_type'] == '双床房'
                            assert result['session']['slots']['budget'] == 450
                            assert result['session']['slots']['check_in'] == start
                        elif kind == 'total_budget':
                            result = await send('总预算'+str(550+i)+'元')
                            assert result['session']['slots']['budget_scope'] == 'stay'
                            assert not result['session']['candidates']
                        elif kind == 'card_revision':
                            result = await agent.commit_card(chat, 'card', initial['session']['revision'], {'budget': 450+i})
                            steps.append({'action': 'card', 'revision': result['session']['revision']})
                            assert result['session']['slots']['budget'] == 450+i
                            try:
                                await agent.commit_card(chat, 'stale-card', initial['session']['revision'], {'budget': 500})
                            except ValueError:
                                pass
                            else:
                                raise AssertionError('Stale card accepted')
                        elif kind == 'relaxation':
                            result = await send('预算改成'+str(100+i)+'元')
                            assert result['session']['slots']['budget'] == 100+i
                            proposal = result['session']['relaxations'][0]
                            accepted = await agent.accept_relaxation(chat, 'accept', proposal['id'])
                            steps.append({'action': 'accept', 'revision': accepted['session']['revision']})
                            assert accepted['session']['slots']['budget'] == 300
                            assert accepted['session']['task_status'] == 'ready'
                        elif kind == 'unknown_evidence':
                            result = await send(('必须安静', '一定要安静')[i % 2])
                            assert result['session']['task_status'] == 'quote_pending'
                            assert not result['session']['candidates'] and not result['session']['claims']
                        else:
                            result = await send('选第一个')
                            draft = result['session']['draft']['id']
                            snapshot = await agent.approvals.graph.aget_state(agent.approvals.config(draft))
                            assert snapshot.next == ('human_review',)
                            first = await agent.approve_draft(chat, draft, 'evaluation-reviewer', 'evaluation-reviewer')
                            second = await agent.approve_draft(chat, draft, 'evaluation-reviewer', 'evaluation-reviewer')
                            assert first == second and not first['supplier_order_created']
                            steps.append({'action': 'approve_twice', 'one_receipt': True})
                        records.append({'id': chat, 'category': kind, 'passed': True, 'steps': steps})
                    except Exception as exc:
                        records.append({'id': chat, 'category': kind, 'passed': False, 'steps': steps,
                                        'error': type(exc).__name__+': '+str(exc)})
    report = {'scope': 'synthetic deterministic end-to-end business trajectories',
              'model_called': False, 'real_hotel_inventory': False, 'total': len(records),
              'passed': sum(record['passed'] for record in records), 'records': records}
    target = ROOT / 'docs/evaluation-results.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{report['passed']}/{report['total']} trajectories passed; report: {target}")
    return report['passed'] == report['total']


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(0 if asyncio.run(evaluate()) else 1)
