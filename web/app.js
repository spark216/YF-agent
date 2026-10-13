const $ = id => document.getElementById(id);
let chatId = sessionStorage.getItem('yf-chat') || crypto.randomUUID();
sessionStorage.setItem('yf-chat', chatId);
let runtime = {}, current = null, lastRequest = null, sequence = 0, renderedSequence = 0, active = 0, cardRevision = 0;
const labels = {city:'城市',check_in:'入住日期',check_out:'离店日期',budget:'预算',room_type:'房型'};
const statuses = {collecting:'收集需求',ready:'候选已更新',no_match:'待协商条件',quote_pending:'待核验房态 / 偏好',awaiting_confirmation:'等待人工审批',confirming_requirements:'等待需求确认',approved_handoff:'已生成交接回执'};
function escapeHtml(value) { const el=document.createElement('span'); el.textContent=String(value??''); return el.innerHTML; }
function render(data, requestSequence = sequence) {
  const s=data.session;
  if (!s || (current && (s.revision<current.revision || (s.revision===current.revision && (s.turn_epoch||0)<(current.turn_epoch||0)))) || requestSequence<renderedSequence) return;
  renderedSequence=requestSequence; current=s; runtime=data.meta||runtime;
  $('catalog').textContent=Number(runtime.catalog_count).toLocaleString(); $('revision').textContent='v'+s.revision;
  $('cache').textContent=runtime.cache==='redis'?'Redis':'SQLite 回退';
  $('cache-note').textContent=runtime.redis_simulated?'Redis API 模拟 · TTL 1800s':'持久状态优先 · 活跃会话缓存';
  $('task-status').textContent=statuses[s.task_status]||s.task_status;
  $('candidate-version').textContent=s.candidate_revision===null?'候选尚未生成':'候选版本 v'+s.candidate_revision;
  $('changed').textContent=data.duplicate?'消息 ID 去重':data.stale?'旧结果已丢弃':'可编辑 · v'+s.revision;
  // Do not overwrite a card while the user is editing it.
  if (!$('slots').contains(document.activeElement)) {cardRevision=s.revision; $('slots').innerHTML=Object.entries(labels).map(([key,label])=>`<div class="slot"><label for="field-${key}">${key==='budget'?(s.slots.budget_scope==='stay'?'住宿总预算':'每晚预算'):label}</label><input id="field-${key}" data-field="${key}" type="${key.startsWith('check_')?'date':key==='budget'?'number':'text'}" value="${escapeHtml(s.slots[key]??'')}" style="width:100%;border:0;background:transparent;color:#17233b;font:inherit"></div>`).join('');}
  const candidates=s.candidates||[]; $('candidate-count').textContent=candidates.length+' 条结果';
  $('hotels').innerHTML=candidates.length?candidates.slice(0,3).map((h,i)=>`<div class="hotel"><em>¥${h.total_price} / ${h.nights}晚</em><b>${escapeHtml(h.name)}</b><div>${escapeHtml(h.city)} · ${escapeHtml(h.room_type)} · 样例库存</div>${(h.claims||[]).filter(c=>c.expires*1000>Date.now()).map(c=>`<div>✓ ${escapeHtml(c.text)}<br><small>证据 ${escapeHtml(c.evidence_id.slice(0,8))} · ${escapeHtml(c.source)}</small></div>`).join('')}${h.unknown_preferences?.length?'<div>部分偏好资料未知，不能承诺满足</div>':''}<button class="outline" data-select="${i+1}">选择并生成待审批草稿</button></div>`).join(''):'<p class="empty">当前无可承诺的候选。缺少证据时保留未知状态。</p>';
  $('hotels').querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>send('选第'+b.dataset.select+'个'));
  const traces=data.traces||[]; $('trace-count').textContent=traces.length+' 条记录';
  $('traces').innerHTML=traces.map(t=>`<div class="trace-row"><span class="trace-icon">${t.status==='error'?'!':'✓'}</span><span class="trace-name">${escapeHtml(t.node)}</span><span class="trace-ms">${Number(t.duration_ms).toFixed(1)}ms</span></div>`).join('');
  if(s.history.length) {$('chat').innerHTML=s.history.map(m=>`<div class="message ${m.role==='user'?'user':''}"><div class="avatar">${m.role==='user'?'你':'YF'}</div><div class="bubble">${escapeHtml(m.content)}</div></div>`).join(''); $('chat').scrollTop=$('chat').scrollHeight;}
  $('preferences').textContent=(s.preferences||[]).map(p=>`${p.field}=${p.value}（${p.hard?'必需':'偏好'}）`).join(' · ')||'未添加偏好；可输入“希望安静、带孩子、步行10分钟内到地铁”。';
  const pending=s.pending_proposal;
  $('decisions').innerHTML=(pending?`<p>模型提出修改，需要您确认：${escapeHtml(JSON.stringify(pending.operations.map(o=>({field:o.field,value:o.value,origin:o.origin}))))}</p><button class="primary" id="accept-pending">确认需求提案</button>`:'')+(s.relaxations||[]).map(p=>`<div class="hotel"><b>可选调整：${escapeHtml(JSON.stringify(p.changes))}</b><div>已查询样例库存：${p.candidate_count} 家 · 仅接受后生效</div><button class="outline" data-relax="${escapeHtml(p.id)}">接受此方案并重新检索</button></div>`).join('');
  if(pending) $('accept-pending').onclick=()=>act('/api/requirements/accept',{id:pending.id});
  $('decisions').querySelectorAll('[data-relax]').forEach(b=>b.onclick=()=>act('/api/relaxations/accept',{id:b.dataset.relax}));
  const draft=s.draft;
  $('approval').innerHTML=draft?`<p>草稿 ${escapeHtml(draft.id.slice(0,10))} · v${draft.revision} · ${escapeHtml(draft.status)}</p>${draft.receipt?`<p>回执：${escapeHtml(draft.receipt.id)}<br>已人工确认交接；供应商订单未创建。</p>`:'<input id="reviewer" placeholder="审批人身份" value="演示客服"><input id="reviewer-token" type="password" placeholder="审批令牌（本地演示：demo-reviewer）" autocomplete="off"><button id="approve" class="primary">人工确认并恢复工作流</button>'}`:'<p class="empty">选定酒店后生成与需求版本、报价有效期绑定的审批草稿。</p>';
  if($('approve')) $('approve').onclick=()=>act('/api/drafts/approve',{draft_id:draft.id,actor:$('reviewer').value},{'X-Reviewer-Token':$('reviewer-token').value});
}
async function request(path,body,headers={}) {
  const n=++sequence; active++; $('send').dataset.active=active; $('error').textContent='';
  try {const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(body)}); const data=await response.json(); if(!response.ok)throw Error(data.detail); render(data,n); return data;}
  catch(e){$('error').textContent=e.message; throw e;}
  finally{active--; $('send').dataset.active=active;}
}
function act(path,fields,headers={}) {return request(path,{chat_id:chatId,message_id:crypto.randomUUID(),...fields},headers).catch(()=>{});}
function send(text,replay=false) { const body=replay&&lastRequest?lastRequest:{chat_id:chatId,message_id:crypto.randomUUID(),text}; lastRequest=body; $('message').value=''; return request('/api/chat',body).catch(()=>{}); }
$('form').onsubmit=e=>{e.preventDefault();if($('message').value.trim())send($('message').value.trim());};
$('sample-city').onclick=()=>send('想去上海住酒店'); $('sample-full').onclick=()=>send(runtime.start+'入住，'+runtime.end+'离店，预算每晚400元，大床房');
$('sample-change').onclick=()=>send('预算改成300元，换双床房'); $('sample-none').onclick=()=>send('预算改成100元'); $('sample-select').onclick=()=>send('选第一个');
$('repeat').onclick=()=>{if(lastRequest)send(lastRequest.text,true);};
$('save-card').onclick=()=>{if(!current)return;const changes={};document.querySelectorAll('[data-field]').forEach(el=>{const key=el.dataset.field;const value=key==='budget'?(el.value?Number(el.value):null):(el.value||null);if(value!==(current.slots[key]??null))changes[key]=value;});if(Object.keys(changes).length)act('/api/requirements/commit',{expected_revision:cardRevision,changes});};
$('reset').onclick=()=>{sessionStorage.setItem('yf-chat',crypto.randomUUID());location.reload();};
fetch('/api/state?chat_id='+encodeURIComponent(chatId)).then(r=>r.json()).then(data=>render(data)).catch(e=>$('error').textContent=e.message);
const events=new EventSource('/api/events?chat_id='+encodeURIComponent(chatId)); events.onmessage=e=>render(JSON.parse(e.data));
