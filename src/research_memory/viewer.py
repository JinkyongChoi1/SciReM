"""Offline viewer for a recorded retrieval; no scores are computed here."""

import json


def render_dashboard(trace: dict, tasks: list | None = None, trials: list | None = None) -> str:
    """Return a self-contained dashboard with only supplied evidence and metrics."""
    payload = json.dumps(
        {"trace": trace, "tasks": tasks or [], "trials": trials or []},
        ensure_ascii=False, allow_nan=False,
    ).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return _HTML.replace("__PAYLOAD__", payload)


_HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shared Research Memory</title>
<style>
:root{color-scheme:light;--ink:#172a35;--muted:#50636d;--line:#d9e1e4;--accent:#216d64}
*{box-sizing:border-box}body{margin:0;background:#f6f8f8;color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:28px 22px}h1{font-size:24px;margin:0}h2{font-size:19px;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px}
p{margin:8px 0 16px}.muted{color:var(--muted);font-size:14px}nav{display:flex;gap:8px;margin:22px 0}
button{font:inherit;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink);padding:8px 16px;min-height:44px;cursor:pointer}
button[aria-selected=true],button[aria-pressed=true]{background:var(--accent);color:white;border-color:var(--accent)}
button:focus-visible,[tabindex]:focus-visible{outline:3px solid #a04b16;outline-offset:3px}
section[hidden]{display:none}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;margin:0 0 16px}
.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(260px,1fr);gap:16px}.row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.badge{font-size:13px;padding:2px 8px;border-radius:4px;background:#edf2f2}.wrap{overflow-wrap:anywhere}pre{white-space:pre-wrap;font:14px/1.6 system-ui,sans-serif}
svg{width:100%;height:auto;min-height:240px}svg text{font:12px system-ui,sans-serif;fill:var(--ink)}svg .node{cursor:pointer}svg .node:focus{outline:none;stroke:#a04b16;stroke-width:4}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);vertical-align:top}
.scroll{overflow:auto}details{margin-top:12px}summary{cursor:pointer}dl{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:14px}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
@media(max-width:720px){main{padding:20px 14px}.grid{grid-template-columns:1fr}.card{padding:16px}nav button{flex:1;padding:8px}h1{font-size:21px}#graph{display:none}#node-list{display:grid;grid-template-columns:1fr}#node-list button{text-align:left}}
</style></head><body><main>
<h1>Shared Research Memory</h1><p class="muted">기록된 retrieval · source evidence · 연구 진행</p>
<nav role="tablist" aria-label="연구 화면">
<button id="tab-research" role="tab" aria-selected="true" aria-controls="research">연구</button>
<button id="tab-memory" role="tab" aria-selected="false" aria-controls="memory" tabindex="-1">메모리</button>
<button id="tab-experiment" role="tab" aria-selected="false" aria-controls="experiment" tabindex="-1">실험</button>
</nav>
<section id="research" role="tabpanel" aria-labelledby="tab-research"><div class="card"><h2>현재 질문</h2><p id="question" class="wrap"></p><details><summary>Trace 정보</summary><div id="identity" class="muted wrap"></div></details></div><div id="tasks"></div></section>
<section id="memory" role="tabpanel" aria-labelledby="tab-memory" hidden>
<div class="card"><h2>질문에서 evidence까지</h2><p id="memory-question" class="wrap"></p>
<div class="row" id="phases"><button aria-pressed="true" data-phase="anchors">1 Anchor</button><button aria-pressed="false" data-phase="rank">2 PPR relevance</button><button aria-pressed="false" data-phase="context">3 Context</button></div>
<p class="muted" id="legend"></p><p class="muted" id="trace-note"></p></div>
<div class="grid"><div class="card"><svg id="graph" viewBox="0 0 620 440" role="group" aria-label="Retrieval graph"></svg><details open><summary>Node 목록 · 선택하여 evidence 확인</summary><div id="node-list" class="row"></div></details><details><summary>관계 목록</summary><div id="edge-list"></div></details></div>
<div class="card" id="node-detail" aria-live="polite"><h2>Evidence</h2><p class="muted">Node를 선택하면 source와 점수가 표시됩니다.</p></div></div>
<div class="card"><h2>Agent context</h2><p class="muted">실제 포함된 memory만 표시합니다. Relevance는 truth나 실행 승인을 뜻하지 않습니다.</p><div id="packet"></div></div></section>
<section id="experiment" role="tabpanel" aria-labelledby="tab-experiment" hidden><div class="card"><h2>실험 기록</h2><p class="muted">제공된 trial metrics만 표시합니다. 측정하지 않은 값은 pending입니다.</p><div id="trials" class="scroll"></div></div></section>
</main><script id="data" type="application/json">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), trace=data.trace;
if(trace.synthetic)document.querySelector('h1').after(Object.assign(document.createElement('p'),{textContent:'Synthetic demo · 실제 empirical result 아님',className:'muted'}));
const allNodes=(trace.nodes||trace.graph?.nodes||[]).map(n=>({...n,id:n.id??n.memory_id,label:n.label??n.title??n.aliases?.[0]??String(n.text??n.memory_id??n.id).slice(0,70),kind:n.kind??n.type??n.memory_type,semantic_score:n.semantic_score??n.similarity,reason:n.reason??n.selection_reason}));
const nodes=allNodes.slice(0,40), edges=(trace.edges||trace.graph?.edges||[]).map(e=>({...e,source:e.origin??e.source}));
const byId=new Map(nodes.map(n=>[String(n.id),n]));
const $=id=>document.getElementById(id);
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);if(cls)e.className=cls;return e}
function display(value){return value===undefined||value===null?'pending':typeof value==='object'?JSON.stringify(value):String(value)}
function metric(n,k){return n[k]??n.scores?.[k]}
function score(n){const v=Number(metric(n,'ppr_score')??n.ppr);return Number.isFinite(v)?v:0}
function seed(n){return Number(metric(n,'seed_weight')??n.anchor_weight??0)>0}
const contextIds=new Set((trace.included_ids||trace.context_ids||trace.selected_ids||[]).map(String));
function included(n){return n.included===true||contextIds.has(String(n.id))}
const query=trace.query?.text??trace.query??trace.question??'질문 기록 없음';
$('question').textContent=display(query);$('memory-question').textContent=display(query);
$('identity').textContent='Snapshot '+display(trace.snapshot_version??trace.snapshot_id??trace.snapshot)+' · Request '+display(trace.request_id);
const tabs=[...document.querySelectorAll('[role=tab]')];
function selectTab(tab){tabs.forEach(t=>{const active=t===tab;t.setAttribute('aria-selected',String(active));t.tabIndex=active?0:-1;$(t.getAttribute('aria-controls')).hidden=!active})}
tabs.forEach((tab,i)=>{tab.onclick=()=>selectTab(tab);tab.onkeydown=e=>{let next;if(e.key==='ArrowRight')next=(i+1)%tabs.length;if(e.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;if(e.key==='Home')next=0;if(e.key==='End')next=tabs.length-1;if(next!==undefined){e.preventDefault();selectTab(tabs[next]);tabs[next].focus()}}});
if(!data.tasks.length)$('tasks').append(el('p','Task 상태 기록 없음. 이 viewer는 job을 제출하거나 승인을 처리하지 않습니다.','muted'));
data.tasks.forEach(task=>{const c=el('article',undefined,'card');c.append(el('h2',task.title??task.question??task.id??'Research task'));c.append(el('span',task.status??'pending','badge'));c.append(el('p',task.next_action??task.next_step??'다음 action 기록 없음'));if(task.decision){c.append(el('h3','다음 결정'));c.append(el('p',display(task.decision)))}$('tasks').append(c)});
function detail(n){const d=$('node-detail');d.replaceChildren(el('h2',n.label??n.title??n.id));d.append(el('pre',n.text??n.summary??n.excerpt??'Source excerpt 기록 없음'));const list=el('dl');for(const [label,value] of [['Source',n.source_refs??n.sources??n.source],['PPR relevance',metric(n,'ppr_score')??n.ppr],['Context',included(n)?'included':'not included']]){list.append(el('dt',label),el('dd',display(value)))}d.append(list);const more=el('details'),meta=el('dl');more.append(el('summary','기록 세부정보'));for(const [label,value] of [['Type',n.kind??n.type],['Anchor weight',metric(n,'seed_weight')??n.anchor_weight],['Semantic score',metric(n,'semantic_score')],['Validation status',n.validation_status??n.status],['선택 근거',n.inclusion_reason??n.reason]]){meta.append(el('dt',label),el('dd',display(value)))}more.append(meta);d.append(more)}
nodes.forEach(n=>{const b=el('button',n.label??n.title??n.id);b.onclick=()=>detail(n);$('node-list').append(b)});
edges.filter(edge=>byId.has(String(edge.source))&&byId.has(String(edge.target))).forEach(edge=>$('edge-list').append(el('p',String(edge.source)+' → '+String(edge.target)+' · '+(edge.relation??edge.type??'relation'),'muted')));
const positions=new Map(nodes.map((n,i)=>{const a=2*Math.PI*i/Math.max(nodes.length,1)-Math.PI/2;return[String(n.id),[310+235*Math.cos(a),215+160*Math.sin(a)]]}));
let phase='anchors';const NS='http://www.w3.org/2000/svg';
function svgEl(tag,attrs){const e=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));return e}
function draw(){const graph=$('graph');graph.replaceChildren();if(!nodes.length){const t=svgEl('text',{x:30,y:70});t.textContent='Graph 기록 없음';graph.append(t)}
edges.forEach(e=>{const a=positions.get(String(e.source)),b=positions.get(String(e.target));if(!a||!b)return;const line=svgEl('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#c7d2d5','stroke-width':1});const title=svgEl('title',{});title.textContent=e.relation??e.type??'relation';line.append(title);graph.append(line)});
const maxScore=Math.max(0,...nodes.map(score));nodes.forEach(n=>{const [x,y]=positions.get(String(n.id)),active=phase==='anchors'?seed(n):phase==='context'?included(n):true;const radius=phase==='rank'&&maxScore?9+18*Math.sqrt(Math.max(0,score(n))/maxScore):13;const concept=['concept','entity','method','variable','dataset'].includes(n.kind??n.type);const attrs={fill:seed(n)?'#e2a34e':'#4b8981',stroke:included(n)?'#163c37':'#fff','stroke-width':included(n)?3:1,opacity:active?1:0.22,tabindex:0,role:'button','aria-label':String(n.label??n.title??n.id),class:'node'};const shape=concept?svgEl('circle',{...attrs,cx:x,cy:y,r:radius}):svgEl('rect',{...attrs,x:x-radius,y:y-radius,width:radius*2,height:radius*2,rx:3});shape.onclick=()=>detail(n);shape.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();detail(n)}};const title=svgEl('title',{});title.textContent=n.label??n.title??n.id;shape.append(title);graph.append(shape);const label=svgEl('text',{x,y:y+radius+17,'text-anchor':'middle'});const name=String(n.label??n.title??n.id);label.textContent=name.length>24?name.slice(0,23)+'…':name;graph.append(label)});
$('legend').textContent=phase==='anchors'?'주황색: query anchor · 원: concept · 사각형: evidence':phase==='rank'?'크기: 기록된 PPR relevance (현재 graph 내 상대 크기) · 진한 테두리: context 포함':'진한 node: 실제 context 포함 · 흐린 node: 미포함';}
document.querySelectorAll('[data-phase]').forEach(b=>b.onclick=()=>{phase=b.dataset.phase;document.querySelectorAll('[data-phase]').forEach(p=>p.setAttribute('aria-pressed',String(p===b)));draw()});
$('trace-note').textContent=trace.display_truncated||trace.truncated||allNodes.length>40?'일부 node만 표시된 graph입니다 (최대 40개). 전체 trace는 원본 artifact를 확인하세요.':'제공된 trace의 graph를 표시합니다. 단계 전환은 저장된 seed·score·inclusion을 설명하며 propagation 재계산이나 animation이 아닙니다.';
draw();
const packet=trace.context_packet??trace.brief;
if(packet!==undefined)$('packet').append(el('pre',display(packet)));else{const selected=allNodes.filter(included);if(!selected.length)$('packet').append(el('p','포함된 evidence 기록 없음','muted'));selected.forEach(n=>{const article=el('article');article.append(el('h3',n.label??n.title??n.id),el('pre',n.text??n.excerpt??n.summary??'Excerpt 기록 없음'));$('packet').append(article)})}
if(!data.trials.length)$('trials').append(el('p','Trial 기록 없음','muted'));
else{const cols=[['Condition','condition'],['첫 faithful (s)','time_to_first_faithful_seconds'],['Queue (s)','queue_time'],['Compute (s)','compute_time'],['Errors','errors'],['Fidelity','fidelity'],['Mechanism review','mechanism_validity']];const table=el('table'),head=el('tr'),body=el('tbody');cols.forEach(([label])=>{const th=el('th',label);th.scope='col';head.append(th)});const thead=el('thead');thead.append(head);data.trials.forEach(t=>{const row=el('tr');cols.forEach(([,key])=>row.append(el('td',display(t[key]??t.metrics?.[key]))));body.append(row)});table.append(thead,body);$('trials').append(table)}
</script></body></html>'''
