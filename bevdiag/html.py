"""Interactive single-page failure report (docs/index.html + an artifact-ready body)."""
from __future__ import annotations

import json
import os

TEMPLATE = r"""<title>Long-Range BEV Failure Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
/* Layout: one 1120px reading column; sections separated by rules, charts full-width, gallery as a 2-up grid. */
:root{
  --bg:#f7f8f7; --panel:#ffffff; --fg:#111413; --fg2:#4d5450; --muted:#7f8682; --rule:#dfe3e0;
  --accent:#2a78d6; --car:#2a78d6; --large:#eb6834; --vru:#1baf7a; --miss:#6250d6; --bad:#d03b3b; --good:#0b7a0b;
  --chip:#eef1ef; --seq0:#f0efec; --seq1:#cde2fb; --seq2:#86b6ef; --seq3:#3987e5; --seq4:#1c5cab; --seq5:#0d366b;
  --f-display:"Archivo","Segoe UI",system-ui,sans-serif; --f-body:"Inter","Segoe UI",system-ui,sans-serif;
  --f-mono:"JetBrains Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#131514; --panel:#1b1e1c; --fg:#eef0ee; --fg2:#bcc3be; --muted:#8a918c; --rule:#2c312e;
  --accent:#3987e5; --car:#3987e5; --large:#d95926; --vru:#199e70; --miss:#9085e9; --bad:#e66767; --good:#3bbf3b;
  --chip:#232825; --seq0:#262624; --seq1:#104281; --seq2:#1c5cab; --seq3:#3987e5; --seq4:#6da7ec; --seq5:#b7d3f6; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#131514; --panel:#1b1e1c; --fg:#eef0ee; --fg2:#bcc3be; --muted:#8a918c; --rule:#2c312e;
  --accent:#3987e5; --car:#3987e5; --large:#d95926; --vru:#199e70; --miss:#9085e9; --bad:#e66767; --good:#3bbf3b;
  --chip:#232825; --seq0:#262624; --seq1:#104281; --seq2:#1c5cab; --seq3:#3987e5; --seq4:#6da7ec; --seq5:#b7d3f6; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--f-body);margin:0}
.wrap{max-width:1120px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:40px 80px}
h1,h2,h3{font-family:var(--f-display);text-wrap:balance;margin:0;letter-spacing:-0.01em}
h1{font-size:clamp(28px,4.4vw,44px);font-weight:800;line-height:1.08}
h2{font-size:24px;font-weight:700;margin-bottom:6px}
h3{font-size:16px;font-weight:700}
p{margin:0;max-width:68ch;color:var(--fg2)}
.eyebrow{font:600 12px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{font-size:17px;color:var(--fg2);max-width:70ch}
header{display:grid;gap:14px;padding-bottom:28px;border-bottom:1px solid var(--rule)}
.meta{display:flex;flex-wrap:wrap;gap:8px 18px;font:13px var(--f-mono);color:var(--muted)}
section{padding-block:36px;border-bottom:1px solid var(--rule);display:grid;gap:16px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:16px 16px 14px;display:grid;gap:4px;min-width:0}
.kpi b{font:700 30px/1.1 var(--f-display);font-variant-numeric:tabular-nums}
.kpi span{font-size:13px;color:var(--fg2)}
.kpi.alert b{color:var(--bad)}
.ruler{position:relative;height:84px;margin-top:6px}
.ruler .bar{position:absolute;top:30px;height:26px;border-radius:2px}
.ruler .mark{position:absolute;top:18px;height:48px;border-left:2px solid var(--fg)}
.ruler .mark span{position:absolute;top:-18px;left:4px;white-space:nowrap;font:12px var(--f-mono);color:var(--fg2)}
.ruler .mark.stop{border-color:var(--bad)} .ruler .mark.stop span{color:var(--bad);left:auto;right:4px}
.ruler .axis{position:absolute;top:62px;font:11px var(--f-mono);color:var(--muted);transform:translateX(-50%)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.seg{display:inline-flex;border:1px solid var(--rule);border-radius:6px;overflow:hidden;background:var(--panel)}
.seg button{font:500 13px var(--f-body);color:var(--fg2);background:none;border:0;padding:6px 12px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--fg);color:var(--bg)}
.seg button:focus-visible,.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.chart{width:100%;min-height:380px;background:var(--panel);border:1px solid var(--rule);border-radius:8px}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.note{font-size:13px;color:var(--muted);max-width:none}
.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.case{background:var(--panel);border:1px solid var(--rule);border-radius:8px;overflow:hidden;display:grid;grid-template-rows:auto 1fr;min-width:0}
.case img{display:block;width:100%;height:auto;cursor:zoom-in;background:var(--chip)}
.case .body{padding:12px 14px 14px;display:grid;gap:8px;align-content:start}
.case .hd{display:flex;justify-content:space-between;gap:8px;font:12px var(--f-mono);color:var(--muted);flex-wrap:wrap}
.tags{display:flex;flex-wrap:wrap;gap:6px}
.chip{font:500 12px var(--f-body);background:var(--chip);color:var(--fg2);border:1px solid var(--rule);border-radius:999px;padding:3px 10px;cursor:pointer}
.chip[aria-pressed="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.pill{font:600 11px var(--f-mono);padding:2px 8px;border-radius:999px;border:1px solid var(--rule);color:var(--fg2)}
.pill.held{border-color:var(--accent);color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule)}
th{font:600 12px var(--f-mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
td.num,th.num{text-align:right;font-family:var(--f-mono)}
.up{color:var(--good)} .down{color:var(--bad)}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:8px}
ol.recs{margin:0;padding-left:22px;display:grid;gap:10px;max-width:78ch}
ol.recs li{color:var(--fg2)} ol.recs b{color:var(--fg)}
.gif{width:100%;border-radius:8px;border:1px solid var(--rule);display:block}
dialog{border:0;padding:0;background:transparent;max-width:min(1400px,96vw)}
dialog img{width:100%;border-radius:6px}
dialog::backdrop{background:rgba(0,0,0,.78)}
code{font:13px var(--f-mono);background:var(--chip);padding:1px 6px;border-radius:4px}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.two,.gallery{grid-template-columns:minmax(0,1fr)}}
@media (max-width:640px){.ruler{height:104px;margin-top:20px}.ruler .mark.stop span{top:-34px}.ruler .mark span{font-size:11px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Evaluate · diagnose · improve — camera BEV perception for freight</div>
  <h1>Where a camera-only BEV detector stops seeing trucks</h1>
  <p class="lede">The official BEVFormer-tiny weights, ported to plain PyTorch and run on public nuScenes mini. Errors are sliced by distance, class, size and lighting, and one cheap fix is measured. Within 30&nbsp;m it finds most things. Past 40&nbsp;m it goes blind, and past 51.2&nbsp;m it cannot see anything by construction.</p>
  <div class="meta"><span id="m-model"></span><span id="m-data"></span><span id="m-speed"></span></div>
</header>

<section>
  <div class="kpis" id="kpis"></div>
  <div>
    <h3>Range budget</h3>
    <p class="note">Recall per distance band on held-out scenes, against the BEV grid edge and the roughly 160&nbsp;m (525&nbsp;ft) a loaded truck needs to stop from 65&nbsp;mph.</p>
    <div class="ruler" id="ruler" aria-label="Recall by distance compared with stopping distance"></div>
  </div>
  <img class="gif" src="assets/scene.gif" alt="Animated camera and bird's-eye view of a held-out scene with true positives, false positives and misses drawn">
</section>

<section id="range">
  <div><h2>1 · Performance by distance</h2>
  <p>Recall counts ground-truth objects bucketed by their true distance. Precision counts predictions bucketed by their own predicted distance. AP uses the nuScenes 2&nbsp;m centre-distance match inside each band. The shaded region is outside the model's ±51.2&nbsp;m BEV grid.</p></div>
  <div class="controls">
    <div class="seg" role="group" aria-label="Dataset"><button id="set-heldout" aria-pressed="true">Held-out (2 scenes)</button><button id="set-all_mini" aria-pressed="false">All mini (10 scenes)</button></div>
    <div class="seg" role="group" aria-label="Metric"><button id="met-recall" aria-pressed="true">Recall</button><button id="met-precision" aria-pressed="false">Precision</button><button id="met-ap" aria-pressed="false">AP</button></div>
  </div>
  <div class="chart" id="c-range"></div>
  <p class="note">"All mini" adds 8 scenes that are part of BEVFormer's training split, so its numbers are optimistic. The drop-off still matches.</p>
</section>

<section>
  <div><h2>2 · Class × distance</h2><p>Recall per class and distance band. Hover a cell for the object count. Cells with fewer than 5 objects are left blank.</p></div>
  <div class="chart" id="c-heat" style="min-height:400px"></div>
</section>

<section>
  <div><h2>3 · Size, localisation and lighting</h2>
  <p>Long vehicles (8&nbsp;m and up) are found reliably up close but almost never at 40&nbsp;m or more. Centre error grows about fourfold from 0–10&nbsp;m to 40–50&nbsp;m, mostly along the camera ray.</p></div>
  <div class="two"><div class="chart" id="c-size"></div><div class="chart" id="c-loc"></div></div>
  <div class="chart" id="c-light"></div>
</section>

<section>
  <div><h2>4 · Failure taxonomy</h2>
  <p>Every miss and every false positive at the operating threshold gets one tag. Rules, in priority order: outside the grid, found below threshold, wrong class, same-class box 2–4&nbsp;m away (depth), ≤40% visible, else missed. False positives: duplicate, wrong class, depth-shifted, or a ghost with or without LiDAR support.</p></div>
  <div class="controls"><div class="seg" role="group" aria-label="Error kind"><button id="fk-fn_tags" aria-pressed="true">Misses (FN)</button><button id="fk-fp_tags" aria-pressed="false">False positives</button></div></div>
  <div class="chart" id="c-fail"></div>
</section>

<section>
  <div><h2>5 · Worst cases</h2><p>Frames ranked by missed large vehicles (×3), other in-grid misses and confident false positives. Each comes with an automatic one-line diagnosis. Click an image to enlarge it.</p></div>
  <div class="tags" id="filters"></div>
  <div class="gallery" id="gallery"></div>
</section>

<section>
  <div><h2>6 · Tested fix: LiDAR sanity filter</h2>
  <p>Drop any camera box whose box, grown by 0.5&nbsp;m per side, contains no LiDAR return. LiDAR only vetoes boxes; it never creates or moves one. The setting was chosen before looking at results, and a sweep shows how sensitive it is.</p></div>
  <div class="kpis" id="fixkpis"></div>
  <div class="chart" id="c-fix"></div>
  <div class="tablewrap"><table id="t-sweep"></table></div>
  <p class="note" id="fixb"></p>
</section>

<section>
  <div><h2>7 · Recommendations</h2></div>
  <ol class="recs">
    <li><b>Widen the BEV grid before anything else.</b> Every object past 51.2&nbsp;m is a structural miss (<span id="r-grid"></span> of held-out misses). For highway freight, train with at least ±100&nbsp;m forward range, using a non-uniform or polar grid so near-field resolution is not lost.</li>
    <li><b>Add a long-range camera and higher input resolution.</b> This model sees images at half resolution (800×450). At that scale a 2.5&nbsp;m-wide truck 80&nbsp;m away is about 20 pixels wide. A narrow-FOV forward camera at full resolution is the cheapest way to get more pixels on target.</li>
    <li><b>Fix depth before thresholds.</b> "Real object, 2–4&nbsp;m off" is the largest false-positive class, and centre error grows fourfold with range. Depth supervision from LiDAR (BEVDepth-style) or radar fusion goes after the root cause.</li>
    <li><b>Ship the LiDAR veto as a runtime guard.</b> It costs one point-in-box test per detection and removes confident ghosts. Grow the box first: without the 0.5&nbsp;m margin it deletes correct but depth-shifted boxes and loses 3.3 mAP points.</li>
    <li><b>Don't tune thresholds per range on small data.</b> Range-aware thresholds overfit when tested across scenes (F1 dropped). Recalibrate scores per range on a large held-out set instead.</li>
    <li><b>Treat articulated vehicles as their own slice.</b> Recall on boxes longer than 12&nbsp;m lags shorter vehicles even up close in the training scenes. Track this in CI with the same slicing code.</li>
  </ol>
</section>

<section style="border-bottom:0">
  <div><h2>Reproduce</h2>
  <p>One command reruns inference, the official nuScenes eval, slicing, the fix and every figure. It is capped at 90% of CPU threads and GPU memory.</p></div>
  <div class="tablewrap" style="padding:12px 14px"><code>python -m bevdiag run --dataroot data/nuscenes</code></div>
  <p class="note">Metrics: nuScenes devkit DetectionEval (detection_cvpr_2019) on mini_val. Slices: the repo's own nuScenes-style matcher, cross-checked against the devkit in unit tests. All numbers on this page come from report_data.json, written by that run.</p>
</section>
</div>
<dialog id="zoom"><img alt="" id="zoomimg"></dialog>

<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const isDark = () => css('--bg') !== '#f7f8f7';
const pct = v => v == null ? '–' : (v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0) + '%';
const RB = ["0-10","10-20","20-30","30-40","40-50","50-60","60-80","80+"];
const G = {car:['Cars','--car'], large_vehicle:['Trucks · buses · trailers','--large'], vru:['Pedestrians & cyclists','--vru']};
const H = D.headline, held = D.sets.heldout;
let state = {set:'heldout', metric:'recall', fk:'fn_tags', tag:null};

document.getElementById('m-model').textContent = 'BEVFormer-tiny · official epoch-24 weights · pure PyTorch port';
document.getElementById('m-data').textContent = `nuScenes v1.0-mini · ${held.n_frames} held-out frames · ${held.n_gt.toLocaleString()} objects`;
document.getElementById('m-speed').textContent = `${D.timing ? D.timing.median_ms.toFixed(0)+' ms/frame on '+D.timing.gpu : ''}`;

function kpis(el, items){ el.innerHTML = items.map(([v,l,c]) => `<div class="kpi ${c||''}"><b>${v}</b><span>${l}</span></div>`).join(''); }
kpis(document.getElementById('kpis'), [
  [H.baseline_mAP_mini_val.toFixed(3) + ' / ' + H.baseline_NDS_mini_val.toFixed(3), 'official mAP / NDS (mini_val; paper on full val: 0.252 / 0.354)'],
  [pct(H.recall_0_30m_heldout), 'recall on objects within 30 m'],
  [pct(H.recall_40_50m_heldout), 'recall at 40–50 m', 'alert'],
  [pct(H.recall_beyond_50m_heldout), 'recall beyond 50 m', 'alert'],
]);

function ruler(){
  const el = document.getElementById('ruler'), max = 170;
  const rp = held.range_profile.filter(r => r.group === 'all');
  const edges = [0,10,20,30,40,50,60,80,170];
  const seq = ['--seq1','--seq2','--seq3','--seq4','--seq5'].map(css);
  let h = '';
  rp.forEach((r,i) => { const v = r.recall || 0, c = seq[Math.min(4, Math.floor(v*5))];
    h += `<div class="bar" title="${r.range} m: recall ${pct(r.recall)} (n=${r.n_gt})" style="left:${edges[i]/max*100}%;width:calc(${(edges[i+1]-edges[i])/max*100}% - 2px);background:${c}"></div>`; });
  h += `<div class="mark" style="left:${51.2/max*100}%"><span>BEV grid 51.2 m</span></div>`;
  h += `<div class="mark stop" style="left:${160/max*100}%"><span>≈160 m truck stopping distance</span></div>`;
  [0,40,80,120,160].forEach(v => h += `<div class="axis" style="left:${v/max*100}%">${v} m</div>`);
  el.innerHTML = h;
}

function layout(extra){
  const f = css('--fg2'), g = css('--rule');
  return Object.assign({paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{family:'Inter, Segoe UI, sans-serif', color:f, size:12},
    margin:{l:56,r:18,t:40,b:96}, xaxis:{gridcolor:g, zeroline:false, linecolor:g, title:{standoff:8}}, yaxis:{gridcolor:g, zeroline:false, linecolor:g},
    legend:{orientation:'h', y:-0.32, x:0, yanchor:'top'}, hoverlabel:{font:{family:'JetBrains Mono, monospace'}}}, extra || {});
}
const cfg = {displayModeBar:false, responsive:true};
const shadeGrid = () => ({type:'rect', xref:'x', yref:'paper', x0:4.62, x1:7.5, y0:0, y1:1, fillcolor:css('--rule'), opacity:0.6, line:{width:0}, layer:'below'});

function drawRange(){
  const rp = D.sets[state.set].range_profile;
  const tr = Object.entries(G).map(([g,[name,col]]) => {
    const rows = rp.filter(r => r.group === g);
    return {x:RB, y:rows.map(r => r[state.metric]), name, mode:'lines+markers', line:{color:css(col), width:2.5}, marker:{size:8, line:{color:css('--panel'), width:2}},
      customdata:rows.map(r => [r.n_gt, r.n_pred]), hovertemplate:`%{x} m · ${name}<br>${state.metric}: %{y:.1%}<br>GT n=%{customdata[0]} · preds n=%{customdata[1]}<extra></extra>`};
  });
  Plotly.react('c-range', tr, layout({title:{text:`${{recall:'Recall',precision:'Precision',ap:'AP'}[state.metric]} by distance`, x:0.01, font:{size:14,color:css('--fg')}},
    yaxis:{range:[-0.02,1.05], tickformat:'.0%', gridcolor:css('--rule'), title:''}, xaxis:{title:'metres from ego', gridcolor:css('--rule')}, shapes:[shadeGrid()],
    annotations:[{x:5.3, y:0.95, xref:'x', yref:'paper', text:'outside ±51.2 m grid', showarrow:false, font:{size:11, color:css('--muted')}}]}), cfg);
}

function drawHeat(){
  const order = ['car','truck','bus','trailer','construction_vehicle','pedestrian','motorcycle','bicycle','traffic_cone','barrier'];
  const rows = D.sets[state.set].class_range_recall;
  const z = order.map(c => RB.map(rb => { const r = rows.find(r => r.cls===c && r.range_bin===rb); return r && r.size>=5 ? r.mean : null; }));
  const n = order.map(c => RB.map(rb => { const r = rows.find(r => r.cls===c && r.range_bin===rb); return r ? r.size : 0; }));
  const scale = ['--seq0','--seq1','--seq2','--seq3','--seq4','--seq5'].map((v,i) => [i/5, css(v)]);
  Plotly.react('c-heat', [{type:'heatmap', z, x:RB, y:order.map(o => o.replace('_',' ')), zmin:0, zmax:1, colorscale:scale, customdata:n, xgap:2, ygap:2,
    text:z.map(r => r.map(v => v==null ? '' : Math.round(v*100)+'%')), texttemplate:'%{text}', textfont:{size:11},
    hovertemplate:'%{y} · %{x} m<br>recall %{z:.1%} · n=%{customdata}<extra></extra>', colorbar:{tickformat:'.0%', outlinewidth:0, thickness:10}}],
    layout({margin:{l:130,r:10,t:40,b:48}, title:{text:`Recall by class × distance — ${state.set==='heldout'?'held-out':'all mini'}`, x:0.01, font:{size:14,color:css('--fg')}},
      xaxis:{title:'metres from ego', gridcolor:'rgba(0,0,0,0)'}, yaxis:{autorange:'reversed', gridcolor:'rgba(0,0,0,0)'}}), cfg);
}

function drawSize(){
  const rows = D.sets[state.set].size_recall_by_range;
  const L = ['<2 m','2-5 m','5-8 m','8-12 m','>12 m'].filter(l => rows.some(r => r.length_bin===l));
  const tr = [['<40 m','--car'],['>=40 m','--large']].map(([far,col]) => ({type:'bar', name:'objects '+far.replace('>=','≥'), x:L,
    y:L.map(l => { const r = rows.find(r => r.length_bin===l && r.far===far); return r && r.size>=5 ? r.mean : null; }),
    customdata:L.map(l => { const r = rows.find(r => r.length_bin===l && r.far===far); return r ? r.size : 0; }),
    marker:{color:css(col), line:{color:css('--panel'), width:2}}, hovertemplate:'%{x} · '+far+'<br>recall %{y:.1%} · n=%{customdata}<extra></extra>'}));
  Plotly.react('c-size', tr, layout({barmode:'group', title:{text:'Vehicle recall by true length', x:0.01, font:{size:14,color:css('--fg')}}, yaxis:{tickformat:'.0%', range:[0,1.05], gridcolor:css('--rule')}}), cfg);
}

function drawLoc(){
  const rows = D.sets.all_mini.tp_errors;
  const tr = Object.entries(G).map(([g,[name,col]]) => { const r = rows.filter(x => x.group===g && x.n>=8);
    return {x:r.map(x => x.range), y:r.map(x => x.trans_med), name, mode:'lines+markers', line:{color:css(col), width:2.5}, marker:{size:7},
      customdata:r.map(x => [x.trans_p90, x.radial_bias, x.n]), hovertemplate:`%{x} m · ${name}<br>median %{y:.2f} m · p90 %{customdata[0]:.2f} m<br>radial bias %{customdata[1]:+.2f} m · n=%{customdata[2]}<extra></extra>`}; });
  Plotly.react('c-loc', tr, layout({title:{text:'Median BEV centre error of true positives', x:0.01, font:{size:14,color:css('--fg')}}, yaxis:{title:'metres', gridcolor:css('--rule')}, xaxis:{title:'true distance (m)', gridcolor:css('--rule')}}), cfg);
}

function drawLight(){
  const rows = D.sets.all_mini.lighting_range_recall;
  const tr = [['day','--large'],['night','--miss']].map(([l,col]) => ({x:RB, name:l, mode:'lines+markers', line:{color:css(col), width:2.5}, marker:{size:7},
    y:RB.map(rb => { const r = rows.find(r => r.lighting===l && r.range_bin===rb); return r && r.size>=10 ? r.mean : null; }),
    customdata:RB.map(rb => { const r = rows.find(r => r.lighting===l && r.range_bin===rb); return r ? r.size : 0; }),
    hovertemplate:`%{x} m · ${l}<br>recall %{y:.1%} · n=%{customdata}<extra></extra>`}));
  Plotly.react('c-light', tr, layout({title:{text:'Day vs night recall (all mini; the 3 night scenes were in training)', x:0.01, font:{size:14,color:css('--fg')}},
    yaxis:{tickformat:'.0%', range:[-0.02,1.05], gridcolor:css('--rule')}, xaxis:{title:'metres from ego', gridcolor:css('--rule')}, shapes:[shadeGrid()]}), cfg);
}

const FN = {low_confidence:['found, below threshold','--seq3'], depth_error:['2–4 m off (depth)','--large'], missed:['missed outright','--bad'],
  heavily_occluded:['≤40% visible','--miss'], class_confusion:['wrong class','--vru'], beyond_bev_grid:['outside ±51.2 m grid','--muted']};
const FP = {depth_error:['real object, 2–4 m off','--large'], duplicate:['duplicate box','--miss'], ghost_no_lidar:['ghost, 0 LiDAR pts','--bad'],
  ghost_with_lidar:['ghost, has LiDAR pts','--seq3'], class_confusion:['wrong class','--vru']};
function drawFail(){
  const rows = D.sets[state.set][state.fk], M = state.fk === 'fn_tags' ? FN : FP;
  const tr = Object.entries(M).map(([t,[name,col]]) => ({type:'bar', name, x:RB, marker:{color:css(col), line:{color:css('--panel'), width:1.5}},
    y:RB.map(rb => rows.filter(r => r.fail_tag===t && r.range_bin===rb).reduce((a,r) => a + r.n, 0)), hovertemplate:`%{x} m · ${name}: %{y}<extra></extra>`}));
  Plotly.react('c-fail', tr, layout({barmode:'stack', title:{text:(state.fk==='fn_tags'?'Why objects were missed':'Why boxes were wrong')+` — ${state.set==='heldout'?'held-out':'all mini'}`, x:0.01, font:{size:14,color:css('--fg')}},
    yaxis:{title:'count', gridcolor:css('--rule')}, xaxis:{title:'metres from ego', gridcolor:css('--rule')}}), cfg);
}

const TAGN = {beyond_bev_grid:'outside grid', low_confidence:'below threshold', depth_error:'depth error', ghost_no_lidar:'ghost (no LiDAR)',
  duplicate:'duplicates', missed:'missed', heavily_occluded:'occluded', class_confusion:'wrong class'};
function gallery(){
  const tags = [...new Set(D.gallery.flatMap(c => c.tags))];
  document.getElementById('filters').innerHTML = ['all', ...tags].map(t => `<button class="chip" data-t="${t}" aria-pressed="${(state.tag||'all')===t}">${t==='all'?'All cases':TAGN[t]||t}</button>`).join('');
  document.querySelectorAll('#filters .chip').forEach(b => b.onclick = () => { state.tag = b.dataset.t === 'all' ? null : b.dataset.t; gallery(); });
  const th = isDark() ? 'image_dark' : 'image_light';
  document.getElementById('gallery').innerHTML = D.gallery.filter(c => !state.tag || c.tags.includes(state.tag)).map(c => `
    <article class="case"><img loading="lazy" src="assets/gallery/${c[th]}" alt="${c.scene} frame ${c.frame}: camera and bird's-eye view">
    <div class="body"><div class="hd"><span>${c.scene} · frame ${c.frame} · ${c.camera.replace(/_/g,' ')}</span><span class="pill ${c.split==='held-out'?'held':''}">${c.split}</span></div>
    <p style="color:var(--fg)">${c.diagnosis}</p>
    <div class="tags">${c.tags.map(t => `<span class="pill">${TAGN[t]||t}</span>`).join('')}<span class="pill">${c.n_fn} FN · ${c.n_fp} FP</span></div></div></article>`).join('');
  document.querySelectorAll('.case img').forEach(i => i.onclick = () => { document.getElementById('zoomimg').src = i.src; document.getElementById('zoom').showModal(); });
}
document.getElementById('zoom').onclick = e => e.currentTarget.close();

function fix(){
  const fa = D.fix_lidar_filter, ch = fa.chosen, base = held.operating_point, op = ch.op_heldout, b0 = D.baseline_official;
  const d = (a,b,f=3,p=false) => { const x = b - a; const s = p ? (x*100).toFixed(1)+' pts' : x.toFixed(f); return `<span class="${x>=0?'up':'down'}">${x>=0?'+':''}${s}</span>`; };
  kpis(document.getElementById('fixkpis'), [
    [`${pct(base.precision)} → ${pct(op.precision)}`, 'precision at threshold (held-out)'],
    [`${pct(base.recall)} → ${pct(op.recall)}`, 'recall at threshold (held-out)'],
    [`${b0.mAP.toFixed(3)} → ${ch.mAP.toFixed(3)}`, 'official mAP (mini_val)'],
    [`${b0.NDS.toFixed(3)} → ${ch.NDS.toFixed(3)}`, 'official NDS (mini_val)'],
  ]);
  const bp = held.range_profile.filter(r => r.group==='all').slice(0,6), ap = ch.range_heldout.filter(r => r.group==='all').slice(0,6);
  const X = RB.slice(0,6);
  Plotly.react('c-fix', [
    {type:'bar', name:'precision · baseline', x:X, y:bp.map(r=>r.precision), marker:{color:css('--rule')}, hovertemplate:'%{x} m · baseline precision %{y:.1%}<extra></extra>'},
    {type:'bar', name:'precision · + LiDAR filter', x:X, y:ap.map(r=>r.precision), marker:{color:css('--accent')}, hovertemplate:'%{x} m · filtered precision %{y:.1%}<extra></extra>'},
    {type:'scatter', mode:'lines+markers', name:'recall · baseline', x:X, y:bp.map(r=>r.recall), line:{color:css('--muted'), dash:'dot', width:2}, hovertemplate:'%{x} m · baseline recall %{y:.1%}<extra></extra>'},
    {type:'scatter', mode:'lines+markers', name:'recall · + LiDAR filter', x:X, y:ap.map(r=>r.recall), line:{color:css('--large'), width:2.5}, hovertemplate:'%{x} m · filtered recall %{y:.1%}<extra></extra>'},
  ], layout({barmode:'group', title:{text:'Before / after by distance (held-out, inside the grid)', x:0.01, font:{size:14,color:css('--fg')}}, yaxis:{tickformat:'.0%', range:[0,1.05], gridcolor:css('--rule')}, xaxis:{title:'metres from ego'}}), cfg);
  const rows = fa.sweep.map(r => `<tr${(r.min_points===ch.min_points && r.enlarge_m===ch.enlarge_m)?' style="font-weight:700;background:var(--chip)"':''}><td>${r.min_points}</td><td class="num">${r.enlarge_m.toFixed(1)} m</td>
    <td class="num">${r.mAP.toFixed(4)} ${d(b0.mAP, r.mAP, 4)}</td><td class="num">${r.NDS.toFixed(4)} ${d(b0.NDS, r.NDS, 4)}</td>
    <td class="num">${pct(r.op_heldout.precision)}</td><td class="num">${pct(r.op_heldout.recall)}</td><td class="num">${r.op_heldout.f1.toFixed(3)}</td>
    <td class="num">${r.removed_fp_heldout}</td><td class="num">${r.removed_tp_heldout}</td></tr>`).join('');
  document.getElementById('t-sweep').innerHTML = `<thead><tr><th>min pts</th><th class="num">box grown</th><th class="num">mAP</th><th class="num">NDS</th><th class="num">precision</th><th class="num">recall</th><th class="num">F1</th><th class="num">FPs removed</th><th class="num">TPs removed</th></tr></thead><tbody>${rows}</tbody>`;
  const fb = D.fix_range_threshold;
  document.getElementById('fixb').innerHTML = `Highlighted row is the pre-chosen setting. Baseline F1 ${base.f1.toFixed(3)}. Also tried and rejected: per-range score thresholds, fitted on one held-out scene and tested on the other. F1 went from ${fb.global.f1.toFixed(3)} with one global threshold to ${fb.range_aware.f1.toFixed(3)}. With two scenes the per-band thresholds overfit.`;
  const fnTot = held.fn_tags.reduce((a,r)=>a+r.n,0), grid = held.fn_tags.filter(r=>r.fail_tag==='beyond_bev_grid').reduce((a,r)=>a+r.n,0);
  document.getElementById('r-grid').textContent = pct(grid/fnTot);
}

function setBtns(prefix, key){ document.querySelectorAll(`[id^="${prefix}-"]`).forEach(b => { b.setAttribute('aria-pressed', b.id === `${prefix}-${state[key]}`);
  b.onclick = () => { state[key] = b.id.slice(prefix.length+1); renderAll(); }; }); }
function renderAll(){ setBtns('set','set'); setBtns('met','metric'); setBtns('fk','fk'); ruler(); drawRange(); drawHeat(); drawSize(); drawLoc(); drawLight(); drawFail(); gallery(); fix(); }
renderAll();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);
new MutationObserver(renderAll).observe(document.documentElement, {attributes:true, attributeFilter:['data-theme']});
</script>
"""


def _slim(summary: dict) -> dict:
    s = json.loads(json.dumps(summary, default=float))
    for r in s.get("fix_lidar_filter", {}).get("sweep", []):
        r.pop("range_all_mini", None)
        r.pop("class_ap", None)
    s.get("baseline_official", {}).pop("label_aps", None)
    return s


def write_html(summary: dict, path: str) -> None:
    data = _slim(summary)
    body = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    # artifact body (the host adds the document skeleton) ...
    with open(path.replace(".html", "_body.html"), "w", encoding="utf-8") as f:
        f.write(body)
    # ... and a standalone page for GitHub Pages / local viewing
    with open(path, "w", encoding="utf-8") as f:
        f.write('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                "</head><body>\n" + body + "\n</body></html>\n")
