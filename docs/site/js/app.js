(()=>{
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const rm=matchMedia('(prefers-reduced-motion: reduce)').matches;
document.documentElement.classList.add('js');
$('#mb').onclick=e=>{const o=$('nav').classList.toggle('open');e.target.setAttribute('aria-expanded',o)};
$$('nav a').forEach(a=>a.onclick=()=>$('nav').classList.remove('open'));
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)$$('nav a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+e.target.id))}),{rootMargin:'-40% 0px -55% 0px'});
$$('section[id]').forEach(s=>io.observe(s));
const ro=new IntersectionObserver(es=>es.forEach(e=>e.isIntersecting&&e.target.classList.add('in')));
$$('.rv').forEach(e=>rm?e.classList.add('in'):ro.observe(e));
function stepper(id,data){
 const root=$(id),bar=$('.stp',root),pk=$('.pk',root),dt=$('.dt',root),c=$('.ctl',root);let i=0,t=null;
 bar.style.setProperty('--n',data.length);
 bar.innerHTML=data.map((d,k)=>`<button type="button">${d.n}</button>`).join('');
 const bs=$$('button',bar);
 c.innerHTML='<button type="button" class="a">Previous</button><button type="button" class="b">Play</button><button type="button" class="c">Next</button><button type="button" class="d">Replay</button>';
 const pp=$('.b',c);
 function show(k){i=k;bs.forEach((b,j)=>{b.className=j<k?'done':j===k?'on':'';b.setAttribute('aria-current',j===k)});
  if(pk)pk.style.left=((k+.5)/data.length*100)+'%';const d=data[k];
  dt.innerHTML=`<h3>${d.n}</h3><dl><dt>enters</dt><dd>${d.i}</dd><dt>does</dt><dd>${d.o}</dd><dt>produces</dt><dd>${d.p}</dd><dt>why</dt><dd>${d.w}</dd></dl>`}
 const stop=()=>{clearInterval(t);t=null;pp.textContent='Play'};
 const play=()=>{pp.textContent='Pause';t=setInterval(()=>{i<data.length-1?show(i+1):stop()},1500)};
 bs.forEach((b,k)=>b.onclick=()=>{stop();show(k)});
 $('.a',c).onclick=()=>{stop();show(Math.max(0,i-1))};$('.c',c).onclick=()=>{stop();show(Math.min(data.length-1,i+1))};
 pp.onclick=()=>t?stop():play();$('.d',c).onclick=()=>{stop();show(0);if(!rm)play()};
 show(0);}
stepper('#rt',[
{n:'Sensors',i:'OS counters via psutil',o:'Samples CPU, memory, swap, disk I/O and process data every 2 s.',p:'One raw sample per interval',w:'Utilisation alone cannot say how slow the machine feels.'},
{n:'Probe',i:'A fixed micro-workload',o:'Times a short CPU loop, memory allocation and disk write with a high-resolution clock, every 4 s.',p:'Probe latency, compared with this machine\u2019s baseline for the current power state',w:'A physical measurement of sluggishness, not a hand-made label.'},
{n:'Features',i:'Raw samples and probe results',o:'Rolls them over 6 s and 12 s windows (12 s of history required), using past data only.',p:'One numeric feature row (27 features per the design spec)',w:'The same code runs in training and live inference.'},
{n:'Models',i:'Feature row',o:'Regression predicts slowdown 20 s ahead; the classifier names the likely cause; Isolation Forest flags unusual states.',p:'Predicted ratio, cause with confidence, anomaly flag',w:'Three tasks, three kinds of model, never mixed.'},
{n:'Score + cause',i:'Predicted slowdown ratio',o:'Maps the ratio to 0\u2013100 (100 at 1.0\u00d7 or less, 0 at 3.0\u00d7) and adds contributing factors.',p:'Health score, likely issue, evidence',w:'Every number traces to a measurement or a model artifact.'},
{n:'Fix',i:'Diagnosis',o:'Fix Center lists safe actions; you choose; each runs only after a safety check.',p:'Changed system state, logged',w:'Diagnosis is only useful if action is safe.'},
{n:'Re-scan',i:'System after the fix',o:'Waits, re-samples and probes again.',p:'Before/after comparison',w:'If the score does not improve, it says so.'}]);
stepper('#tp',[
{n:'Collect',i:'Real machine, controlled scenarios and normal use',o:'Runs CPU, memory, disk and background-load scenarios with known start and stop times, plus normal-use sessions.',p:'JSONL runs with telemetry and probe times',w:'No Kaggle data; ground truth is measured.'},
{n:'Dataset',i:'Runs',o:'Builds feature rows, slowdown targets (probe at t+20 s divided by baseline), issue labels, session ids and power state.',p:'Validated dataset with a SHA-256',w:'Per-session baselines keep AC and battery separate.'},
{n:'Train',i:'Training sessions',o:'Fits Ridge, Random Forest and XGBoost (regression); Logistic Regression, Random Forest and XGBoost (classification); Isolation Forest on healthy rows.',p:'Candidate models',w:'Preprocessing is fitted on training data only.'},
{n:'Cross-validate',i:'Candidates and small tuning grids',o:'Grouped cross-validation by run, so one run never sits on both sides of a fold.',p:'Chosen hyperparameters, primary model',w:'Prevents leakage between neighbouring samples.'},
{n:'Evaluate',i:'Held-out sessions (newest)',o:'Touches the test set once; 95% bootstrap intervals; compares with Persistence and Rolling Mean.',p:'Metrics and intervals',w:'A model beats a baseline only if the interval excludes zero.'},
{n:'Export',i:'Chosen models and metrics',o:'Writes the model set and a model card with hashes, counts, sessions, machines and the synthetic flag.',p:'Model artifact with provenance',w:'The CLI and this site read provenance from here.'}]);
const M={collector:['Samples CPU, memory, swap, disk, processes and runs the probe.','OS counters','Raw samples, probe times'],platform:['Windows and macOS differences behind one interface.','Common calls','Platform-specific readings'],features:['Rolling-window feature builder shared by training and runtime.','Samples','Numeric feature row'],ml:['Train, predict, evaluate, explain.','Feature rows','Predictions, model artifacts'],health:['Score and issue reasoning.','Predicted ratio, class','Score, evidence'],remediation:['Fix actions and safety checks.','User selection','Logged changes'],evaluation:['Live accuracy tracker and report data.','Predictions, probes','Per-model live metrics'],commands:['CLI entry points.','Arguments','Orchestrated runs'],ui:['Rich terminal rendering.','Results','Terminal output']};
$('#mods').innerHTML=Object.keys(M).map(k=>`<button type="button" aria-pressed="false" data-k="${k}">${k}/</button>`).join('');
const sm=k=>{$$('#mods button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.k===k));const m=M[k];$('#mdd').innerHTML=`<h3 class="mono">${k}/</h3><dl><dt>role</dt><dd>${m[0]}</dd><dt>input</dt><dd>${m[1]}</dd><dt>output</dt><dd>${m[2]}</dd></dl>`};
$$('#mods button').forEach(b=>b.onclick=()=>sm(b.dataset.k));sm('collector');
const r=$('#rr'),sc=()=>{const v=+r.value;$('#rv').textContent=v.toFixed(1)+'\u00d7';$('#sv').textContent=Math.round(100*Math.min(1,Math.max(0,(3-v)/2)))};r.oninput=sc;sc();
$$('.tabs button').forEach(b=>b.onclick=()=>{$$('.tabs button').forEach(x=>x.setAttribute('aria-selected',x===b));$$('.tp').forEach(p=>p.hidden=p.id!==b.dataset.t)});
$$('[data-copy]').forEach(b=>b.onclick=async()=>{try{await navigator.clipboard.writeText($('#'+b.dataset.copy).textContent);b.textContent='Copied'}catch{b.textContent='Select and copy manually'}setTimeout(()=>b.textContent='Copy',1800)});
const lab=$('#lab'),NR=t=>`<div class="nr"><b>NO REAL DATA YET</b><p style="margin:10px 0 0">${t}</p></div>`;
function bars(rows){const mx=Math.max(...rows.map(x=>x.ci?x.ci[1]:x.mae)),W=560,s=v=>150+v/mx*(W-200);
 return `<svg viewBox="0 0 ${W} ${rows.length*40+40}" role="img" aria-label="Held-out MAE by model with 95% intervals" style="width:100%;font:12px Consolas,monospace">`+
 rows.map((x,k)=>{const y=20+k*40;return `<text x="0" y="${y+14}" fill="#F2EFE8">${x.name}</text><rect x="150" y="${y}" width="${s(x.mae)-150}" height="18" fill="${x.baseline?'#394653':'#B79A6B'}"/>`+(x.ci?`<line x1="${s(x.ci[0])}" x2="${s(x.ci[1])}" y1="${y+9}" y2="${y+9}" stroke="#F2EFE8"/>`:'')+`<text x="${s(x.mae)+6}" y="${y+14}" fill="#89939A">${x.mae}</text>`}).join('')+
 `<text x="0" y="${rows.length*40+34}" fill="#89939A">MAE, slowdown ratio, lower is better. Whisker = 95% bootstrap interval. Grey = baseline.</text></svg>`}
const loadData=async()=>{
 for(const p of ['data/project-data.json','project-data.json']){
  try{const r=await fetch(p);if(r.ok)return await r.json()}catch{}
 }
 throw 0;
};
loadData().then(d=>{
 if(d.status!=='ok')return lab.innerHTML=NR('No model card or report data was supplied to this site build, so no metrics are shown. Generate them with <code>python scripts/build_site_data.py model_card.json report_data.json</code>.');
 const p=`<p class="mu mono" style="font-size:12px">Provenance: train rows ${d.train_rows??'n/a'}, test rows ${d.test_rows??'n/a'}, sessions ${d.sessions??'n/a'}, machines ${d.machines??'n/a'}. Dataset SHA-256: ${d.dataset_sha256??'n/a'}</p>`;
 if(d.synthetic!==false)return lab.innerHTML=`<p class="mono" style="color:var(--wa)">DEMO MODEL \u2014 TRAINED ON SYNTHETIC DATA</p>`+NR('The bundled model demonstrates the inference pipeline. Metrics from synthetic training data are not evidence of real-world performance, so none are shown. Run <code>deskflow full</code> on your machine to measure real accuracy.')+p;
 lab.innerHTML=(d.regression&&d.regression.length?bars(d.regression):NR('Evaluation data is missing from the report, so the comparison is unavailable.'))+p;
}).catch(()=>{lab.innerHTML=NR('data/project-data.json could not be loaded. If you opened the file directly, preview with <code>python -m http.server 8000 --directory docs/site</code>.')});
})();
