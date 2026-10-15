/* 응용 세 가지: graphscape.sequence / path / transition을 브라우저에서 직접 호출 */
(function () {
  const App = window.App;
  const D = App.D;
  const q = (field, extra) => Object.assign({ field, type: 'quantitative' }, extra || {});

  /* ---------- 탭 ---------- */
  const tabs = document.getElementById('app-tabs');
  const panels = { seq: 'app-seq', path: 'app-path', alt: 'app-alt' };
  const inited = {};
  tabs.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    tabs.querySelectorAll('button').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    Object.entries(panels).forEach(([k, id]) => document.getElementById(id).classList.toggle('hidden', k !== b.dataset.tab));
    initTab(b.dataset.tab);
  });
  function initTab(k) {
    if (inited[k]) return;
    inited[k] = true;
    ({ seq: initSeq, path: initPath, alt: initAlt })[k]();
  }

  /* ---------- 1. 시퀀스 추천 ---------- */
  const PICKS0 = [0, 2, 3, 1, 4];
  let picks = PICKS0.slice();
  let drawPool = null;
  function initSeq() {
    const pool = document.getElementById('seq-pool');
    const draw = (drawPool = () => {
      pool.innerHTML = '';
      App.POOL.forEach((c, i) => {
        const on = picks.includes(i);
        const lab = App.el(`<label class="thumb ${on ? 'on' : ''}"><input type="checkbox" ${on ? 'checked' : ''} aria-label="${c.cap}"></label>`);
        const slot = App.el('<div class="chart-slot"></div>');
        lab.appendChild(slot);
        lab.appendChild(App.el(`<div class="cap">${on ? `<b>${picks.indexOf(i) + 1}.</b> ` : ''}${c.cap}</div>`));
        App.mount(slot, c.spec, { w: 150, h: 96, small: true, noAxisTitle: true, label: c.cap });
        lab.querySelector('input').addEventListener('change', (e) => {
          if (e.target.checked) { if (picks.length >= 7) { e.target.checked = false; status('최대 7장까지 고를 수 있습니다 (7! = 5,040가지).'); return; } picks.push(i); }
          else picks = picks.filter((x) => x !== i);
          draw();
        });
        pool.appendChild(lab);
      });
    });
    draw();
    document.getElementById('seq-go').addEventListener('click', runSeq);
  }
  const status = (t) => (document.getElementById('seq-status').innerHTML = t);
  async function runSeq() {
    if (picks.length < 3) return status('3장 이상 골라 주세요.');
    const fix = document.getElementById('seq-fix').checked;
    const items = picks.map((i) => App.POOL[i]);
    const n = items.length;
    const nPerm = [1, 1, 2, 6, 24, 120, 720, 5040][fix ? n - 1 : n];
    status(`<span class="spinner"></span> ${App.fmtInt(nPerm)}가지 순서를 평가하는 중`);
    await new Promise((r) => setTimeout(r, 30));
    const t0 = performance.now();
    const res = await graphscape.sequence(App.clone(items.map((x) => x.spec)), { fixFirst: fix });
    const ms = performance.now() - t0;
    status(`${App.fmtInt(res.length)}가지 순서를 ${App.fmtInt(Math.round(ms))}ms에 평가했습니다.`);
    const out = document.getElementById('seq-out');
    out.innerHTML = '';
    res.slice(0, 3).forEach((r, k) => {
      const pat = r.patterns && r.patterns[0] && r.globalWeightingTerm < 1 ? `반복 패턴 ${r.patterns[0].appear.length}회, 커버리지 ${App.fmt(r.patterns[0].coverage, 2)}` : '반복 패턴 없음';
      const box = App.el(`<div style="border:1px solid var(--rule);background:var(--paper);padding:10px 12px;margin-bottom:10px">
        <div class="row" style="justify-content:space-between"><strong>${k + 1}위 순서</strong>
        <span class="small">비용 ${App.fmt(r.sequenceCost, 3)} = W ${App.fmt(r.globalWeightingTerm, 3)} × (ΣT ${App.fmt(r.sumOfTransitionCosts)} + F ${App.fmt(r.filterSequenceCost, 3)} ÷ 1000), ${pat}</span></div></div>`);
      const strip = App.el('<div class="strip"></div>');
      box.appendChild(strip);
      out.appendChild(box);
      App.renderSeq(strip, r, items);
    });
    out.appendChild(App.el('<p class="small">논문은 완전 자동화보다 비용이 낮은 순서 여러 개를 함께 보여주고 사람이 고르는 방식을 제안합니다. 실험 2 세트 3에서 사람들이 고른 순서가 GraphScape 2, 3위였던 것을 떠올리면 설득력 있는 설계입니다. 공개 코드는 비용에 F를 1000으로 나눠 더합니다.</p>'));
  }

  /* ---------- 2. 경로 상세화 ---------- */
  const scat = { data: D, mark: 'point', encoding: { x: q('Horsepower', { title: '마력' }), y: q('Miles_per_Gallon', { title: '연비' }) } };
  const PAIRS = [
    { name: '산점도 → 지역별 평균 연비 곡선', s: scat, e: { data: D, mark: 'line', encoding: { x: q('Horsepower', { bin: true, title: '마력' }), y: q('Miles_per_Gallon', { aggregate: 'mean', title: '평균 연비' }), color: { field: 'Origin', type: 'nominal' } } } },
    { name: '지역별 개수 막대 → 지역 색을 입힌 산점도', s: { data: D, mark: 'bar', encoding: { x: { field: 'Origin', type: 'nominal' }, y: { aggregate: 'count', field: '*', type: 'quantitative' } } }, e: { data: D, mark: 'point', encoding: { x: q('Horsepower', { title: '마력' }), y: q('Miles_per_Gallon', { title: '연비' }), color: { field: 'Origin', type: 'nominal' } } } },
    { name: '산점도 → 축 교환 + 로그 + 일본 차만', s: scat, e: { data: D, mark: 'point', transform: [{ filter: { field: 'Origin', equal: 'Japan' } }], encoding: { x: q('Miles_per_Gallon', { title: '연비' }), y: q('Horsepower', { title: '마력', scale: { type: 'log' } }) } } },
  ];
  const clean = (s) => { const c = App.clone(s); delete c.distance; delete c.prev; delete c.$schema; return c; };
  let pathOps = 0, playTimer = null;
  function initPath() {
    const sel = document.getElementById('path-pair');
    sel.innerHTML = PAIRS.map((p, i) => `<option value="${i}">${p.name}</option>`).join('');
    const m = document.getElementById('path-m');
    m.addEventListener('input', () => (document.getElementById('path-m-out').textContent = m.value));
    sel.addEventListener('change', showEnds);
    document.getElementById('path-go').addEventListener('click', runPath);
    showEnds();
  }
  async function showEnds() {
    const p = PAIRS[+document.getElementById('path-pair').value];
    const ends = document.getElementById('path-ends');
    ends.innerHTML = '';
    document.getElementById('path-out').innerHTML = '';
    const tr = await App.gsTransition(p.s, p.e);
    const ops = App.opsOf(tr);
    pathOps = ops.length;
    const m = document.getElementById('path-m');
    m.max = pathOps;
    if (+m.value > pathOps) m.value = pathOps;
    document.getElementById('path-m-out').textContent = m.value;
    const strip = App.el('<div class="strip"></div>');
    strip.appendChild(App.thumb(p.s, '시작', { w: 140, h: 100, noAxisTitle: false }));
    const ed = App.edge(ops, tr.cost, false);
    ed.style.width = '150px';
    strip.appendChild(ed);
    strip.appendChild(App.thumb(p.e, '끝', { w: 140, h: 100, noAxisTitle: false }));
    ends.appendChild(strip);
    ends.appendChild(App.el(`<p class="small">graphscape.transition이 찾은 최소 비용 편집은 ${pathOps}개입니다: ${ops.map((o) => App.opLabel(o)).join(', ')}. 이 편집들을 M단계로 나누면 중간 차트가 생깁니다.</p>`));
  }
  async function runPath() {
    const p = PAIRS[+document.getElementById('path-pair').value];
    const M = +document.getElementById('path-m').value;
    const st = document.getElementById('path-status');
    const out = document.getElementById('path-out');
    st.innerHTML = '<span class="spinner"></span> graphscape.path 실행 중 (편집이 많으면 몇 초 걸립니다)';
    out.innerHTML = '';
    await new Promise((r) => setTimeout(r, 30));
    const t0 = performance.now();
    let paths;
    try { paths = await graphscape.path(App.clone(p.s), App.clone(p.e), M); } catch (e) { console.error(e); st.textContent = '경로를 만들지 못했습니다. 다른 M을 골라 보세요.'; return; }
    const ms = performance.now() - t0;
    paths = (paths || []).slice().sort((a, b) => b.eval.score - a.eval.score);
    st.textContent = `${paths.length}개 경로를 ${App.fmtInt(Math.round(ms))}ms에 만들었습니다.`;
    if (!paths.length) { out.innerHTML = '<p class="small">이 단계 수로는 유효한 경로가 없습니다. 중간 차트가 Vega-Lite로 그릴 수 없는 상태가 되는 분할은 라이브러리가 걸러냅니다.</p>'; return; }
    const totals = paths.map((pt) => pt.editOpPartition.flat().reduce((s, o) => s + o.cost, 0));
    const same = totals.every((t) => Math.abs(t - totals[0]) < 1e-9);
    const listBox = App.el('<div></div>');
    out.appendChild(App.el(`<p class="note ${same ? 'check' : ''}"><b>${paths.length}개 분할의 총비용이 ${same ? '모두 ' + App.fmt(totals[0]) + '로 같습니다.' : '서로 다릅니다.'}</b> 같은 편집을 나눌 뿐이라 비용 모델로는 분할의 좋고 나쁨을 가를 수 없습니다. 그래서 라이브러리는 “스케일은 인코딩 변경과 함께” 같은 휴리스틱 규칙 점수로 경로 순위를 매깁니다.</p>`));
    out.appendChild(listBox);
    const detail = App.el('<div style="margin-top:12px"></div>');
    out.appendChild(detail);
    const show = (k) => {
      listBox.querySelectorAll('.cand').forEach((c, i) => c.setAttribute('aria-pressed', String(i === k)));
      drawPath(detail, paths[k]);
    };
    paths.slice(0, 6).forEach((pt, k) => {
      const rules = (pt.eval.satisfiedRules || []).map((r) => r.name).join(', ') || '만족한 규칙 없음';
      const steps = pt.editOpPartition.map((g) => g.map((o) => App.opLabel(o, true)).join(' + ')).join('  |  ');
      const b = App.el(`<button type="button" class="cand"><span class="rk">${k + 1}</span><span>${steps}<br><span class="meta">규칙: ${rules}</span></span><span class="meta">점수 ${pt.eval.score}</span><span></span></button>`);
      b.addEventListener('click', () => show(k));
      listBox.appendChild(b);
    });
    show(0);
  }
  function drawPath(box, pt) {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    box.innerHTML = '';
    const strip = App.el('<div class="strip"></div>');
    pt.sequence.forEach((s, k) => {
      if (k) { const g = pt.editOpPartition[k - 1]; strip.appendChild(App.edge(g, g.reduce((a, o) => a + o.cost, 0), false)); }
      strip.appendChild(App.thumb(clean(s), k === 0 ? '시작' : k === pt.sequence.length - 1 ? '끝' : `중간 ${k}`, { w: 120, h: 88 }));
    });
    box.appendChild(strip);
    const ctl = App.el('<div class="row" style="margin:10px 0"><button class="btn" type="button">단계별로 재생</button><span class="small">애니메이션 전환의 키프레임처럼 한 단계씩 넘깁니다.</span></div>');
    const stage = App.el('<div class="stage"></div>');
    const lbl = App.el('<p class="small" style="margin-top:6px"></p>');
    box.appendChild(ctl); box.appendChild(stage); box.appendChild(lbl);
    let k = 0;
    const frame = () => {
      App.mount(stage, clean(pt.sequence[k]), { w: 380, h: 230, small: false });
      lbl.textContent = k === 0 ? '시작 차트' : `단계 ${k}: ${pt.editOpPartition[k - 1].map((o) => App.opLabel(o)).join(', ')}`;
    };
    frame();
    ctl.querySelector('button').addEventListener('click', () => {
      if (playTimer) clearInterval(playTimer);
      k = 0; frame();
      playTimer = setInterval(() => { k++; if (k >= pt.sequence.length) { clearInterval(playTimer); playTimer = null; return; } frame(); }, 1400);
    });
  }

  /* ---------- 3. 설계 대안 ---------- */
  const FL = { url: 'flights.json' };
  const ax = q('distance', { title: '비행 거리' }), ay = q('delay', { title: '지연 (분)' });
  const ORIG = { data: FL, mark: 'point', encoding: { x: ax, y: ay } };
  const CANDS = [
    { name: '무작위 10% 표본만 표시', spec: { data: FL, mark: 'point', transform: [{ filter: { field: 'sample', equal: 0 } }], encoding: { x: ax, y: ay } }, count: 'sample' },
    { name: '2차원 히스토그램 (크기 = 개수)', spec: { data: FL, mark: 'point', encoding: { x: Object.assign({}, ax, { bin: true }), y: Object.assign({}, ay, { bin: true }), size: { aggregate: 'count', field: '*', type: 'quantitative' } } }, count: 'bin2' },
    { name: '거리 구간별 평균 지연 (선)', spec: { data: FL, mark: 'line', encoding: { x: Object.assign({}, ax, { bin: true }), y: Object.assign({}, ay, { aggregate: 'mean', title: '평균 지연 (분)' }) } }, count: 'binx' },
    { name: '거리 구간별 평균 지연 (크기 = 개수)', spec: { data: FL, mark: 'point', encoding: { x: Object.assign({}, ax, { bin: true }), y: Object.assign({}, ay, { aggregate: 'mean', title: '평균 지연 (분)' }), size: { aggregate: 'count', field: '*', type: 'quantitative' } } }, count: 'binx' },
    { name: '틱 마크로 바꾸기', spec: { data: FL, mark: 'tick', encoding: { x: ax, y: ay } }, count: 'all' },
    { name: 'x를 로그 스케일로', spec: { data: FL, mark: 'point', encoding: { x: Object.assign({}, ax, { scale: { type: 'log' } }), y: ay } }, count: 'all' },
  ];
  let altSel = -1;
  function countMarks(kind) {
    const F = window.GSDATA.flights;
    if (kind === 'all') return F.length;
    if (kind === 'sample') return F.filter((d) => d.sample === 0).length;
    const binner = (field) => {
      const v = F.map((d) => d[field]);
      const b = vega.bin({ extent: [Math.min(...v), Math.max(...v)], maxbins: 10 });
      return (x) => Math.floor((x - b.start) / b.step);
    };
    const bx = binner('distance');
    if (kind === 'binx') return new Set(F.map((d) => bx(d.distance))).size;
    const by = binner('delay');
    return new Set(F.map((d) => bx(d.distance) + ',' + by(d.delay))).size;
  }
  async function initAlt() {
    const list = document.getElementById('alt-list');
    list.innerHTML = '<p class="small"><span class="spinner"></span> 후보마다 graphscape.transition으로 편집 비용 계산 중</p>';
    for (const c of CANDS) {
      c.n = countMarks(c.count);
      const tr = await App.gsTransition(ORIG, c.spec);
      c.cost = tr.cost;
      c.ops = App.opsOf(tr);
    }
    const th = document.getElementById('alt-th');
    th.addEventListener('input', drawAlt);
    drawAlt();
    selectAlt(-1);
  }
  function drawAlt() {
    const th = +document.getElementById('alt-th').value;
    document.getElementById('alt-th-out').textContent = App.fmtInt(th);
    const list = document.getElementById('alt-list');
    const ok = CANDS.filter((c) => c.n <= th).sort((a, b) => a.cost - b.cost);
    const no = CANDS.filter((c) => c.n > th).sort((a, b) => a.cost - b.cost);
    list.innerHTML = '';
    const orig = App.el(`<button type="button" class="cand ${10000 > th ? 'fail' : ''}" aria-pressed="${altSel === -1}"><span class="rk">0</span><span>원래 설계: 산점도<br><span class="meta">마크 ${App.fmtInt(10000)}개${10000 > th ? ', 상한 초과' : ''}</span></span><span class="meta">비용 0</span><span></span></button>`);
    orig.addEventListener('click', () => selectAlt(-1));
    list.appendChild(orig);
    [...ok, ...no].forEach((c, k) => {
      const i = CANDS.indexOf(c);
      const pass = c.n <= th;
      const b = App.el(`<button type="button" class="cand ${pass ? '' : 'fail'}" aria-pressed="${altSel === i}"><span class="rk">${pass ? k + 1 : '–'}</span><span>${c.name}<br><span class="meta">마크 ${App.fmtInt(c.n)}개${pass ? '' : ', 상한 초과'}</span></span><span class="meta">편집 비용 ${App.fmt(c.cost)}</span><span></span></button>`);
      b.addEventListener('click', () => selectAlt(i));
      list.appendChild(b);
    });
    document.getElementById('alt-note').innerHTML = ok.length
      ? `상한 ${App.fmtInt(th)}개를 만족하는 대안 ${ok.length}개를 원래 설계에서 덜 바뀐 순서(편집 비용)로 정렬했습니다. 1위는 <b>${ok[0].name}</b>입니다. 상한을 낮추면 표본 추출이 탈락하고 집계형 대안만 남습니다.`
      : '상한을 만족하는 대안이 없습니다. 상한을 올려 보세요.';
  }
  async function selectAlt(i) {
    altSel = i;
    drawAlt();
    const c = i < 0 ? null : CANDS[i];
    document.getElementById('alt-sel-title').textContent = c ? c.name : '원래 설계: 산점도';
    document.getElementById('alt-sel-meta').textContent = `마크 ${App.fmtInt(c ? c.n : 10000)}개`;
    document.getElementById('alt-ops').innerHTML = c ? `편집: ${c.ops.map((o) => `${App.opLabel(o)} (${o.cost})`).join(', ')}` : '비행 1만 편의 거리와 지연. 점이 겹쳐서 밀도를 읽을 수 없습니다.';
    const stage = document.getElementById('alt-stage');
    const spec = App.clone(c ? c.spec : ORIG);
    if (spec.mark === 'point' && !spec.encoding.size) spec.mark = { type: 'point', size: 8, opacity: 0.5 };
    await App.canvas(stage, spec, { w: 420, h: 260 });
  }
  App.onTheme(() => { if (inited.alt) selectAlt(altSel); });

  /* 응용 탭마다 초기화 */
  function tools(id, fn) {
    const panel = document.getElementById(id);
    const bar = App.el('<div class="panel-tools"></div>');
    panel.insertBefore(bar, panel.firstChild);
    App.addReset(bar, fn);
  }
  tools('app-seq', () => {
    picks = PICKS0.slice(); document.getElementById('seq-fix').checked = false;
    document.getElementById('seq-out').innerHTML = ''; status('');
    if (drawPool) drawPool();
  });
  tools('app-path', () => {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    document.getElementById('path-pair').value = '0';
    document.getElementById('path-m').value = 2;
    document.getElementById('path-status').textContent = '';
    if (inited.path) showEnds();
  });
  tools('app-alt', () => {
    document.getElementById('alt-th').value = 5000;
    if (inited.alt) selectAlt(-1);
  });

  App.lazy('s-apps', () => initTab('seq'));
})();
