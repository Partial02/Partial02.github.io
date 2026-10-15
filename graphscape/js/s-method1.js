/* 방법론 1: 삼중 비교 → 부등식 → 선형계획 해 (체험 2), 범주 제약 검산(그림 2), 임베딩 문제(그림 3) */
(function () {
  const App = window.App;
  const D = App.D;
  const X = { data: D, mark: 'point', encoding: { x: { field: 'Horsepower', type: 'quantitative', title: '마력' }, y: { field: 'Miles_per_Gallon', type: 'quantitative', title: '연비' } } };
  const variant = (f) => { const s = App.clone(X); f(s); return s; };
  const OPS = {
    mark: { label: '마크 변경', detail: 'point → tick', cat: 0, spec: variant((s) => { s.mark = 'tick'; }) },
    scale: { label: '스케일 변경', detail: 'x를 로그 스케일로', cat: 1, spec: variant((s) => { s.encoding.x.scale = { type: 'log' }; }) },
    filter: { label: '필터 추가', detail: '미국 차만', cat: 1, spec: variant((s) => { s.transform = [{ filter: { field: 'Origin', equal: 'USA' } }]; }) },
    transpose: { label: '전치', detail: 'x와 y 교환', cat: 2, spec: variant((s) => { const t = s.encoding.x; s.encoding.x = s.encoding.y; s.encoding.y = t; }) },
    add: { label: '필드 추가', detail: '색 ← 지역', cat: 2, spec: variant((s) => { s.encoding.color = { field: 'Origin', type: 'nominal' }; }) },
  };
  const KEYS = Object.keys(OPS);
  const CAT = ['마크', '변환', '인코딩'];
  const PAIRS = [['mark', 'scale'], ['scale', 'filter'], ['filter', 'transpose'], ['transpose', 'add'], ['mark', 'filter'], ['scale', 'add']];
  const rnd = App.seeded(11);
  const QS = PAIRS.map(([p, q]) => (rnd() < 0.5 ? [p, q] : [q, p])); // A/B 위치 섞기
  let answers = []; // {a, b, rel: '<' | '='}
  let libCost = null;

  const elQ = document.getElementById('tri-q');
  const elDots = document.getElementById('tri-dots');
  const elCharts = document.getElementById('tri-charts');
  const elBtns = document.getElementById('tri-btns');
  const elHint = document.getElementById('tri-hint');
  const elIneq = document.getElementById('tri-ineqs');
  const elBars = document.getElementById('tri-bars');
  const elNote = document.getElementById('tri-note');
  const margin = document.getElementById('tri-margin');
  const mOut = document.getElementById('tri-m-out');
  const catChk = document.getElementById('tri-cat');

  function drawQuestion() {
    const i = answers.length;
    elDots.innerHTML = QS.map((_, k) => `<span class="${k < i ? 'on' : ''}"></span>`).join('');
    if (i >= QS.length) {
      elQ.textContent = '여섯 질문 완료';
      elCharts.innerHTML = '';
      elBtns.classList.add('hidden');
      elHint.innerHTML = '모든 판단이 반영되었습니다. 오른쪽의 여유폭과 범주 가정을 바꿔 보세요. <button class="btn sm quiet" type="button" id="tri-again">판단 다시 하기</button>';
      document.getElementById('tri-again').addEventListener('click', () => { answers = []; elBtns.classList.remove('hidden'); drawQuestion(); solve(); });
      return;
    }
    const [a, b] = QS[i];
    elQ.textContent = `질문 ${i + 1} / ${QS.length}`;
    elCharts.innerHTML = '';
    const card = (role, spec, cap) => {
      const box = App.el(`<div><div class="role">${role}</div></div>`);
      box.appendChild(App.thumb(spec, cap, { w: 168, h: 122, noAxisTitle: false }));
      return box;
    };
    elCharts.appendChild(card('원본 X', X, '마력과 연비 산점도'));
    elCharts.appendChild(card('A', OPS[a].spec, `${OPS[a].label}: ${OPS[a].detail}`));
    elCharts.appendChild(card('B', OPS[b].spec, `${OPS[b].label}: ${OPS[b].detail}`));
    elHint.textContent = 'X를 방금 본 사람에게 A와 B 중 어느 쪽이 해석하기 쉬울까요?';
  }
  elBtns.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    const [a, b] = QS[answers.length];
    const v = btn.dataset.ans;
    if (v === 'a') answers.push({ a, b, rel: '<' });
    else if (v === 'b') answers.push({ a: b, b: a, rel: '<' });
    else answers.push({ a, b, rel: '=' });
    drawQuestion();
    solve();
  });

  /* 차분 제약 선형계획: min Σx, x ≥ 1, x_b − x_a ≥ m. 최적해는 판단 그래프의 최장 경로. */
  function solve() {
    const m = +margin.value;
    mOut.textContent = m;
    const useCat = catChk.checked;
    // 같다고 답한 연산은 하나로 묶기
    const parent = {}; KEYS.forEach((k) => (parent[k] = k));
    const find = (k) => (parent[k] === k ? k : (parent[k] = find(parent[k])));
    answers.filter((x) => x.rel === '=').forEach((x) => { parent[find(x.a)] = find(x.b); });
    const comp = (k) => find(k);
    const edges = answers.filter((x) => x.rel === '<').map((x) => ({ from: comp(x.a), to: comp(x.b), src: x }));
    const bad = new Set();
    edges.forEach((e) => { if (e.from === e.to) bad.add(e.src); });
    // 순환 찾기
    const nodes = [...new Set(KEYS.map(comp))];
    const adj = {}; nodes.forEach((n) => (adj[n] = []));
    edges.forEach((e) => { if (e.from !== e.to) adj[e.from].push(e); });
    const state = {}; let cycle = null; const stack = [];
    const dfs = (n) => {
      state[n] = 1; stack.push(n);
      for (const e of adj[n]) {
        if (cycle) return;
        if (state[e.to] === 1) { cycle = stack.slice(stack.indexOf(e.to)).concat(e.to); return; }
        if (!state[e.to]) dfs(e.to);
      }
      stack.pop(); state[n] = 2;
    };
    nodes.forEach((n) => { if (!state[n] && !cycle) dfs(n); });
    if (cycle) {
      const cs = new Set(cycle);
      edges.forEach((e) => { if (cs.has(e.from) && cs.has(e.to)) bad.add(e.src); });
    }
    // 위상 순서
    const indeg = {}; nodes.forEach((n) => (indeg[n] = 0));
    edges.forEach((e) => { if (e.from !== e.to) indeg[e.to]++; });
    let x = null; const catConflict = new Set();
    if (!cycle && !bad.size) {
      x = {};
      const order = []; const qu = nodes.filter((n) => !indeg[n]);
      const ind = Object.assign({}, indeg);
      while (qu.length) { const n = qu.shift(); order.push(n); adj[n].forEach((e) => { if (--ind[e.to] === 0) qu.push(e.to); }); }
      const preds = {}; nodes.forEach((n) => (preds[n] = []));
      edges.forEach((e) => preds[e.to].push(e));
      if (!useCat) {
        order.forEach((n) => { x[n] = Math.max(1, ...preds[n].map((e) => x[e.from] + m)); });
      } else {
        // 범주별로 아래에서부터: 범주 k의 하한 = 범주 k−1 비용의 합 + m
        const catOf = (n) => Math.min(...KEYS.filter((k) => comp(k) === n).map((k) => OPS[k].cat));
        let base = 1;
        for (let c = 0; c < 3; c++) {
          const inCat = order.filter((n) => catOf(n) === c);
          inCat.forEach((n) => {
            let v = base;
            preds[n].forEach((e) => {
              if (catOf(e.from) > c) catConflict.add(e.src);
              else if (x[e.from] != null) v = Math.max(v, x[e.from] + m);
            });
            x[n] = v;
          });
          // 같은 범주 안의 뒤늦은 선행자 반영(위상 순서라 이미 처리됨)
          base = inCat.reduce((s, n) => s + x[n] * KEYS.filter((k) => comp(k) === n).length, 0) + m;
        }
        catConflict.forEach((s) => bad.add(s));
      }
    }
    // 부등식 목록
    if (!answers.length) {
      elIneq.innerHTML = '아직 판단이 없습니다. 왼쪽 질문에 답하면 여기에 부등식이 쌓입니다.';
    } else {
      elIneq.innerHTML = answers.map((s) => {
        const t = `비용(${OPS[s.a].label}) ${s.rel === '<' ? '<' : '='} 비용(${OPS[s.b].label})`;
        const why = catConflict.has(s) ? ' : 범주 가정과 충돌' : bad.has(s) ? ' : 순환' : '';
        return `<div class="ineq ${bad.has(s) ? 'bad' : ''}">${t}${why}</div>`;
      }).join('');
    }
    if (cycle) {
      const names = cycle.map((n) => KEYS.filter((k) => comp(k) === n).map((k) => OPS[k].label).join('='));
      elBars.innerHTML = `<p class="verdict" style="color:var(--red)">해가 없습니다</p><p class="small">판단이 순환합니다: ${names.join(' < ')}. 선형계획이 불가능(infeasible)해지는 순간입니다. 저자들이 자기 실험을 반복하며 일관된 순서를 찾아야 했던 이유이기도 합니다.</p>`;
      elNote.textContent = '';
      return;
    }
    if (catConflict.size) {
      elBars.innerHTML = `<p class="verdict" style="color:var(--red)">범주 가정과 판단이 충돌합니다</p><p class="small">빨간 부등식은 상위 범주의 연산을 하위 범주보다 쉽다고 판단한 경우입니다. 논문은 이런 판단 자체를 묻지 않고 범주 순서를 가정으로 고정했습니다. 실험 1의 과제 2, 3이 바로 이 가정을 사람에게 검증한 것입니다.</p>`;
      elNote.textContent = '';
      return;
    }
    const vals = KEYS.map((k) => ({ op: `${OPS[k].label}`, cat: CAT[OPS[k].cat], v: x[comp(k)], lib: libCost ? libCost[k] : null, judged: answers.some((s) => s.a === k || s.b === k) }));
    const p = App.palette();
    App.mountRaw(elBars, {
      width: 330, height: 150,
      data: { values: vals },
      encoding: { y: { field: 'op', type: 'nominal', sort: KEYS.map((k) => OPS[k].label), title: null } },
      layer: [
        { mark: { type: 'bar', height: 16 }, encoding: { x: { field: 'v', type: 'quantitative', title: '선형계획 해 (비용)', axis: { tickMinStep: 1 } }, color: { field: 'cat', type: 'nominal', scale: { domain: CAT, range: [p.forest, p.contour, p.water] }, legend: { title: null, orient: 'bottom', direction: 'horizontal' } }, opacity: { condition: { test: 'datum.judged', value: 1 }, value: 0.35 } } },
        { mark: { type: 'text', align: 'left', dx: 4, fontWeight: 600 }, encoding: { x: { field: 'v', type: 'quantitative' }, text: { field: 'v', type: 'quantitative' } } },
      ],
    });
    const judged = vals.filter((v) => v.judged);
    let note = '';
    if (judged.length >= 2) {
      const lo = Math.min(...judged.map((v) => v.v)), hi = Math.max(...judged.map((v) => v.v));
      note = `지금 가장 비싼 연산은 가장 싼 연산의 ${App.fmt(hi / lo, 1)}배입니다. 여유폭을 바꾸면 이 배율이 바뀌지만 순서는 그대로입니다. 선형계획의 숫자는 비율이 아니라 <b>순서</b>를 담고 있습니다.`;
    } else note = '흐린 막대는 아직 판단에 등장하지 않은 연산입니다(최솟값 1).';
    if (libCost) {
      const libOrder = KEYS.slice().sort((a, b) => libCost[a] - libCost[b]).map((k) => `${OPS[k].label} ${libCost[k]}`).join(' < ');
      note += `<br>참고로 공개 라이브러리의 실제 비용: ${libOrder}.`;
    }
    elNote.innerHTML = note;
  }
  margin.addEventListener('input', solve);
  catChk.addEventListener('change', solve);

  /* 그림 2: 공개 코드 비용표의 범주 제약 검산 */
  const MARK = { AREA_BAR: .03, AREA_LINE: .02, AREA_POINT: .04, AREA_TEXT: .08, AREA_TICK: .04, BAR_LINE: .04, BAR_POINT: .02, BAR_TEXT: .06, BAR_TICK: .02, LINE_POINT: .03, LINE_TEXT: .07, LINE_TICK: .03, POINT_TEXT: .05, POINT_TICK: .01, TEXT_TICK: .05 };
  const TRANS = { SCALE: .6, SORT: .61, BIN: .62, AGGREGATE: .63, MODIFY_FILTER: .64, ADD_FILTER: .65, REMOVE_FILTER: .65 };
  function catCheck() {
    const p = App.palette();
    const rows = [];
    Object.entries(MARK).forEach(([k, v]) => rows.push({ g: '마크 → 변환', bar: '마크 15개의 합', k, v, kind: 'sum' }));
    rows.push({ g: '마크 → 변환', bar: '변환 최솟값 (SCALE)', k: 'SCALE', v: .6, kind: 'min' });
    Object.entries(TRANS).forEach(([k, v]) => rows.push({ g: '변환 → 인코딩', bar: '변환 7개의 합', k, v, kind: 'sum' }));
    rows.push({ g: '변환 → 인코딩', bar: '인코딩 최솟값 (SWAP_ROW_COLUMN)', k: 'SWAP_ROW_COLUMN', v: 4.41, kind: 'min' });
    const one = (g) => ({
      width: 300, height: 70, title: { text: g, anchor: 'start', fontSize: 13, fontWeight: 600 },
      transform: [{ filter: { field: 'g', equal: g } }],
      mark: { type: 'bar', stroke: p.surface, strokeWidth: 0.6, height: 22 },
      encoding: {
        y: { field: 'bar', type: 'nominal', title: null, sort: null, axis: { labelLimit: 200 } },
        x: { aggregate: 'sum', field: 'v', type: 'quantitative', title: '비용', stack: 'zero' },
        detail: { field: 'k' },
        color: { field: 'kind', type: 'nominal', scale: { domain: ['sum', 'min'], range: [p.contour, p.water] }, legend: null },
        tooltip: [{ field: 'k' }, { field: 'v' }],
      },
    });
    App.mountRaw(document.getElementById('cat-check'), { data: { values: rows }, hconcat: [one('마크 → 변환'), one('변환 → 인코딩')], spacing: 40, autosize: { type: 'pad' } });
  }

  /* 그림 3: 임베딩(L2) vs 가법 비용(L1) */
  const angle = document.getElementById('emb-angle');
  function emb() {
    const th = (+angle.value * Math.PI) / 180;
    document.getElementById('emb-out').textContent = angle.value + '°';
    const a = 3, b = 4, s = 30, ox = 40, oy = 200;
    const ax = ox + a * s, ay = oy;
    const bx = ax + b * s * Math.cos(Math.PI - th) * -1, by = ay - b * s * Math.sin(th);
    const l2 = Math.sqrt((a + b * Math.cos(th)) ** 2 + (b * Math.sin(th)) ** 2);
    document.getElementById('emb-l1').textContent = '7.00';
    document.getElementById('emb-l2').textContent = App.fmt(l2);
    const p = App.palette();
    const minX = Math.min(ox, ax, bx) - 20, maxX = Math.max(ox, ax, bx) + 30;
    document.getElementById('emb-svg').innerHTML = `<svg viewBox="${minX} 40 ${maxX - minX} 190" role="img" aria-label="두 편집 벡터와 직선 거리">
      <defs><marker id="ea" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="${p.ink}"/></marker></defs>
      <line x1="${ox}" y1="${oy}" x2="${bx}" y2="${by}" stroke="${p.red}" stroke-width="2.5" stroke-dasharray="6 5"/>
      <line x1="${ox}" y1="${oy}" x2="${ax}" y2="${ay}" stroke="${p.ink}" stroke-width="2.5" marker-end="url(#ea)"/>
      <line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" stroke="${p.ink}" stroke-width="2.5" marker-end="url(#ea)"/>
      <circle cx="${ox}" cy="${oy}" r="5" fill="${p.water}"/><circle cx="${bx}" cy="${by}" r="5" fill="${p.water}"/>
      <text x="${(ox + ax) / 2}" y="${oy + 18}" text-anchor="middle" font-size="13" fill="${p.ink}">편집 A = 3</text>
      <text x="${(ax + bx) / 2 + 8}" y="${(ay + by) / 2}" font-size="13" fill="${p.ink}">편집 B = 4</text>
      <text x="${(ox + bx) / 2 - 6}" y="${(oy + by) / 2 - 8}" text-anchor="end" font-size="13" fill="${p.red}">L2 = ${App.fmt(l2)}</text>
    </svg>`;
  }
  angle.addEventListener('input', emb);

  async function init() {
    drawQuestion();
    catCheck();
    emb();
    libCost = {};
    for (const k of KEYS) libCost[k] = (await App.gsTransition(X, OPS[k].spec)).cost;
    solve();
  }
  App.onTheme(() => { catCheck(); emb(); solve(); });
  App.lazy('s-method1', init);
})();
