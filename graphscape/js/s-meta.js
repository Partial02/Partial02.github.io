/* 마무리: 이 리뷰 페이지의 섹션 순서를 GraphScape식 비용 함수로 평가 (체험 6)
   섹션 = 명세. 형식(마크급), 다루는 단위(변환급), 개념 집합(인코딩급), 논문 쪽수(필터 값). */
(function () {
  const App = window.App;
  const CON = { Q: '문제의식', E: '편집 비용', H: '사람의 판단', S: '시퀀스 비용', A: '응용', C: '비판' };
  const SEC = [
    { id: 's-problem', n: 1, name: '문제 상황', fmt: '설명', unit: '시퀀스', con: ['Q'], need: [], page: 1 },
    { id: 's-method1', n: 2, name: '방법론 1', fmt: '설명', unit: '연산', con: ['E'], need: ['Q'], page: 3 },
    { id: 's-study1', n: 3, name: '실험 1', fmt: '실험', unit: '연산', con: ['E', 'H'], need: ['Q', 'E'], page: 5 },
    { id: 's-model', n: 4, name: '방법론 2', fmt: '설명', unit: '시퀀스', con: ['S'], need: ['Q', 'E'], page: 6 },
    { id: 's-study2', n: 5, name: '실험 2', fmt: '실험', unit: '시퀀스', con: ['S', 'H'], need: ['Q', 'S'], page: 7 },
    { id: 's-apps', n: 6, name: '응용', fmt: '데모', unit: '시퀀스', con: ['S', 'A'], need: ['Q', 'S'], page: 8 },
    { id: 's-eval', n: 7, name: '평가', fmt: '토론', unit: '논문', con: ['C'], need: ['Q', 'E', 'H', 'S', 'A'], page: 9 },
  ];
  const COST = { 형식: 0.04, 단위: 0.63, 교체: 4.70, 제거: 4.55, 추가: 4.55 };
  const EMPTY = { fmt: null, unit: null, con: [] };
  const OURS = [0, 1, 2, 3, 4, 5, 6];
  const DEFAULT = { need: true, s0: true, f: false, w: false };
  let st = Object.assign({ order: OURS.slice() }, DEFAULT);
  let all = null;

  function ops(a, b) {
    const o = [];
    if (a.fmt && a.fmt !== b.fmt) o.push({ k: '형식', t: `${a.fmt}→${b.fmt}` });
    if (a.unit && a.unit !== b.unit) o.push({ k: '단위', t: `${a.unit}→${b.unit}` });
    const rem = a.con.filter((c) => !b.con.includes(c)).sort();
    const add = b.con.filter((c) => !a.con.includes(c)).sort();
    const m = Math.min(rem.length, add.length);
    for (let i = 0; i < m; i++) o.push({ k: '교체', t: `${CON[rem[i]]}→${CON[add[i]]}` });
    rem.slice(m).forEach((c) => o.push({ k: '제거', t: `−${CON[c]}` }));
    add.slice(m).forEach((c) => o.push({ k: '추가', t: `+${CON[c]}` }));
    return o;
  }
  const opsCost = (o) => o.reduce((s, x) => s + COST[x.k], 0);
  const sigOf = (o) => o.map((x) => x.k + ':' + x.t).join('|');
  function wTerm(sig) {
    const n = sig.length; let best = 0, bestPat = null;
    for (let L = 1; L < n; L++) {
      for (let i = 0; i + L <= n; i++) {
        const pat = sig.slice(i, i + L).join('#');
        const ch = [];
        for (let j = 0; j + L <= n; j++) {
          if (sig.slice(j, j + L).join('#') === pat && (!ch.length || j >= ch[ch.length - 1] + L)) ch.push(j);
        }
        if (ch.length > 1 && (ch.length * L) / n > best) { best = (ch.length * L) / n; bestPat = { at: ch, L }; }
      }
    }
    return { W: 1 - best, pat: bestPat };
  }
  function fTerm(order) {
    let d = 0;
    for (let i = 1; i < order.length; i++) { const a = SEC[order[i - 1]].page, b = SEC[order[i]].page; d += b > a ? 1 : b < a ? -1 : 0; }
    return 1 - Math.abs(d + 0.1) / (order.length - 1 + 0.1);
  }
  function valid(order) {
    const seen = new Set();
    for (const i of order) {
      if (!SEC[i].need.every((c) => seen.has(c))) return false;
      SEC[i].con.forEach((c) => seen.add(c));
    }
    return true;
  }
  function evaluate(order) {
    const seq = [EMPTY].concat(order.map((i) => SEC[i]));
    const tr = [];
    for (let i = 1; i < seq.length; i++) { const o = ops(seq[i - 1], seq[i]); tr.push({ o, c: opsCost(o), sig: sigOf(o) }); }
    const T0 = tr[0].c, Tall = tr.reduce((s, x) => s + x.c, 0);
    const w = wTerm(tr.map((x) => x.sig));
    return { order, tr, T0, Tall, W: w.W, pat: w.pat, F: fTerm(order), ok: valid(order) };
  }
  function total(r) {
    const T = st.s0 ? r.Tall : r.Tall - r.T0;
    return (st.w ? r.W : 1) * (T + (st.f ? r.F : 0));
  }

  const $ = (id) => document.getElementById(id);
  function chip(i) {
    const s = SEC[i];
    return `<div class="thumb meta-chip"><div class="meta-n">${s.n}</div><div class="meta-name">${s.name}</div><div class="cap">${s.fmt}, ${s.unit}<br>${s.con.map((c) => CON[c]).join(', ')}</div></div>`;
  }
  function strip(el, r, movable) {
    const inPat = new Set();
    if (st.w && r.pat) r.pat.at.forEach((a) => { for (let k = 0; k < r.pat.L; k++) inPat.add(a + k); });
    let h = '';
    if (st.s0) h += `<div class="thumb meta-chip" style="opacity:.7"><div class="meta-name" style="margin-top:14px">빈 시작</div><div class="cap">S₀</div></div>`;
    r.order.forEach((i, k) => {
      const t = r.tr[k];
      if (k > 0 || st.s0) h += `<div class="edge ${inPat.has(k) ? 'pat' : ''}" style="width:92px"><span>${t.o.map((x) => x.t).join(', ')}</span><div class="arrow"></div><span class="c">${App.fmt(t.c)}</span></div>`;
      h += chip(i);
    });
    el.innerHTML = h;
    if (movable) {
      el.querySelectorAll('.meta-chip').forEach((c, k) => {
        const idx = st.s0 ? k - 1 : k;
        if (idx < 0) return;
        const row = App.el(`<div class="row" style="justify-content:center;gap:4px;margin-top:4px"><button type="button" class="btn sm quiet" aria-label="왼쪽으로" ${idx === 0 ? 'disabled' : ''}>◀</button><button type="button" class="btn sm quiet" aria-label="오른쪽으로" ${idx === 6 ? 'disabled' : ''}>▶</button></div>`);
        const [l, rr] = row.querySelectorAll('button');
        l.addEventListener('click', () => move(idx, idx - 1));
        rr.addEventListener('click', () => move(idx, idx + 1));
        c.appendChild(row);
      });
    }
  }
  function move(a, b) {
    if (b < 0 || b > 6) return;
    const o = st.order.slice(); const [x] = o.splice(a, 1); o.splice(b, 0, x); st.order = o; update();
  }
  const key = (o) => o.join(',');
  function update() {
    if (!all) return;
    ['need', 's0', 'f', 'w'].forEach((k) => ($('meta-' + k).checked = st[k]));
    const pool = st.need ? all.filter((r) => r.ok) : all;
    const ranked = pool.map((r) => ({ r, c: total(r) })).sort((a, b) => a.c - b.c);
    const cur = all.find((r) => key(r.order) === key(st.order));
    const cc = total(cur);
    const eps = 1e-9;
    const rank = ranked.filter((x) => x.c < cc - eps).length + 1;
    const ties = ranked.filter((x) => Math.abs(x.c - cc) < eps).length;
    const isOurs = key(st.order) === key(OURS);
    strip($('meta-cur'), cur, true);
    strip($('meta-best'), ranked[0].r, false);
    const ttl = isOurs ? '이 리뷰의 실제 순서' : '내가 바꾼 순서';
    $('meta-cur-title').textContent = ttl;
    let verdict;
    if (!cur.ok && st.need) verdict = `${ttl}는 <strong>전제 조건 위반</strong>입니다. 아직 소개하지 않은 개념이 필요한 섹션이 앞에 나옵니다.`;
    else verdict = `${ttl}: ${App.fmtInt(pool.length)}가지 중 <strong>${rank === 1 ? (ties > 1 ? `공동 1위 (${ties}개)` : '단독 1위') : rank + '위'}</strong>`;
    $('meta-verdict').innerHTML = verdict;
    const cell = (k, v, on = true) => `<div><div class="k">${k}</div><div class="v ${on ? '' : 'off'}">${v}</div></div>`;
    $('meta-readout').innerHTML = cell('전환 비용 합 ΣT', App.fmt(st.s0 ? cur.Tall : cur.Tall - cur.T0)) + cell('필터 항 F (쪽 순서)', App.fmt(cur.F, 3), st.f) + cell('전역 가중치 W', App.fmt(cur.W, 3), st.w) + cell('최종 비용', App.fmt(cc, 3));
    let msg;
    if (!st.need) msg = '전제 조건을 끄면 “방법론 2를 방법론 1보다 먼저” 같은 순서가 이깁니다. 차트에서 Vega-Lite가 그릴 수 없는 명세를 걸러내듯, 섹션에서도 성립하지 않는 순서는 먼저 걸러내야 합니다.';
    else if (!st.w && !st.f) msg = `전환 비용만 보면 이 리뷰의 순서는 1위가 아닙니다. 현재 최적은 개념을 덜 바꾸는 순서입니다. 논문에서 T만으로는 사람의 판단을 설명하지 못했던 것(ρ = 0.35)과 같은 상황입니다. 이제 W를 켜 보세요.`;
    else if (st.w && !st.f) msg = 'W를 켜는 순간 단독 1위가 됩니다. “설명 → 실험, +사람의 판단”이라는 같은 전환이 방법론 1 → 실험 1, 방법론 2 → 실험 2에서 두 번 반복되는 구조(민트 화살표)가 보상받기 때문입니다. F도 켜서 논문의 흐름과도 맞는지 확인해 보세요.';
    else if (st.f && !st.w) msg = 'F는 논문 쪽수 순서(오름차순)를 보상합니다. 하지만 반복 구조를 보지 못하면 여전히 다른 순서와 경쟁합니다. W도 켜 보세요.';
    else msg = `완전한 모델(전제 조건 + S₀ + F + W)에서 이 리뷰의 순서는 ${App.fmtInt(pool.length)}가지 성립 가능한 순서 중 <strong>단독 1위</strong>입니다. 결정적인 것은 반복되는 “방법 → 실험” 구조(W)였고, 논문의 흐름(F)도 같은 순서를 가리킵니다. 전환 비용만으로는 2위였다는 점에서, 논문이 실험 2에서 보인 결론(쌍별 비용만으로는 좋은 순서를 고를 수 없다)이 이 리뷰에서도 그대로 재현됩니다.`;
    if (!isOurs) msg += ' <span class="muted">순서를 바꿔 본 상태입니다. 초기화를 누르면 이 리뷰의 실제 순서로 돌아갑니다.</span>';
    $('meta-explain').innerHTML = msg;
    const p = App.palette();
    if (ranked.length <= 40) {
      // 성립하는 순서가 적으면 순서 하나하나를 막대로 보여준다
      const curKey = key(st.order);
      const vals = ranked.map((x, i) => ({ rank: i + 1, c: x.c, who: key(x.r.order) === curKey ? ttl : i === 0 ? '최적' : '다른 순서', lbl: x.r.order.map((k) => SEC[k].n).join('-') }));
      if (!vals.some((v) => v.who === ttl)) vals.push({ rank: '현재', c: cc, who: ttl, lbl: st.order.map((k) => SEC[k].n).join('-') });
      App.mountRaw($('meta-hist'), {
        width: 380, height: 170, data: { values: vals },
        mark: { type: 'bar', cornerRadiusEnd: 3 },
        encoding: {
          x: { field: 'rank', type: 'ordinal', title: '순위 (성립하는 순서 전체)', axis: { labelAngle: 0 } },
          y: { field: 'c', type: 'quantitative', title: '비용' },
          color: { field: 'who', type: 'nominal', scale: { domain: [ttl, '최적', '다른 순서'], range: [p.red, p.forest, p.water] }, legend: { title: null, orient: 'top', direction: 'horizontal' } },
          opacity: { condition: { test: "datum.who === '다른 순서'", value: 0.45 }, value: 1 },
          tooltip: [{ field: 'lbl', title: '섹션 순서' }, { field: 'c', title: '비용', format: '.2f' }],
        },
      });
    } else {
      App.mountRaw($('meta-hist'), {
        width: 380, height: 150,
        layer: [
          { data: { values: ranked.map((x) => ({ c: x.c })) }, mark: { type: 'bar', color: p.water, opacity: 0.55, cornerRadiusEnd: 2 }, encoding: { x: { field: 'c', bin: { maxbins: 30 }, type: 'quantitative', title: '비용 (낮을수록 좋은 순서)' }, y: { aggregate: 'count', type: 'quantitative', title: '순서 개수' } } },
          { data: { values: [{ c: ranked[0].c, w: '최적' }, { c: cc, w: ttl }] }, mark: { type: 'rule', strokeWidth: 3 }, encoding: { x: { field: 'c', type: 'quantitative' }, color: { field: 'w', type: 'nominal', scale: { domain: [ttl, '최적'], range: [p.red, p.forest] }, legend: { title: null, orient: 'top-right' } } } },
        ],
      });
    }
  }
  function reset() { st = Object.assign({ order: OURS.slice() }, DEFAULT); update(); }

  function init() {
    // 명세 표
    $('meta-table').innerHTML = `<table class="data"><thead><tr><th>섹션</th><th>형식 (마크급)</th><th>다루는 단위 (변환급)</th><th>다루는 개념 (인코딩급)</th><th>필요한 개념 (전제)</th><th>논문 쪽</th></tr></thead><tbody>${SEC.map((s) => `<tr><td><a href="#${s.id}">${s.n}. ${s.name}</a></td><td>${s.fmt}</td><td>${s.unit}</td><td>${s.con.map((c) => CON[c]).join(', ')}</td><td>${s.need.length ? s.need.map((c) => CON[c]).join(', ') : '없음'}</td><td class="num">${s.page}</td></tr>`).join('')}</tbody></table>`;
    const perms = App.permutations(OURS);
    all = perms.map(evaluate);
    ['need', 's0', 'f', 'w'].forEach((k) => $('meta-' + k).addEventListener('change', (e) => { st[k] = e.target.checked; update(); }));
    $('meta-full').addEventListener('click', () => { st = Object.assign({ order: OURS.slice() }, { need: true, s0: true, f: true, w: true }); update(); });
    App.addReset(document.getElementById('plate-meta'), reset);
    update();
  }
  App.onTheme(update);
  App.lazy('s-meta', init);
})();
