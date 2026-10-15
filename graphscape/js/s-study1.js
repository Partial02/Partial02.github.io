/* 실험 1: 강제 선택 과제(체험 3)와 7개 과제 결과표(그림 4) */
(function () {
  const App = window.App;
  const D = App.D;
  const ST = { url: 'stocks.json' };
  const line = { data: ST, mark: 'line', encoding: { x: { field: 'date', type: 'temporal', title: null }, y: { field: 'price', type: 'quantitative', title: '주가' } } };
  const pointS = App.clone(line); pointS.mark = 'point';
  const logLine = App.clone(line); logLine.encoding.y.scale = { type: 'log' };
  const acc = { data: D, mark: 'point', encoding: { x: { field: 'Displacement', type: 'quantitative', title: '배기량' }, y: { field: 'Acceleration', type: 'quantitative', title: '가속도' } } };
  const accT = App.clone(acc); [accT.encoding.x, accT.encoding.y] = [acc.encoding.y, acc.encoding.x];
  const accEU = App.clone(acc); accEU.transform = [{ filter: { field: 'Origin', equal: 'Europe' } }];
  const F4 = App.F4;

  const TASKS = [
    {
      id: 't2', name: '과제 2: 마크 vs 스케일',
      intro: '아마존 주가 데이터입니다. 첫 차트(선 차트)는 고정입니다. 나머지 두 차트를 어떤 순서로 보여줄까요?',
      options: [
        { lbl: '산점도를 먼저', items: [[line, '선 차트 (고정)'], [pointS, '같은 데이터의 산점도'], [logLine, '로그 스케일 선 차트']], hyp: '마크 변경 두 번 + 스케일 변경 한 번', n: 44 },
        { lbl: '로그 스케일을 먼저', items: [[line, '선 차트 (고정)'], [logLine, '로그 스케일 선 차트'], [pointS, '같은 데이터의 산점도']], hyp: '스케일 변경 두 번 + 마크 변경 한 번', n: 7 },
      ],
      take: '압도적 다수가 마크를 두 번 바꾸는 쪽을 골랐습니다. 마크 변경이 데이터 변환보다 가볍다는 범주 가정이 지지되었습니다.',
    },
    {
      id: 't3', name: '과제 3: 필터 vs 전치',
      intro: '자동차의 배기량과 가속도입니다. 첫 차트는 고정입니다. 전치(축 교환)는 인코딩 중 가장 싼 연산이고, 필터는 변환 중 가장 비싼 연산입니다. 두 범주가 가장 가깝게 맞닿는 경계를 시험하는 과제입니다.',
      options: [
        { lbl: '전치를 먼저', items: [[acc, '전체 (고정)'], [accT, '축을 바꾼 차트'], [accEU, '유럽 차만']], hyp: '전치 두 번 + 필터 한 번', n: 14 },
        { lbl: '필터를 먼저', items: [[acc, '전체 (고정)'], [accEU, '유럽 차만'], [accT, '축을 바꾼 차트']], hyp: '필터 두 번 + 전치 한 번', n: 37 },
      ],
      take: '필터를 두 번 바꾸는 쪽이 더 많았습니다. 축을 뒤집는 것이 데이터 일부를 숨기는 것보다 무겁다는 뜻이고, 데이터 변환 < 인코딩 가정이 경계에서도 유지되었습니다.',
    },
    {
      id: 't7', name: '과제 7: 병렬 구조 vs 편집 최소화',
      intro: '체험 1과 같은 여섯 차트입니다. 논문에서는 자유롭게 배열했지만, 여기서는 대표적인 두 순서 중에서 고르게 단순화했습니다. 화살표 위의 비용 합도 비교해 보세요.',
      options: [
        { lbl: '연도별로 묶어 반복', items: [0, 2, 4, 1, 3, 5].map((i) => [F4[i].spec, F4[i].cap]), hyp: '병렬 구조 (한 단계에서 두 개를 바꿈)', n: 36 },
        { lbl: '한 번에 하나씩만 바꿈', items: [0, 1, 3, 2, 4, 5].map((i) => [F4[i].spec, F4[i].cap]), hyp: '매 단계 편집 최소화 (지그재그)', n: 15 },
      ],
      take: '편집 비용의 합은 지그재그 쪽이 더 작은데도, 사람들은 같은 구조가 반복되는 순서를 더 좋아했습니다. 쌍별 비용만 최소화하면 사람의 선호를 놓친다는 이 발견이 전역 가중치 W를 만든 직접적 이유입니다.',
    },
  ];
  const picked = {};
  const tabs = document.getElementById('s1-tabs');
  const panel = document.getElementById('s1-panel');
  let cur = 0;

  function drawTabs() {
    tabs.innerHTML = '';
    TASKS.forEach((t, i) => {
      const b = App.el(`<button type="button" role="tab" aria-selected="${i === cur}">${t.name}${picked[t.id] != null ? ' (완료)' : ''}</button>`);
      b.addEventListener('click', () => { cur = i; drawTabs(); drawPanel(); });
      tabs.appendChild(b);
    });
  }
  async function stripFor(items) {
    const strip = App.el('<div class="strip"></div>');
    for (let k = 0; k < items.length; k++) {
      if (k > 0) {
        const tr = await App.gsTransition(items[k - 1][0], items[k][0]);
        strip.appendChild(App.edge(App.opsOf(tr), tr.cost, false));
      }
      strip.appendChild(App.thumb(items[k][0], items[k][1], { w: 110, h: 80 }));
    }
    return strip;
  }
  async function drawPanel() {
    const t = TASKS[cur];
    panel.innerHTML = `<p>${t.intro}</p>`;
    const grid = App.el('<div style="display:grid;gap:12px;grid-template-columns:minmax(0,1fr)"></div>');
    panel.appendChild(grid);
    const sums = [];
    for (let i = 0; i < t.options.length; i++) {
      const o = t.options[i];
      const btn = App.el(`<button type="button" class="choice" aria-pressed="${picked[t.id] === i}"><div class="lbl">선택 ${i + 1}: ${o.lbl}</div></button>`);
      const strip = await stripFor(o.items);
      btn.appendChild(strip);
      let s = 0; strip.querySelectorAll('.edge .c').forEach((c) => (s += +c.textContent));
      sums.push(s);
      btn.appendChild(App.el(`<div class="small">편집 비용 합 ${App.fmt(s)}</div>`));
      btn.addEventListener('click', () => { picked[t.id] = i; drawTabs(); drawPanel(); });
      grid.appendChild(btn);
    }
    if (picked[t.id] != null) {
      const tot = t.options.reduce((a, o) => a + o.n, 0);
      const me = picked[t.id];
      const split = App.el('<div class="split" role="img"></div>');
      t.options.forEach((o, i) => {
        split.appendChild(App.el(`<div class="${i === 0 ? 'a' : 'b'} ${i === me ? 'me' : ''}" style="width:${(o.n / tot) * 100}%">${o.n}명${i === me ? ' (나)' : ''}</div>`));
      });
      split.setAttribute('aria-label', t.options.map((o) => `${o.lbl} ${o.n}명`).join(', '));
      const res = App.el(`<div style="margin-top:16px"><h4 style="margin:0">참가자 51명의 선택</h4></div>`);
      res.appendChild(split);
      res.appendChild(App.el(`<div class="small">${t.options.map((o) => `${o.lbl}: ${o.hyp}`).join('<br>')}</div>`));
      const agree = t.options[me].n >= Math.max(...t.options.map((o) => o.n));
      res.appendChild(App.el(`<p class="note ${agree ? 'check' : 'warn'}" style="margin-top:12px"><b>${agree ? '다수와 같은 선택입니다.' : '소수 쪽 선택입니다.'}</b> ${t.take}</p>`));
      panel.appendChild(res);
    }
  }

  /* 그림 4: 결과표와 보정 */
  const ROWS = [
    { n: 1, q: '필드 추가 vs 제거', obs: '여러 번 제거 28, 여러 번 추가 23', chi: 0.49, df: 1, model: '추가 = 제거 비용' },
    { n: 2, q: '마크 vs 스케일', obs: '마크 두 번 44, 스케일 두 번 7', chi: 26.84, df: 1, model: '마크 < 변환' },
    { n: 3, q: '필터 vs 전치', obs: '필터 두 번 37, 전치 두 번 14', chi: 10.37, df: 1, model: '변환 < 인코딩' },
    { n: 4, q: '필터 추가 vs 제거', obs: '28, 23', chi: 0.49, df: 1, model: '필터 비용 대칭' },
    { n: 5, q: '롤업 vs 드릴다운', obs: '롤업 26, 드릴다운 11, 혼합 14', chi: 7.41, df: 2, model: '빈 차트 S₀' },
    { n: 6, q: '단계별 편집 최소화', obs: '최소화 32, 기타 19 (축 방향 무시 시 44, 7). 시간순 42', chi: 3.31, df: 1, model: '전환 비용 T, 필터 항 F' },
    { n: 7, q: '병렬 구조 vs 편집 최소화', obs: '병렬 36, 최소화 15', chi: 8.65, df: 1, model: '전역 가중치 W' },
  ];
  ROWS.forEach((r) => (r.p = App.chiP(r.chi, r.df)));
  const seg = document.getElementById('corr-seg');
  let mode = 'none';
  function decide() {
    const a = 0.05, m = ROWS.length;
    const sig = {};
    if (mode === 'none') ROWS.forEach((r) => (sig[r.n] = r.p < a));
    if (mode === 'bonf') ROWS.forEach((r) => (sig[r.n] = r.p < a / m));
    if (mode === 'holm') {
      const sorted = ROWS.slice().sort((x, y) => x.p - y.p);
      let stop = false;
      sorted.forEach((r, i) => { const ok = !stop && r.p < a / (m - i); if (!ok) stop = true; sig[r.n] = ok; });
    }
    return sig;
  }
  const fp = (p) => (p < 0.001 ? '< .001' : p.toFixed(3).replace(/^0/, ''));
  function table() {
    const base = {}; ROWS.forEach((r) => (base[r.n] = r.p < 0.05));
    const sig = decide();
    const thr = mode === 'bonf' ? `α = .05 / 7 ≈ .0071` : mode === 'holm' ? 'α = .05 / (7 − 순위 + 1), 작은 p부터 차례로' : 'α = .05';
    let h = `<table class="data"><thead><tr><th>과제</th><th>질문</th><th>관찰</th><th>χ²(df)</th><th>p</th><th>판정 (${thr})</th><th>모델에 반영</th></tr></thead><tbody>`;
    ROWS.forEach((r) => {
      const s = sig[r.n], lost = base[r.n] && !s;
      const tag = lost ? '<span class="tag lost">보정 후 유의하지 않음</span>' : s ? '<span class="tag sig">유의</span>' : '<span class="tag ns">유의하지 않음</span>';
      h += `<tr><td class="num">${r.n}</td><td>${r.q}</td><td>${r.obs}</td><td class="num">${r.chi.toFixed(2)} (${r.df})</td><td class="num">${fp(r.p)}</td><td>${tag}</td><td>${r.model}</td></tr>`;
    });
    h += '</tbody></table>';
    document.getElementById('s1-table').innerHTML = h;
    const note = document.getElementById('corr-note');
    if (mode === 'none') note.innerHTML = '과제 1, 4의 “유의하지 않음”을 저자들은 “차이가 없다”로 해석해 같은 비용을 두었습니다. 엄밀히 동등함을 보이려면 동등성 검정이 필요하지만, 모델의 기본값을 정해야 하는 상황에서는 실용적인 선택입니다.';
    else note.innerHTML = '보정하면 과제 5(롤업 선호, p = .025)가 유의하지 않게 됩니다. 다만 과제마다 서로 다른 모델 가정을 검증하므로 하나의 가설 가족으로 묶어야 하는지는 논쟁적입니다. “롤업 선호는 다른 결과보다 근거가 약하다” 정도가 공정한 결론입니다.';
  }
  seg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; mode = b.dataset.v; App.setPressed(seg, mode); table(); });

  App.addReset(document.getElementById('plate-study1'), () => { Object.keys(picked).forEach((k) => delete picked[k]); cur = 0; drawTabs(); drawPanel(); });
  App.addReset(seg.closest('.plate'), () => { mode = 'none'; App.setPressed(seg, mode); table(); });
  App.lazy('s-study1', () => { drawTabs(); drawPanel(); table(); });
})();
