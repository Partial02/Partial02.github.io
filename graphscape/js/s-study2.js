/* 실험 2: 시퀀스 다섯 개 평가(체험 5) → 내 평점과 모델 버전별 스피어만 상관 */
(function () {
  const App = window.App;
  const D = App.D;
  const mpg = { field: 'Miles_per_Gallon', type: 'quantitative', title: '연비' };
  const mean = Object.assign({}, mpg, { aggregate: 'mean', title: '평균 연비' });
  const ori = { field: 'Origin', type: 'nominal', title: null };
  const cyl = { field: 'Cylinders', type: 'ordinal', title: '실린더' };
  const C = [
    { cap: '지역별 연비 (원자료)', spec: { data: D, mark: 'tick', encoding: { x: mpg, y: ori } } },
    { cap: '지역별 평균 연비', spec: { data: D, mark: 'point', encoding: { x: mean, y: ori } } },
    { cap: '실린더별 연비 (원자료)', spec: { data: D, mark: 'tick', encoding: { x: mpg, y: cyl } } },
    { cap: '실린더별 평균 연비', spec: { data: D, mark: 'point', encoding: { x: mean, y: cyl } } },
    { cap: '지역 × 실린더 (원자료)', spec: { data: D, mark: 'tick', encoding: { x: mpg, y: cyl, color: { field: 'Origin', type: 'nominal' } } } },
    { cap: '지역 × 실린더 평균', spec: { data: D, mark: 'point', encoding: { x: mean, y: cyl, color: { field: 'Origin', type: 'nominal' } } } },
  ];
  const LABELS = ['가', '나', '다', '라', '마'];
  const SCALE = ['매우 나쁨', '나쁨', '보통', '좋음', '매우 좋음'];
  const list = document.getElementById('rate-list');
  const go = document.getElementById('rate-go');
  const status = document.getElementById('rate-status');
  let seqs = [], res = null, byKey = null;
  const ratings = {};

  async function init() {
    list.innerHTML = '<p class="small"><span class="spinner"></span> graphscape.sequence로 후보 시퀀스를 만드는 중</p>';
    res = await App.gsSequence(C.map((c) => c.spec));
    byKey = new Map(res.map((r) => [r.sequence.join(','), r]));
    const best = res[0].sequence.slice(1).map((i) => i - 1);
    const key = (o) => o.join(',');
    const cands = [
      { o: best, kind: 'GraphScape 1위' },
      { o: [0, 2, 4, 1, 3, 5], kind: '저자식 수제: 원자료를 모두 본 뒤 평균' },
      { o: [1, 3, 5, 0, 2, 4], kind: '저자식 수제: 평균부터 (드릴다운)' },
      { o: [0, 1, 2, 3, 4, 5], kind: '저자식 수제: 원자료 → 평균 쌍 반복' },
      { o: [4, 5, 2, 3, 0, 1], kind: '저자식 수제: 복잡한 차트부터' },
    ];
    const seen = new Set();
    seqs = [];
    cands.forEach((c) => { if (!seen.has(key(c.o))) { seen.add(key(c.o)); seqs.push(c); } });
    const rnd = App.seeded(3);
    let r;
    do { r = App.shuffle([0, 1, 2, 3, 4, 5], rnd); } while (seen.has(key(r)));
    seqs.push({ o: r, kind: '무작위 순열' });
    seqs = seqs.slice(0, 5);
    if (!seqs.some((s) => s.kind === '무작위 순열')) seqs[4] = { o: r, kind: '무작위 순열' };
    seqs = App.shuffle(seqs, App.seeded(5));
    seqs.forEach((s, i) => { s.label = LABELS[i]; s.r = byKey.get('0,' + s.o.map((x) => x + 1).join(',')); });
    draw();
  }
  function draw() {
    list.innerHTML = '';
    seqs.forEach((s, i) => {
      const box = App.el(`<div style="border:1px solid var(--rule);background:var(--paper);padding:10px 12px;margin-bottom:10px"></div>`);
      box.appendChild(App.el(`<div class="row" style="justify-content:space-between"><strong>시퀀스 ${s.label}</strong><span class="small reveal-${i}"></span></div>`));
      const strip = App.el('<div class="strip"></div>');
      s.o.forEach((ci, k) => {
        if (k) strip.appendChild(App.el('<div class="edge" style="width:22px"><div class="arrow"></div></div>'));
        strip.appendChild(App.thumb(C[ci].spec, C[ci].cap, { w: 108, h: 78 }));
      });
      box.appendChild(strip);
      const fs = App.el(`<fieldset style="border:0;padding:0;margin:4px 0 0"><legend class="sr">시퀀스 ${s.label} 평가</legend><div class="row" style="gap:14px"></div></fieldset>`);
      SCALE.forEach((lab, v) => {
        const l = App.el(`<label class="chk"><input type="radio" name="rate-${i}" value="${v + 1}"> ${v + 1} ${lab}</label>`);
        l.querySelector('input').addEventListener('change', () => { ratings[i] = v + 1; check(); });
        fs.querySelector('.row').appendChild(l);
      });
      box.appendChild(fs);
      list.appendChild(box);
    });
  }
  function check() {
    const n = Object.keys(ratings).length;
    go.disabled = n < seqs.length;
    status.textContent = n < seqs.length ? `${n} / ${seqs.length} 평가함` : '모두 평가했습니다.';
  }
  go.addEventListener('click', reveal);
  function reveal() {
    const rate = seqs.map((_, i) => ratings[i]);
    const full = seqs.map((s) => -s.r.sequenceCost);
    const tOnly = seqs.map((s) => -s.r.sumOfTransitionCosts);
    const wOnly = seqs.map((s) => -s.r.globalWeightingTerm);
    const rho = { full: App.spearman(rate, full), t: App.spearman(rate, tOnly), w: App.spearman(rate, wOnly) };
    seqs.forEach((s, i) => {
      const el = list.querySelector('.reveal-' + i);
      el.innerHTML = `${s.kind}, 비용 ${App.fmt(s.r.sequenceCost, 2)} (ΣT ${App.fmt(s.r.sumOfTransitionCosts, 2)}, W ${App.fmt(s.r.globalWeightingTerm, 2)})`;
      if (s.kind === 'GraphScape 1위') el.style.color = 'var(--forest)';
    });
    const p = App.palette();
    const vals = [];
    [['완전한 모델', 'full', 0.8], ['W만', 'w', 0.6], ['T만', 't', 0.35]].forEach(([n, k, paper]) => {
      vals.push({ m: n, who: '내 평가 (시퀀스 5개)', v: isNaN(rho[k]) ? 0 : rho[k] });
      vals.push({ m: n, who: '논문 (55명)', v: paper });
    });
    App.mountRaw(document.getElementById('rate-rho'), groupBars(vals, p));
    const gsIdx = seqs.findIndex((s) => s.kind === 'GraphScape 1위');
    const top = Math.max(...rate);
    const myTop = seqs.filter((_, i) => rate[i] === top).map((s) => s.label);
    const allSame = rate.every((v) => v === rate[0]);
    let t = '';
    if (allSame) t = '다섯 개를 모두 같은 점수로 평가해서 순위 상관을 계산할 수 없습니다. 평점을 다르게 매겨 보세요.';
    else {
      t = `GraphScape 1위는 <strong>시퀀스 ${seqs[gsIdx].label}</strong>였고, 내가 가장 높게 준 것은 <strong>${myTop.join(', ')}</strong>입니다. `;
      t += `내 평점과의 상관은 완전한 모델 ${App.fmt(rho.full)}, W만 ${App.fmt(rho.w)}, T만 ${App.fmt(rho.t)}입니다. `;
      t += '시퀀스가 다섯 개뿐이라 ρ는 한두 개 순위만 바뀌어도 크게 출렁입니다. 논문의 ρ도 6개 과제 × 5개 시퀀스라는 작은 표본에서 나온 값이라는 점을 함께 기억해 둘 만합니다.';
    }
    document.getElementById('rate-text').innerHTML = t;
    document.getElementById('rate-result').classList.remove('hidden');
  }
  function groupBars(vals, p) {
    return {
      data: { values: vals },
      facet: { row: { field: 'm', type: 'nominal', title: null, sort: ['완전한 모델', 'W만', 'T만'], header: { labelAngle: 0, labelAlign: 'left', labelColor: p.ink2, labelFontSize: 12 } } },
      spacing: 6,
      spec: {
        width: 300, height: 34,
        mark: { type: 'bar' },
        encoding: {
          y: { field: 'who', type: 'nominal', title: null, axis: null },
          x: { field: 'v', type: 'quantitative', title: '스피어만 ρ (평점과 모델 순위)', scale: { domain: [-1, 1] } },
          color: { field: 'who', type: 'nominal', scale: { domain: ['내 평가 (시퀀스 5개)', '논문 (55명)'], range: [p.red, p.ink3] }, legend: { title: null, orient: 'bottom' } },
        },
      },
      autosize: { type: 'pad' },
    };
  }
  App.onTheme(() => { if (!document.getElementById('rate-result').classList.contains('hidden')) reveal(); });
  App.lazy('s-study2', init);
})();
