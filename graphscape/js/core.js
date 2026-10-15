/* 공통 기능: 테마에 맞춘 Vega 렌더링, graphscape 호출 래퍼, 편집 연산 한국어 라벨, 통계 함수 */
(function () {
  const App = (window.App = { state: {}, modules: {} });
  const root = document.documentElement;

  /* ---------- 테마 ---------- */
  App.css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  App.palette = () => ({
    ink: App.css('--ink'), ink2: App.css('--ink-2'), ink3: App.css('--ink-3'), rule: App.css('--rule'),
    water: App.css('--water'), contour: App.css('--contour'), forest: App.css('--forest'), red: App.css('--red'),
    surface: App.css('--surface'), paper: App.css('--paper'),
  });
  App.originColors = () => { const p = App.palette(); return { domain: ['Europe', 'Japan', 'USA'], range: [p.water, p.contour, p.red] }; };
  App.vgConfig = (small) => {
    const p = App.palette();
    const fs = small ? 9 : 11;
    return {
      background: null,
      font: 'IBM Plex Sans KR, Apple SD Gothic Neo, sans-serif',
      view: { stroke: null },
      axis: {
        labelColor: p.ink2, titleColor: p.ink2, domainColor: p.ink3, tickColor: p.ink3, gridColor: p.rule,
        gridOpacity: 0.6, labelFontSize: fs, titleFontSize: fs + 0.5, titleFontWeight: 500, labelLimit: 70,
        titlePadding: 4, labelPadding: 2, tickSize: 3,
      },
      legend: { labelColor: p.ink2, titleColor: p.ink2, labelFontSize: fs, titleFontSize: fs + 0.5, symbolSize: 40, padding: 0, rowPadding: 1 },
      title: { color: p.ink, fontSize: fs + 2 },
      range: { category: [p.water, p.contour, p.red, p.forest, p.ink3, p.ink2] },
      mark: { color: p.water },
      point: { size: small ? 14 : 24, strokeWidth: 1.1 },
      circle: { size: small ? 14 : 24 },
      tick: { thickness: 1, opacity: 0.75 },
      line: { strokeWidth: 1.8 },
      bar: { color: p.water },
      rule: { color: p.ink },
      text: { color: p.ink },
    };
  };
  const themeKey = () => App.css('--paper') + App.css('--ink');
  const themeHandlers = [];
  App.onTheme = (fn) => themeHandlers.push(fn);
  let lastTheme = '';
  function themeChanged() {
    const k = themeKey();
    if (k === lastTheme) return;
    lastTheme = k;
    svgCache.clear();
    mounted.forEach((m) => { if (m.el.isConnected) App.mount(m.el, m.spec, m.opts, true); });
    themeHandlers.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
  }
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => setTimeout(themeChanged, 30));
  new MutationObserver(() => setTimeout(themeChanged, 30)).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  lastTheme = themeKey();

  /* ---------- 데이터 연결과 렌더링 ---------- */
  const DATA = window.GSDATA;
  const clone = (o) => JSON.parse(JSON.stringify(o));
  App.clone = clone;
  function prepare(spec, opts) {
    const s = clone(spec);
    const url = s.data && s.data.url;
    const name = url ? url.replace('.json', '') : 'cars';
    s.data = { values: DATA[name] };
    for (const ch of Object.keys(s.encoding || {})) {
      const e = s.encoding[ch];
      if (e.field === '*') delete e.field;
      if (e.field === 'Origin' && (ch === 'color')) e.scale = Object.assign({}, App.originColors(), e.scale || {});
      if (e.field === 'Year' && e.type === 'ordinal' && !e.axis) e.axis = { labelAngle: 0, values: [1970, 1974, 1978, 1982] };
      if (e.field === 'Origin' && ch === 'color' && opts.small) e.legend = null;
    }
    /* 그림 4 차트 묶음은 비교하기 쉽게 축 범위를 고정한다(그리기 전용: 모델 명세에는 넣지 않는다.
       명세에 domain을 넣으면 graphscape가 빈 차트와의 비교에서 스케일 편집을 추가로 센다) */
    const e = s.encoding || {};
    if (e.x && e.y && e.x.field === 'Horsepower' && e.y.field === 'Weight_in_lbs' && !e.x.scale) {
      e.x.scale = { domain: [40, 240] }; e.y.scale = { domain: [1500, 5200] };
    }
    if (opts.noAxisTitle) {
      for (const ch of ['x', 'y']) if (s.encoding && s.encoding[ch]) s.encoding[ch].title = null;
    }
    if (opts.mark) s.mark = Object.assign(typeof s.mark === 'string' ? { type: s.mark } : s.mark, opts.mark);
    s.width = opts.w || 120;
    s.height = opts.h || 90;
    s.autosize = { type: 'fit', contains: 'padding' };
    s.config = App.vgConfig(opts.small !== false && (opts.w || 120) < 220);
    return s;
  }
  App.prepare = prepare;
  const svgCache = new Map();
  App.svg = async function (spec, opts = {}) {
    const key = JSON.stringify(spec) + '|' + JSON.stringify(opts) + '|' + themeKey();
    if (svgCache.has(key)) return svgCache.get(key);
    const p = (async () => {
      const vgSpec = vl.compile(prepare(spec, opts)).spec;
      const view = new vega.View(vega.parse(vgSpec), { renderer: 'none' });
      const out = await view.toSVG();
      view.finalize();
      return out;
    })();
    svgCache.set(key, p);
    try { return await p; } catch (e) { svgCache.delete(key); throw e; }
  };
  const mounted = [];
  App.mount = async function (el, spec, opts = {}, rerender) {
    if (!rerender) {
      const i = mounted.findIndex((m) => m.el === el);
      if (i >= 0) mounted.splice(i, 1);
      mounted.push({ el, spec, opts });
      if (mounted.length > 600) mounted.splice(0, mounted.length - 600);
    }
    el.classList.add('loading');
    try {
      const svg = await App.svg(spec, opts);
      el.innerHTML = svg;
      const s = el.querySelector('svg');
      if (s) { s.setAttribute('role', 'img'); if (opts.label) s.setAttribute('aria-label', opts.label); }
    } catch (e) {
      console.error(e, spec);
      el.innerHTML = '<span class="small muted">차트를 그리지 못했습니다</span>';
    }
    el.classList.remove('loading');
  };
  /* 캔버스 렌더링(점이 많은 차트용) */
  App.canvas = async function (el, spec, opts = {}) {
    const vgSpec = vl.compile(prepare(spec, Object.assign({ small: false }, opts))).spec;
    if (el.__view) el.__view.finalize();
    el.innerHTML = '';
    const view = new vega.View(vega.parse(vgSpec), { renderer: 'canvas', container: el, hover: false });
    el.__view = view;
    await view.runAsync();
    return view;
  };
  App.mountRaw = async function (el, vlSpec, opts = {}) {
    /* 이미 data.values를 가진 일반 Vega-Lite 차트(분포도 등) */
    const s = clone(vlSpec);
    s.config = Object.assign(App.vgConfig(false), s.config || {});
    s.autosize = s.autosize || { type: 'fit', contains: 'padding' };
    const vgSpec = vl.compile(s).spec;
    const view = new vega.View(vega.parse(vgSpec), { renderer: 'none' });
    el.innerHTML = await view.toSVG();
    view.finalize();
  };

  /* ---------- 공통 차트 명세 ---------- */
  const D = { url: 'cars.json' };
  App.D = D;
  const scatter = (filters, x = 'Horsepower', y = 'Weight_in_lbs') => ({
    data: D, mark: 'point',
    transform: filters.map((f) => ({ filter: f })),
    encoding: { x: { field: x, type: 'quantitative' }, y: { field: y, type: 'quantitative' } },
  });
  const Y = (y) => ({ field: 'Year', equal: y });
  const R = (r) => ({ field: 'Origin', equal: r });
  /* 논문 그림 4와 실험 1 과제 7의 여섯 차트 */
  App.F4 = [
    { cap: '1976년 전체', spec: scatter([Y(1976)]) },
    { cap: '1980년 전체', spec: scatter([Y(1980)]) },
    { cap: '1976년 유럽', spec: scatter([Y(1976), R('Europe')]) },
    { cap: '1980년 유럽', spec: scatter([Y(1980), R('Europe')]) },
    { cap: '1976년 미국', spec: scatter([Y(1976), R('USA')]) },
    { cap: '1980년 미국', spec: scatter([Y(1980), R('USA')]) },
  ];
  App.F4.forEach((c) => { c.spec.encoding.x.title = '마력'; c.spec.encoding.y.title = '무게(lb)'; });

  /* ---------- graphscape 래퍼 ---------- */
  const seqCache = new Map();
  App.gsSequence = function (specs, fixFirst = false) {
    const key = JSON.stringify(specs) + fixFirst;
    if (!seqCache.has(key)) seqCache.set(key, graphscape.sequence(clone(specs), { fixFirst }));
    return seqCache.get(key);
  };
  App.gsTransition = (a, b) => graphscape.transition(clone(a), clone(b));
  App.opsOf = (t) => [...(t.mark || []), ...(t.transform || []), ...(t.encoding || [])];

  const CH = { x: 'x', y: 'y', color: '색', shape: '모양', size: '크기', row: '행', column: '열', text: '텍스트' };
  const unq = (v) => { try { return JSON.parse(v); } catch (e) { return v; } };
  const fval = (arr) => (arr && arr[0] != null ? unq(arr[0]) : '');
  const short = (v) => {
    if (typeof v === 'number' && v > 1900 && v < 2100) return "'" + String(v).slice(2);
    if (/^\d{4}$/.test(String(v))) return "'" + String(v).slice(2);
    const m = { Europe: '유럽', USA: '미국', Japan: '일본' };
    return m[v] || v;
  };
  App.opLabel = function (op, compact) {
    const n = op.name, d = op.detail || {};
    const mk = { area: 'area', bar: 'bar', line: 'line', point: 'point', text: 'text', tick: 'tick' };
    if (/^(AREA|BAR|LINE|POINT|TEXT|TICK)_(AREA|BAR|LINE|POINT|TEXT|TICK)$/.test(n)) {
      return compact ? `마크 ${d.before}→${d.after}` : `마크 변경 ${mk[d.before] || d.before} → ${mk[d.after] || d.after}`;
    }
    const chs = (arr) => (Array.isArray(arr) ? arr.map((x) => (x.how === 'removed' ? '−' : '+') + (CH[x.channel] || x.channel)).join(' ') : '');
    switch (n) {
      case 'SCALE': return compact ? '스케일' : `스케일 변경 (${chs(d)})`;
      case 'SORT': return compact ? '정렬' : `정렬 (${chs(d)})`;
      case 'BIN': return compact ? '비닝' : `비닝 (${chs(d)})`;
      case 'AGGREGATE': return compact ? '집계' : `집계 (${chs(d)})`;
      case 'SETTYPE': return '타입 변경';
      case 'ADD_FILTER': return compact ? `+필터 ${short(fval(d.after && d.after[2]))}` : `필터 추가 ${d.id} = ${fval(d.after && d.after[2])}`;
      case 'REMOVE_FILTER': return compact ? `−필터 ${d.id === 'Origin' ? '지역' : d.id}` : `필터 제거 ${d.id}`;
      case 'MODIFY_FILTER': return compact ? `${d.id === 'Year' ? '연도' : d.id === 'Origin' ? '지역' : d.id} ${short(fval(d.before && d.before[0]))}→${short(fval(d.after && d.after[0]))}` : `필터 수정 ${d.id}: ${fval(d.before && d.before[0])} → ${fval(d.after && d.after[0])}`;
    }
    let m;
    if ((m = n.match(/^SWAP_(\w+)_(\w+)$/))) return compact ? '전치' : `전치 (${m[1].toLowerCase()} ↔ ${m[2].toLowerCase()})`;
    if ((m = n.match(/^MOVE_([A-Z]+)_([A-Z]+)$/))) return `이동 ${CH[m[1].toLowerCase()]}→${CH[m[2].toLowerCase()]}`;
    if ((m = n.match(/^(ADD|REMOVE|MODIFY)_([A-Z]+)(_COUNT|_ADD_COUNT|_REMOVE_COUNT)?$/))) {
      const verb = { ADD: '필드 추가', REMOVE: '필드 제거', MODIFY: '필드 교체' }[m[1]];
      const ch = CH[m[2].toLowerCase()] || m[2];
      const f = (d.after && d.after.field) || (d.before && d.before.field) || '';
      const cnt = m[3] ? ' (count)' : '';
      return compact ? `${verb.replace('필드 ', '')} ${ch}` : `${verb}: ${ch}${f && f !== '*' ? ' ← ' + f : ''}${cnt}`;
    }
    return n;
  };
  App.fmt = (x, d = 2) => (x == null || isNaN(x) ? '–' : Number(x).toFixed(d));
  App.fmtInt = (x) => Number(x).toLocaleString('ko-KR');

  /* ---------- 시퀀스 스트립 ---------- */
  App.thumb = function (spec, cap, opts = {}) {
    const box = document.createElement('div');
    box.className = 'thumb';
    const slot = document.createElement('div');
    slot.className = 'chart-slot';
    box.appendChild(slot);
    if (cap) { const c = document.createElement('div'); c.className = 'cap'; c.textContent = cap; box.appendChild(c); }
    App.mount(slot, spec, Object.assign({ w: 116, h: 86, small: true, noAxisTitle: true, label: cap }, opts));
    return box;
  };
  App.edge = function (ops, cost, isPattern) {
    const e = document.createElement('div');
    e.className = 'edge' + (isPattern ? ' pat' : '');
    const lbl = ops.map((o) => App.opLabel(o, true)).join(' + ');
    e.innerHTML = `<span>${lbl || '변화 없음'}</span><div class="arrow"></div><span class="c">${App.fmt(cost)}</span>`;
    e.title = ops.map((o) => App.opLabel(o) + ` (${o.cost})`).join('\n');
    return e;
  };
  /* result: graphscape.sequence 결과 하나, items: [{spec, cap}] (null 명세 제외 순서) */
  App.renderSeq = function (container, result, items, opts = {}) {
    container.innerHTML = '';
    const seq = result.sequence;
    const offset = result.charts.length > items.length ? 1 : 0; // fixFirst=false면 0번이 빈 명세
    const pat = (result.patterns && result.patterns[0]) || null;
    const inPat = new Set();
    if (pat && result.globalWeightingTerm < 1) pat.appear.forEach((a) => { for (let k = 0; k < pat.pattern.length; k++) inPat.add(a + k); });
    const start = offset ? 1 : 0;
    for (let i = start; i < seq.length; i++) {
      if (i > start) {
        const t = result.transitions[i - 1];
        container.appendChild(App.edge(App.opsOf(t), t.cost, inPat.has(i - 1)));
      }
      const it = items[seq[i] - offset];
      container.appendChild(App.thumb(it.spec, it.cap, opts.thumb || {}));
    }
  };

  /* ---------- 통계 ---------- */
  App.rank = function (arr) { // 평균 순위(동점 처리)
    const idx = arr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const r = new Array(arr.length);
    for (let i = 0; i < idx.length;) {
      let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
      i = j + 1;
    }
    return r;
  };
  App.spearman = function (a, b) {
    const ra = App.rank(a), rb = App.rank(b);
    const n = a.length, ma = ra.reduce((s, v) => s + v, 0) / n, mb = rb.reduce((s, v) => s + v, 0) / n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - mb); da += (ra[i] - ma) ** 2; db += (rb[i] - mb) ** 2; }
    if (da === 0 || db === 0) return NaN;
    return num / Math.sqrt(da * db);
  };
  function erfc(x) { // Numerical Recipes
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  App.chiP = (x, df) => (df === 1 ? erfc(Math.sqrt(x / 2)) : Math.exp(-x / 2));
  App.seeded = function (seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); };
  App.shuffle = function (arr, rnd = Math.random) { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  App.permutations = function (arr) {
    if (arr.length <= 1) return [arr.slice()];
    const out = [];
    arr.forEach((x, i) => { App.permutations(arr.slice(0, i).concat(arr.slice(i + 1))).forEach((p) => out.push([x].concat(p))); });
    return out;
  };

  /* ---------- 지연 초기화: 섹션이 가까워지면 실행 ---------- */
  App.lazy = function (id, fn) {
    const el = document.getElementById(id);
    if (!el) return;
    let done = false;
    const run = () => { if (done) return; done = true; Promise.resolve().then(fn).catch((e) => console.error(id, e)); };
    if (!('IntersectionObserver' in window)) return run();
    const io = new IntersectionObserver((ents) => { if (ents.some((e) => e.isIntersecting)) { io.disconnect(); run(); } }, { rootMargin: '900px 0px' });
    io.observe(el);
  };
  App.el = function (html) { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  App.setPressed = function (seg, v) { seg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === v))); };
})();
