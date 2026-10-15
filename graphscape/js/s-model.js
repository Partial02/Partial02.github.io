/* 방법론 2: 비용 함수 해부대 (체험 4). 720개 순서의 ΣT, W, F는 graphscape.sequence 결과를 그대로 쓴다 */
(function () {
  const App = window.App;
  const F4 = App.F4;
  const PRESET = { a: [4, 5, 3, 2, 0, 1], b: [0, 1, 3, 2, 4, 5], c: [0, 2, 4, 1, 3, 5] };
  const MULT = [1, 2, 5, 10, 20, 50, 100];
  const st = { order: PRESET.a.slice(), s0: false, f: false, w: false, fmode: 'paper', fw: 0 };
  let res = null, byKey = null;

  const strip = document.getElementById('model-strip');
  const chkS0 = document.getElementById('m-s0'), chkF = document.getElementById('m-f'), chkW = document.getElementById('m-w');
  const fmode = document.getElementById('m-fmode');
  const fw = document.getElementById('m-fw'), fwOut = document.getElementById('m-fw-out');
  const readout = document.getElementById('m-readout');
  const formula = document.getElementById('m-formula');
  const rankEl = document.getElementById('m-rank');
  const hist = document.getElementById('m-hist');
  const explain = document.getElementById('m-explain');

  const cost = (r) => {
    const T = r.sumOfTransitionCosts - (st.s0 ? 0 : r.transitions[0].cost);
    const Fraw = r.filterSequenceCost;
    const F = st.f ? (st.fmode === 'paper' ? Fraw : Fraw / 1000) * MULT[st.fw] : 0;
    const W = st.w ? r.globalWeightingTerm : 1;
    return { T, F, W, total: W * (T + F), Fraw };
  };
  const capList = (ord) => ord.map((i) => F4[i].cap).join(' → ');

  function drawStrip(r) {
    strip.innerHTML = '';
    const shown = st.w ? r : Object.assign({}, r, { globalWeightingTerm: 1 });
    const pat = shown.globalWeightingTerm < 1 && r.patterns && r.patterns[0];
    const inPat = new Set();
    if (pat) pat.appear.forEach((a) => { for (let k = 0; k < pat.pattern.length; k++) inPat.add(a + k); });
    if (st.s0) {
      const t0 = r.transitions[0];
      const z = App.el('<div class="thumb" style="width:84px;display:grid;place-items:center;color:var(--ink-3);font-size:12.5px;text-align:center">빈 차트<br>S₀</div>');
      strip.appendChild(z);
      const e0 = App.edge([], t0.cost, inPat.has(0));
      e0.querySelector('span').textContent = '첫 차트 만들기';
      strip.appendChild(e0);
    }
    st.order.forEach((ci, k) => {
      if (k > 0) {
        const t = r.transitions[k];
        strip.appendChild(App.edge(App.opsOf(t), t.cost, inPat.has(k)));
      }
      const th = App.thumb(F4[ci].spec, F4[ci].cap);
      th.draggable = true;
      th.dataset.k = k;
      const ctl = App.el(`<div class="row" style="justify-content:center;gap:4px;margin-top:2px"><button type="button" class="btn sm quiet" aria-label="${F4[ci].cap}를 왼쪽으로" ${k === 0 ? 'disabled' : ''}>◀</button><button type="button" class="btn sm quiet" aria-label="${F4[ci].cap}를 오른쪽으로" ${k === 5 ? 'disabled' : ''}>▶</button></div>`);
      const [l, rr] = ctl.querySelectorAll('button');
      l.addEventListener('click', () => move(k, k - 1));
      rr.addEventListener('click', () => move(k, k + 1));
      th.appendChild(ctl);
      th.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/plain', String(k)); th.style.opacity = '.4'; });
      th.addEventListener('dragend', () => { th.style.opacity = ''; });
      th.addEventListener('dragover', (e) => { e.preventDefault(); th.style.borderColor = 'var(--red)'; });
      th.addEventListener('dragleave', () => { th.style.borderColor = ''; });
      th.addEventListener('drop', (e) => { e.preventDefault(); th.style.borderColor = ''; const from = +e.dataTransfer.getData('text/plain'); move(from, k); });
      strip.appendChild(th);
    });
  }
  function move(from, to) {
    if (to < 0 || to > 5 || from === to) return;
    const o = st.order.slice();
    const [x] = o.splice(from, 1);
    o.splice(to, 0, x);
    st.order = o;
    update();
  }

  function update() {
    if (!res) return;
    const r = byKey.get(App.seqKey(st.order));
    const c = cost(r);
    const all = res.map((x) => ({ r: x, c: cost(x).total }));
    all.sort((a, b) => a.c - b.c);
    const best = all[0];
    const eps = 1e-9;
    const rank = all.filter((x) => x.c < c.total - eps).length + 1;
    const ties = all.filter((x) => Math.abs(x.c - best.c) < eps).length;
    drawStrip(r);
    const cell = (k, v, on = true) => `<div><div class="k">${k}</div><div class="v ${on ? '' : 'off'}">${v}</div></div>`;
    readout.innerHTML =
      cell('전환 비용 합 ΣT', App.fmt(c.T)) +
      cell('그중 S₀ → 첫 차트', App.fmt(r.transitions[0].cost), st.s0) +
      cell('필터 항 F', App.fmt(st.f ? c.F : c.Fraw, 3), st.f) +
      cell('전역 가중치 W', App.fmt(r.globalWeightingTerm, 3), st.w) +
      cell('최종 비용', App.fmt(c.total, 3));
    formula.innerHTML = `Cost = ${st.w ? App.fmt(c.W, 3) + ' × ' : ''}(${st.f ? App.fmt(c.F, 3) + ' + ' : ''}${App.fmt(c.T)}) = <strong>${App.fmt(c.total, 3)}</strong>`;
    rankEl.innerHTML = rank === 1 ? `720가지 중 <strong>최적</strong>${ties > 1 ? ` (같은 비용 ${ties}개 공동 1위)` : ''}` : `720가지 중 <strong>${rank}위</strong>`;
    const p = App.palette();
    App.mountRaw(hist, {
      width: 400, height: 150,
      layer: [
        { data: { values: all.map((x) => ({ c: x.c })) }, mark: { type: 'bar', color: p.ink3 }, encoding: { x: { field: 'c', bin: { maxbins: 36 }, type: 'quantitative', title: '현재 설정의 비용' }, y: { aggregate: 'count', type: 'quantitative', title: '순서 개수' } } },
        { data: { values: [{ c: best.c, w: '최적' }, { c: c.total, w: '현재 순서' }] }, mark: { type: 'rule', strokeWidth: 3 }, encoding: { x: { field: 'c', type: 'quantitative' }, color: { field: 'w', type: 'nominal', scale: { domain: ['현재 순서', '최적'], range: [p.red, p.forest] }, legend: { title: null, orient: 'top-right' } } } },
      ],
    });
    const bestOrd = best.r.sequence.slice(1).map((i) => i - 1);
    let msg = '';
    if (!st.s0 && !st.f && !st.w) msg = `전환 비용만 보면 매 단계 하나씩만 바꾸는 지그재그가 최적이고, 공동 1위가 ${ties}개나 됩니다. 시작점이 공짜라서 어디서 시작하든 상관없기 때문입니다. 그림 4(a)가 이 상태입니다. 이제 빈 차트 S₀를 켜 보세요.`;
    else if (st.s0 && !st.w) msg = `S₀를 켜면 첫 차트를 만드는 비용이 생겨서, 필터가 적은 “전체” 차트에서 시작하는 순서가 유리해집니다.${st.f ? ' F는 이 중 연도가 오름차순인 쪽을 고르는 타이브레이커로만 작동합니다.' : ' F를 켜면 동점 중에서 값이 오름차순인 쪽이 골라집니다.'} 하지만 구조는 여전히 지그재그입니다. 이제 W를 켜 보세요.`;
    else if (st.w) msg = `W를 켜면 “+필터 유럽 → 지역 유럽→미국”이 반복되는 병렬 구조(초록 굵은 화살표)가 이깁니다. 한 단계에서 두 개를 바꾸느라 ΣT는 오히려 커지지만, 비용 전체에 1/3이 곱해지기 때문입니다.`;
    else msg = 'S₀ 없이 F나 W만 켠 조합입니다. 항마다 시퀀스의 다른 측면을 결정한다는 점을 확인해 보세요.';
    if (st.f && st.fw > 0) msg += ` F에 ×${MULT[st.fw]}을 곱했습니다. 값의 방향(오름차순)은 더 강하게 고정되지만, “연도로 묶을지 지역으로 묶을지”라는 그룹 기준은 W가 구분하지 못합니다. 실험 2 세트 3의 실패와 연결되는 지점입니다.`;
    msg += `<br><span class="muted">현재 설정의 최적: ${capList(bestOrd)}</span>`;
    explain.innerHTML = msg;
  }

  function sync() {
    st.s0 = chkS0.checked; st.f = chkF.checked; st.w = chkW.checked;
    st.fw = +fw.value; fwOut.textContent = '×' + MULT[st.fw];
    fw.disabled = !st.f;
    update();
  }
  [chkS0, chkF, chkW].forEach((c) => c.addEventListener('change', sync));
  fw.addEventListener('input', sync);
  fmode.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; st.fmode = b.dataset.v; App.setPressed(fmode, st.fmode); update(); });
  document.querySelectorAll('#plate-model [data-preset]').forEach((b) => {
    b.addEventListener('click', () => {
      const p = b.dataset.preset;
      if (p === 'mine') {
        if (!App.state.userOrder) { explain.innerHTML = '<b>체험 1에서 아직 순서를 만들지 않았습니다.</b> 맨 위로 가서 여섯 차트를 배열하면 여기로 불러올 수 있습니다.'; return; }
        st.order = App.state.userOrder.slice();
      } else if (p === 'best') {
        const all = res.map((x) => ({ r: x, c: cost(x).total })).sort((a, b) => a.c - b.c);
        st.order = all[0].r.sequence.slice(1).map((i) => i - 1);
      } else {
        st.order = PRESET[p].slice();
        // 그림 4의 단계와 같은 항 조합으로 맞춤
        chkS0.checked = p !== 'a'; chkF.checked = p !== 'a'; chkW.checked = p === 'c';
        fw.value = 0;
        return sync();
      }
      update();
    });
  });
  const mineBtn = document.querySelector('#plate-model [data-preset="mine"]');
  const refreshMine = () => { mineBtn.disabled = !App.state.userOrder; };
  document.addEventListener('userorder', () => { refreshMine(); if (res) { st.order = App.state.userOrder.slice(); update(); } });

  async function init() {
    chkS0.checked = false; chkF.checked = false; chkW.checked = false;
    refreshMine();
    if (App.state.userOrder) st.order = App.state.userOrder.slice();
    strip.innerHTML = '<p class="small"><span class="spinner"></span> graphscape.sequence 실행 중</p>';
    res = await App.gsSequence(F4.map((x) => x.spec));
    byKey = new Map(res.map((r) => [r.sequence.join(','), r]));
    sync();
  }
  App.onTheme(update);
  App.lazy('s-model', init);
})();
