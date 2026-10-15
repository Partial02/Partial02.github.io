/* 체험 1: 여섯 차트를 직접 배열하고 720가지 순서 전체와 비교 */
(function () {
  const App = window.App;
  const pool = document.getElementById('hook-pool');
  const slots = document.getElementById('hook-slots');
  const status = document.getElementById('hook-status');
  const result = document.getElementById('hook-result');
  const items = App.F4;
  const shownOrder = App.shuffle([0, 1, 2, 3, 4, 5], App.seeded(7)); // 발표 때마다 같은 배치
  let order = [];

  App.seqKey = (ord) => '0,' + ord.map((i) => i + 1).join(',');

  function drawPool() {
    pool.innerHTML = '';
    shownOrder.forEach((i) => {
      const t = App.thumb(items[i].spec, items[i].cap, { w: 140, h: 104 });
      t.tabIndex = 0;
      t.setAttribute('role', 'button');
      const used = order.indexOf(i);
      if (used >= 0) {
        t.classList.add('used');
        t.setAttribute('aria-disabled', 'true');
        t.appendChild(App.el(`<span class="badge">${used + 1}</span>`));
      } else {
        t.setAttribute('aria-label', `${items[i].cap} 차트를 ${order.length + 1}번째로 놓기`);
      }
      const act = () => { if (order.includes(i)) return; order.push(i); render(); };
      t.addEventListener('click', act);
      t.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } });
      pool.appendChild(t);
    });
  }
  function drawSlots() {
    slots.innerHTML = '';
    if (!order.length) {
      slots.innerHTML = '<span class="empty">위의 차트를 눌러서 하나씩 놓으세요. 놓인 차트를 누르면 빠집니다.</span>';
      return;
    }
    order.forEach((i, k) => {
      const b = App.el(`<button type="button" class="btn sm" title="빼기">${k + 1}. ${items[i].cap}</button>`);
      b.addEventListener('click', () => { order.splice(k, 1); render(); });
      slots.appendChild(b);
    });
  }
  async function render() {
    drawPool();
    drawSlots();
    status.textContent = order.length < 6 ? `${order.length} / 6` : '';
    if (order.length === 6) await score(); else result.classList.add('hidden');
  }
  async function score() {
    status.innerHTML = '<span class="spinner"></span> 720가지 순서를 계산하는 중';
    const res = await App.gsSequence(items.map((x) => x.spec));
    status.textContent = '';
    const key = App.seqKey(order);
    const mine = res.find((r) => r.sequence.join(',') === key);
    const best = res[0];
    const better = res.filter((r) => r.sequenceCost < mine.sequenceCost - 1e-9).length;
    const same = Math.abs(mine.sequenceCost - best.sequenceCost) < 1e-9;
    App.state.userOrder = order.slice();
    document.dispatchEvent(new CustomEvent('userorder'));
    document.getElementById('hook-verdict').innerHTML = same
      ? `내 순서는 720가지 중 비용이 가장 낮은 순서와 <strong>같은 비용</strong>입니다. (비용 ${App.fmt(mine.sequenceCost, 3)})`
      : `내 순서는 720가지 중 <strong>${better + 1}번째</strong>로 비용이 낮습니다. (내 비용 ${App.fmt(mine.sequenceCost, 2)}, 최적 ${App.fmt(best.sequenceCost, 2)})`;
    const p = App.palette();
    App.mountRaw(document.getElementById('hook-hist'), {
      width: 460, height: 170,
      layer: [
        { data: { values: res.map((r) => ({ c: r.sequenceCost })) }, mark: { type: 'bar', color: p.ink3 },
          encoding: { x: { field: 'c', bin: { maxbins: 40 }, type: 'quantitative', title: '시퀀스 비용 (낮을수록 좋음)' }, y: { aggregate: 'count', type: 'quantitative', title: '순서 개수' } } },
        { data: { values: [{ c: best.sequenceCost, w: '최적' }, { c: mine.sequenceCost, w: '내 순서' }] }, mark: { type: 'rule', strokeWidth: 3 },
          encoding: { x: { field: 'c', type: 'quantitative' }, color: { field: 'w', type: 'nominal', scale: { domain: ['내 순서', '최적'], range: [p.red, p.forest] }, legend: { title: null, orient: 'top-right' } } } },
      ],
    });
    App.renderSeq(document.getElementById('hook-best'), best, items);
    result.classList.remove('hidden');
  }
  document.getElementById('hook-reset').addEventListener('click', () => { order = []; render(); });
  App.onTheme(() => { if (order.length === 6) score(); });
  App.lazy('s-abstract', render);
})();
