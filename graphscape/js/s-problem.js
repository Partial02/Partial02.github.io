/* 문제 상황: 추천 갤러리 vs 시퀀스. 응용 1에서도 같은 차트 묶음(App.POOL)을 쓴다 */
(function () {
  const App = window.App;
  const D = App.D;
  const q = (field, extra) => Object.assign({ field, type: 'quantitative' }, extra || {});
  App.POOL = [
    { cap: '연비 분포 (히스토그램)', spec: { data: D, mark: 'bar', encoding: { x: q('Miles_per_Gallon', { bin: true }), y: { aggregate: 'count', field: '*', type: 'quantitative' } } } },
    { cap: '지역별 평균 연비', spec: { data: D, mark: 'bar', encoding: { x: { field: 'Origin', type: 'nominal' }, y: q('Miles_per_Gallon', { aggregate: 'mean' }) } } },
    { cap: '마력과 연비', spec: { data: D, mark: 'point', encoding: { x: q('Horsepower'), y: q('Miles_per_Gallon') } } },
    { cap: '마력과 연비, 지역별 색', spec: { data: D, mark: 'point', encoding: { x: q('Horsepower'), y: q('Miles_per_Gallon'), color: { field: 'Origin', type: 'nominal' } } } },
    { cap: '연도별 평균 연비', spec: { data: D, mark: 'line', encoding: { x: { field: 'Year', type: 'ordinal' }, y: q('Miles_per_Gallon', { aggregate: 'mean' }) } } },
    { cap: '실린더 수별 평균 연비', spec: { data: D, mark: 'bar', encoding: { x: { field: 'Cylinders', type: 'ordinal' }, y: q('Miles_per_Gallon', { aggregate: 'mean' }) } } },
    { cap: '마력(로그)과 연비', spec: { data: D, mark: 'point', encoding: { x: q('Horsepower', { scale: { type: 'log' } }), y: q('Miles_per_Gallon') } } },
    { cap: '일본 차의 마력과 연비', spec: { data: D, mark: 'point', transform: [{ filter: { field: 'Origin', equal: 'Japan' } }], encoding: { x: q('Horsepower'), y: q('Miles_per_Gallon') } } },
  ];
  const GAL = [0, 1, 2, 3, 4, 5].map((i) => App.POOL[i]);
  const box = document.getElementById('gal-box');
  const cap = document.getElementById('gal-cap');
  const seg = document.getElementById('gal-toggle');
  const galleryOrder = [3, 0, 5, 2, 4, 1];

  function gallery() {
    box.innerHTML = '';
    const g = App.el('<div class="pick-grid" style="grid-template-columns:repeat(auto-fill,minmax(150px,1fr))"></div>');
    galleryOrder.forEach((i) => g.appendChild(App.thumb(GAL[i].spec, GAL[i].cap, { w: 140, h: 100 })));
    box.appendChild(g);
    cap.textContent = '추천 시스템은 이런 차트들을 격자로 늘어놓습니다. 어떤 차트를 먼저 봐야 하는지, 서로 무엇이 다른지는 보는 사람이 알아서 해석해야 합니다.';
  }
  async function sequence() {
    box.innerHTML = '<p class="small"><span class="spinner"></span> graphscape.sequence로 720가지 순서를 비교하는 중</p>';
    const res = await App.gsSequence(GAL.map((x) => x.spec));
    const strip = App.el('<div class="strip"></div>');
    box.innerHTML = '';
    box.appendChild(strip);
    App.renderSeq(strip, res[0], GAL);
    cap.textContent = '같은 여섯 장을 GraphScape가 정렬한 결과입니다. 화살표마다 무엇이 바뀌는지(편집 연산)와 그 비용이 드러납니다. 원자료에서 시작해 요약으로 올라가고, 비슷한 차트끼리 붙어 있습니다.';
  }
  seg.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    App.setPressed(seg, b.dataset.v);
    (b.dataset.v === 'g' ? gallery : sequence)();
  });
  App.lazy('s-problem', gallery);
})();
