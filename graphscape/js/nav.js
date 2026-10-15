/* 경로 레일(진행률·목차), 흐름도 다이얼로그, 발표 타이머, 키보드 이동 */
(function () {
  const secs = Array.from(document.querySelectorAll('main > section.sec'));
  const route = document.getElementById('route');
  const fill = document.getElementById('route-fill');
  const pct = document.getElementById('pct');
  const pctWhere = document.getElementById('pct-where');
  const topWhere = document.getElementById('top-where');
  const topPct = document.getElementById('top-pct');
  const gauge = document.getElementById('top-gauge');

  const wps = secs.map((s, i) => {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'wp';
    b.innerHTML = `<span class="dot"></span><span>${s.dataset.title}</span><span class="min">${s.dataset.min}분</span>`;
    b.addEventListener('click', () => go(i));
    li.appendChild(b);
    route.appendChild(li);
    return b;
  });

  function go(i) {
    const s = secs[Math.max(0, Math.min(secs.length - 1, i))];
    if (s) s.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  }

  let current = 0;
  function update() {
    const max = document.documentElement.scrollHeight - innerHeight;
    const p = max > 0 ? Math.min(1, Math.max(0, scrollY / max)) : 0;
    const pc = Math.round(p * 100);
    const probe = scrollY + innerHeight * 0.35;
    let cur = 0;
    secs.forEach((s, i) => { if (s.offsetTop <= probe) cur = i; });
    if (p > 0.995) cur = secs.length - 1;
    current = cur;
    // 현재 섹션 안에서의 진행 비율
    const s = secs[cur];
    const within = Math.min(1, Math.max(0, (probe - s.offsetTop) / Math.max(1, s.offsetHeight)));
    pct.textContent = pc + '%';
    topPct.textContent = pc + '%';
    pctWhere.textContent = s.dataset.title;
    topWhere.textContent = `${cur}. ${s.dataset.title}`.replace(/^0\. /, '');
    gauge.style.width = pc + '%';
    wps.forEach((w, i) => {
      w.classList.toggle('done', i < cur);
      w.classList.toggle('here', i === cur);
      if (i === cur) w.setAttribute('aria-current', 'step'); else w.removeAttribute('aria-current');
    });
    // 경로 선 채우기: 첫 점 중심부터 현재 점 + 섹션 내 진행만큼
    const dots = wps.map((w) => w.offsetTop + w.offsetHeight / 2);
    if (dots.length > 1) {
      const first = dots[0];
      route.style.setProperty('--first', first + 'px');
      route.style.setProperty('--span', dots[dots.length - 1] - first + 'px');
      fill.style.top = first + 'px';
      const next = cur < dots.length - 1 ? dots[cur + 1] : dots[cur];
      const y = dots[cur] + (next - dots[cur]) * within;
      fill.style.height = Math.max(0, y - first) + 'px';
    }
    updateFlow();
  }
  addEventListener('scroll', () => requestAnimationFrame(update), { passive: true });
  addEventListener('resize', update);
  setTimeout(update, 50);
  new ResizeObserver(() => update()).observe(document.querySelector('main'));

  /* ---------- 흐름도 ---------- */
  const dlg = document.getElementById('map');
  const flow = document.getElementById('flow');
  const subs = ['순서 짜기', '갤러리 vs 시퀀스', '비용 만들기', '강제 선택', '비용 해부', '시퀀스 평가', '데모 세 개', '장단점, 토론'];
  const W = 1000, NW = 114, GAP = (W - NW * secs.length) / (secs.length - 1), NY = 14, NH = 74;
  const cx = (i) => i * (NW + GAP) + NW / 2;
  // [출발, 도착, 라벨, 곡선 깊이]: 아래쪽에 깊이를 달리해 겹치지 않게 그린다
  const deps = [
    [2, 3, '검증', 46],
    [3, 4, 'F, W, S₀ 근거', 46],
    [4, 5, '평가', 46],
    [0, 4, '체험 1의 순서를 다시 해부', 112],
    [4, 6, '비용 함수로 데모 구동', 84],
    [3, 7, '다중 비교 보정 문제', 176],
    [5, 7, '세트 3 실패', 140],
  ];
  const H = NY + NH + 4 + 176 + 24;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="리뷰 흐름도"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>`;
  for (let i = 0; i < secs.length - 1; i++) {
    svg += `<path class="edge-main" style="color:var(--ink)" marker-end="url(#ah)" d="M${cx(i) + NW / 2} ${NY + NH / 2} L${cx(i + 1) - NW / 2 - 2} ${NY + NH / 2}"/>`;
  }
  let labels = '';
  deps.forEach(([a, b, lbl, h]) => {
    const x1 = cx(a) + (a < b ? 10 : -10), x2 = cx(b) - 10, y = NY + NH + 2;
    // 3차 베지어의 꼭짓점 깊이는 제어점 깊이의 3/4
    const ctrl = (h * 4) / 3;
    svg += `<path class="edge-dep" style="color:var(--contour)" marker-end="url(#ah)" d="M${x1} ${y} C ${x1} ${y + ctrl}, ${x2} ${y + ctrl}, ${x2} ${y + 2}"/>`;
    labels += `<text class="dep-lbl" x="${(x1 + x2) / 2}" y="${y + h + 4}" text-anchor="middle" style="paint-order:stroke;stroke:var(--surface);stroke-width:5px;stroke-linejoin:round">${lbl}</text>`;
  });
  svg += labels;
  secs.forEach((s, i) => {
    const x = cx(i) - NW / 2;
    svg += `<g class="node" data-i="${i}" tabindex="0" role="link" aria-label="${s.dataset.title}로 이동"><rect x="${x}" y="${NY}" width="${NW}" height="${NH}" rx="2"/>` +
      `<text class="t2" x="${x + 8}" y="${NY + 16}">${i === 0 ? '시작' : i + '단계'}, ${s.dataset.min}분</text>` +
      wrap(s.dataset.title, x + 8, NY + 37) +
      `<text class="t2" x="${x + 8}" y="${NY + 65}">${subs[i]}</text></g>`;
  });
  svg += '</svg>';
  function wrap(t, x, y) {
    const parts = t.split(': ');
    if (parts.length === 1) return `<text class="t1" x="${x}" y="${y + 4}">${t}</text>`;
    return `<text class="t1" x="${x}" y="${y - 2}" style="font-size:14.5px">${parts[0]}</text><text class="t2" x="${x}" y="${y + 13}" style="font-size:12.5px;fill:var(--ink)">${parts[1]}</text>`;
  }
  flow.innerHTML = svg;
  flow.querySelectorAll('.node').forEach((n) => {
    const i = +n.dataset.i;
    const goto = () => { dlg.close(); setTimeout(() => go(i), 30); };
    n.addEventListener('click', goto);
    n.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goto(); } });
  });
  function updateFlow() {
    flow.querySelectorAll('.node').forEach((n) => {
      const i = +n.dataset.i;
      n.classList.toggle('done', i < current);
      n.classList.toggle('here', i === current);
    });
  }
  const open = () => { updateFlow(); dlg.showModal(); };
  document.getElementById('open-map').addEventListener('click', open);
  document.getElementById('open-map-m').addEventListener('click', open);
  document.getElementById('close-map').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });

  /* ---------- 키보드: [ 이전, ] 다음, m 흐름도 ---------- */
  addEventListener('keydown', (e) => {
    if (e.target.closest && e.target.closest('input, select, textarea, [contenteditable]')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === ']') go(current + 1);
    else if (e.key === '[') go(current - 1);
    else if (e.key === 'm' && !dlg.open) open();
  });

  /* ---------- 발표 타이머 (15분) ---------- */
  const tOut = document.getElementById('timer-out');
  const tBtn = document.getElementById('timer-btn');
  let remain = 15 * 60, tick = null;
  const show = () => {
    const neg = remain < 0, a = Math.abs(remain);
    tOut.textContent = (neg ? '+' : '') + Math.floor(a / 60) + ':' + String(a % 60).padStart(2, '0');
    tOut.style.color = neg ? 'var(--red)' : '';
  };
  tBtn.addEventListener('click', () => {
    if (tick) { clearInterval(tick); tick = null; tBtn.textContent = '계속'; return; }
    tick = setInterval(() => { remain--; show(); }, 1000);
    tBtn.textContent = '멈춤';
  });
  tBtn.addEventListener('dblclick', () => { clearInterval(tick); tick = null; remain = 15 * 60; show(); tBtn.textContent = '시작'; });
  tBtn.title = '두 번 누르면 15:00으로 초기화';
})();
