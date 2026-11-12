/*
 * 진입 화면 배경: 그래프 신경망(GNN) 메시지 전달 시각화
 * - 마우스를 올린 노드 기준 3-hop 이웃(receptive field)을 강조하고,
 *   바깥 hop에서 안쪽으로 메시지 펄스가 모이는 모습(aggregation)을 보여줍니다.
 * - 빈 곳을 클릭하면 가장 가까운 노드에서 BFS 순서대로 신호가 퍼집니다.
 * - prefers-reduced-motion 이면 움직임 없이 hover 강조만 합니다.
 */
(function () {
  "use strict";

  var hero = document.querySelector("[data-gnn-hero]");
  if (!hero) return;
  var canvas = hero.querySelector("canvas");
  var ctx = canvas && canvas.getContext("2d");
  if (!ctx) return;

  var reduceMQ = window.matchMedia("(prefers-reduced-motion: reduce)");
  var coarse = window.matchMedia("(pointer: coarse)").matches;
  var hint = hero.querySelector("[data-hint]");
  if (hint && coarse) {
    hint.textContent = "노드를 탭하면 3-hop 이웃이 강조되고, 신호가 그래프 전체로 퍼집니다.";
  }

  var MAX_HOP = 3;
  var HOP_MS = 210; // 클릭 신호가 한 hop 이동하는 시간
  var W = 0, H = 0, dpr = 1;
  var nodes = [], edges = [], adj = [];
  var mouse = { x: -1e4, y: -1e4, active: false };
  var focus = -1, hops = null;
  var pulses = [];
  var ripple = null;
  var visible = true;
  var raf = 0;
  var lastSpawn = 0, lastAmbient = 0;

  function reduced() { return reduceMQ.matches; }

  /* ---------- 그래프 만들기 ---------- */
  function build() {
    var r = hero.getBoundingClientRect();
    W = r.width; H = r.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    var target = Math.round(Math.min(170, Math.max(42, (W * H) / 8200)));
    var minD = Math.sqrt((W * H) / target) * 0.6;
    var minD2 = minD * minD;
    nodes = [];
    var tries = 0;
    while (nodes.length < target && tries < target * 80) {
      tries++;
      var x = Math.random() * W, y = Math.random() * H, ok = true;
      for (var i = 0; i < nodes.length; i++) {
        var dx = nodes[i].ax - x, dy = nodes[i].ay - y;
        if (dx * dx + dy * dy < minD2) { ok = false; break; }
      }
      if (ok) {
        nodes.push({
          ax: x, ay: y, x: x, y: y,
          ph: Math.random() * Math.PI * 2,
          sp: 0.18 + Math.random() * 0.3,
          amp: 3 + Math.random() * 7,
          deg: 0
        });
      }
    }

    // k-최근접 이웃으로 간선 연결
    var seen = {};
    edges = [];
    adj = nodes.map(function () { return []; });
    var maxLen2 = (minD * 2.7) * (minD * 2.7);
    nodes.forEach(function (a, i) {
      var d = [];
      for (var j = 0; j < nodes.length; j++) {
        if (j === i) continue;
        var dx = a.ax - nodes[j].ax, dy = a.ay - nodes[j].ay;
        d.push([j, dx * dx + dy * dy]);
      }
      d.sort(function (p, q) { return p[1] - q[1]; });
      var k = Math.random() < 0.3 ? 3 : 2;
      for (var m = 0; m < k && m < d.length; m++) {
        if (d[m][1] > maxLen2) break;
        var j2 = d[m][0];
        var key = i < j2 ? i + "-" + j2 : j2 + "-" + i;
        if (seen[key]) continue;
        seen[key] = true;
        edges.push([i, j2]);
        adj[i].push(j2); adj[j2].push(i);
      }
    });
    nodes.forEach(function (n, i) { n.deg = adj[i].length; });

    focus = -1; hops = null; pulses = []; ripple = null;
  }

  function bfs(src, limit) {
    var h = new Int16Array(nodes.length).fill(-1);
    h[src] = 0;
    var q = [src];
    for (var d = 1; q.length && d <= limit; d++) {
      var nq = [];
      for (var a = 0; a < q.length; a++) {
        var nb = adj[q[a]];
        for (var b = 0; b < nb.length; b++) {
          if (h[nb[b]] < 0) { h[nb[b]] = d; nq.push(nb[b]); }
        }
      }
      q = nq;
    }
    return h;
  }

  function nearest(x, y, within) {
    var best = -1, bd = within * within;
    for (var i = 0; i < nodes.length; i++) {
      var dx = nodes[i].x - x, dy = nodes[i].y - y, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  /* ---------- 상태 갱신 ---------- */
  function move(t) {
    var s = t / 1000, still = reduced();
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i], x = n.ax, y = n.ay;
      if (!still) {
        x += Math.sin(s * n.sp + n.ph) * n.amp;
        y += Math.cos(s * n.sp * 0.8 + n.ph * 1.3) * n.amp;
      }
      if (mouse.active && !still) {
        var dx = mouse.x - x, dy = mouse.y - y, d = Math.sqrt(dx * dx + dy * dy);
        if (d > 1 && d < 200) {
          var f = Math.pow(1 - d / 200, 2) * 16;
          x += (dx / d) * f; y += (dy / d) * f;
        }
      }
      n.x = x; n.y = y;
    }
  }

  function updateFocus() {
    var f = mouse.active ? nearest(mouse.x, mouse.y, 150) : -1;
    if (f !== focus) {
      focus = f;
      hops = f >= 0 ? bfs(f, MAX_HOP) : null;
    }
  }

  function spawn(t) {
    if (reduced()) return;
    // 이웃에서 중심 노드로 모이는 메시지
    if (hops && t - lastSpawn > 70) {
      lastSpawn = t;
      for (var tries = 0; tries < 6; tries++) {
        var v = (Math.random() * nodes.length) | 0;
        var hv = hops[v];
        if (hv < 1) continue;
        var nb = adj[v].filter(function (u) { return hops[u] === hv - 1; });
        if (!nb.length) continue;
        var u = nb[(Math.random() * nb.length) | 0];
        pulses.push({ a: v, b: u, start: t, dur: 520, s: 1 - (hv - 1) * 0.22 });
        break;
      }
    }
    // 잔잔하게 흐르는 배경 신호
    if (t - lastAmbient > 380 && edges.length) {
      lastAmbient = t;
      var e = edges[(Math.random() * edges.length) | 0];
      var fwd = Math.random() < 0.5;
      pulses.push({ a: fwd ? e[0] : e[1], b: fwd ? e[1] : e[0], start: t, dur: 1400, s: 0.35 });
    }
  }

  function fireRipple(src, t) {
    var dist = bfs(src, 999);
    ripple = { dist: dist, start: t };
    if (reduced()) return;
    for (var i = 0; i < edges.length; i++) {
      var a = edges[i][0], b = edges[i][1];
      if (dist[a] < 0 || dist[b] < 0 || dist[a] === dist[b]) continue;
      var from = dist[a] < dist[b] ? a : b, to = from === a ? b : a;
      pulses.push({ a: from, b: to, start: t + dist[from] * HOP_MS, dur: HOP_MS, s: 0.95 });
    }
  }

  /* ---------- 그리기 ---------- */
  var HOP_EDGE = [0.9, 0.62, 0.4, 0.26];
  var HOP_FILL = ["#eef6ff", "#9cc8ff", "#6299f5", "#4c7fdc"];
  var HOP_GROW = [3.6, 1.8, 0.9, 0.4];

  function rippleGlow(i, t) {
    if (!ripple || ripple.dist[i] < 0) return 0;
    if (reduced()) return ripple.dist[i] === 0 ? 1 : 0;
    var phase = (t - ripple.start) / HOP_MS - ripple.dist[i];
    return phase < -0.5 || phase > 3 ? 0 : Math.exp(-phase * phase * 1.6);
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);

    // 기본 간선
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = "rgba(96, 140, 222, 0.2)";
    ctx.beginPath();
    for (var i = 0; i < edges.length; i++) {
      var a = nodes[edges[i][0]], b = nodes[edges[i][1]];
      ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();

    // 수용 영역(receptive field) 간선
    if (hops) {
      ctx.lineWidth = 1.3;
      for (var e = 0; e < edges.length; e++) {
        var ia = edges[e][0], ib = edges[e][1];
        var ha = hops[ia], hb = hops[ib];
        if (ha < 0 || hb < 0) continue;
        var na = nodes[ia], nb = nodes[ib];
        ctx.strokeStyle = "rgba(140, 190, 255," + HOP_EDGE[Math.max(ha, hb)] + ")";
        ctx.beginPath(); ctx.moveTo(na.x, na.y); ctx.lineTo(nb.x, nb.y); ctx.stroke();
      }
    }

    // 메시지 펄스
    var alive = [];
    for (var p = 0; p < pulses.length; p++) {
      var pu = pulses[p];
      var k = (t - pu.start) / pu.dur;
      if (k > 1) continue;
      alive.push(pu);
      if (k < 0) continue;
      var A = nodes[pu.a], B = nodes[pu.b];
      if (!A || !B) continue;
      var k0 = Math.max(0, k - 0.22);
      var x0 = A.x + (B.x - A.x) * k0, y0 = A.y + (B.y - A.y) * k0;
      var x1 = A.x + (B.x - A.x) * k, y1 = A.y + (B.y - A.y) * k;
      var g = ctx.createLinearGradient(x0, y0, x1, y1);
      g.addColorStop(0, "rgba(190, 220, 255, 0)");
      g.addColorStop(1, "rgba(200, 228, 255," + pu.s + ")");
      ctx.strokeStyle = g; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.fillStyle = "rgba(225, 240, 255," + pu.s + ")";
      ctx.beginPath(); ctx.arc(x1, y1, 1.9, 0, Math.PI * 2); ctx.fill();
    }
    pulses = alive;

    // 노드
    for (var n = 0; n < nodes.length; n++) {
      var nd = nodes[n];
      var r = 1.5 + Math.min(nd.deg, 6) * 0.38;
      var h = hops ? hops[n] : -1;
      var glow = rippleGlow(n, t);
      var fill = "rgba(126, 170, 245, 0.62)";
      if (h >= 0) { r += HOP_GROW[h]; fill = HOP_FILL[h]; }
      if (glow > 0.02) {
        ctx.fillStyle = "rgba(120, 180, 255," + (glow * 0.28) + ")";
        ctx.beginPath(); ctx.arc(nd.x, nd.y, r + 9 * glow, 0, Math.PI * 2); ctx.fill();
        r += 1.6 * glow;
        if (h < 0) fill = "rgba(200, 225, 255," + (0.62 + glow * 0.38) + ")";
      }
      if (h === 0) {
        ctx.fillStyle = "rgba(156, 200, 255, 0.18)";
        ctx.beginPath(); ctx.arc(nd.x, nd.y, r + 10, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "rgba(200, 228, 255, 0.7)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(nd.x, nd.y, r + 5, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.fillStyle = fill;
      ctx.beginPath(); ctx.arc(nd.x, nd.y, r, 0, Math.PI * 2); ctx.fill();
    }

    if (ripple && t - ripple.start > (maxDist(ripple.dist) + 4) * HOP_MS) ripple = null;
  }

  function maxDist(d) {
    var m = 0;
    for (var i = 0; i < d.length; i++) if (d[i] > m) m = d[i];
    return m;
  }

  /* ---------- 루프 ---------- */
  function frame(t) {
    raf = 0;
    move(t);
    updateFocus();
    spawn(t);
    draw(t);
    if (visible && !document.hidden && !reduced()) raf = requestAnimationFrame(frame);
  }
  function kick() { if (!raf) raf = requestAnimationFrame(frame); }

  /* ---------- 입력 ---------- */
  function local(e) {
    var r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  hero.addEventListener("pointermove", function (e) {
    if (e.pointerType === "touch") return;
    var p = local(e);
    mouse.x = p.x; mouse.y = p.y; mouse.active = true;
    kick();
  });
  hero.addEventListener("pointerleave", function () {
    mouse.active = false;
    kick();
  });
  hero.addEventListener("pointerdown", function (e) {
    if (e.target.closest("a, button")) return;
    var p = local(e);
    if (e.pointerType === "touch") { mouse.x = p.x; mouse.y = p.y; mouse.active = true; }
    var src = nearest(p.x, p.y, 1e5);
    if (src >= 0) fireRipple(src, performance.now());
    kick();
  });

  var lastW = 0, lastH = 0, resizeTimer = 0;
  function onResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      var r = hero.getBoundingClientRect();
      // 모바일 주소창 때문에 생기는 작은 높이 변화는 무시
      if (Math.abs(r.width - lastW) < 2 && Math.abs(r.height - lastH) < 140) return;
      lastW = r.width; lastH = r.height;
      build(); kick();
    }, 180);
  }
  window.addEventListener("resize", onResize);

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
      if (visible) kick();
    }).observe(hero);
  }
  document.addEventListener("visibilitychange", function () { if (!document.hidden) kick(); });
  if (reduceMQ.addEventListener) reduceMQ.addEventListener("change", kick);

  var r0 = hero.getBoundingClientRect();
  lastW = r0.width; lastH = r0.height;
  build();
  kick();
})();
