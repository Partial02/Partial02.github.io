// /melon-chart/songs 같은 주소로 들어오면, 실제 FastAPI 서버(Cloudflare 터널 주소)의 같은 경로로 이동
// 터널 주소는 api.json 에 있음 (내 컴퓨터에서 start_public.py 가 켤 때마다 갱신)
(async () => {
  const path = document.body.dataset.path;
  const msg = document.getElementById("msg");
  try {
    const cfg = await (await fetch("/melon-chart/api.json", { cache: "no-store" })).json();
    if (!cfg.base) throw new Error("API 주소가 아직 설정되지 않았습니다.");
    // 서버가 켜져 있는지 먼저 확인 (꺼져 있으면 Cloudflare 오류 페이지 대신 안내 문구 표시)
    const alive = await fetch(cfg.base + "/openapi.json", { cache: "no-store" });
    if (!alive.ok) throw new Error(`HTTP ${alive.status}`);
    location.replace(cfg.base + path + location.search);
  } catch (err) {
    console.error(err);
    msg.className = "status error";
    msg.innerHTML =
      "<strong>API 서버가 꺼져 있습니다.</strong><br>" +
      "이 API는 사이트 주인의 컴퓨터에서 실행됩니다. 서버가 켜져 있을 때 다시 시도해 주세요.";
  }
})();
