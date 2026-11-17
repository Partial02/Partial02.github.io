// 로컬 FastAPI 서버에서 멜론 Top 10과 차트 한 줄 평을 불러와 화면에 표시
// localhost 는 IPv6(::1)로 먼저 풀릴 수 있어, 서버가 듣는 IPv4 주소를 직접 사용
const API_BASE = "http://127.0.0.1:8000";

const FAIL_HINT =
  "① API 서버가 켜져 있는지 확인해 주세요. (터미널에서 <code>uvicorn main:app --reload</code>)<br>" +
  "② Chrome 주소창 왼쪽 아이콘 → 사이트 설정 → <b>로컬 네트워크 액세스</b>(또는 '로컬 네트워크의 기기')를 <b>허용</b>으로 바꾼 뒤 새로고침하세요.";

async function getJSON(path) {
  // targetAddressSpace: Chrome의 '로컬 네트워크 접근(LNA)' 정책에 내 컴퓨터(loopback)로 가는 요청임을 알림
  const res = await fetch(API_BASE + path, { targetAddressSpace: "loopback" });
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch (e) { /* 본문 없음 */ }
    throw new Error(`HTTP ${res.status}${detail ? " · " + detail : ""}`);
  }
  return res.json();
}

function showError(el, title, err, retry) {
  console.error(title, err); // F12 콘솔에서 자세한 원인 확인용
  el.hidden = false;
  el.className = "status error";
  el.innerHTML = `<strong>${title}</strong><br>${FAIL_HINT}`;
  const small = document.createElement("small");
  small.textContent = err.message === "Failed to fetch"
    ? "서버에 연결할 수 없습니다. (서버 꺼짐 또는 브라우저의 로컬 네트워크 차단)"
    : err.message;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "retry";
  btn.textContent = "다시 시도";
  btn.addEventListener("click", retry);
  el.append(document.createElement("br"), small, document.createElement("br"), btn);
}

function showLoading(el, text) {
  el.hidden = false;
  el.className = "status loading";
  el.innerHTML = `<span class="spinner" aria-hidden="true"></span> ${text}`;
}

async function loadTop10() {
  const status = document.getElementById("top10-status");
  const list = document.getElementById("top10");
  showLoading(status, "차트를 불러오는 중…");
  list.hidden = true;
  try {
    const songs = await getJSON("/songs/top10");
    list.replaceChildren(...songs.map((s) => {
      const li = document.createElement("li");
      const rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = s["순위"];
      const info = document.createElement("span");
      info.className = "info";
      const title = document.createElement("span");
      title.className = "title";
      title.textContent = s["곡명"];
      const artist = document.createElement("span");
      artist.className = "artist";
      artist.textContent = s["가수"];
      info.append(title, artist);
      li.append(rank, info);
      return li;
    }));
    status.hidden = true;
    list.hidden = false;
  } catch (err) {
    showError(status, "Top 10을 불러오지 못했습니다.", err, loadTop10);
  }
}

async function loadInsight() {
  const el = document.getElementById("insight");
  showLoading(el, "Gemini가 차트를 분석하는 중…");
  try {
    const data = await getJSON("/insight");
    el.className = "insight";
    el.textContent = data.insight;
    const meta = document.createElement("small");
    meta.className = "meta";
    meta.textContent = `${data.date} · ${data.model}`;
    el.append(meta);
  } catch (err) {
    showError(el, "한 줄 평을 불러오지 못했습니다.", err, loadInsight);
  }
}

// 두 요청을 동시에 시작 (Top 10은 빠르고, Gemini 분석은 몇 초 걸림)
loadTop10();
loadInsight();
