// 로컬 FastAPI 서버에서 멜론 Top 10과 차트 한 줄 평을 불러와 화면에 표시
const API_BASE = "http://localhost:8000";

const FAIL_HINT =
  "API 서버가 켜져 있는지 확인해 주세요. (터미널에서 <code>uvicorn main:app --reload</code>) " +
  "브라우저가 '로컬 네트워크 접근' 권한을 물으면 허용해야 합니다.";

async function getJSON(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch (e) { /* 본문 없음 */ }
    throw new Error(`HTTP ${res.status}${detail ? " · " + detail : ""}`);
  }
  return res.json();
}

function showError(el, title, err) {
  el.className = "status error";
  el.innerHTML = `<strong>${title}</strong><br>${FAIL_HINT}`;
  const small = document.createElement("small");
  small.textContent = err.message === "Failed to fetch" ? "서버에 연결할 수 없습니다." : err.message;
  el.append(document.createElement("br"), small);
}

async function loadTop10() {
  const status = document.getElementById("top10-status");
  const list = document.getElementById("top10");
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
    showError(status, "Top 10을 불러오지 못했습니다.", err);
  }
}

async function loadInsight() {
  const el = document.getElementById("insight");
  try {
    const data = await getJSON("/insight");
    el.className = "insight";
    el.textContent = data.insight;
    const meta = document.createElement("small");
    meta.className = "meta";
    meta.textContent = `${data.date} · ${data.model}`;
    el.append(meta);
  } catch (err) {
    showError(el, "한 줄 평을 불러오지 못했습니다.", err);
  }
}

// 두 요청을 동시에 시작 (Top 10은 빠르고, Gemini 분석은 몇 초 걸림)
loadTop10();
loadInsight();
