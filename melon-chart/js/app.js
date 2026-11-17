// melon-chart/data/ 폴더의 JSON 파일(내 컴퓨터에서 export_static.py 로 생성)을 불러와 화면에 표시
// 서버 없이 정적 파일만 쓰므로 누구나 볼 수 있습니다.
const DATA = "data/";

async function getJSON(name) {
  const res = await fetch(DATA + name, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${name} — HTTP ${res.status}`);
  return res.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showLoading(box, text) {
  box.hidden = false;
  box.className = "status loading";
  box.innerHTML = `<span class="spinner" aria-hidden="true"></span> ${text}`;
}

function showError(box, title, err, retry) {
  console.error(title, err);
  box.hidden = false;
  box.className = "status error";
  box.replaceChildren(
    el("strong", "", title),
    document.createElement("br"),
    el("span", "", "데이터 파일을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."),
    document.createElement("br"),
    el("small", "", err.message),
    document.createElement("br"),
  );
  const btn = el("button", "retry", "다시 시도");
  btn.type = "button";
  btn.addEventListener("click", retry);
  box.append(btn);
}

async function loadTop10() {
  const status = document.getElementById("top10-status");
  const list = document.getElementById("top10");
  showLoading(status, "차트를 불러오는 중…");
  list.hidden = true;
  try {
    const songs = await getJSON("top10.json");
    list.replaceChildren(...songs.map((s) => {
      const li = el("li");
      const info = el("span", "info");
      info.append(el("span", "title", s["곡명"]), el("span", "artist", s["가수"]));
      li.append(el("span", "rank", s["순위"]), info);
      return li;
    }));
    status.hidden = true;
    list.hidden = false;
  } catch (err) {
    showError(status, "Top 10을 불러오지 못했습니다.", err, loadTop10);
  }
}

async function loadInsight() {
  const box = document.getElementById("insight");
  showLoading(box, "한 줄 평을 불러오는 중…");
  try {
    const data = await getJSON("insight.json");
    box.className = "insight";
    box.textContent = data.insight;
    box.append(el("small", "meta", `${data.date} · ${data.model}`));
  } catch (err) {
    showError(box, "한 줄 평을 불러오지 못했습니다.", err, loadInsight);
  }
}

const ARTIST_LIMIT = 10;

async function loadArtists() {
  const status = document.getElementById("artists-status");
  const list = document.getElementById("artists");
  const more = document.getElementById("artists-more");
  showLoading(status, "가수별 곡 수를 불러오는 중…");
  list.hidden = true;
  more.hidden = true;
  try {
    const artists = await getJSON("artists.json");
    const max = artists[0] ? artists[0]["곡수"] : 1;
    list.replaceChildren(...artists.slice(0, ARTIST_LIMIT).map((a) => {
      const li = el("li");
      const bar = el("span", "bar");
      bar.style.width = `${(a["곡수"] / max) * 100}%`;
      const track = el("span", "track");
      track.append(bar);
      li.append(el("span", "name", a["가수"]), track, el("span", "count", `${a["곡수"]}곡`));
      return li;
    }));
    status.hidden = true;
    list.hidden = false;
    if (artists.length > ARTIST_LIMIT) {
      more.textContent = `외 ${artists.length - ARTIST_LIMIT}명 · 전체 ${artists.length}명이 Top 100에 올랐습니다.`;
      more.hidden = false;
    }
  } catch (err) {
    showError(status, "가수별 곡 수를 불러오지 못했습니다.", err, loadArtists);
  }
}

async function loadMeta() {
  try {
    const meta = await getJSON("meta.json");
    document.getElementById("updated").textContent =
      `데이터 기준: ${meta.exported_at.replace("T", " ")}`;
  } catch (err) { /* 갱신 시각은 없어도 됨 */ }
}

loadTop10();
loadInsight();
loadArtists();
loadMeta();
