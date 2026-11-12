/*
 * 프로젝트 관리 페이지
 * 폼 내용을 _projects/<파일이름>.md 로 만들고, GitHub의 새 파일/편집 화면으로 보냅니다.
 * 실제 저장(커밋)은 GitHub에서 하므로 저장소 주인만 반영할 수 있습니다.
 */
(function () {
  "use strict";

  var root = document.querySelector(".lp-admin");
  if (!root) return;
  var repo = root.getAttribute("data-repo");
  var branch = root.getAttribute("data-branch") || "master";
  var form = root.querySelector("[data-form]");
  var preview = root.querySelector("[data-preview]");
  var fileOut = root.querySelector("[data-file]");
  var updatesEl = root.querySelector("[data-updates]");
  var errorEl = root.querySelector("[data-error]");
  var noteEl = root.querySelector("[data-note]");
  var modeTitle = root.querySelector("[data-mode-title]");
  var saveBtn = root.querySelector("[data-save]");
  var pctOut = root.querySelector("[data-pct]");

  var projects = [];
  try { projects = JSON.parse(document.getElementById("lp-projects-data").textContent); } catch (e) { projects = []; }

  var editing = null; // 불러온 기존 프로젝트의 경로
  var slugTouched = false;

  var STATUS = { active: "진행 중", planned: "계획", paused: "보류", done: "완료" };

  /* ---------- 유틸 ---------- */
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function q(s) { return JSON.stringify(String(s)); } // YAML에서도 유효한 큰따옴표 문자열
  function ym(d) { return d ? d.slice(0, 7).replace("-", ".") : ""; }
  function slugify(s) {
    return String(s).toLowerCase().normalize("NFKD")
      .replace(/[^a-z0-9\s-]/g, "").trim().replace(/[\s_]+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
  }
  function topicsOf(v) {
    return String(v || "").split(",").map(function (t) { return t.trim(); }).filter(Boolean);
  }

  /* ---------- 진행 기록 줄 ---------- */
  function addUpdate(date, text) {
    var li = document.createElement("li");
    li.innerHTML =
      '<input type="date" aria-label="날짜">' +
      '<input type="text" aria-label="내용" placeholder="무엇이 진행됐는지">' +
      '<button type="button" class="lp-abtn lp-abtn--quiet" aria-label="이 기록 지우기"><i class="fas fa-times" aria-hidden="true"></i></button>';
    var inputs = li.querySelectorAll("input");
    inputs[0].value = date || new Date().toISOString().slice(0, 10);
    inputs[1].value = text || "";
    li.querySelector("button").addEventListener("click", function () { li.remove(); render(); });
    li.addEventListener("input", render);
    updatesEl.appendChild(li);
    return inputs[1];
  }
  function readUpdates() {
    var out = [];
    updatesEl.querySelectorAll("li").forEach(function (li) {
      var i = li.querySelectorAll("input");
      if (i[1].value.trim()) out.push({ date: i[0].value, text: i[1].value.trim() });
    });
    return out;
  }

  /* ---------- 폼 → 데이터 ---------- */
  function read() {
    var f = form.elements;
    return {
      title: f.title.value.trim(),
      icon: f.icon.value.trim(),
      summary: f.summary.value.trim(),
      status: f.status.value,
      progress: parseInt(f.progress.value, 10) || 0,
      start: f.start.value,
      end: f.end.value,
      category: f.category.value.trim(),
      role: f.role.value.trim(),
      topics: topicsOf(f.topics.value),
      details: f.details.value.replace(/\s+$/, ""),
      updates: readUpdates(),
      slug: f.slug.value.trim()
    };
  }

  function toFile(d) {
    var L = ["---"];
    L.push("title: " + q(d.title));
    if (d.icon) L.push("icon: " + q(d.icon));
    L.push("status: " + d.status + "            # active(진행 중) / planned(계획) / paused(보류) / done(완료)");
    if (d.start) L.push("start: " + d.start);
    L.push("end: " + (d.end || "") + (d.end ? "" : "                # 진행 중이면 비워 두세요"));
    L.push("progress: " + d.progress + "           # 0~100");
    if (d.category) L.push("category: " + q(d.category));
    L.push("topics: [" + d.topics.map(q).join(", ") + "]");
    if (d.role) L.push("role: " + q(d.role));
    if (d.summary) L.push("summary: " + q(d.summary));
    if (d.updates.length) {
      L.push("updates:");
      d.updates.slice().sort(function (a, b) { return a.date < b.date ? 1 : -1; }).forEach(function (u) {
        L.push("  - date: " + u.date);
        L.push("    text: " + q(u.text));
      });
    }
    if (d.details) {
      L.push("details: |");
      d.details.split("\n").forEach(function (line) { L.push(line ? "  " + line : ""); });
    }
    L.push("---");
    return L.join("\n") + "\n";
  }

  /* ---------- 미리보기 ---------- */
  function tagColor(t) { return Array.from(String(t)).length % 5; } // Liquid의 size | modulo: 5 와 같은 규칙
  function rowHTML(d) {
    var st = STATUS[d.status] ? d.status : "active";
    var period = (d.start ? d.start.replace(/-/g, ".") : "") + (d.end ? " → " + d.end.replace(/-/g, ".") : (st === "active" ? " → 진행 중" : ""));
    var tags = (d.category ? '<span class="nt-tag nt-c' + tagColor(d.category) + '">' + esc(d.category) + "</span>" : "") +
      d.topics.map(function (t) { return '<span class="nt-tag nt-c' + tagColor(t) + '">' + esc(t) + "</span>"; }).join("");
    var icon = d.icon ? esc(d.icon) : '<i class="far fa-file-alt"></i>';
    return '<div class="nt-col nt-col--single" aria-hidden="true">' +
      '<div class="nt-col__head"><span class="nt-status nt-status--' + st + '">' + STATUS[st] + "</span></div>" +
      '<div class="nt-card">' +
      '<span class="nt-card__title"><span class="nt-icon">' + icon + "</span>" + esc(d.title || "프로젝트 이름") + "</span>" +
      (d.summary ? '<span class="nt-card__sum">' + esc(d.summary) + "</span>" : "") +
      '<span class="nt-prog"><span class="nt-meter"><span style="width:' + d.progress + '%"></span></span><span class="nt-pct">' + d.progress + "%</span></span>" +
      (tags ? '<span class="nt-tags">' + tags + "</span>" : "") +
      (period ? '<span class="nt-date">' + esc(period) + "</span>" : "") +
      (d.role ? '<span class="nt-card__sum"><i class="far fa-user" aria-hidden="true"></i> ' + esc(d.role) + "</span>" : "") +
      "</div></div>";
  }

  function render() {
    var d = read();
    if (!slugTouched && !editing) {
      form.elements.slug.value = slugify(d.title) || (d.title ? "project-" + (d.start || new Date().toISOString().slice(0, 10)).replace(/-/g, "") : "");
      d.slug = form.elements.slug.value;
    }
    pctOut.textContent = d.progress + "%";
    preview.innerHTML = rowHTML(d);
    fileOut.textContent = toFile(d);
  }

  /* ---------- 불러오기 / 초기화 ---------- */
  function fill(pr) {
    var f = form.elements;
    f.title.value = pr.title || "";
    f.icon.value = pr.icon || "";
    f.summary.value = pr.summary || "";
    f.status.value = STATUS[pr.status] ? pr.status : "active";
    f.progress.value = pr.progress || 0;
    f.start.value = pr.start || "";
    f.end.value = pr.end || "";
    f.category.value = pr.category || "";
    f.role.value = pr.role || "";
    f.topics.value = (Array.isArray(pr.topics) ? pr.topics : topicsOf(pr.topics)).join(", ");
    f.details.value = pr.details || "";
    f.slug.value = pr.slug || "";
    updatesEl.innerHTML = "";
    (pr.updates || []).forEach(function (u) { addUpdate(u.date, u.text); });
  }

  function setMode(pr) {
    editing = pr ? pr.path : null;
    form.elements.slug.readOnly = !!pr;
    modeTitle.textContent = pr ? "고치는 중: " + pr.title : "새 프로젝트";
    saveBtn.textContent = pr ? "복사하고 GitHub 편집 열기" : "GitHub에 저장";
    noteEl.textContent = pr
      ? "파일 내용이 복사됩니다. 열린 GitHub 편집 화면에서 전체 선택 후 붙여넣고 Commit changes를 누르세요."
      : "GitHub 새 파일 화면에 내용이 채워진 채로 열립니다. Commit changes를 누르면 1~2분 뒤 사이트에 반영됩니다.";
    hideError();
  }

  function reset() {
    form.reset();
    updatesEl.innerHTML = "";
    slugTouched = false;
    setMode(null);
    render();
  }

  root.querySelectorAll("[data-load]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var pr = projects[parseInt(btn.getAttribute("data-load"), 10)];
      if (!pr) return;
      fill(pr);
      setMode(pr);
      render();
      form.scrollIntoView({ behavior: "smooth", block: "start" });
      form.elements.title.focus({ preventScroll: true });
    });
  });

  /* ---------- 저장 ---------- */
  function showError(msg, field) {
    errorEl.textContent = msg; errorEl.hidden = false;
    if (field) field.focus();
  }
  function hideError() { errorEl.hidden = true; }

  function validate(d) {
    if (!d.title) return showError("프로젝트 이름을 입력하세요.", form.elements.title), false;
    if (!/^[a-z0-9-]+$/.test(d.slug)) return showError("파일 이름은 영문 소문자, 숫자, 하이픈(-)만 쓸 수 있어요.", form.elements.slug), false;
    if (d.start && d.end && d.end < d.start) return showError("종료일이 시작일보다 앞서 있어요.", form.elements.end), false;
    if (!editing && projects.some(function (p) { return p.slug === d.slug; }))
      return showError("같은 파일 이름의 프로젝트가 이미 있어요. 파일 이름을 바꾸거나 목록에서 불러와 고치세요.", form.elements.slug), false;
    hideError();
    return true;
  }

  function copyFallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } finally { ta.remove(); }
  }
  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(function () { copyFallback(text); });
    }
    copyFallback(text);
    return Promise.resolve();
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var d = read();
    if (!validate(d)) return;
    var text = toFile(d);
    if (editing) {
      var editUrl = "https://github.com/" + repo + "/edit/" + branch + "/" + editing;
      copy(text).then(function () { noteEl.textContent = "복사했어요. GitHub 편집 화면에서 전체 선택 후 붙여넣고 Commit changes를 누르세요."; });
      window.open(editUrl, "_blank", "noopener");
    } else {
      var url = "https://github.com/" + repo + "/new/" + branch +
        "?filename=" + encodeURIComponent("_projects/" + d.slug + ".md") +
        "&value=" + encodeURIComponent(text);
      window.open(url, "_blank", "noopener");
    }
  });

  root.querySelector("[data-copy]").addEventListener("click", function () {
    copy(fileOut.textContent).then(function () { noteEl.textContent = "파일 내용을 복사했어요."; });
  });
  root.querySelector("[data-reset]").addEventListener("click", reset);
  root.querySelector("[data-add-update]").addEventListener("click", function () { addUpdate().focus(); render(); });

  form.elements.slug.addEventListener("input", function () { slugTouched = true; });
  form.elements.status.addEventListener("change", function () {
    if (form.elements.status.value === "done") {
      form.elements.progress.value = 100;
      if (!form.elements.end.value) form.elements.end.value = new Date().toISOString().slice(0, 10);
    }
    render();
  });
  form.addEventListener("input", render);

  reset();
})();
