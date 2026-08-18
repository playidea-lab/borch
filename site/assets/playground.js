/**
 * 플레이그라운드 화면.
 *
 * 코드를 치고, 브라우저의 GPU 에서 돌리고, 나온 것을 본다. 서버로 아무것도 안
 * 보낸다 — 보낼 곳이 없다. 이 페이지는 정적 파일이고 계산은 전부 이 탭 안에서 난다.
 *
 * 언어가 둘이다. **밑바닥은 하나다** — 자바스크립트는 borch.ts 를 직접 부르고,
 * 파이썬은 Pyodide 위의 `borch_webgpu` 가 같은 커널을 부른다. 그 차이(파이썬을 한 번
 * 지나는 값)가 얼마인지는 저장소의 벤치가 재놓았다: 배치 64 에서 118.5ms 대 123.4ms.
 */

import { EXAMPLES } from "./examples.js";
import { t } from "./i18n.js";
import {
  formatBytes, highlight, probeDevice, requestStop, runCode, runPython,
} from "./runner.js";

const $ = (sel) => document.querySelector(sel);

const editor = $("#code");
const consoleOut = $("#console");
const picker = $("#example");
const langPicker = $("#lang");
const runBtn = $("#run");
const stopBtn = $("#stop");
const badge = $("#device-badge");
const badgeText = $("#device-text");
const chart = $("#chart");
const painted = $("#highlight").firstElementChild;
const surface = $("#surface");
const filename = $("#filename");

const STORE_KEY = "borch.playground.code";
let lang = "js";

/* ── 예시 목록 ──────────────────────────────────────────────────────── */

function fillPicker() {
  picker.textContent = "";
  for (const ex of EXAMPLES[lang]) {
    const opt = document.createElement("option");
    opt.value = ex.id;
    opt.textContent = ex.title;
    picker.append(opt);
  }
}

function findExample(id) {
  for (const set of Object.values(EXAMPLES)) {
    const hit = set.find((e) => e.id === id);
    if (hit) return hit;
  }
  return null;
}

function langOf(id) {
  return EXAMPLES.py.some((e) => e.id === id) ? "py" : "js";
}

/** 편집기의 언어를 바꾼다. 저장된 코드도 언어별로 따로 둔다. */
function setLang(next, { keepCode = false } = {}) {
  lang = next;
  langPicker.value = next;
  fillPicker();
  surface.textContent = t(next === "js" ? "editor.surfaceJs" : "editor.surfacePy");
  filename.textContent = t(next === "js" ? "editor.fileJs" : "editor.filePy");
  editor.setAttribute("aria-label", next === "js" ? "자바스크립트 코드" : "파이썬 코드");
  repaint();
  if (!keepCode) {
    const ex = EXAMPLES[lang][0];
    picker.value = ex.id;
    load(ex);
  }
}

function load(ex) {
  editor.value = ex.code;
  repaint();
  saveCode();
  clearConsole();
  resetChart();
  say(ex.blurb, "note");
  if (lang === "py") {
    say(t("editor.pyFirstRun"), "note");
  }
}

function saveCode() {
  localStorage.setItem(`${STORE_KEY}.${lang}`, editor.value);
}

/* ── 강조 층 ────────────────────────────────────────────────────────
 *
 * 뒤에 색을 칠한 층을 깔고 위의 textarea 는 글자를 투명하게 둔다. 입력·선택·
 * 되돌리기·한글 조합은 전부 브라우저의 것이라 우리가 흉내 낼 것이 없다 —
 * 직접 그리는 편집기가 그 자리에서 커서를 잃는다.
 */

function repaint() {
  // **끝의 줄바꿈 하나가 안 그려진다.** 마지막 줄이 비어 있으면 아래 층의 높이가
  // 한 줄 모자라고, 그러면 그 줄에서 스크롤이 어긋난다. 공백 하나를 붙여 막는다.
  painted.innerHTML = highlight(`${editor.value}\n `, lang);
  syncScroll();
}

function syncScroll() {
  const layer = painted.parentElement;
  layer.scrollTop = editor.scrollTop;
  layer.scrollLeft = editor.scrollLeft;
}

editor.addEventListener("scroll", syncScroll);

/* ── 처음에 무엇을 열 것인가 ────────────────────────────────────────── */

function boot() {
  const hash = new URLSearchParams(location.hash.slice(1));

  const shared = hash.get("code");
  if (shared) {
    try {
      setLang(hash.get("lang") === "py" ? "py" : "js", { keepCode: true });
      editor.value = decodeCode(shared);
      repaint();
      saveCode();
      say(t("editor.opened", t("editor.fromLink")), "note");
      return;
    } catch { /* 망가진 링크는 없는 셈 친다 */ }
  }

  const want = hash.get("example");
  const ex = want ? findExample(want) : null;
  if (ex) {
    setLang(langOf(ex.id), { keepCode: true });
    picker.value = ex.id;
    load(ex);
    return;
  }

  const wantLang = hash.get("lang") === "py" ? "py" : "js";
  setLang(wantLang, { keepCode: true });
  const saved = localStorage.getItem(`${STORE_KEY}.${wantLang}`);
  if (saved) {
    editor.value = saved;
    repaint();
    say(t("editor.opened", t("editor.lastTime")), "note");
  } else {
    const first = EXAMPLES[wantLang][0];
    picker.value = first.id;
    load(first);
  }
}

langPicker.addEventListener("change", () => setLang(langPicker.value));

picker.addEventListener("change", () => {
  const ex = findExample(picker.value);
  if (ex) load(ex);
});

editor.addEventListener("input", () => { repaint(); saveCode(); });

// 같은 문서 안에서 주소만 바뀌는 경우에도 예시가 바뀌어야 한다 — 해시만 바뀌면
// 브라우저는 페이지를 다시 안 읽는다.
window.addEventListener("hashchange", () => {
  const want = new URLSearchParams(location.hash.slice(1)).get("example");
  const ex = want ? findExample(want) : null;
  if (!ex) return;
  setLang(langOf(ex.id), { keepCode: true });
  picker.value = ex.id;
  load(ex);
});

// 탭이 초점을 옮기면 코드 편집기가 아니다.
editor.addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    e.preventDefault();
    const { selectionStart: s, selectionEnd: t, value } = editor;
    const pad = lang === "py" ? "    " : "  ";
    editor.value = `${value.slice(0, s)}${pad}${value.slice(t)}`;
    editor.selectionStart = editor.selectionEnd = s + pad.length;
    repaint();
  }
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    run();
  }
});

/* ── 출력 ───────────────────────────────────────────────────────────── */

function say(text, kind = "") {
  const line = document.createElement("div");
  if (kind) line.className = kind;
  line.textContent = text;
  consoleOut.append(line);
  consoleOut.scrollTop = consoleOut.scrollHeight;
}

function clearConsole() { consoleOut.textContent = ""; }

/* ── 손실 그래프 ────────────────────────────────────────────────────── */

let series = [];
let seriesName = "";

function resetChart() {
  series = [];
  seriesName = "";
  drawChart();
}

function pushPoint(name, value) {
  if (!Number.isFinite(value)) return;
  seriesName = name;
  series.push(value);
  if (series.length > 4000) series = series.slice(-2000);
  scheduleDraw();
}

let pending = false;
function scheduleDraw() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; drawChart(); });
}

function drawChart() {
  const dpr = window.devicePixelRatio || 1;
  const w = chart.clientWidth, h = chart.clientHeight;
  chart.width = Math.max(1, Math.round(w * dpr));
  chart.height = Math.max(1, Math.round(h * dpr));
  const ctx = chart.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--ember").trim() || "#ff7a3d";
  const faint = css.getPropertyValue("--fg-faint").trim() || "#6b788c";

  ctx.font = "11px ui-monospace, monospace";
  ctx.fillStyle = faint;

  if (series.length < 2) {
    ctx.fillText(t("chart.empty"), 10, h / 2);
    return;
  }

  const lo = Math.min(...series), hi = Math.max(...series);
  const pad = 14;
  const span = hi - lo || 1;
  const xAt = (i) => pad + (i / (series.length - 1)) * (w - pad * 2);
  const yAt = (v) => h - pad - ((v - lo) / span) * (h - pad * 2);

  ctx.strokeStyle = faint;
  ctx.globalAlpha = 0.3;
  ctx.beginPath();
  ctx.moveTo(pad, h - pad); ctx.lineTo(w - pad, h - pad);
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.strokeStyle = ink;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  series.forEach((v, i) => (i ? ctx.lineTo(xAt(i), yAt(v)) : ctx.moveTo(xAt(i), yAt(v))));
  ctx.stroke();

  ctx.fillStyle = faint;
  ctx.fillText(`${seriesName} ${series[series.length - 1].toPrecision(4)}`, pad, 14);
  ctx.fillText(hi.toPrecision(3), w - pad - 52, 14);
  ctx.fillText(lo.toPrecision(3), w - pad - 52, h - pad - 4);
}

window.addEventListener("resize", scheduleDraw);

/* ── 장치 ───────────────────────────────────────────────────────────── */

(async function showDevice() {
  try {
    const p = await probeDevice();
    if (p.ok) {
      badge.className = "badge on";
      badgeText.textContent = p.adapter;
    } else {
      badge.className = "badge off";
      badgeText.textContent = t(p.why === "no-api" ? "device.noApi" : "device.noAdapter");
      say(p.message, "err");
      say(t("device.noFallback"), "note");
    }
  } catch (err) {
    badge.className = "badge off";
    badgeText.textContent = t("device.notLoaded");
    say(String(err.message ?? err), "err");
  }
})();

/* ── 실행 ───────────────────────────────────────────────────────────── */

let running = false;

async function run() {
  if (running) return;
  running = true;
  runBtn.disabled = true;
  stopBtn.disabled = false;
  clearConsole();
  resetChart();

  const hooks = { onLog: (text, kind) => say(text, kind), onPlot: pushPoint };
  try {
    const result = lang === "js"
      ? await runCode(editor.value, hooks)
      : await runPython(editor.value, hooks);
    setStats(result);
    say("");
    say(t("run.done", result.ms.toFixed(0)), "ok");
    if (result.stats && result.stats.faults > 0) {
      say(t("run.faults", result.stats.faults, result.stats.firstFault), "err");
    }
  } catch (err) {
    say("");
    say(String(err && err.stack ? err.stack : err), "err");
  } finally {
    running = false;
    runBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function setStats(result) {
  const s = result.stats;
  $("#stat-time").textContent = `${result.ms.toFixed(0)} ms`;
  $("#stat-dispatch").textContent = s ? String(result.dispatchDelta) : "—";
  $("#stat-mem").textContent = s ? t("stats.tensors", formatBytes(s.bytes), s.tensors) : "—";
}

runBtn.addEventListener("click", run);
stopBtn.addEventListener("click", () => {
  requestStop();
  say(t("run.stopping"), "note");
});

/* ── 공유 ───────────────────────────────────────────────────────────── */

$("#share").addEventListener("click", async () => {
  const frag = `#lang=${lang}&code=${encodeCode(editor.value)}`;
  const url = `${location.origin}${location.pathname}${frag}`;
  try {
    await navigator.clipboard.writeText(url);
    say(t("share.copied"), "ok");
  } catch {
    say(url, "note");
  }
  history.replaceState(null, "", frag);
});

/** UTF-8 을 base64 로. 한글 주석이 들어가므로 그냥 btoa 는 안 된다. */
function encodeCode(text) {
  const bytes = new TextEncoder().encode(text);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function decodeCode(encoded) {
  const b64 = encoded.replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64);
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

// **맨 끝에서 연다.** 위쪽의 그래프 상태(`series`)보다 먼저 부르면 그 자리에서
// 초기화 전 접근으로 터지고, 화면은 그냥 빈 채로 남는다 — 실제로 그렇게 겪었다.
boot();
