/**
 * The playground screen.
 *
 * Type code, run it on the browser's GPU, look at what comes out. Nothing is sent to a
 * server — there is nowhere to send it. This page is a static file and every
 * computation happens inside this tab.
 *
 * There are two languages. **There is one floor beneath them** — JavaScript calls
 * borch.ts directly, and Python calls the same kernels through `borch_webgpu` on
 * Pyodide. What that difference costs (the value passing through Python once) has been
 * measured by the repository's benchmark: 118.5ms against 123.4ms at batch 64.
 */

import { EXAMPLES } from "./examples.js";
import { t } from "./i18n.js";
import {
  decodeCode, describeError, encodeCode, formatBytes, highlight, probeDevice, requestStop,
  runCode, runPython,
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

/* ── the example list ───────────────────────────────────────────────── */

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

/** Switches the editor's language, keeping the saved code apart per language. */
function setLang(next, { keepCode = false } = {}) {
  lang = next;
  langPicker.value = next;
  fillPicker();
  surface.textContent = t(next === "js" ? "editor.surfaceJs" : "editor.surfacePy");
  filename.textContent = t(next === "js" ? "editor.fileJs" : "editor.filePy");
  // **Korean was hardcoded here.** The English page's editor introduced itself to a
  // screen reader in Korean — a place the eye never reaches, so nobody saw it.
  const name = t(next === "js" ? "editor.nameJs" : "editor.namePy");
  editor.setAttribute("aria-label", `${name}. ${t("editor.hint")}`);
  editor.setAttribute("title", t("editor.hint"));
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

/* ── the painted layer ──────────────────────────────────────────────
 *
 * A coloured layer sits behind and the textarea on top keeps its text transparent.
 * Typing, selection, undo and IME composition all stay the browser's, so there is
 * nothing for us to imitate — an editor that draws its own text is exactly where the
 * caret gets lost.
 */

function repaint() {
  // **A trailing newline does not get drawn.** With an empty last line the layer below
  // is one line short and the scroll goes out of step there. One space prevents it.
  painted.innerHTML = highlight(`${editor.value}\n `, lang);
  syncScroll();
}

function syncScroll() {
  const layer = painted.parentElement;
  layer.scrollTop = editor.scrollTop;
  layer.scrollLeft = editor.scrollLeft;
}

editor.addEventListener("scroll", syncScroll);

/* ── what to open with ──────────────────────────────────────────────── */

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
    } catch { /* a broken link counts as no link */ }
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

// The example has to change even when only the address moves within the same document —
// on a hash-only change the browser does not reload the page.
window.addEventListener("hashchange", () => {
  const want = new URLSearchParams(location.hash.slice(1)).get("example");
  const ex = want ? findExample(want) : null;
  if (!ex) return;
  setLang(langOf(ex.id), { keepCode: true });
  picker.value = ex.id;
  load(ex);
});

// If Tab moves focus it is not a code editor. **But there has to be a way out.**
//
// Without one, someone who arrives by keyboard cannot leave — Tab only grows the
// spaces. Escape arms it once and the next Tab leaves. The same door as the lesson
// blocks (`runnable.js`), so both editors take the same gesture.
let leaving = false;
editor.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { leaving = true; return; }
  if (e.key === "Tab" && leaving) { leaving = false; return; }
  if (e.key !== "Tab") leaving = false;
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

/* ── output ─────────────────────────────────────────────────────────── */

function say(text, kind = "") {
  const line = document.createElement("div");
  if (kind) line.className = kind;
  line.textContent = text;
  consoleOut.append(line);
  consoleOut.scrollTop = consoleOut.scrollHeight;
}

/** The same line, with a way out of it. `textContent` everywhere else keeps user output
 *  from becoming markup; this one place builds the anchor itself rather than parsing a
 *  string, so that stays true. */
function sayLink(text, href) {
  const line = document.createElement("div");
  line.className = "note";
  const a = document.createElement("a");
  a.href = href;
  a.textContent = text;
  line.append(a);
  consoleOut.append(line);
  consoleOut.scrollTop = consoleOut.scrollHeight;
}

function clearConsole() { consoleOut.textContent = ""; }

/* ── the loss plot ──────────────────────────────────────────────────── */

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

/* ── the device ─────────────────────────────────────────────────────── */


/** Is this adapter a CPU? The four names `tests/browser/launch.py` refuses on — one rule,
 *  two places, and the day they disagree is the day one of them lies about a number. */
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|software/i;

/** Linux, where Chrome's blocklist is what stands between the page and the driver. */
const ON_LINUX = /linux/i.test(navigator.userAgent) && !/android/i.test(navigator.userAgent);

(async function showDevice() {
  try {
    const p = await probeDevice();
    if (p.ok) {
      badge.className = SOFTWARE.test(p.adapter) ? "badge off" : "badge on";
      badgeText.textContent = p.adapter;
      if (SOFTWARE.test(p.adapter)) {
        say(t("device.software"), "err");
        sayLink(t("device.setupSay"), t("device.setupHref"));
      }
    } else {
      badge.className = "badge off";
      badgeText.textContent = t(p.why === "no-api" ? "device.noApi" : "device.noAdapter");
      say(p.message, "err");
      if (p.why === "no-adapter" && ON_LINUX) say(t("device.linuxFlags"), "note");
      say(t("device.noFallback"), "note");
      sayLink(t("device.setupSay"), t("device.setupHref"));
    }
  } catch (err) {
    badge.className = "badge off";
    badgeText.textContent = t("device.notLoaded");
    say(String(err.message ?? err), "err");
  }
})();

/* ── running ────────────────────────────────────────────────────────── */

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
    say(describeError(err), "err");
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

/* ── sharing ────────────────────────────────────────────────────────── */

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

// **Opened at the very end.** Called before the plot state above it (`series`), it
// blows up there on access-before-initialisation and the screen simply stays blank —
// which happened twice. (The second time was deleting this one line while truncating
// the end of the file. The symptom was the same: no exception, an empty editor, and no
// code in the share link.)
boot();
