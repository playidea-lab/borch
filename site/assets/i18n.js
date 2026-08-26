/**
 * The wording that reaches the screen. **English is the default and Korean lives under `/ko/`.**
 *
 * A page's prose is written out inside its own HTML — it reads with JavaScript off and
 * it appears in a link preview. What lives here is only **the wording code produces**:
 * device state, run results, errors, the examples' titles.
 *
 * Which one is in force is decided by `<html lang>`. It travels with the path (`/ko/`)
 * but does not read the path — do that and moving a file quietly drops the wording back
 * to English.
 */

export const LANG = document.documentElement.lang === "ko" ? "ko" : "en";

const STRINGS = {
  "device.checking": {
    en: "checking device…", ko: "장치 확인 중…" },
  "device.noApi": {
    en: "no WebGPU (browser)", ko: "WebGPU 없음 (브라우저)" },
  "device.noAdapter": {
    en: "no adapter", ko: "어댑터 없음" },
  "device.notLoaded": {
    en: "not loaded", ko: "안 실렸다" },
  "device.ready": {
    en: "WebGPU is up. Run executes on this tab's GPU.",
    ko: "WebGPU 를 잡았다. Run 을 누르면 이 탭의 GPU 에서 돈다." },
  "device.noFallback": {
    en: "borch does not fall back — better not to run than to quietly go slow.",
    ko: "borch 는 폴백하지 않는다 — 조용히 느려지느니 안 도는 편이 낫다." },
  // **The message said "a driver blocklist" and stopped there.** That guess was right on
  // the machine this was written for, and it still cost a day, because naming a cause is
  // not naming a next step. Measured on Ubuntu 24.04 with an RTX 5080 (driver
  // 580.159.04): with neither flag there is no adapter at all; with the first alone the
  // adapter is SwiftShader, which is the CPU; with both it is `nvidia / blackwell`.
  // Where the flags, the ladder and the reason live. Relative from the page that says it,
  // which is why it is a key rather than a literal — `ko/` needs its own.
  "device.setupHref": {
    en: "setup.html", ko: "setup.html" },
  "device.setupSay": {
    en: "What each platform asks for, measured →", ko: "플랫폼마다 무엇이 필요한지, 실측 →" },
  "device.linuxFlags": {
    en: "On Linux with an NVIDIA card, Chrome refuses before it asks the driver. "
      + "Which switch gets past that differed on the two cards measured here, so turn on "
      + "all of them: #enable-unsafe-webgpu, #ignore-gpu-blocklist, #enable-vulkan, and "
      + "the ANGLE backend set to Vulkan — then relaunch, because a window already open "
      + "keeps the old settings. The setup page has the table.",
    ko: "리눅스에 NVIDIA 카드면, 크롬이 드라이버에 묻기 전에 먼저 거절한다. "
      + "그 거절을 어느 스위치가 지나가는지가 여기서 잰 두 카드에서 서로 달랐으니 전부 켜라 — "
      + "#enable-unsafe-webgpu, #ignore-gpu-blocklist, #enable-vulkan, 그리고 ANGLE 백엔드를 "
      + "Vulkan 으로. 그 다음 다시 띄워라, 이미 떠 있는 창은 옛 설정을 그대로 쓴다. "
      + "표는 설정 페이지에 있다." },
  // **A software adapter is the failure that looks like a success.** It runs, it agrees
  // with the golden, and every number it prints is a CPU's. The same four names the
  // runners refuse on (`tests/browser/launch.py`).
  "device.software": {
    en: "That adapter is a CPU rasteriser, not a GPU. It will run and it will agree with "
      + "the golden — and every speed it prints is the CPU's.",
    ko: "저 어댑터는 GPU 가 아니라 CPU 래스터라이저다. 돌기도 하고 골든과 값도 맞는다 — "
      + "다만 여기서 나오는 속도는 전부 CPU 의 것이다." },

  "run.done": {
    en: "done — {0} ms", ko: "끝 — {0} ms" },
  "run.doneLocal": {
    en: "done — {0} ms · no server was involved",
    ko: "끝 — {0} ms · 서버는 안 지났다" },
  "run.faults": {
    en: "WebGPU validation errors: {0} — {1}",
    ko: "WebGPU 검증 오류 {0} 건: {1}" },
  "run.stopping": {
    en: "stop requested — the loop halts where it checks stopped().",
    ko: "중지를 걸었다 — 루프가 stopped() 를 보는 자리에서 멈춘다." },

  "run.unknownError": { en: "unknown error", ko: "알 수 없는 오류" },
  "draw.rank": {
    en: "show() draws [H,W], [C,H,W] or [N,C,H,W] — this was [{0}].",
    ko: "show() 는 [H,W]·[C,H,W]·[N,C,H,W] 를 그린다 — 받은 것은 [{0}] 다." },
  "draw.channels": {
    en: "drawing needs 1 or 3 channels — this had {0}.",
    ko: "채널이 1 이나 3 이어야 그린다 — 받은 것은 {0} 다." },
  "draw.empty": { en: "nothing to plot yet", ko: "아직 그릴 것이 없다" },
  "editor.nameJs": { en: "JavaScript code", ko: "자바스크립트 코드" },
  "editor.namePy": { en: "Python code", ko: "파이썬 코드" },
  "editor.hint": {
    en: "Tab indents; press Escape then Tab to leave.",
    ko: "Tab 은 들여쓰기이고, 나가려면 Escape 를 누른 뒤 Tab 을 누른다." },
  "editor.opened": {
    en: "{0} — ⌘/Ctrl + Enter to run.", ko: "{0} — ⌘/Ctrl + Enter 로 돌린다." },
  "editor.fromLink": {
    en: "Opened code from a link", ko: "링크로 받은 코드를 열었다" },
  "editor.lastTime": {
    en: "Opened what you were typing last time", ko: "지난번에 치던 것을 열었다" },
  "editor.pyFirstRun": {
    en: "Python loads Pyodide once — a few seconds. After that it runs instantly.",
    ko: "파이썬은 처음 한 번 Pyodide 를 올린다 — 몇 초 걸린다. 그 다음부터는 즉시 돈다." },
  "editor.fileJs": {
    en: "code.js — runs inside this tab", ko: "code.js — 이 탭 안에서 돈다" },
  "editor.filePy": {
    en: "code.py — runs inside this tab", ko: "code.py — 이 탭 안에서 돈다" },
  "stats.tensors": {
    en: "{0} · {1} tensors", ko: "{0} · {1}개" },

  "editor.surfaceJs": {
    en: "init · Tensor · nn · optim · scope · keepAlive · log() · plot() · stopped()",
    ko: "init · Tensor · nn · optim · scope · keepAlive · log() · plot() · stopped()" },
  "editor.surfacePy": {
    en: "borch_webgpu as torch · print() · plot() · stopped()",
    ko: "borch_webgpu as torch · print() · plot() · stopped()" },

  "share.copied": {
    en: "Copied a URL that carries this code. Whoever opens it runs it with no install.",
    ko: "이 코드가 담긴 주소를 복사했다. 받는 쪽은 설치 없이 그대로 돌린다." },

  "chart.empty": {
    en: 'numbers passed to plot("loss", value) are drawn here',
    ko: 'plot("loss", value) 로 찍은 수가 여기 그려진다' },

  "load.noDist": {
    en: "Could not load the borch.ts build — browsers do not read TypeScript.\n"
      + "  From the repository root: npm run build:ts\n"
      + "  Looked for: {0}\n  (original error: {1})",
    ko: "borch.ts 방출물을 못 읽었다 — 브라우저는 TypeScript 를 그대로 못 읽는다.\n"
      + "  from the repository root: npm run build:ts\n"
      + "  looked in: {0}\n  (original error: {1})" },
  "load.borchTs": {
    en: "loading borch.ts and acquiring the adapter…",
    ko: "borch.ts 를 올리고 어댑터를 잡는 중…" },
  "load.pyodide": {
    en: "loading Pyodide… (served from this repository, once)",
    ko: "Pyodide 를 올리는 중… (저장소 안에서 온다, 처음 한 번만)" },
  "load.numpy": {
    en: "loading numpy…", ko: "numpy 를 올리는 중…" },
  "load.binding": {
    en: "loading borch_webgpu…", ko: "borch_webgpu 를 싣는 중…" },
  // **What loads when there is no adapter.** The core is numpy and does not want one,
  // so Python mode runs without WebGPU; `borch_webgpu` is what is missing, and the
  // message names it rather than saying something general about the browser.
  "load.core": {
    en: "no WebGPU — loading the core (borch) on wasm, without borch_webgpu…",
    ko: "WebGPU 가 없다 — 코어(borch)를 wasm 위에 싣는 중, borch_webgpu 없이…" },
  "load.moduleFailed": {
    en: "could not fetch {0} (HTTP {1})", ko: "{0} 를 못 받았다 (HTTP {1})" },
  "data.missing": {
    en: "The tutorial data is not here ({0}).\n"
      + "  From the repository root: python3 site/fetch_data.py\n"
      + "  (add --download if you do not have the CIFAR binaries)",
    ko: "튜토리얼 데이터가 없다 ({0}).\n"
      + "  from the repository root: python3 site/fetch_data.py\n"
      + "  (add --download if the CIFAR binaries are not there)" },
  "load.scriptFailed": {
    en: "could not load {0}", ko: "못 올렸다: {0}" },
};

/** `t("run.done", 12)` — the placeholders are `{0}`, `{1}`. */
export function t(key, ...args) {
  const entry = STRINGS[key];
  if (!entry) return key;                    // an unknown key comes back as itself — it does not go quietly blank
  const text = entry[LANG] ?? entry.en;
  return text.replace(/\{(\d+)\}/g, (m, i) => String(args[Number(i)] ?? m));
}

/** Picks the current language out of something written as `{en, ko}` — the examples' titles and prose are that shape. */
export function pick(pair) {
  if (pair === null || typeof pair !== "object") return pair;
  return pair[LANG] ?? pair.en;
}
