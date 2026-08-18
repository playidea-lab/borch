/**
 * 화면에 나가는 문구. **기본은 영어이고 한국어가 `/ko/` 에 있다.**
 *
 * 페이지의 본문은 각 HTML 안에 그대로 적혀 있다 — 자바스크립트가 꺼져도 읽히고,
 * 링크를 걸었을 때 미리보기에도 뜬다. 여기 있는 것은 **코드가 만드는 문구**뿐이다:
 * 장치 상태, 실행 결과, 오류, 예시의 제목.
 *
 * 어느 쪽인지는 `<html lang>` 이 정한다. 경로(`/ko/`)와 같이 가지만 경로를 읽지
 * 않는다 — 그러면 파일을 옮기는 순간 문구가 조용히 영어로 돌아간다.
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
      + "  저장소 루트에서: npm run build:ts\n"
      + "  찾던 자리: {0}\n  (원래 오류: {1})" },
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
  "load.moduleFailed": {
    en: "could not fetch {0} (HTTP {1})", ko: "{0} 를 못 받았다 (HTTP {1})" },
  "load.scriptFailed": {
    en: "could not load {0}", ko: "못 올렸다: {0}" },
};

/** `t("run.done", 12)` — 자리표시자는 `{0}`, `{1}` 이다. */
export function t(key, ...args) {
  const entry = STRINGS[key];
  if (!entry) return key;                    // 없는 열쇠는 열쇠 그대로 — 조용히 비지 않는다
  const text = entry[LANG] ?? entry.en;
  return text.replace(/\{(\d+)\}/g, (m, i) => String(args[Number(i)] ?? m));
}

/** `{en, ko}` 로 적힌 것에서 지금 언어를 고른다. 예시의 제목·본문이 그 꼴이다. */
export function pick(pair) {
  if (pair === null || typeof pair !== "object") return pair;
  return pair[LANG] ?? pair.en;
}
