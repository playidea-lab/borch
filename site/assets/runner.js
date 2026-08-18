/**
 * 이 페이지에서 borch 를 싣고 사용자 코드를 돌리는 자리.
 *
 * 히어로의 작은 데모와 플레이그라운드가 **같은 것을 쓴다** — 두 벌로 두면 한쪽만
 * 고쳐지고, 랜딩에서 도는 코드가 플레이그라운드에서 안 도는 상태가 조용히 생긴다.
 *
 * 싣는 것은 `borch-ts/dist` 다. 그것은 `.gitignore` 라 어느 커밋에도 없으므로,
 * 없을 때 "먼저 npm run build:ts" 라고 말해 주는 것이 이 파일의 일 중 하나다.
 * (빈 화면과 안 만든 방출물은 구별이 안 된다.)
 */

import { t } from "./i18n.js";

/** 방출물의 절대 URL. 상대 경로만 쓴다 — 호스트를 박으면 한 군데서만 맞는다. */
export const BORCH_URL = new URL("../../borch-ts/dist/src/index.js", import.meta.url).href;

let cached = null;

/** borch 모듈을 한 번만 싣는다. */
export async function loadBorch() {
  if (cached) return cached;
  try {
    cached = await import(BORCH_URL);
  } catch (err) {
    throw new Error(t("load.noDist", BORCH_URL, err && err.message ? err.message : err));
  }
  return cached;
}

/**
 * WebGPU 가 있는가 — **없으면 폴백하지 않는다.**
 *
 * 이 저장소가 TF.js 판을 걷어낸 이유가 그것이다. 조용히 WebGL·소프트웨어로
 * 내려가면 거기서 잰 수를 GPU 의 수로 읽게 된다. 여기서도 같은 규칙을 지킨다:
 * 안 되면 왜 안 되는지 말하고 멈춘다.
 */
export async function probeDevice() {
  const borch = await loadBorch();
  const result = await borch.probe();
  if (result.ok) return { ok: true, adapter: result.adapter };
  return { ok: false, why: result.why, message: result.message };
}

/** 지금 잡고 있는 장치의 수를 읽는다. 초기화 전이면 null. */
export function readStats(borch) {
  try {
    if (!borch.currentDevice()) return null;
    const d = borch.device();
    const mem = d.memory;
    const pool = d.pooled;
    return {
      dispatches: d.dispatches,
      tensors: mem.tensors,
      bytes: mem.bytes,
      pooledBytes: pool.bytes,
      pipelines: d.pipelineCount,
      faults: d.faults.count,
      firstFault: d.faults.first,
      adapter: borch.Device.adapterInfo,
    };
  } catch {
    return null;
  }
}

export function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/** 실행 중인 코드가 들여다보는 자리. 중지 버튼이 여기에 표시를 남긴다. */
const bridge = {
  stopped: false,
  log: () => {},
  plot: () => {},
};
globalThis.__borch_playground__ = bridge;
// 파이썬 쪽에서 `js.borchPG` 로 잡는 자리. 던더 이름은 `js` 모듈에서 집기 나쁘다.
globalThis.borchPG = bridge;

/**
 * 사용자 코드를 돌린다.
 *
 * ES 모듈로 만들어 `import()` 로 건다 — `new Function` 은 최상위 `await` 과
 * `import` 를 못 받는데, borch 의 첫 줄이 `await init()` 이라 그 둘이 필요하다.
 *
 * @param code   사용자가 친 자바스크립트
 * @param hooks  {onLog, onPlot} — 출력과 그래프를 받는 쪽
 */
export async function runCode(code, hooks = {}) {
  const borch = await loadBorch();
  const onLog = hooks.onLog ?? (() => {});
  const onPlot = hooks.onPlot ?? (() => {});

  bridge.stopped = false;
  bridge.log = onLog;
  bridge.plot = onPlot;

  // 콘솔을 가로챈다. 사용자는 `console.log` 를 칠 것이고(문서의 예시가 그렇다),
  // 그것이 브라우저 개발자 도구에만 나오면 이 페이지는 아무것도 안 보여준다.
  const real = { log: console.log, warn: console.warn, error: console.error };
  const relay = (kind) => (...args) => {
    real[kind](...args);
    onLog(args.map(render).join(" "), kind === "log" ? "" : kind === "warn" ? "note" : "err");
  };
  console.log = relay("log");
  console.warn = relay("warn");
  console.error = relay("error");

  const before = readStats(borch);
  const t0 = performance.now();
  let url = null;
  try {
    const source = [
      `import * as borch from ${JSON.stringify(BORCH_URL)};`,
      // 문서의 예시가 그대로 붙게 이름을 펼쳐 둔다.
      "const { init, Tensor, nn, optim, data, vision, fft, scope, keepAlive,",
      "        noGrad, manualSeed, einsum, slice, save, load, isAvailable, probe,",
      "        currentDevice, device, Device } = borch;",
      "const __pg = globalThis.__borch_playground__;",
      "const log = (...a) => __pg.log(a.map(v => typeof v === 'string' ? v : JSON.stringify(v)).join(' '));",
      "const plot = (name, value) => __pg.plot(name, value);",
      "const stopped = () => __pg.stopped;",
      "",
      code,
    ].join("\n");
    url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
    await import(url);
  } finally {
    console.log = real.log;
    console.warn = real.warn;
    console.error = real.error;
    if (url) URL.revokeObjectURL(url);
  }

  const ms = performance.now() - t0;
  const after = readStats(borch);
  return {
    ms,
    stats: after,
    dispatchDelta: after && before ? after.dispatches - before.dispatches
                 : after ? after.dispatches : 0,
  };
}

/** 중지 표시. 실행 중인 루프가 `stopped()` 로 이것을 본다. */
export function requestStop() {
  bridge.stopped = true;
}

function render(v) {
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean" || v == null) return String(v);
  try { return JSON.stringify(v); } catch { return String(v); }
}

/**
 * 아주 얇은 코드 강조. 라이브러리를 받지 않는다 — 이 저장소가 런타임 의존성 0 을
 * 지키는 것과 같은 이유이고, 여기 필요한 것은 다섯 색뿐이다.
 *
 * **파서가 아니다.** 정규식 하나로 훑으므로 중첩된 것까지는 안 본다. 여는 자리가
 * 왼쪽부터 잡히므로 흔한 경우는 맞고, 틀려도 색이 하나 어긋날 뿐 코드는 안 바뀐다 —
 * 이것은 편집기가 아니라 **표시**다. 진짜 파서가 필요해지면 그때 받아 온다.
 */

/** 언어마다 다른 것은 셋뿐이다: 주석 기호, 문자열의 꼴, 예약어. */
const LANGS = {
  js: {
    comment: "\\/\\/[^\\n]*",
    string: '"(?:[^"\\\\\\n]|\\\\.)*"' + "|'(?:[^'\\\\\\n]|\\\\.)*'" + "|`(?:[^`\\\\]|\\\\.)*`",
    keywords: [
      "await", "async", "const", "let", "var", "function", "class", "new",
      "return", "if", "else", "for", "of", "in", "while", "break", "continue",
      "try", "catch", "finally", "throw", "typeof", "instanceof", "import",
      "from", "export", "default", "this", "null", "undefined", "true", "false",
    ],
  },
  py: {
    comment: "#[^\\n]*",
    // **삼중 따옴표가 먼저다.** 홑따옴표 규칙이 먼저 물면 여는 자리에서 끊기고,
    // 그 뒤가 전부 문자열 색으로 흐른다 — docstring 하나에 화면이 다 물든다.
    string: '[frbu]{0,2}"""[\\s\\S]*?"""'
      + "|[frbu]{0,2}'''[\\s\\S]*?'''"
      + '|[frbu]{0,2}"(?:[^"\\\\\\n]|\\\\.)*"'
      + "|[frbu]{0,2}'(?:[^'\\\\\\n]|\\\\.)*'",
    keywords: [
      "import", "from", "as", "def", "class", "return", "if", "elif", "else",
      "for", "in", "while", "break", "continue", "with", "try", "except",
      "finally", "raise", "lambda", "yield", "pass", "global", "nonlocal",
      "assert", "del", "and", "or", "not", "is", "None", "True", "False",
      "self",
    ],
  },
};

/** 정규식은 언어마다 한 번만 만든다 — 글쇠 하나마다 다시 만들 이유가 없다. */
const PATTERNS = new Map();

function patternFor(lang) {
  const hit = PATTERNS.get(lang);
  if (hit) return hit;
  const spec = LANGS[lang] ?? LANGS.js;
  // 다섯 무리를 한 번에 훑는다. **주석·문자열이 앞에** 있어야 그 안의 예약어가
  // 따로 물리지 않는다 — 같은 자리에서 겨루면 앞에 적힌 쪽이 이긴다.
  const re = new RegExp(
    [
      "(" + spec.comment + ")",
      "(" + spec.string + ")",
      "\\b(" + spec.keywords.join("|") + ")\\b",
      "\\b(\\d[\\w.]*)\\b",
      "\\b([A-Za-z_$][\\w$]*)(?=\\s*\\()",
    ].join("|"),
    "g",
  );
  PATTERNS.set(lang, re);
  return re;
}

export function highlight(code, lang = "js") {
  const esc = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return esc.replace(patternFor(lang), (m, com, str, key, num, fn) => {
    if (com !== undefined) return '<span class="tok-com">' + com + "</span>";
    if (str !== undefined) return '<span class="tok-str">' + str + "</span>";
    if (key !== undefined) return '<span class="tok-key">' + key + "</span>";
    if (num !== undefined) return '<span class="tok-num">' + num + "</span>";
    if (fn !== undefined) return '<span class="tok-fn">' + fn + "</span>";
    return m;
  });
}

/* ── 파이썬 쪽 — Pyodide 위에서 borch.ts 를 부른다 ──────────────────────
 *
 * 같은 커널 위에 파이썬 표면을 얹는 것이 `borch_webgpu` 다. 여기서 하는 일은
 * `tests/browser/runner.html` 이 하던 것과 같다: borch.ts 를 먼저 올려 어댑터를
 * 잡고 전역에 두면, 결속이 `js.borch` 로 그것을 집는다.
 *
 * Pyodide 도 numpy 도 **저장소 안(`vendor/`)에서 온다.** CDN 은 그 시점에 살아
 * 있어야 하는 의존이고 이 저장소는 실제로 한 번 끊겨 봤다.
 */

const PYODIDE_DIR = new URL("../../vendor/pyodide/", import.meta.url).href;

/**
 * 가상 파일시스템에 얹을 파이썬 패키지들. 빠뜨린 모듈은 ImportError 로 시끄럽게
 * 터진다 — 조용히 반쪽만 실리는 것보다 낫다.
 *
 * **코어(`borch`)도 같이 얹는다.** 결속이 GPU 로 안 도는 자리(순수 파이썬 유틸)를
 * 코어에서 빌려 쓴다 — 없으면 `No module named 'borch'` 로 멈춘다(실측).
 */
const PACKAGES = {
  borch: ["__init__", "_base", "_tensor", "_ops", "_fft", "_nn", "_optim",
          "_data", "_rnn", "_serialize"],
  borch_webgpu: ["__init__", "_base", "_ops", "_nn", "_optim", "_data",
                 "_serialize"],
};

let pyodide = null;

/** Pyodide + numpy + borch_webgpu 를 한 번만 올린다. */
export async function loadPython(say = () => {}) {
  if (pyodide) return pyodide;

  const borch = await loadBorch();
  const probed = await borch.probe();
  if (!probed.ok) throw new Error(probed.message);

  say(t("load.borchTs"));
  await borch.init();
  // 결속이 여기를 본다. 없으면 그쪽이 조용히 다른 것으로 안 돌고 멈춘다.
  globalThis.borch = borch;

  say(t("load.pyodide"));
  await loadScript(`${PYODIDE_DIR}pyodide.js`);
  const py = await globalThis.loadPyodide({ indexURL: PYODIDE_DIR });

  say(t("load.numpy"));
  await py.loadPackage("numpy");

  say(t("load.binding"));
  const repo = new URL("../../", import.meta.url).href;
  const jobs = [];
  for (const [pkg, modules] of Object.entries(PACKAGES)) {
    py.FS.mkdirTree(`/work/${pkg}`);
    for (const name of modules) {
      jobs.push((async () => {
        const res = await fetch(`${repo}${pkg}/${name}.py`);
        // fetch 는 404 에 예외를 안 던진다. 확인 안 하면 오류 페이지의 HTML 이 그대로
        // 파이썬 파일로 써지고, 터지는 자리는 원인에서 아주 멀다.
        if (!res.ok) throw new Error(t("load.moduleFailed", `${pkg}/${name}.py`, res.status));
        py.FS.writeFile(`/work/${pkg}/${name}.py`, await res.text());
      })());
    }
  }
  // torchvision 자리의 transforms. 예시가 아직 안 쓰지만 결속이 임포트할 수 있다.
  jobs.push((async () => {
    const res = await fetch(`${repo}borch_vision.py`);
    if (res.ok) py.FS.writeFile("/work/borch_vision.py", await res.text());
  })());
  await Promise.all(jobs);

  await py.runPythonAsync(`
import sys
if "/work" not in sys.path:
    sys.path.insert(0, "/work")
`);

  pyodide = py;
  return py;
}

/** 파이썬 코드를 돌린다. `await` 도 `scope()` 도 안 나온다 — 결속이 감춘다. */
export async function runPython(code, hooks = {}) {
  const onLog = hooks.onLog ?? (() => {});
  const onPlot = hooks.onPlot ?? (() => {});
  bridge.stopped = false;
  bridge.log = onLog;
  bridge.plot = onPlot;

  const py = await loadPython((line) => onLog(line, "note"));
  const borch = await loadBorch();

  // print 가 개발자 도구로만 가면 이 페이지는 아무것도 안 보여준다.
  py.setStdout({ batched: (line) => onLog(line) });
  py.setStderr({ batched: (line) => onLog(line, "err") });

  const before = readStats(borch);
  const t0 = performance.now();
  try {
    await py.runPythonAsync([PY_PRELUDE, code].join("\n"));
  } finally {
    py.setStdout({});
    py.setStderr({});
  }
  const ms = performance.now() - t0;
  const after = readStats(borch);
  return {
    ms,
    stats: after,
    dispatchDelta: after && before ? after.dispatches - before.dispatches : after ? after.dispatches : 0,
  };
}

/** 자바스크립트 쪽의 `log`·`plot`·`stopped` 를 파이썬에서도 같은 이름으로. */
const PY_PRELUDE = `
import js as _js

def log(*args):
    print(*args)

def plot(name, value):
    _js.borchPG.plot(name, float(value))

def stopped():
    return bool(_js.borchPG.stopped)
`;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = src;
    el.onload = () => resolve();
    el.onerror = () => reject(new Error(t("load.scriptFailed", src)));
    document.head.append(el);
  });
}
