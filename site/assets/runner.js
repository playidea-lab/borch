/**
 * Where this page loads borch and runs the user's code.
 *
 * The hero's small demo and the playground **use the same thing** — kept as two copies,
 * one of them gets fixed and a state quietly appears where code that runs on the
 * landing does not run in the playground.
 *
 * What it loads is `borch-ts/dist`, which is gitignored and therefore in no commit, so
 * saying "run npm run build:ts first" when it is absent is one of this file's jobs.
 * (A blank screen and an unbuilt bundle are indistinguishable.)
 */

import { cifar10 } from "./datasets.js";
import { t } from "./i18n.js";

/** The bundle's absolute URL. Relative paths only — a pinned host is right in one place. */
export const BORCH_URL = new URL("../../borch-ts/dist/src/index.js", import.meta.url).href;

let cached = null;

/** Loads the borch module once. */
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
 * Whether WebGPU is there — **and it does not fall back.**
 *
 * That is why this repository removed the TF.js version. Sliding quietly down to WebGL
 * or software means numbers measured there get read as the GPU's. The same rule holds
 * here: when it cannot, it says why and stops.
 */
export async function probeDevice() {
  const borch = await loadBorch();
  const result = await borch.probe();
  // `software` travels with the name. Callers used to test the name themselves against
  // their own copy of the list; the library judges it now, in one place.
  if (result.ok) return { ok: true, adapter: result.adapter, software: result.software };
  return { ok: false, why: result.why, message: result.message };
}

/** Reads the counters of the device currently held. Null before initialisation. */
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

/** What running code looks into. The stop button leaves its mark here. */
const bridge = {
  stopped: false,
  log: () => {},
  plot: () => {},
  // **Showing is sometimes shorter than explaining.** Drawing the image beats writing
  // about an image classification. With no receiver it quietly does nothing.
  show: async () => {},
  cifar10,
};
globalThis.__borch_playground__ = bridge;
// What Python reaches through `js.borchPG`. A dunder name is awkward to pick up from the `js` module.
globalThis.borchPG = bridge;

/**
 * Runs the user's code.
 *
 * It is built as an ES module and hung on `import()` — `new Function` takes neither
 * top-level `await` nor `import`, and borch's first line is `await init()`, which needs
 * both.
 *
 * @param code   the JavaScript the user typed
 * @param hooks  {onLog, onPlot} — where output and plots are received
 */
export async function runCode(code, hooks = {}) {
  const borch = await loadBorch();
  const onLog = hooks.onLog ?? (() => {});
  const onPlot = hooks.onPlot ?? (() => {});

  bridge.stopped = false;
  bridge.log = onLog;
  bridge.plot = onPlot;
  bridge.show = hooks.onShow ?? (async () => {});

  // The console is intercepted. The user will type `console.log` (the documentation's
  // examples do), and if that only reaches devtools this page shows nothing at all.
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
      // The names are spread out so the documentation's examples paste in as they are.
      "const { init, Tensor, nn, optim, data, vision, ops, fft, linalg, scope, keepAlive,",
      "        noGrad, manualSeed, einsum, slice, save, load, isAvailable, probe,",
      "        currentDevice, device, Device } = borch;",
      "const __pg = globalThis.__borch_playground__;",
      "const log = (...a) => __pg.log(a.map(v => typeof v === 'string' ? v : JSON.stringify(v)).join(' '));",
      "const plot = (name, value) => __pg.plot(name, value);",
      "const show = (t, opts) => __pg.show(t, opts);",
      "const stopped = () => __pg.stopped;",
      // The data the tutorials use. Tensor is handed over, so the caller only thinks about shapes.
      "const datasets = { cifar10: (split, opts) => __pg.cifar10(Tensor, split, opts) };",
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

/** The stop flag. A running loop watches it through `stopped()`. */
export function requestStop() {
  bridge.stopped = true;
}

function render(v) {
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean" || v == null) return String(v);
  try { return JSON.stringify(v); } catch { return String(v); }
}

/**
 * A very thin syntax highlight. No library is fetched — the same reason this repository
 * keeps its runtime dependencies at zero, and what is needed here is five colours.
 *
 * **It is not a parser.** One regular expression sweeps it, so nesting is not seen.
 * Openers are caught from the left, which is right in the common case, and when it is
 * wrong one colour is off and the code is unchanged — this is **display**, not an
 * editor. The day a real parser is needed, one gets fetched.
 */

/** Only three things differ per language: the comment marker, the shape of strings, and the keywords. */
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
    // **Triple quotes come first.** Let the single-quote rule bite first and it stops at
    // the opener, and everything after it runs in the string colour — one docstring
    // stains the whole screen.
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

/** The regular expression is built once per language — no reason to rebuild it on every keystroke. */
const PATTERNS = new Map();

function patternFor(lang) {
  const hit = PATTERNS.get(lang);
  if (hit) return hit;
  const spec = LANGS[lang] ?? LANGS.js;
  // All five groups are swept at once. **Comments and strings have to come first** so
  // that keywords inside them are not caught separately — competing at the same position,
  // whichever is written first wins.
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

/* ── the Python side — calling borch.ts on Pyodide ──────────────────────
 *
 * `borch_webgpu` is the Python surface laid over the same kernels. What happens here is
 * what `tests/browser/runner.html` does: borch.ts goes up first, acquires the adapter and
 * sits on the global, and the binding picks it up through `js.borch`.
 *
 * Pyodide and numpy both **come from inside the repository (`vendor/`).** A CDN is a
 * dependency that has to be alive at that moment, and this repository has had one go
 * down.
 */

const PYODIDE_DIR = new URL("../../vendor/pyodide/", import.meta.url).href;

/**
 * The Python packages laid onto the virtual filesystem. A module left out blows up
 * loudly as an ImportError — better than half of it loading quietly.
 *
 * **The core (`borch`) goes on too.** The binding borrows from the core wherever it does
 * not run on the GPU (pure-Python utilities) — without it, it stops with
 * `No module named 'borch'` (measured).
 */
const PACKAGES = {
  borch: ["__init__", "_base", "_tensor", "_ops", "_fft", "_nn", "_optim",
          "_data", "_rnn", "_serialize", "autograd"],
  borch_webgpu: ["__init__", "_base", "_ops", "_nn", "_optim", "_data",
                 "_serialize", "_onnx", "autograd"],
};

let pyodide = null;
/** The load in flight, so a click during the warm-up joins it rather than starting a
 *  second Pyodide (two loads would lay two `/work` trees and the later one would win). */
let loadingPython = null;

/** Brings up Pyodide, numpy and borch_webgpu once — and only once at a time. */
export function loadPython(say = () => {}) {
  if (pyodide) return Promise.resolve(pyodide);
  if (!loadingPython) {
    loadingPython = loadPythonFresh(say).catch((err) => {
      loadingPython = null;
      throw err;
    });
  }
  return loadingPython;
}

async function loadPythonFresh(say) {

  // **The GPU is required by `borch_webgpu` and not by Python mode**, and this gate
  // used to be in front of both. Without WebGPU the whole thing stopped — including
  // the core, which is numpy and has never touched a GPU.
  //
  // Measured before changing it: Pyodide, numpy and the core's ten modules, no
  // adapter anywhere, `nn.Linear` trained 200 steps and the loss went 17.2945 →
  // 0.000001 with the weight landing on 1.9992 against a true 2.0. Pyodide **is**
  // wasm, so the fallback the site's table called "planned" was already loaded and
  // held behind a check about a different thing.
  //
  // So the adapter is asked for and its absence is carried rather than thrown. What
  // is lost is `borch_webgpu`; what remains is `import borch as torch`.
  const borch = await loadBorch();
  const probed = await borch.probe();
  const onGpu = probed.ok;

  if (onGpu) {
    say(t("load.borchTs"));
    await borch.init();
    // The binding looks here. Without it, that side stops rather than quietly running on something else.
    globalThis.borch = borch;
  }

  say(t("load.pyodide"));
  await loadScript(`${PYODIDE_DIR}pyodide.js`);
  const py = await globalThis.loadPyodide({ indexURL: PYODIDE_DIR });

  say(t("load.numpy"));
  await py.loadPackage("numpy");

  say(onGpu ? t("load.binding") : t("load.core"));
  const repo = new URL("../../", import.meta.url).href;
  const jobs = [];
  // **Without an adapter the binding's modules are left off.** Written, they import
  // fine and stop at the first call with a message about a device, which reads as a
  // defect in whatever line the reader was running. Absent, `import borch_webgpu`
  // says there is no such module, which is the true sentence.
  const wanted = onGpu ? Object.entries(PACKAGES)
                       : Object.entries(PACKAGES).filter(([pkg]) => pkg === "borch");
  for (const [pkg, modules] of wanted) {
    py.FS.mkdirTree(`/work/${pkg}`);
    for (const name of modules) {
      jobs.push((async () => {
        const res = await fetch(`${repo}${pkg}/${name}.py`);
        // fetch does not throw on a 404. Unchecked, an error page's HTML gets written as
        // a Python file, and where it blows up is a long way from the cause.
        if (!res.ok) throw new Error(t("load.moduleFailed", `${pkg}/${name}.py`, res.status));
        py.FS.writeFile(`/work/${pkg}/${name}.py`, await res.text());
      })());
    }
  }
  // The transforms that stand where torchvision would.
  //
  // **This swallowed its own failure while the loop three lines up refused to.** That
  // loop carries the reason — `fetch` does not throw on a 404, so an unchecked response
  // writes an error page's HTML into a `.py` — and this call, in the same function,
  // wrote nothing at all and carried on. The visible result was a playground that came
  // up fine and answered `ModuleNotFoundError: borchvision` on the reader's own import,
  // a long way from the missing file.
  //
  // Same check as the modules, and the same message.
  jobs.push((async () => {
    const res = await fetch(`${repo}borchvision.py`);
    if (!res.ok) throw new Error(t("load.moduleFailed", "borchvision.py", res.status));
    py.FS.writeFile("/work/borchvision.py", await res.text());
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

/** Runs Python code. Neither `await` nor `scope()` appears — the binding hides them. */
export async function runPython(code, hooks = {}) {
  const onLog = hooks.onLog ?? (() => {});
  const onPlot = hooks.onPlot ?? (() => {});
  bridge.stopped = false;
  bridge.log = onLog;
  bridge.plot = onPlot;
  bridge.show = hooks.onShow ?? (async () => {});

  const py = await loadPython((line) => onLog(line, "note"));
  const borch = await loadBorch();

  // With print reaching devtools only, this page shows nothing at all.
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

/** The JavaScript side's `log`, `plot` and `stopped`, under the same names in Python. */
const PY_PRELUDE = `
import js as _js

def log(*args):
    print(*args)

def plot(name, value):
    _js.borchPG.plot(name, float(value))

def show(tensor, **options):
    # A tensor as a picture. It calls the same thing the JavaScript side's show does.
    #
    # This Python sits inside a JS template literal — **no backticks.** One of them closes
    # the string right there and the whole file dies in parsing. The symptom is an
    # "Unexpected identifier" with nothing to do with this line, and not a single runnable
    # block appears. tests/browser/runner.html had written the same warning in a comment
    # and it was stepped on anyway.
    return _js.borchPG.show(tensor, _js.Object.fromEntries(list(options.items())))

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

/* ── carrying code in the address ───────────────────────────────────────
 *
 * The playground's share link and the lessons' "open in playground" use the same code.
 * Kept as two copies, one gets fixed, and what breaks then is **a link someone else
 * received.**
 */

/** UTF-8 into a URL-safe base64. Korean comments go through it, so plain btoa will not do. */
export function encodeCode(text) {
  const bytes = new TextEncoder().encode(text);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function decodeCode(encoded) {
  const b64 = encoded.replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64);
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

/**
 * An error as something a person can read.
 *
 * **Showing `err.stack` alone loses the explanation in Safari.** V8's stack begins with
 * `Error: what went wrong`, so printing that alone still read; WebKit's stack has **the
 * call sites only** and no message (measured: `module code@…:12:22`). So a Safari visitor
 * saw a filename and a line number instead of "there is no WebGPU" — and at the first
 * error Safari meets on this site, at that.
 *
 * The message is written first and the stack appended. Both are needed: one belongs to
 * the person reading and one to the person fixing.
 */
export function describeError(err) {
  if (!err) return t("run.unknownError");
  const head = err.name && err.message ? `${err.name}: ${err.message}`
             : String(err.message ?? err);
  const stack = typeof err.stack === "string" ? err.stack : "";
  // In V8 the stack's first line is already the message — it is not written twice.
  if (stack && stack.startsWith(head)) return stack;
  return stack ? `${head}\n${stack}` : head;
}
