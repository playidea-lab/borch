/**
 * Code blocks that run where they sit, inside the prose.
 *
 * **Code beside an explanation that does not run is a picture.** This repository has
 * twice caught documentation code going stale, and on a site there is only one way to
 * close that gap — the reader presses it and it really runs. So every example under
 * Learn **can be edited and can be run.**
 *
 * It is written like this:
 *
 *     <div class="runnable" data-lang="js">
 *       <script type="text/plain">await init(); …</script>
 *     </div>
 *
 * One block can hold **two languages**. Writing the same work on both surfaces lets a
 * reader press and compare the numbers — better than the sentence "they agree to the
 * last digit":
 *
 *     <div class="runnable" data-lang="js">
 *       <script type="text/plain" data-lang="js">await init(); …</script>
 *       <script type="text/plain" data-lang="py">import borch_webgpu as torch …</script>
 *     </div>
 *
 * A source without `data-lang` counts as the outer `div`'s language, and the outer
 * `div`'s language is the one shown first.
 *
 * It sits in `<script type="text/plain">` because the browser **leaves the inside
 * alone**. Put it in a `<pre>` and every `<` and `&` has to be escaped by hand, and one
 * missed escape changes the code quietly.
 */

import { drawSeries, drawTensor } from "./render.js";
import { describeError, encodeCode, highlight, requestStop, runCode, runPython } from "./runner.js";
import { t } from "./i18n.js";

const LABEL = {
  run: { en: "▶ Run", ko: "▶ 실행" },
  running: { en: "running…", ko: "도는 중…" },
  reset: { en: "reset", ko: "되돌리기" },
  open: { en: "open in playground", ko: "플레이그라운드에서" },
  stop: { en: "■ Stop", ko: "■ 중지" },
  editorHint: {
    en: "Code editor. Tab indents; press Escape then Tab to leave.",
    ko: "코드 편집기. Tab 은 들여쓰기이고, 나가려면 Escape 를 누른 뒤 Tab 을 누른다." },
  // A lesson block has no separate name for what it is, so the sentence above serves as
  // the name too. The playground already has one ("JavaScript code"), so it appends only
  // the instruction.
};

const NAME = { js: "javascript", py: "python" };

/** Where the big editor lives. **It must not be resolved from how deep the page is** —
 *  `../playground.html` is right only from `learn/` and points outside the site from a
 *  top-level page. This file is always in `site/assets/`, so measuring from here gives
 *  the same place whichever page loaded it. */
const PLAYGROUND = new URL("../playground.html", import.meta.url).href;

const LANG = document.documentElement.lang === "ko" ? "ko" : "en";
const say = (key) => LABEL[key][LANG];

/** One at a time — two blocks stepping on the same device at once mix the numbers. */
let busy = false;

export function mountRunnables(root = document) {
  for (const box of root.querySelectorAll(".runnable")) mount(box);
}

function mount(box) {
  const sources = [...box.querySelectorAll('script[type="text/plain"]')];
  if (!sources.length) return;
  const first = box.dataset.lang === "py" ? "py" : "js";

  // language → the code as first written. A second source without `data-lang` overwrites
  // the first one's slot and a copy disappears quietly — `test_site.py` blocks that.
  const original = new Map();
  for (const el of sources) {
    original.set(el.dataset.lang === "py" ? "py" : el.dataset.lang === "js" ? "js" : first,
      dedent(el.textContent));
    el.remove();
  }
  // What is in the editor right now. Edits have to survive switching languages, or there
  // is nothing to compare.
  const draft = new Map(original);
  let lang = original.has(first) ? first : [...original.keys()][0];

  box.innerHTML = `
    <div class="runnable-head">
      ${original.size > 1
        ? [...original.keys()].map((k) => `<button class="tab" type="button" data-lang="${k}"
            aria-pressed="${k === lang}">${NAME[k]}</button>`).join("")
        : `<span>${NAME[lang]}</span>`}
      <span class="grow"></span>
      <a class="open" href="#" title="${say("open")}">${say("open")}</a>
      <button class="reset" type="button">${say("reset")}</button>
      <button class="go" type="button">${say("run")}</button>
    </div>
    <div class="edit"><pre tabindex="-1"><code></code></pre><textarea spellcheck="false"
      autocomplete="off" autocapitalize="off" autocorrect="off" wrap="off"
      aria-label="${say("editorHint")}" title="${say("editorHint")}"></textarea></div>
    <pre class="out"></pre>
    <div class="canvas-out"></div>`;

  const painted = box.querySelector(".edit code");
  const editor = box.querySelector("textarea");
  const out = box.querySelector(".out");
  const stage = box.querySelector(".canvas-out");
  const runBtn = box.querySelector("button.go");
  const resetBtn = box.querySelector("button.reset");
  const openLink = box.querySelector("a.open");

  const paint = () => {
    // The trailing `\n ` is a line the painted layer carries and the value does not.
    painted.innerHTML = highlight(`${editor.value}\n `, lang);
    // **The box is no longer measured in arithmetic.** It used to be
    // `rows * 1.6 * 12.5 + 28`, which is the line height, the font size and the padding
    // copied out of `style.css` — three numbers in two places, and it was short by a
    // line on every block: the count missed the line painted just above, and a block
    // whose code is wider than the box loses another ten pixels to the horizontal
    // scrollbar. Measured on `learn/10-vit.html`: 288px of box against 307px of text on
    // the first block, 1648 against 1657 on the third, so the last line was cut on all
    // four. The `<pre>` is in flow and already knows all of it — the textarea over it is
    // `inset: 0`, so it follows whatever the `<pre>` comes out as.
  };
  const setCode = (text) => {
    editor.value = text;
    paint();
    // It hands over **the code currently on screen**. Carrying your edit into the big
    // editor as it stands is the point — handing over the original loses what you just
    // did.
    openLink.href = `${PLAYGROUND}#lang=${lang}&code=${encodeCode(editor.value)}`;
  };

  editor.addEventListener("input", () => {
    paint();
    openLink.href = `${PLAYGROUND}#lang=${lang}&code=${encodeCode(editor.value)}`;
  });
  editor.addEventListener("scroll", () => {
    const layer = painted.parentElement;
    layer.scrollTop = editor.scrollTop;
    layer.scrollLeft = editor.scrollLeft;
  });
  // **Tab indents here, so there has to be a separate way out.**
  //
  // Without one, someone who arrives by keyboard cannot leave — however many times Tab
  // is pressed, only the spaces grow (measured: two presses took 822 characters to 826
  // with focus unmoved). Without a mouse there is nothing to do but close the tab, and
  // this failure has a name — WCAG 2.1.2, keyboard trap.
  //
  // Escape arms it once and the next Tab leaves — the door CodeMirror and Monaco use,
  // so it is already a known gesture to anyone who has touched a code editor. Any other
  // key locks it again.
  let leaving = false;
  editor.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { leaving = true; return; }
    if (e.key === "Tab" && leaving) { leaving = false; return; }   // default action = move focus
    if (e.key !== "Tab") leaving = false;
    if (e.key === "Tab") {
      e.preventDefault();
      const { selectionStart: s, selectionEnd: t2, value } = editor;
      const pad = lang === "py" ? "    " : "  ";
      editor.value = `${value.slice(0, s)}${pad}${value.slice(t2)}`;
      editor.selectionStart = editor.selectionEnd = s + pad.length;
      paint();
    }
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); go(); }
  });

  resetBtn.addEventListener("click", () => { setCode(original.get(lang)); out.textContent = ""; });

  // The language tabs. Edits are kept per language and **the output is cleared** — a
  // header reading python above numbers JavaScript printed is a wrong label.
  for (const tab of box.querySelectorAll("button.tab")) {
    tab.addEventListener("click", () => {
      if (tab.dataset.lang === lang) return;
      draft.set(lang, editor.value);
      lang = tab.dataset.lang;
      for (const other of box.querySelectorAll("button.tab")) {
        other.setAttribute("aria-pressed", String(other.dataset.lang === lang));
      }
      setCode(draft.get(lang));
      out.textContent = "";
      stage.textContent = "";
    });
  }

  const write = (text, kind = "") => {
    // **It does not take the reader's place away.** Always dragging to the bottom sends
    // someone who scrolled up back down the instant the next line prints — and with code
    // that keeps printing, such as a training loop, scrolling up becomes impossible.
    const following = out.scrollHeight - out.scrollTop - out.clientHeight < 24;
    const line = document.createElement("div");
    if (kind) line.className = kind;
    line.textContent = text;
    out.append(line);
    if (following) out.scrollTop = out.scrollHeight;
  };

  // The numbers one run printed. `plot()` accumulates and it is drawn once at the end —
  // redrawing per step costs more time than the training does.
  let series = [];
  let seriesName = "";

  async function go() {
    if (busy) return;
    busy = true;
    runBtn.disabled = true;
    runBtn.textContent = say("running");
    out.textContent = "";
    out.scrollTop = 0;
    stage.textContent = "";
    series = [];
    seriesName = "";
    try {
      const hooks = {
        onLog: write,
        onPlot: (name, value) => {
          if (!Number.isFinite(value)) return;
          seriesName = name;
          series.push(value);
        },
        onShow: async (tensor, options) => {
          // Arriving from Python, the options may not be a plain object. Take the values only.
          const opts = options ? JSON.parse(JSON.stringify(options)) : {};
          // How many fit per row is decided by **this box's real width**. Pinned at 720,
          // the right edge was cut off on a narrow screen.
          //
          // **The empty box's width must not be measured** — `.canvas-out:empty` is
          // `display: none` and gives 0. Before the first image that is always the state,
          // so everything came out one per row. Measuring the whole block avoids it.
          if (!opts.width) opts.width = box.clientWidth - 32;
          stage.append(await drawTensor(tensor, opts));
        },
      };
      const result = lang === "js"
        ? await runCode(editor.value, hooks)
        : await runPython(editor.value, hooks);
      if (series.length > 1) stage.prepend(drawSeries(series, { name: seriesName }));
      write("");
      write(t("run.done", result.ms.toFixed(0)), "ok");
      verdict();
    } catch (err) {
      write(describeError(err), "err");
    } finally {
      busy = false;
      runBtn.disabled = false;
      runBtn.textContent = say("run");
    }
  }

  /**
   * **A lesson that asks the reader to fix something has to say whether it is fixed.**
   * `data-verdict="loss<0.05"` on the box: the last `loss N` the run printed is read
   * off the output and judged. Not `err` — the runner (`lessons.py`) reads `err` as a
   * broken page, and a wrong answer the reader is meant to reach is not that.
   */
  function verdict() {
    const rule = /^loss<([0-9.]+)$/.exec(box.dataset.verdict ?? "");
    if (!rule) return;
    const limit = Number(rule[1]);
    let last = null;
    for (const line of out.querySelectorAll("div")) {
      const m = /loss\s+(-?[0-9.]+(?:e[-+]?\d+)?|nan|inf)/i.exec(line.textContent);
      if (m) last = m[1];
    }
    if (last === null) { write(t("verdict.noLoss"), "verdict bad"); return; }
    const value = Number(last);
    if (Number.isFinite(value) && value < limit) write(t("verdict.learned", last, String(limit)), "verdict good");
    else write(t("verdict.notYet", last, String(limit)), "verdict bad");
  }
  runBtn.addEventListener("click", go);
  setCode(draft.get(lang));
}

/** Strips the indentation inside `<script>`. HTML indentation mixed into the code kills Python. */
function dedent(text) {
  const lines = text.replace(/^\n/, "").replace(/\s+$/, "").split("\n");
  const pad = Math.min(...lines.filter((l) => l.trim())
    .map((l) => l.match(/^ */)[0].length));
  return lines.map((l) => l.slice(pad)).join("\n");
}

// No stop button yet — every lesson example finishes within a few hundred milliseconds.
// It goes here the day something long-running arrives; added now it is a handle nobody uses.
void requestStop;

mountRunnables();
