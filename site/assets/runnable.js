/**
 * 문서 안에서 그 자리에 도는 코드 블록.
 *
 * **설명 옆의 코드가 안 돌면 그건 그림이다.** 이 저장소는 문서의 코드가 썩는 것을
 * 두 번 잡았고, 사이트에서 그 자리를 막는 방법은 하나뿐이다 — 읽는 사람이 누르면
 * 진짜로 도는 것. 그래서 Learn 의 모든 예제는 **고칠 수 있고 돌릴 수 있다.**
 *
 * 쓰는 쪽은 이렇게 적는다:
 *
 *     <div class="runnable" data-lang="js">
 *       <script type="text/plain">await init(); …</script>
 *     </div>
 *
 * 한 블록이 **두 언어**를 담을 수 있다. 같은 일을 두 표면으로 적어 두면 읽는
 * 사람이 눌러서 수를 맞춰 볼 수 있다 — "자릿수까지 같다" 는 말보다 그게 낫다:
 *
 *     <div class="runnable" data-lang="js">
 *       <script type="text/plain" data-lang="js">await init(); …</script>
 *       <script type="text/plain" data-lang="py">import borch_webgpu as torch …</script>
 *     </div>
 *
 * `data-lang` 이 없는 원본은 바깥 `div` 의 언어로 친다. 바깥 `div` 의 언어가
 * 처음 보이는 쪽이다.
 *
 * `<script type="text/plain">` 에 담는 이유는 브라우저가 그 안을 **건드리지 않기**
 * 때문이다. `<pre>` 에 넣으면 `<`·`&` 를 일일이 escape 해야 하고, 한 번 빠뜨리면
 * 코드가 조용히 달라진다.
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
};

const NAME = { js: "javascript", py: "python" };

/** 큰 편집기의 자리. **페이지가 몇 겹 아래인지로 풀면 안 된다** — `../playground.html`
 *  은 `learn/` 에서만 맞고 최상위 페이지에서는 사이트 밖을 가리킨다. 이 파일은 언제나
 *  `site/assets/` 에 있으므로 여기서 재면 어느 페이지에서 불러도 같은 곳이 나온다. */
const PLAYGROUND = new URL("../playground.html", import.meta.url).href;

const LANG = document.documentElement.lang === "ko" ? "ko" : "en";
const say = (key) => LABEL[key][LANG];

/** 한 번에 하나만 돈다 — 두 블록이 같은 장치를 동시에 밟으면 수가 섞인다. */
let busy = false;

export function mountRunnables(root = document) {
  for (const box of root.querySelectorAll(".runnable")) mount(box);
}

function mount(box) {
  const sources = [...box.querySelectorAll('script[type="text/plain"]')];
  if (!sources.length) return;
  const first = box.dataset.lang === "py" ? "py" : "js";

  // 언어 → 처음 적힌 코드. 두 번째 원본에 `data-lang` 이 없으면 첫 번째와 같은
  // 칸에 덮어써서 한 벌이 조용히 사라진다 — `test_site.py` 가 그 꼴을 막는다.
  const original = new Map();
  for (const el of sources) {
    original.set(el.dataset.lang === "py" ? "py" : el.dataset.lang === "js" ? "js" : first,
      dedent(el.textContent));
    el.remove();
  }
  // 지금 편집기에 있는 글. 언어를 오갈 때 고친 것이 남아 있어야 비교가 된다.
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
    <div class="edit"><pre><code></code></pre><textarea spellcheck="false"
      autocomplete="off" autocapitalize="off" autocorrect="off" wrap="off"></textarea></div>
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
    painted.innerHTML = highlight(`${editor.value}\n `, lang);
    // 글 높이에 맞춘다 — 블록마다 스크롤 막대가 생기면 읽는 흐름이 끊긴다.
    const rows = editor.value.split("\n").length;
    box.querySelector(".edit").style.height = `${rows * 1.6 * 12.5 + 28}px`;
  };
  const setCode = (text) => {
    editor.value = text;
    paint();
    // **지금 화면에 있는 코드**를 넘긴다. 고친 것을 그대로 큰 편집기로 들고 갈 수
    // 있어야 의미가 있다 — 원본을 넘기면 방금 한 일이 사라진다.
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
  editor.addEventListener("keydown", (e) => {
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

  // 언어 탭. 고친 글은 언어별로 남기고, **출력은 지운다** — 자바스크립트가 찍은
  // 수 위에 python 이라고 적힌 머리띠가 걸리면 그건 틀린 이름표다.
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
    // **읽던 자리를 뺏지 않는다.** 무조건 맨 아래로 끌면, 위를 보려고 올린 사람이
    // 다음 줄이 찍히는 순간 도로 내려간다 — 학습 루프처럼 계속 찍는 코드에서는
    // 올라가는 것이 아예 불가능해진다.
    const following = out.scrollHeight - out.scrollTop - out.clientHeight < 24;
    const line = document.createElement("div");
    if (kind) line.className = kind;
    line.textContent = text;
    out.append(line);
    if (following) out.scrollTop = out.scrollHeight;
  };

  // 한 실행이 찍은 수. `plot()` 이 쌓고, 끝나면 한 번 그린다 — 스텝마다 다시
  // 그리면 학습보다 그리는 데 시간이 더 든다.
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
          // 파이썬에서 오면 옵션이 평범한 객체가 아닐 수 있다. 값만 꺼낸다.
          const opts = options ? JSON.parse(JSON.stringify(options)) : {};
          // 몇 장을 한 줄에 놓을지는 **이 칸의 실제 너비**가 정한다. 720 을 박아
          // 두었더니 좁은 화면에서 오른쪽이 잘렸다.
          //
          // **비어 있는 칸의 너비를 재면 안 된다** — `.canvas-out:empty` 는
          // `display: none` 이라 0 을 준다. 첫 장을 그리기 전이 언제나 그 상태라
          // 한 줄에 한 장씩 나왔다. 블록 전체의 너비를 재면 그 함정이 없다.
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
    } catch (err) {
      write(describeError(err), "err");
    } finally {
      busy = false;
      runBtn.disabled = false;
      runBtn.textContent = say("run");
    }
  }

  runBtn.addEventListener("click", go);
  setCode(draft.get(lang));
}

/** `<script>` 안의 들여쓰기를 걷어낸다. HTML 들여쓰기가 코드에 섞이면 파이썬이 죽는다. */
function dedent(text) {
  const lines = text.replace(/^\n/, "").replace(/\s+$/, "").split("\n");
  const pad = Math.min(...lines.filter((l) => l.trim())
    .map((l) => l.match(/^ */)[0].length));
  return lines.map((l) => l.slice(pad)).join("\n");
}

// 중지 버튼은 아직 안 단다 — 강의의 예제는 전부 몇백 밀리초 안에 끝난다.
// 오래 도는 것을 넣게 되면 그때 여기에 붙인다. 지금 달면 안 쓰이는 손잡이가 된다.
void requestStop;

mountRunnables();
