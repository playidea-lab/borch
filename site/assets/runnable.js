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
 * `<script type="text/plain">` 에 담는 이유는 브라우저가 그 안을 **건드리지 않기**
 * 때문이다. `<pre>` 에 넣으면 `<`·`&` 를 일일이 escape 해야 하고, 한 번 빠뜨리면
 * 코드가 조용히 달라진다.
 */

import { encodeCode, highlight, requestStop, runCode, runPython } from "./runner.js";
import { t } from "./i18n.js";

const LABEL = {
  run: { en: "▶ Run", ko: "▶ 실행" },
  running: { en: "running…", ko: "도는 중…" },
  reset: { en: "reset", ko: "되돌리기" },
  open: { en: "open in playground", ko: "플레이그라운드에서" },
  stop: { en: "■ Stop", ko: "■ 중지" },
};

const LANG = document.documentElement.lang === "ko" ? "ko" : "en";
const say = (key) => LABEL[key][LANG];

/** 한 번에 하나만 돈다 — 두 블록이 같은 장치를 동시에 밟으면 수가 섞인다. */
let busy = false;

export function mountRunnables(root = document) {
  for (const box of root.querySelectorAll(".runnable")) mount(box);
}

function mount(box) {
  const source = box.querySelector('script[type="text/plain"]');
  if (!source) return;
  const lang = box.dataset.lang === "py" ? "py" : "js";
  const original = dedent(source.textContent);
  source.remove();

  box.innerHTML = `
    <div class="runnable-head">
      <span>${lang === "js" ? "javascript" : "python"}</span>
      <span class="grow"></span>
      <a class="open" href="#" title="${say("open")}">${say("open")}</a>
      <button class="reset" type="button">${say("reset")}</button>
      <button class="go" type="button">${say("run")}</button>
    </div>
    <div class="edit"><pre><code></code></pre><textarea spellcheck="false"
      autocomplete="off" autocapitalize="off" autocorrect="off" wrap="off"></textarea></div>
    <pre class="out"></pre>`;

  const painted = box.querySelector(".edit code");
  const editor = box.querySelector("textarea");
  const out = box.querySelector(".out");
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
    openLink.href = `../playground.html#lang=${lang}&code=${encodeCode(editor.value)}`;
  };

  editor.addEventListener("input", () => {
    paint();
    openLink.href = `../playground.html#lang=${lang}&code=${encodeCode(editor.value)}`;
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

  resetBtn.addEventListener("click", () => { setCode(original); out.textContent = ""; });

  const write = (text, kind = "") => {
    const line = document.createElement("div");
    if (kind) line.className = kind;
    line.textContent = text;
    out.append(line);
    out.scrollTop = out.scrollHeight;
  };

  async function go() {
    if (busy) return;
    busy = true;
    runBtn.disabled = true;
    runBtn.textContent = say("running");
    out.textContent = "";
    try {
      const result = lang === "js"
        ? await runCode(editor.value, { onLog: write })
        : await runPython(editor.value, { onLog: write });
      write("");
      write(t("run.done", result.ms.toFixed(0)), "ok");
    } catch (err) {
      write(String(err && err.stack ? err.stack : err), "err");
    } finally {
      busy = false;
      runBtn.disabled = false;
      runBtn.textContent = say("run");
    }
  }

  runBtn.addEventListener("click", go);
  setCode(original);
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
