/**
 * The course home, rendered from `curriculum.json` — the learning tree as a table of
 * contents with the reader's own progress on it. No backend, no login: completion lives
 * in `progress.js` (localStorage).
 *
 * **Progress is the reader's to set, not the GPU's to grant.** A lesson's exercise, when
 * it passes, marks that lesson done (runnable.js) — but a visitor with no WebGPU adapter,
 * or on Safari without JSPI for the Python twin, can never make a verdict pass. If that
 * were the only path their tree would sit at 0% forever. So the check in front of every
 * lesson is a real toggle: press it to mark a lesson read, press again to clear it. The
 * verdict is a bonus on top, not the gate.
 */
import * as progress from "./progress.js";

const LANG = document.documentElement.lang === "ko" ? "ko" : "en";
// learn/index.html sits one level below site/ (en) and two below (ko/learn/index.html).
const BASE = LANG === "ko" ? "../../" : "../";
const T = (o) => (o && (o[LANG] ?? o.en)) ?? "";

const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};

function bar(frac) {
  const wrap = el("div", "c-bar");
  const fill = el("i");
  fill.style.width = `${Math.round(frac * 100)}%`;
  wrap.appendChild(fill);
  return wrap;
}

(async () => {
  const root = document.getElementById("course-root");
  const head = document.getElementById("course-head");
  if (!root) return;

  let data;
  try { data = await (await fetch(`${BASE}assets/curriculum.json`)).json(); }
  catch { return; }                       // no manifest → leave the static fallback in place

  const lessons = data.lessons;
  const linkOf = (l) => (LANG === "ko" ? l.ko_url : l.url);
  // "New since your last visit" is judged against the moment the page was opened, so it is
  // captured once here — before render() stamps a fresh visit — and reused across re-renders.
  const seen = progress.lastSeen();
  const isNew = (l) => l.status === "exists" && l.added && seen > 0
    && new Date(l.added).getTime() > seen;

  function render() {
    const done = progress.completed();
    const isDone = (l) => done.has(l.url) || (l.ko_url && done.has(l.ko_url));
    const live = lessons.filter((l) => l.status === "exists");
    const doneCount = live.filter(isDone).length;
    const pct = Math.round(100 * doneCount / (live.length || 1));

    // ── header: overall progress + resume ──
    if (head) {
      head.textContent = "";
      head.appendChild(bar(live.length ? doneCount / live.length : 0));
      const line = el("p", "c-stat",
        LANG === "ko" ? `${doneCount} / ${live.length} 레슨 완료 · ${pct}%`
                      : `${doneCount} / ${live.length} lessons done · ${pct}%`);
      const next = live.find((l) => !isDone(l));
      if (next) {
        line.appendChild(document.createTextNode("  "));
        const a = el("a", "c-resume", (LANG === "ko" ? "이어보기: " : "Resume: ") + T(next.title) + " →");
        a.href = BASE + linkOf(next);
        line.appendChild(a);
      }
      head.appendChild(line);
    }

    // ── parts → lessons ──
    root.textContent = "";
    for (const part of data.parts) {
      const inPart = lessons.filter((l) => l.part === part.id).sort((a, b) => a.order - b.order);
      if (!inPart.length) continue;
      const livePart = inPart.filter((l) => l.status === "exists");
      const donePart = livePart.filter(isDone).length;

      const sec = el("section", "c-part");
      const h = el("div", "c-part-head");
      h.appendChild(el("h2", null, T(part.title)));
      if (livePart.length) h.appendChild(el("span", "c-frac", `${donePart}/${livePart.length}`));
      sec.appendChild(h);
      if (part.blurb) sec.appendChild(el("p", "c-blurb", T(part.blurb)));

      const list = el("div", "c-lessons");
      for (const l of inPart) {
        const soon = l.status !== "exists";
        const row = soon ? el("div", "c-lesson soon") : el("a", "c-lesson");
        if (!soon) row.href = BASE + linkOf(l);

        const done_ = !soon && isDone(l);
        const check = el("button", "c-check " + (soon ? "soon" : done_ ? "done" : "todo"),
          soon ? "◇" : done_ ? "✓" : "○");
        check.type = "button";
        if (soon) {
          check.disabled = true;
          check.setAttribute("aria-label", LANG === "ko" ? "준비 중" : "not written yet");
        } else {
          check.setAttribute("aria-label", (done_ ? (LANG === "ko" ? "완료 해제: " : "mark not done: ")
                                                   : (LANG === "ko" ? "완료 표시: " : "mark done: ")) + T(l.title));
          check.addEventListener("click", (e) => {
            e.preventDefault();          // do not follow the row's link
            e.stopPropagation();
            if (isDone(l)) {
              // clear whichever key carries it (a lesson can be marked from either language)
              const key = done.has(l.url) ? l.url : l.ko_url;
              progress.markUndone(key);
            } else progress.markDone(linkOf(l));
            render();                    // completion changed → redraw bars, fraction, resume
          });
        }
        row.appendChild(check);
        row.appendChild(el("span", "c-title", T(l.title)));
        if (soon) row.appendChild(el("span", "c-badge soon", LANG === "ko" ? "예정" : "soon"));
        else if (isNew(l)) row.appendChild(el("span", "c-badge new", "NEW"));
        list.appendChild(row);
      }
      sec.appendChild(list);
      root.appendChild(sec);
    }
  }

  render();
  progress.touchSeen();   // stamp the visit AFTER the first paint computed "new since last visit"
})();
