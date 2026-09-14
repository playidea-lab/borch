/**
 * The book's table of contents, in every lesson's sidebar.
 *
 * The static `<aside class="sidebar">` in each page lists only that section's own lessons —
 * enough for the section-sidebar guard and for a no-JS reader, but it leaves the rest of the
 * book reachable only through the course home. This replaces that sidebar, at runtime, with
 * the whole tree read from `curriculum.json` (the same SSOT the course home renders), the
 * current lesson marked and completed ones ticked — so a reader can jump between chapters
 * from anywhere. It reuses the sidebar's existing styles (`h4`, `nav a`, `.on`); nothing new.
 */
import * as progress from "./progress.js";

const LANG = document.documentElement.lang === "ko" ? "ko" : "en";
// every lesson page sits one level under site/ (en) or two (ko/…); the book pages have no
// sidebar and never reach here, so these two are the only depths.
const BASE = LANG === "ko" ? "../../" : "../";
const T = (o) => (o && (o[LANG] ?? o.en)) ?? "";

function make(tag, text, href) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (href) e.href = href;
  return e;
}

(async () => {
  const aside = document.querySelector("aside.sidebar");
  if (!aside) return;

  let data;
  try { data = await (await fetch(`${BASE}assets/curriculum.json`)).json(); }
  catch { return; }                      // leave the static section sidebar in place

  // The page's own path as it appears in the manifest (drops origin and any deploy prefix
  // up to the last `site/`), so the current lesson can be marked.
  const here = location.pathname.replace(/.*\/site\//, "").replace(/^\/+/, "").replace(/[?#].*$/, "");
  const done = progress.completed();
  const linkOf = (l) => (LANG === "ko" ? l.ko_url : l.url);
  const isHere = (l) => l.url === here || l.ko_url === here;

  const frag = document.createDocumentFragment();

  // A short header linking the front matter and the full course home.
  frag.append(make("h4", LANG === "ko" ? "목차" : "Contents"));
  const top = document.createElement("nav");
  top.append(make("a", LANG === "ko" ? "코스 홈 (전체 트리)" : "Course home (full tree)", `${BASE}learn/`));
  top.append(make("a", LANG === "ko" ? "이 책을 읽는 법" : "How to read this book", `${BASE}book/preface.html`));
  top.append(make("a", LANG === "ko" ? "표기법" : "Notation", `${BASE}book/notation.html`));
  frag.append(top);

  for (const part of data.parts) {
    const inPart = data.lessons.filter((l) => l.part === part.id).sort((a, b) => a.order - b.order);
    if (!inPart.length) continue;
    frag.append(make("h4", T(part.title)));
    const nav = document.createElement("nav");
    for (const l of inPart) {
      const soon = l.status !== "exists";
      const label = (!soon && (done.has(l.url) || done.has(l.ko_url)) ? "✓ " : "") + T(l.title);
      const item = soon ? make("span", "◇ " + T(l.title)) : make("a", label, BASE + linkOf(l));
      if (isHere(l)) item.className = "on";
      nav.append(item);
    }
    frag.append(nav);
  }

  aside.replaceChildren(frag);
})();
