/**
 * The API reference screen.
 *
 * The index comes from `site/assets/api.json`, which `site/build_api.py` pulls out of
 * `borch-ts/dist/src/*.d.ts`. **Nothing is written here** — a hand-written index is
 * right for the first week only, and this repository has already caught four stale
 * numbers.
 *
 * The descriptions are the source's TSDoc, which is English. The Korean beside them
 * lives in `site/api_ko.json`, each entry stamped with a fingerprint of the English it
 * was made from, so the generator names the ones whose source has moved. Signatures,
 * kinds and the torch name mapping are language-neutral and identical on both pages.
 */

import { pick } from "./i18n.js";
import { highlight } from "./runner.js";

const LANG = document.documentElement.lang === "ko" ? "ko" : "en";
const API_URL = new URL("./api.json", import.meta.url).href;

const S = {
  search: { en: "Search 1,300+ names…", ko: "이름으로 찾기…" },
  members: { en: "{0} members", ko: "멤버 {0}개" },
  torchIs: { en: "torch", ko: "torch" },
  noHits: { en: "Nothing matches that.", ko: "걸리는 것이 없다." },
  hits: { en: "{0} matches", ko: "{0}개 걸림" },
  modules: { en: "Modules", ko: "모듈" },
  inThis: { en: "In this module", ko: "이 모듈 안" },
  indexGone: {
    en: "Could not read the API index — run <code>python3 site/build_api.py</code> first.",
    ko: "API 목록을 못 읽었다 — <code>python3 site/build_api.py</code> 를 먼저 돌린다." },
  sourceNote: {
    // This place read "not translated here — a translation starts drifting from the
    // source the day it is written" for a long time. The worry was right, so every
    // translation carries a hash of the source it was made from, and the generator names
    // the stale ones. Drift is not prevented; it is prevented from being quiet.
    en: "Descriptions are the source's own comments. Fix one in the source, not on this "
      + "page.",
    ko: "설명문은 소스의 주석에서 나온다. 소스가 영어라 여기 한국어는 그 옆에 둔 번역이고, "
      + "번역마다 그때 본 원문이 함께 적혀 있어 낡으면 생성기가 이름을 댄다." },
  notYet: {
    en: "",
    ko: "아직 안 옮겼다 — 소스의 영어를 그대로 보여준다." },
  generated: {
    en: "Generated from borch-ts/dist/src/*.d.ts by site/build_api.py — {0} entries.",
    ko: "site/build_api.py 가 borch-ts/dist/src/*.d.ts 에서 뽑았다 — 항목 {0}개." },
};
const say = (key, ...args) => (S[key][LANG] ?? S[key].en)
  .replace(/\{(\d+)\}/g, (m, i) => String(args[Number(i)] ?? m));

/** The description in this screen's language. **Missing, it shows the source rather than hiding it.**
 *
 *  A blank leaves the name reading as though it has no description — and "written but
 *  not yet carried across" and "never written" are entirely different facts for a
 *  reader. So the source text appears with a mark beside it. */
const prose = (node) => {
  const src = node.doc ?? "";
  if (LANG !== "ko" || !src) return { text: src, untranslated: false };
  return node.doc_ko
    ? { text: node.doc_ko, untranslated: false }
    : { text: src, untranslated: true };
};

/** Draws one description, saying so when it has not been carried across. */
const proseHtml = (node, style = "") => {
  const { text, untranslated } = prose(node);
  if (!text) return "";
  return `<div class="prose"${style ? ` style="${style}"` : ""}>${md(text)}`
    + (untranslated ? `<p class="small muted">${esc(say("notYet"))}</p>` : "")
    + "</div>";
};

const sidebar = document.getElementById("sidebar");
const main = document.getElementById("doc-main");

let api = null;
let index = [];      // a flat list for searching

boot();

async function boot() {
  try {
    const res = await fetch(API_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    api = await res.json();
  } catch (err) {
    // This wording was hardcoded in Korean. A missing index is the normal state right
    // after a clone, so the person opening the English page for the first time meets
    // exactly this line.
    main.innerHTML = `<div class="note-box">${say("indexGone")}`
      + `<br><span class="small muted">${esc(String(err))}</span></div>`;
    return;
  }

  for (const mod of api.modules) {
    for (const sym of mod.symbols) {
      index.push({ mod: mod.name, name: sym.name, id: sym.name, kind: sym.kind,
                   doc: prose(sym).text });
      for (const mem of sym.members) {
        index.push({ mod: mod.name, name: mem.name, id: `${sym.name}.${mem.name}`,
                     kind: "member", of: sym.name, doc: prose(mem).text });
      }
    }
  }

  drawSidebar();
  window.addEventListener("hashchange", route);
  route();
}

/* ── the sidebar ────────────────────────────────────────────────────── */

function drawSidebar() {
  const box = document.createElement("div");
  box.innerHTML = `<input id="search" type="search" placeholder="${say("search")}"
                          autocomplete="off" spellcheck="false">
    <h4>${say("modules")}</h4><nav id="mods"></nav><h4 id="inthis-h" hidden>${say("inThis")}</h4>
    <nav id="inthis"></nav>`;
  sidebar.append(box);

  const mods = box.querySelector("#mods");
  for (const mod of api.modules) {
    const a = document.createElement("a");
    a.href = `#${mod.name}`;
    a.innerHTML = `<span class="n">${esc(mod.title)}</span><span class="c">${mod.count}</span>`;
    a.dataset.mod = mod.name;
    mods.append(a);
  }

  const search = box.querySelector("#search");
  search.addEventListener("input", () => {
    const q = search.value.trim();
    if (q) showHits(q); else route();
  });
}

/* ── routing ────────────────────────────────────────────────────────── */

function route() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const [modName, symName] = raw.split(".");
  const mod = api.modules.find((m) => m.name === modName) ?? api.modules[0];
  drawModule(mod);
  for (const a of sidebar.querySelectorAll("#mods a")) {
    a.classList.toggle("on", a.dataset.mod === mod.name);
  }
  if (symName || (raw && raw !== mod.name)) {
    const target = document.getElementById(raw);
    if (target) target.scrollIntoView({ block: "start" });
  } else {
    main.scrollIntoView({ block: "start" });
  }
}

function drawModule(mod) {
  main.innerHTML = "";
  const head = document.createElement("header");
  head.innerHTML = `
    <p class="eyebrow">API</p>
    <h1>${esc(mod.title)}</h1>
    ${mod.doc
        // Where the source has its own description, that one is the original. Our own
        // line is used only in its absence — both visible is a screen saying the same
        // thing twice.
        ? proseHtml(mod)
        : `<p class="lead">${inline(esc(pick(mod.blurb)))}</p>`}
    <p class="small muted" style="margin-top:1rem">${say("generated", api.total)}
      <br>${esc(say("sourceNote"))}</p>`;
  main.append(head);

  for (const sym of mod.symbols) main.append(card(sym));

  const inthis = document.getElementById("inthis");
  document.getElementById("inthis-h").hidden = false;
  inthis.innerHTML = "";
  for (const sym of mod.symbols) {
    const a = document.createElement("a");
    a.href = `#${sym.name}`;
    a.innerHTML = `<span class="n">${esc(sym.name)}</span>`;
    a.addEventListener("click", (e) => {
      e.preventDefault();
      history.replaceState(null, "", `#${mod.name}.${sym.name}`);
      document.getElementById(sym.name)?.scrollIntoView({ block: "start" });
    });
    inthis.append(a);
  }
}

function card(sym) {
  const box = document.createElement("section");
  box.className = "symbol";
  box.id = sym.name;
  const sigs = [sym.signature, ...(sym.overloads ?? [])];
  box.innerHTML = `
    <h3>${esc(sym.name)}
      <span class="kind ${esc(sym.kind)}">${esc(sym.kind)}</span>
      ${sym.torch ? `<span class="torch-hint">torch: <b>${esc(sym.torch)}</b></span>` : ""}
    </h3>
    ${sigs.map((s) => `<pre class="sig"><code>${highlight(s, "js")}</code></pre>`).join("")}
    ${proseHtml(sym)}
    ${tags(sym)}`;

  if (sym.members.length) {
    const det = document.createElement("details");
    det.className = "members";
    det.open = sym.members.length <= 12;
    det.innerHTML = `<summary>${say("members", sym.members.length)}</summary>`;
    // **The order and grouping the source set are kept as they are.** Four hundred and
    // twenty-eight names in one run is a wall, not a list. They are not regrouped —
    // that would cover the author's classification with our guess, and it would drift
    // quietly the moment the source changed.
    let lastSection = null;
    for (const mem of sym.members) {
      const section = mem.section ? pick(mem.section) : null;
      if (section !== lastSection) {
        lastSection = section;
        if (section) {
          const head = document.createElement("h5");
          head.className = "section-head";
          head.textContent = section;
          det.append(head);
        }
      }
      const el = document.createElement("div");
      el.className = "member";
      el.id = `${sym.name}.${mem.name}`;
      el.innerHTML = `
        <h4>${esc(mem.name)}
          ${mem.protected ? '<span class="kind">protected</span>' : ""}
          ${mem.torch ? `<span class="torch-hint">torch: <b>${esc(mem.torch)}</b></span>` : ""}
        </h4>
        <div class="sigline">${highlight(mem.signature, "js")}</div>
        ${proseHtml(mem, "margin-top:.4rem")}
        ${tags(mem)}`;
      det.append(el);
    }
    box.append(det);
  }
  return box;
}

function tags(item) {
  if (!item.tags || !item.tags.length) return "";
  const rows = item.tags.map((t) =>
    `<tr><td class="t">@${esc(t.tag)}</td><td>${md(t.text)}</td></tr>`).join("");
  return `<table class="tags">${rows}</table>`;
}

/* ── search ─────────────────────────────────────────────────────────── */

function showHits(q) {
  const needle = q.toLowerCase();
  // **A hit in the name comes first.** Ranking was once by match position (indexOf)
  // alone, and something matching only its **owner's** name scored −1 and went to the
  // front — searching `conv` put `bias — nn.ConvND` above `Conv2d`.
  const rank = (e) => {
    const name = e.name.toLowerCase();
    const at = name.indexOf(needle);
    if (at === 0) return [0, name.length];                  // the name starts with it
    if (at > 0) return [1, at];                             // somewhere inside the name
    const owner = (e.of ?? "").toLowerCase().indexOf(needle);
    if (owner >= 0) return [2, owner];                      // only in the owner's name
    return [3, 0];                                          // only in the description
  };
  // **The descriptions are searched too.** Arriving without knowing the name is the
  // common case — for someone searching "leak", "transpose" or "in place", a search
  // that only reads names gives nothing back. Each page searches its own language,
  // since `prose()` above hands over whichever text that page shows.
  const hits = index
    .filter((e) => e.name.toLowerCase().includes(needle)
                || (e.of ?? "").toLowerCase().includes(needle)
                || e.doc.toLowerCase().includes(needle))
    .sort((a, b) => {
      const [ra, sa] = rank(a), [rb, sb] = rank(b);
      return ra - rb || sa - sb || a.name.localeCompare(b.name);
    })
    .slice(0, 200);

  main.innerHTML = `<header><p class="eyebrow">${esc(q)}</p>
    <h1>${hits.length ? say("hits", hits.length) : say("noHits")}</h1></header>`;
  const ul = document.createElement("ul");
  ul.className = "hits";
  for (const h of hits) {
    const li = document.createElement("li");
    const why = !h.name.toLowerCase().includes(needle)
             && !(h.of ?? "").toLowerCase().includes(needle)
      ? snippet(h.doc, needle) : "";
    li.innerHTML = `<a href="#${esc(h.mod)}.${esc(h.id)}">${esc(h.name)}
      <span class="where">— ${esc(h.mod)}${h.of ? `.${esc(h.of)}` : ""}</span></a>`
      + (why ? `<div class="why">${why}</div>` : "");
    ul.append(li);
  }
  main.append(ul);
}

/** Shows only the slice around the matched word. */
function snippet(text, needle) {
  const flat = text.replace(/\s+/g, " ");
  const at = flat.toLowerCase().indexOf(needle);
  if (at < 0) return "";
  const from = Math.max(0, at - 40);
  const cut = flat.slice(from, at + needle.length + 60);
  return (from ? "… " : "") + esc(cut) + " …";
}

/* ── a very thin markdown ───────────────────────────────────────────── */

function md(text) {
  const blocks = esc(text).split(/\n{2,}/);
  return blocks.map((block) => {
    const lines = block.split("\n");
    if (lines[0].startsWith("## ")) {
      return `<h4>${inline(lines[0].slice(3))}</h4>`
        + (lines.length > 1 ? `<p>${inline(lines.slice(1).join(" "))}</p>` : "");
    }
    if (lines.every((l) => l.trim().startsWith("- ") || !l.trim())) {
      const items = lines.filter((l) => l.trim()).map((l) => `<li>${inline(l.trim().slice(2))}</li>`);
      return `<ul>${items.join("")}</ul>`;
    }
    return `<p>${inline(lines.join(" "))}</p>`;
  }).join("");
}

function inline(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

function esc(text) {
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
