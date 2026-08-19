/**
 * API 레퍼런스 화면.
 *
 * 목록의 원본은 `site/assets/api.json` 이고 그것은 `site/build_api.py` 가
 * `borch-ts/dist/src/*.d.ts` 에서 뽑는다. **여기서는 아무것도 안 적는다** — 손으로
 * 적은 목록은 첫 주에만 맞고, 이 저장소는 낡은 수를 이미 네 번 잡았다.
 *
 * 설명문은 소스의 TSDoc 그대로라 **한국어다.** 시그니처·분류·torch 이름 대응은
 * 언어 중립이라 양쪽 페이지에서 같다. 영어 페이지는 그 사정을 화면에 적는다 —
 * 번역본을 지어내면 소스와 갈리기 시작하고, 갈린 뒤에는 어느 쪽이 사실인지 모른다.
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
  koNote: {
    en: "Descriptions are lifted from the source's own comments, which are written "
      + "in Korean. They are not translated here — a translation would start drifting "
      + "from the source the day it was written. Signatures, kinds and the torch name "
      + "mapping are language-neutral.",
    ko: "설명문은 소스의 주석을 그대로 옮긴 것이다. 고칠 곳은 이 페이지가 아니라 소스다." },
  generated: {
    en: "Generated from borch-ts/dist/src/*.d.ts by site/build_api.py — {0} entries.",
    ko: "site/build_api.py 가 borch-ts/dist/src/*.d.ts 에서 뽑았다 — 항목 {0}개." },
};
const say = (key, ...args) => (S[key][LANG] ?? S[key].en)
  .replace(/\{(\d+)\}/g, (m, i) => String(args[Number(i)] ?? m));

const sidebar = document.getElementById("sidebar");
const main = document.getElementById("doc-main");

let api = null;
let index = [];      // 검색용 납작한 목록

boot();

async function boot() {
  try {
    const res = await fetch(API_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    api = await res.json();
  } catch (err) {
    main.innerHTML = `<div class="note-box">API 목록을 못 읽었다 — <code>python3 site/build_api.py</code>
      를 먼저 돌린다.<br><span class="small muted">${esc(String(err))}</span></div>`;
    return;
  }

  for (const mod of api.modules) {
    for (const sym of mod.symbols) {
      index.push({ mod: mod.name, name: sym.name, id: sym.name, kind: sym.kind,
                   doc: sym.doc ?? "" });
      for (const mem of sym.members) {
        index.push({ mod: mod.name, name: mem.name, id: `${sym.name}.${mem.name}`,
                     kind: "member", of: sym.name, doc: mem.doc ?? "" });
      }
    }
  }

  drawSidebar();
  window.addEventListener("hashchange", route);
  route();
}

/* ── 사이드바 ───────────────────────────────────────────────────────── */

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

/* ── 길찾기 ─────────────────────────────────────────────────────────── */

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
        // 소스가 자기 설명을 갖고 있으면 그쪽이 원본이다. 우리 한 줄은 그것이
        // 없을 때만 쓴다 — 둘 다 보이면 같은 말을 두 번 하는 화면이 된다.
        ? `<div class="prose">${md(mod.doc)}</div>`
        : `<p class="lead">${inline(esc(pick(mod.blurb)))}</p>`}
    <p class="small muted" style="margin-top:1rem">${say("generated", api.total)}
      ${LANG === "en" ? `<br>${esc(say("koNote"))}` : ""}</p>`;
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
    ${sym.doc ? `<div class="prose">${md(sym.doc)}</div>` : ""}
    ${tags(sym)}`;

  if (sym.members.length) {
    const det = document.createElement("details");
    det.className = "members";
    det.open = sym.members.length <= 12;
    det.innerHTML = `<summary>${say("members", sym.members.length)}</summary>`;
    // **소스가 나눠 둔 순서와 묶음을 그대로 지킨다.** 428 개가 한 줄로 늘어서면
    // 목록이 아니라 벽이다. 다시 나누지는 않는다 — 그건 저자의 분류를 우리 짐작으로
    // 덮는 것이고, 소스가 바뀌면 조용히 어긋난다.
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
        ${mem.doc ? `<div class="prose" style="margin-top:.4rem">${md(mem.doc)}</div>` : ""}
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

/* ── 검색 ───────────────────────────────────────────────────────────── */

function showHits(q) {
  const needle = q.toLowerCase();
  // **이름에 걸린 것이 먼저다.** 처음에는 걸린 자리(indexOf)로만 줄을 세웠는데,
  // 이름에 없고 **주인 이름**에만 걸린 것이 −1 을 받아 맨 앞에 섰다 — `conv` 를
  // 찾으면 `Conv2d` 보다 `bias — nn.ConvND` 가 먼저 나왔다.
  const rank = (e) => {
    const name = e.name.toLowerCase();
    const at = name.indexOf(needle);
    if (at === 0) return [0, name.length];                  // 이름이 그것으로 시작
    if (at > 0) return [1, at];                             // 이름 안 어딘가
    const owner = (e.of ?? "").toLowerCase().indexOf(needle);
    if (owner >= 0) return [2, owner];                      // 주인 이름에만
    return [3, 0];                                          // 설명문에만
  };
  // **설명문도 찾는다.** 이름을 모르는 채로 오는 것이 흔한 경우다 — "누수"·"전치"·
  // "in place" 로 찾는 사람에게 이름만 보는 검색은 아무것도 못 준다. 설명이 한국어라
  // 한국어 검색이 실제로 듣는다.
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

/** 걸린 낱말 둘레만 잘라 보여 준다. */
function snippet(text, needle) {
  const flat = text.replace(/\s+/g, " ");
  const at = flat.toLowerCase().indexOf(needle);
  if (at < 0) return "";
  const from = Math.max(0, at - 40);
  const cut = flat.slice(from, at + needle.length + 60);
  return (from ? "… " : "") + esc(cut) + " …";
}

/* ── 아주 얇은 마크다운 ─────────────────────────────────────────────── */

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
