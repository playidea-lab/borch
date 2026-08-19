/**
 * 강의 본문의 이름을 API 레퍼런스로 잇는다.
 *
 * 읽다가 `keepAlive` 를 만난 사람이 그 자리에서 시그니처를 볼 수 있어야 한다.
 * 손으로 링크를 걸면 이름이 바뀔 때 조용히 엉뚱한 자리를 가리키므로, **생성된
 * 목록에 있는 이름만** 잇는다 — 없는 이름은 그냥 코드 글씨로 남는다.
 *
 * 실행되는 코드 블록 안은 건드리지 않는다. 거기는 편집기이지 글이 아니다.
 */

const INDEX_URL = new URL("./api-index.json", import.meta.url).href;
const API_BASE = new URL("../api/", document.baseURI).href;

// 이름만 있는 `code` 만 본다. `model.call(x)` 같은 조각은 이름이 아니다.
// **빈 괄호는 붙어 있어도 이름이다** — 글에서는 `backward()`·`keepAlive()` 로 적는
// 것이 자연스럽고, 그것이 정확히 읽는 사람이 누르고 싶은 것이다.
const BARE = /^([A-Za-z_$][\w$]*)(?:\(\))?$/;

// 너무 흔해서 링크가 방해가 되는 것들. 글에서 이 낱말은 대개 API 이야기가 아니다.
const SKIP = new Set(["call", "forward", "step", "shape", "size", "data", "get", "set",
                      "params", "lr", "name", "value", "training", "describe"]);

(async () => {
  let index;
  try {
    const res = await fetch(INDEX_URL);
    if (!res.ok) return;                       // 목록이 없으면 그냥 안 잇는다
    index = await res.json();
  } catch {
    return;
  }

  const scope = document.querySelectorAll(
    ".lesson p code, .lesson li code, .lesson .note-box code");
  for (const el of scope) {
    if (el.closest(".runnable") || el.closest("a")) continue;
    const hit = BARE.exec(el.textContent.trim());
    if (!hit) continue;
    const name = hit[1];
    if (SKIP.has(name)) continue;
    const at = index[name];
    if (!at) continue;
    const link = document.createElement("a");
    link.href = `${API_BASE}#${at}`;
    link.className = "api-link";
    link.title = `API — ${at}`;
    el.replaceWith(link);
    link.append(el);
  }
})();
