/**
 * Links names in lesson prose to the API reference.
 *
 * Someone who meets `keepAlive` while reading should be able to see its signature
 * from there. Hand-written links point quietly at the wrong place once a name
 * changes, so **only names present in the generated index** are linked — anything
 * else stays plain code type.
 *
 * Inside a runnable block nothing is touched. That is an editor, not prose.
 */

const INDEX_URL = new URL("./api-index.json", import.meta.url).href;
const API_BASE = new URL("../api/", document.baseURI).href;

// Only `code` that is a bare name. A fragment like `model.call(x)` is not a name.
// **Empty parentheses still leave it a name** — prose naturally writes `backward()`
// and `keepAlive()`, and that is exactly what a reader wants to click.
const BARE = /^([A-Za-z_$][\w$]*)(?:\(\))?$/;

// Too common for a link to be anything but noise. In prose these words are usually
// not about the API at all.
const SKIP = new Set(["call", "forward", "step", "shape", "size", "data", "get", "set",
                      "params", "lr", "name", "value", "training", "describe"]);

(async () => {
  let index;
  try {
    const res = await fetch(INDEX_URL);
    if (!res.ok) return;                       // no index, no links — that is all
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
