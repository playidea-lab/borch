/**
 * Course progress — per browser, no login, nothing leaves the tab.
 *
 * Completion is keyed by the lesson's **site-relative path** ("learn/01-tensors.html",
 * "ko/learn/01-tensors.html") so it matches a lesson's `url`/`ko_url` in
 * `curriculum.json` without every lesson page having to declare an id. A lesson marks
 * itself done when its exercise verdict passes (see runnable.js) — "your code worked",
 * not a checkbox. `localStorage` is per-origin and can throw (private windows, blocked
 * site data), so every access is guarded and a missing store just reads as "nothing done".
 */

const DONE_KEY = "borch.progress.v1";   // { "learn/01-tensors.html": { at: <ms> } }
const SEEN_KEY = "borch.lastSeen.v1";   // ms of the last course-home visit

/** location.pathname → the path as it appears in curriculum.json (drops origin and any
 *  deploy prefix up to and including the last `site/`; falls back to a trimmed pathname). */
export function pathKey(pathname = location.pathname) {
  const m = /(?:^|\/)site\/(.+)$/.exec(pathname);
  return (m ? m[1] : pathname.replace(/^\/+/, "")).replace(/[?#].*$/, "");
}

function read() {
  try { return JSON.parse(localStorage.getItem(DONE_KEY) || "{}") || {}; } catch { return {}; }
}
function write(obj) {
  try { localStorage.setItem(DONE_KEY, JSON.stringify(obj)); } catch { /* private window / blocked */ }
}

/** All completed paths as a Set. */
export function completed() {
  return new Set(Object.keys(read()));
}

/** Is any of these paths complete? Pass a lesson's [url, ko_url]. */
export function isDone(...paths) {
  const done = read();
  return paths.some((p) => p && done[p]);
}

/** Mark a path complete (idempotent). Default: the current page. */
export function markDone(path = pathKey()) {
  if (!path) return;
  const done = read();
  if (!done[path]) { done[path] = { at: Date.now() }; write(done); }
}

/** Clear a path's completion. */
export function markUndone(path = pathKey()) {
  const done = read();
  if (done[path]) { delete done[path]; write(done); }
}

/** ms of the last time the learner opened the course home (0 if never). */
export function lastSeen() {
  try { return Number(localStorage.getItem(SEEN_KEY)) || 0; } catch { return 0; }
}
/** Record a course-home visit — read this *before* calling, then stamp now. */
export function touchSeen() {
  try { localStorage.setItem(SEEN_KEY, String(Date.now())); } catch { /* ignore */ }
}
