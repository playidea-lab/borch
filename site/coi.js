/*
 * Cross-origin isolation for a page whose host cannot set headers — GitHub Pages, or a
 * folder served by `python3 -m http.server`.
 *
 * `SharedArrayBuffer`, and with it the cpu device's worker pool (`borch-ts/src/cpu/threads.ts`),
 * exists only on a page delivered with `Cross-Origin-Opener-Policy: same-origin` and
 * `Cross-Origin-Embedder-Policy: require-corp`. `site/serve.py` sends both. A static host
 * that cannot is given them by this file, which is two things in one:
 *
 *   as a page script  — if the page is not isolated, it registers itself as a service
 *                       worker and reloads once, so the second load comes through the
 *                       worker;
 *   as that worker    — it adds the two headers to every response it hands the page.
 *
 * Opaque responses (`no-cors` fetches of another origin, a plain `<img src>` from
 * elsewhere) cannot be given headers and are blocked under `require-corp`; the site has
 * none, and `tests/browser/coi_sweep.py` visits every page under the worker to keep it so.
 * What this cannot do: run on `file://` or on plain `http://` from another machine —
 * a service worker needs a secure context, and `localhost` is the one insecure origin
 * browsers treat as secure. Without it the same pages run, on one thread.
 *
 * Reload happens at most once per tab (sessionStorage), so a browser that installs the
 * worker and still does not isolate — an extension stripping headers, a corporate proxy —
 * gets the page on one thread rather than a reload loop.
 */
(() => {
  const COOP = "same-origin", COEP = "require-corp";
  if (typeof ServiceWorkerGlobalScope !== "undefined" && self instanceof ServiceWorkerGlobalScope) {
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
    self.addEventListener("fetch", (event) => {
      const request = event.request;
      // The browser's own back/forward cache probe; answering it breaks the probe.
      if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
      event.respondWith(fetch(request).then((response) => {
        if (response.status === 0) return response; // opaque — no headers can be added
        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Embedder-Policy", COEP);
        headers.set("Cross-Origin-Opener-Policy", COOP);
        return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
      }));
    });
    return;
  }
  if (typeof window === "undefined" || window.crossOriginIsolated) return; // the host sent the headers
  if (!("serviceWorker" in navigator)) return;                             // insecure context or no support: one thread
  const script = document.currentScript && document.currentScript.src;
  if (!script) return;
  const KEY = "borch-coi-reloaded";
  navigator.serviceWorker.register(script).then(async (registration) => {
    if (navigator.serviceWorker.controller) return; // already through a worker and still not isolated: stop here
    if (sessionStorage.getItem(KEY)) return;        // reloaded once already
    sessionStorage.setItem(KEY, "1");
    const worker = registration.installing || registration.waiting || registration.active;
    if (worker && worker.state !== "activated") {
      await new Promise((resolve) => worker.addEventListener("statechange", () => { if (worker.state === "activated") resolve(); }));
    }
    location.reload();
  }).catch(() => { /* registration refused: the page runs on one thread */ });
})();
