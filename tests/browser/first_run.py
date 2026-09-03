"""Time to first run — the landing page, opened cold, Run pressed, until "done".

    uv run --with playwright python tests/browser/first_run.py [--headed] [--url=<page>]

**This is the product's first number and nothing measured it.** The PRD names
time-to-first-run as the success metric; the landing page has timed its own hero run
since the start (`home.js` prints `done — N ms`), and that number went to the person
who pressed the button and nowhere else. This opens the real page the way a visitor
does — page load, device probe, one click — and writes down what a visitor waits for,
and **where the wait is**:

    first visit   load → device ready → click → done, with the count of shaders the
                  click compiled and the time they took (hooked on `Device.pipeline`
                  from outside the page, so the site's code is not touched)
    revisit       the same page in the same browser profile, so Chrome's GPU cache
                  is warm — what the second visit costs

The split is the reason the two rows exist. On NVIDIA the first run was 3.2 s in the
page where Apple's was 46 ms, and the two numbers are not comparable until the shader
compile is separated from the rest: macOS caches compiled Metal shaders system-wide,
so a fresh browser profile on the laptop is warm, while Vulkan on Linux has only
Chrome's own cache, cold in every fresh profile. The first-visit row is the honest
cold number on both; the revisit row is what a returning visitor sees.

The adapter is the part that makes the number mean something. A software adapter
answers every WebGPU call correctly and slowly, so the same procedure on a machine with
no GPU prints a true number about the wrong thing — this refuses that number, as the
golden runners do, rather than logging it beside a real one.

`--url=<page>` measures a deployed copy instead of this checkout; the network is then
inside the clock, which is the number a visitor actually waits for. Served locally, what
this measures is the page and the device — the part this repository can change.

First measurements, 2026-09-03, this checkout served locally: apple / metal-3 180 ms in
the page and 0.7 s from opening the page to the done line (46 ms / 0.2 s on the quiet
machine at 04:30); nvidia / blackwell 3.2 s / 3.3 s. The deployed site: 763 ms / 1.8 s
on Apple, 3.0 s / 7.2 s on NVIDIA — the difference is the transfer, and that row is the
visitor's number.

Run nightly (`tests/browser/nightly.py`) so the number has a history, not a moment.
"""
import os
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from launch import FLAGS, refuse_if_software                     # noqa: E402
from run import serve                                            # noqa: E402

# A cold page on a slow adapter can take this long before a person gives up; past it the
# number is a failure, not a measurement.
GIVE_UP_S = 120

# Installed from outside the page once the device is ready: counts the shaders the click
# compiles and the time `createComputePipeline` takes, which is what a first visit pays on
# a driver with no warm cache.
# Installed before any page script runs (`add_init_script`): every adapter request, every
# device request, every submit and the first readback wait, page probe included.
GPU_HOOK = """
(() => {
  const stat = { adapterMs: 0, deviceMs: 0, adapters: [], submits: 0, firstMapMs: 0 };
  window.__gpuStat = stat;
  const install = () => {
    if (!("gpu" in navigator) || !window.GPU) return;
    const ra = GPU.prototype.requestAdapter;
    GPU.prototype.requestAdapter = async function (...a) {
      const t0 = performance.now(); const r = await ra.apply(this, a);
      const ms = performance.now() - t0; stat.adapterMs += ms;
      stat.adapters.push({ opts: JSON.stringify(a[0] ?? null), ms: Math.round(ms) });
      return r;
    };
    const rd = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...a) {
      const t0 = performance.now(); const r = await rd.apply(this, a); stat.deviceMs += performance.now() - t0; return r;
    };
    const sub = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function (...a) { stat.submits += 1; return sub.apply(this, a); };
    const map = GPUBuffer.prototype.mapAsync;
    let firstMap = true;
    GPUBuffer.prototype.mapAsync = async function (...a) {
      const t0 = performance.now(); const r = await map.apply(this, a);
      if (firstMap) { stat.firstMapMs = Math.round(performance.now() - t0); firstMap = false; }
      return r;
    };
  };
  install();
})();
"""

HOOK = """
(lib) => import(lib).then((m) => {
  // Hooked on the class, not an instance: the page probes at load and creates its
  // device on the first click, so there is nothing to hook until then. The WebGPU
  // entry points are wrapped too, so the adapter and device requests are on the clock.
  const stat = { compiled: 0, compileMs: 0 };
  const proto = m.Device.prototype;
  const orig = proto.pipeline;
  proto.pipeline = function (sig, src) {
    const had = this.pipelineCount;
    const t0 = performance.now();
    const p = orig.call(this, sig, src);
    if (this.pipelineCount > had) { stat.compiled += 1; stat.compileMs += performance.now() - t0; }
    return p;
  };
  window.__firstRun = stat;
  return 0;
})
"""


def _lib_url(url):
    return url.rsplit("/site/", 1)[0] + "/borch-ts/dist/src/index.js" if url else "/borch-ts/dist/src/index.js"


def _visit(context, url, label):
    """One visit: opens the page, waits for the device, hooks the shader compiler, clicks."""
    page = context.new_page()
    page.add_init_script(GPU_HOOK)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    opened = time.time()
    page.goto(url, wait_until="load")
    loaded = time.time()
    # The button is disabled until the device probe answers — that wait is part of what
    # a visitor experiences, so it is inside the clock.
    page.wait_for_function("!document.getElementById('hero-run').disabled",
                           timeout=GIVE_UP_S * 1000)
    ready = time.time()
    page.evaluate(HOOK, _lib_url(url))
    page.click("#hero-run")
    page.wait_for_function(
        "[...document.querySelectorAll('#hero-out div')].some(d => d.textContent.startsWith('done'))",
        timeout=GIVE_UP_S * 1000)
    done = time.time()
    said = page.evaluate(
        "[...document.querySelectorAll('#hero-out div')].map(d => d.textContent).find(t => t.startsWith('done'))")
    stat = page.evaluate("Object.assign({}, window.__gpuStat, window.__firstRun)")
    adapter = page.evaluate(
        f"import('{_lib_url(url)}').then(m => m.probe()).then(p => p.ok ? p.adapter : null)")
    in_page = said.split("—", 1)[1].split("ms")[0].strip() if "—" in said else "?"
    print(f"{label}: {in_page} ms in the page · {done - opened:.1f} s from opening the page to done")
    print(f"  load {loaded - opened:.2f} s · probe +{ready - loaded:.2f} s · click→done "
          f"{done - ready:.2f} s = adapter {stat['adapterMs']:.0f} ms + device {stat['deviceMs']:.0f} ms + "
          f"{stat['compiled']} shaders {stat['compileMs']:.0f} ms + the rest")
    print(f"  adapter requests: {stat['adapters']} · submits {stat['submits']} · "
          f"first readback waited {stat['firstMapMs']} ms")
    if errors:
        print("  page errors: " + " | ".join(errors[:3]))
    page.close()
    return adapter, errors


def main():
    from playwright.sync_api import sync_playwright

    headed = "--headed" in sys.argv
    url = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), None)
    port, shutdown = serve(ROOT)
    target = url or f"http://127.0.0.1:{port}/site/index.html"
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    # **A persistent profile, and two visits in it.** The second visit is what Chrome's
    # GPU cache makes of the first; a fresh profile per visit would measure the cold
    # number twice and call it the product.
    profile = tempfile.mkdtemp(prefix="borch-first-run-")
    try:
        with sync_playwright() as pw:
            # Opened headed unless `--headless` is asked for — the launcher's rule, kept
            # here because a persistent context is not the launcher's door.
            want_headed = headed or not (os.environ.get("BORCH_HEADLESS") or "--headless" in sys.argv)
            context = pw.chromium.launch_persistent_context(
                profile, headless=not want_headed, channel=channel, args=list(FLAGS),
                timeout=60_000)
            try:
                adapter, errors = _visit(context, target, "first visit")
                if errors:
                    return 1
                if refuse_if_software(adapter, "time to first run"):
                    return 1
                # Same profile, page closed and reopened: the driver and Chrome's shader
                # cache are whatever the first visit left them.
                adapter2, errors2 = _visit(context, target, "revisit")
                if errors2:
                    return 1
            finally:
                context.close()
            print(f"  page: {url or 'this checkout, served locally (network not in the clock)'}")
            print(f"  measured on: {adapter}")
            return 0
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
