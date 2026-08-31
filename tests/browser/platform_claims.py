"""**The reasons in `torch_gap.py` that make a claim about the browser, re-measured.**

    uv run --with playwright python tests/browser/platform_claims.py

A declined name carries a sentence saying why it is all right for it to be absent. Some
of those sentences are judgements — *outside the curriculum* — and a person settles those.
Others are **claims about the platform**, and a claim about the platform can stop being
true without anybody touching this repository.

Four of them were false when this file was written, and had been for the whole life of
the library:

    get_num_threads          "inside one tab there is no thread count to choose"
    set_num_threads          same
    get_num_interop_threads  "it is inside one tab"
    set_num_interop_threads  same

Measured: `navigator.hardwareConcurrency` is 16, `Worker` is a function, and a worker
handed `9.5` answered `20` — a second thread ran arithmetic. A fifth said a browser has
no file layer; OPFS wrote three bytes and read them back.

Nobody had asked. `Web Worker`, `hardwareConcurrency` and `SharedArrayBuffer` appear
nowhere in `tests/`, `borch/`, `borch_webgpu/`, `borch-ts/src/` or `site/` — zero
occurrences before this file. **A reason that asserts an absence nobody looked for is
worse than a name with no reason at all**: the missing reason draws the eye and the false
one answers the question so the eye moves on.

## What this can and cannot watch

It re-runs the measurements the corrected sentences rest on and fails when one flips. It
cannot watch the half of `from_file` that is about *semantics* — that a browser reads a
file into an ArrayBuffer whole, with no paging on touch and no write-through to one
buffer two readers share. No feature probe detects the arrival of an API that does not
exist; that half is a person's to re-check, and it is named here so the omission is on
the page rather than in the gap.

It also checks the other direction: **a claim whose row is gone is deleted, not kept.**
An assertion defending a decision nobody makes any more is how a check starts passing
about nothing.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# The page to measure in. **A site page, not the test runner** — cross-origin isolation
# is a property of what the server sends for the document people actually open, and the
# runner loads Pyodide for a minute to tell us nothing more.
PAGE = "site/index.html"

PROBE = """() => new Promise((resolve) => {
  const out = {
    hardwareConcurrency: navigator.hardwareConcurrency,
    hasWorker: typeof Worker,
    hasSharedArrayBuffer: typeof SharedArrayBuffer,
    crossOriginIsolated: self.crossOriginIsolated === true,
    workerAnswered: null,
    opfsRoundTrip: null,
  };
  const opfs = (async () => {
    // A handle existing is not the layer working. Write bytes, read them back.
    const dir = await navigator.storage.getDirectory();
    const fh = await dir.getFileHandle("platform_claims.bin", {create: true});
    const w = await fh.createWritable();
    await w.write(new Uint8Array([7, 8, 9]));
    await w.close();
    const back = new Uint8Array(await (await fh.getFile()).arrayBuffer());
    await dir.removeEntry("platform_claims.bin");
    return Array.from(back).join(",");
  })().then((v) => { out.opfsRoundTrip = v; },
            (e) => { out.opfsRoundTrip = "failed: " + e.message; });

  const worker = new Promise((done) => {
    try {
      // `2n + 1` rather than a constant: a reply of 20 to 9.5 cannot be produced by an
      // echo, a default, or a message that never left.
      const src = "onmessage = (e) => postMessage(e.data * 2 + 1)";
      const url = URL.createObjectURL(new Blob([src], {type: "text/javascript"}));
      const w = new Worker(url);
      const timer = setTimeout(() => { out.workerAnswered = "timed out"; done(); }, 10000);
      w.onmessage = (e) => {
        clearTimeout(timer);
        out.workerAnswered = e.data;
        w.terminate();
        URL.revokeObjectURL(url);
        done();
      };
      w.onerror = (e) => {
        clearTimeout(timer);
        out.workerAnswered = "error: " + e.message;
        done();
      };
      w.postMessage(9.5);
    } catch (err) {
      out.workerAnswered = "could not be created: " + err.message;
      done();
    }
  });

  Promise.all([opfs, worker]).then(() => resolve(out));
})"""


def _claims(got):
    """(what was claimed, whether it holds, what was seen, what changes if it flips)."""
    return [
        ("a tab has a thread count to read",
         isinstance(got["hardwareConcurrency"], int) and got["hardwareConcurrency"] > 1,
         f"navigator.hardwareConcurrency = {got['hardwareConcurrency']}",
         "If this ever came back 1 or absent, the *original* reason for the four thread "
         "rows would be true again and these sentences would be the wrong ones."),

        ("a second thread actually runs",
         got["workerAnswered"] == 20,
         f"a worker sent 9.5 answered {got['workerAnswered']!r} (expected 20 = 2n+1)",
         "A Worker constructor that exists but cannot run would put the thread rows "
         "back on their old footing."),

        ("there is no shared buffer for a thread pool to work over",
         got["hasSharedArrayBuffer"] == "undefined" and not got["crossOriginIsolated"],
         f"typeof SharedArrayBuffer = {got['hasSharedArrayBuffer']}, "
         f"crossOriginIsolated = {got['crossOriginIsolated']}",
         "**This is the one to watch.** `set_num_threads` and its interop twin are "
         "declined because threads without shared memory can only copy, and torch's "
         "intra-op pool is threads over one buffer. If this page becomes "
         "cross-origin-isolated, that reason is spent and the pair is buildable."),

        ("a browser has a working file layer",
         got["opfsRoundTrip"] == "7,8,9",
         f"OPFS wrote [7, 8, 9] and read back {got['opfsRoundTrip']!r}",
         "`from_file`'s reason opens by granting this. If OPFS stopped working the "
         "sentence would be defending the wrong thing — the old reason said a browser "
         "has no file layer, and it was this measurement that showed otherwise."),
    ]


# The rows above defend these, by name. **When a row is gone, its claim goes with it** —
# an assertion protecting a decision nobody makes any more passes forever about nothing,
# which is the failure this repository keeps finding in its own checks.
DEFENDS = ("get_num_threads", "set_num_threads",
           "get_num_interop_threads", "set_num_interop_threads", "from_file")


def _still_declined():
    """The names above that `torch_gap.py` no longer declines."""
    import importlib.util                                             # noqa: PLC0415
    spec = importlib.util.spec_from_file_location(
        "bt_gap_claims", ROOT / "tests" / "torch_gap.py")
    gap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gap)
    return [name for name in DEFENDS if not gap._look(gap.SKIPPED, name)]


def main():
    from playwright.sync_api import sync_playwright                   # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "tests" / "browser"))
    from run import serve                                             # noqa: PLC0415
    from launch import browser                                        # noqa: PLC0415

    stale = _still_declined()

    port, shutdown = serve(ROOT)
    try:
        with sync_playwright() as pw, browser(pw, headed=False) as b:
            page = b.new_page()
            page.set_default_timeout(30000)
            page.goto(f"http://127.0.0.1:{port}/{PAGE}", wait_until="load")
            got = page.evaluate(PROBE)
    finally:
        shutdown()

    rows = _claims(got)
    print(f"platform claims behind {len(DEFENDS)} declined names — measured in /{PAGE}\n")
    for what, holds, seen, _ in rows:
        print(f"  {'ok ' if holds else 'NO '} {what}\n        {seen}")

    broken = [(what, seen, sting) for what, holds, seen, sting in rows if not holds]
    if not broken and not stale:
        print(f"\nall {len(rows)} hold — the reasons say what the browser says")
        print("  Not watched here: that a file is read whole, with no paging on touch "
              "and no\n  write-through. No probe sees an API that does not exist — "
              "`from_file`'s second\n  half is a person's to re-check.")
        return 0

    if broken:
        print("\nthe browser no longer says what these reasons say:")
        for what, seen, sting in broken:
            print(f"\n  ✗ {what}\n    measured: {seen}\n    {sting}")
    if stale:
        print("\nclaims defending names that are no longer declined:")
        for name in stale:
            print(f"  ✗ {name} — `torch_gap.py` has no reason for it any more, so "
                  "remove it from\n    DEFENDS and drop whichever claim above was "
                  "holding its place.")
    print("\n  Change the reason in tests/torch_gap.py to what was measured here.\n"
          "  A sentence that outlives its measurement is the defect this file exists "
          "for.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
