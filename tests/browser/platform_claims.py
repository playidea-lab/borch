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
    adapterFeatures: null,
    adapterName: null,
  };
  // **What the GPU actually offers**, which is what three declined rows rest on:
  // `scaled_mm`, `grouped_mm` and `scaled_grouped_mm` read *that hardware is not
  // here*, and `autocast`'s row says its own absence is "a decision, not the
  // hardware" and names `shader-f16` as the thing to re-measure. Asked of the
  // adapter rather than of the shader, because a feature the adapter does not
  // list is one no shader can ask for.
  const features = (async () => {
    if (!navigator.gpu) return "no navigator.gpu";
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return "no adapter";
    out.adapterName = [adapter.info?.vendor, adapter.info?.architecture]
      .filter(Boolean).join(" / ") || "(unnamed)";
    return [...adapter.features].sort().join(" ") || "(none)";
  })().then((v) => { out.adapterFeatures = v; },
            (e) => { out.adapterFeatures = "failed: " + e.message; });
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

  Promise.all([opfs, worker, features]).then(() => resolve(out));
})"""


def _is_software(name):
    """Told apart by name, as `launch.is_software` does — imported there rather than
    copied, so the two cannot drift into two lists of CPU renderers."""
    sys.path.insert(0, str(ROOT / "tests" / "browser"))
    from launch import is_software                                    # noqa: PLC0415
    return is_software(name)


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

        ("there is a shared buffer, and a thread pool works over it",
         got["hasSharedArrayBuffer"] == "function" and got["crossOriginIsolated"],
         f"typeof SharedArrayBuffer = {got['hasSharedArrayBuffer']}, "
         f"crossOriginIsolated = {got['crossOriginIsolated']}",
         "**This one flipped once, on 2026-09-05.** It read *there is no shared buffer* "
         "and said `set_num_threads` was buildable the day the page became "
         "cross-origin-isolated. The page did (COOP/COEP from the servers, `site/coi.js` "
         "where a host cannot send them), the pool was built — `borch_cpu`'s "
         "`WorkerPool` over one wasm memory — and the reason moved to what the knob "
         "sizes: a pool of *tensor ops*, which this library does not have on the CPU. "
         "If this ever came back false, the cpu device would be on one thread again and "
         "`coi_sweep.py` is the check that says which page lost it."),

        ("a browser has a working file layer",
         got["opfsRoundTrip"] == "7,8,9",
         f"OPFS wrote [7, 8, 9] and read back {got['opfsRoundTrip']!r}",
         "`from_file`'s reason opens by granting this. If OPFS stopped working the "
         "sentence would be defending the wrong thing — the old reason said a browser "
         "has no file layer, and it was this measurement that showed otherwise."),

        # **Two claims, not one, and writing them as one was wrong.** The first draft
        # asked whether the adapter offered *any* reduced-precision format and went red
        # on `shader-f16` — which is f16, and the three rows it was defending are about
        # **fp8**. Two formats, two questions: the check was measuring the wrong one and
        # would have sent somebody to rewrite three correct sentences.
        ("the GPU has no fp8 format",
         isinstance(got["adapterFeatures"], str)
         and "f8" not in got["adapterFeatures"],
         f"no `f8` among: {got['adapterFeatures']}",
         "`scaled_mm`, `grouped_mm` and `scaled_grouped_mm` are torch's fp8 and "
         "grouped GEMMs, declined under *that hardware is not here*. WebGPU has no "
         "8-bit float type at all, in the shading language or as a feature; if one "
         "appears the sentence is spent."),

        # **The opposite direction, and the only claim here asserted as a presence.**
        #
        # It went red in CI the first time another machine ran it, and **the red was
        # the check disagreeing with its own last sentence.** That sentence says half
        # precision is declined because `shader-f16` is *optional* and machines with
        # it and without it diverge — and then the row asserted every machine has it.
        # A runner that lacks it does not falsify that reason; it is the reason.
        #
        # What the row is really defending is *declined by choice, **not** by
        # hardware*, and only **hardware** can witness that. SwiftShader is a CPU
        # pretending to be a GPU; what it does not offer says nothing about what GPUs
        # offer. So on a software adapter this reports and abstains, using the same
        # `is_software` the benchmarks have refused to run under since the day a
        # headless Linux box answered 845/845 as `google / swiftshader`.
        ("the GPU does have f16, so `autocast` is declined by choice",
         None if _is_software(got.get("adapterName"))
         else (isinstance(got["adapterFeatures"], str)
               and "shader-f16" in got["adapterFeatures"]),
         f"`shader-f16` present: {'shader-f16' in (got['adapterFeatures'] or '')}"
         f" · adapter {got.get('adapterName')!r}",
         "`autocast`'s row says its absence is **a decision, not the hardware** and "
         "names `shader-f16` as the thing to re-measure. That sentence only holds "
         "while some real adapter actually has it — were it to vanish from hardware, "
         "the row would be right for a reason it does not give, which is the same "
         "defect as being wrong."),
    ]


# The rows above defend these, by name. **When a row is gone, its claim goes with it** —
# an assertion protecting a decision nobody makes any more passes forever about nothing,
# which is the failure this repository keeps finding in its own checks.
DEFENDS = ("get_num_threads", "set_num_threads",
           "get_num_interop_threads", "set_num_interop_threads", "from_file",
           # The three fp8 rows, all under *that hardware is not here*, and `autocast`,
           # whose row rests on f16 being present rather than absent.
           "scaled_mm", "grouped_mm", "scaled_grouped_mm", "autocast")


def _still_declined():
    """The names above that `torch_gap.py` no longer declines.

    **`(rows, None)` when torch is not installed**, and the caller says so rather than
    passing quietly. This step runs under `uv run --with playwright` and nothing else —
    `torch_gap.py` imports torch at module scope, so on the CI runner this raised
    `No module named 'torch'` and turned the whole step red for a reason that has
    nothing to do with the browser. **It took main down with it**, and a peer session
    had to come and say so on the morning of a launch.

    Installing torch here to read one dictionary would cost the step its speed, which
    is why it sits beside the clipping scan rather than with the value comparisons. So
    the half that needs torch is skipped and **named as skipped** — a check that goes
    quiet without saying so is the failure this repository keeps finding, and the
    distinction between *checked and fine* and *not checked* has to survive into the
    output.
    """
    import importlib.util                                             # noqa: PLC0415
    try:
        import torch                                                  # noqa: PLC0415, F401
    except ImportError:
        return None
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
    # **`None` is a third answer and prints as one.** A claim that only hardware can
    # witness, measured on a CPU adapter, is neither held nor broken — and collapsing
    # it into either is how a check comes to disagree with its own reason.
    for what, holds, seen, _ in rows:
        mark = "-- " if holds is None else ("ok " if holds else "NO ")
        print(f"  {mark} {what}\n        {seen}")
    if stale is None:
        print("\n  -- whether these names are still declined was NOT checked --\n"
              "  `torch_gap.py` imports torch and this step does not install it. The\n"
              "  browser half above ran; the half that would catch a claim outliving\n"
              "  its row did not.")

    broken = [(what, seen, sting) for what, holds, seen, sting in rows
              if holds is False]
    abstained = [what for what, holds, *_ in rows if holds is None]
    if abstained:
        print(f"\n  -- {len(abstained)} not asked on this adapter --")
        for what in abstained:
            print(f"     {what}\n     a software adapter cannot witness a claim about "
                  "hardware. Run it on a GPU.")
    if not broken and not stale:
        print(f"\n{len(rows) - len(abstained)} of {len(rows)} hold — "
              "the reasons say what the browser says")
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
