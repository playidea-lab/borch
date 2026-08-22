"""Whether device management runs in a real browser.

    npm run build:ts
    uv run --with playwright python borch-ts/test/device.py [--headed]

**The golden does not catch this.** The golden asks whether the values match torch's,
and what is asked here is not a value but *where it is* and *what it says when it is not
there* — `t.device`, the `cpu()`/`webgpu()` round trip, the message for mixing devices,
`synchronize()`.

These questions cannot be imitated in node. `navigator.gpu` has to exist, an adapter has
to exist, and real buffers have to move.

It does not refuse on a software adapter — the placement rules are the same on any.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/device.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **A stale emit is as bad as none** — edit the source, forget the build, and you
    # measure the old code. `require_fresh_dist` watches that place (`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "device.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        # **`with` closes it too** — put on the last line instead, an exception before
        # it leaves it open, and the leftover Chromium ruins another measurement.
        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [browser] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchDevice !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchDevice")
    finally:
        stop()

    if "error" in result:
        print(f"**the device check blew up**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"adapter: {result.get('adapter', '(unknown)')}")
    # **The optional features are recorded too.** Without `timestamp-query` the route to
    # per-kernel timing is closed outright — and then the cause is hunted with a wall clock
    # alone.
    print(f"features: {result.get('features') or '(none)'}")
    print(result["text"])
    # The page counts the failures. Here that verdict is taken as it stands — counted in
    # two places, there is no way to know which is right when the two disagree.
    return verdict(result, "device management")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
