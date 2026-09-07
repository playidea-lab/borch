"""The fold probe — **the backward of expand, repeat and flip against the walking kernel.**

    npm run build:ts
    uv run --with playwright python tests/browser/fold_probe.py [--side=1024] [--headless]

Opens `fold_probe.html`, which imports `borch-ts/dist` directly. Each case runs the
operation's backward the way autograd does (a fold on existing kernels) and dispatches
the walking kernel by hand on the same rules; both are timed by the wall and compared
with the closed form. Exit 1 on a fault, a WRONG, or a software adapter.
"""
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, refuse_if_screen_off, serve  # noqa: E402
from launch import _headed, refuse_if_software  # noqa: E402

GIVE_UP_MS = 10 * 60 * 1000


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headless" not in argv)
    side = next((a.split("=", 1)[1] for a in argv if a.startswith("--side=")), "1024")
    if refuse_if_screen_off("the fold probe"):
        return 1
    port, shutdown = serve(ROOT)
    url = f"http://127.0.0.1:{port}/tests/browser/fold_probe.html?side={side}"
    profile = tempfile.mkdtemp(prefix="borch-fold-")
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(profile, headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(url, wait_until="load")
                page.wait_for_function("window.__bench !== undefined", timeout=GIVE_UP_MS, polling=200)
                got = page.evaluate("window.__bench")
            finally:
                context.close()
    finally:
        shutdown()
    print(got["text"])
    if got.get("error"):
        print("error: " + got["error"][:600])
        return 1
    if refuse_if_software(got.get("adapter"), "the fold probe"):
        return 1
    print("**the fold is the walk's answer, in O(output)**" if got["ok"] else "**a fault or a wrong answer** — see above")
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
