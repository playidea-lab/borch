"""Whether the examples written in the documentation actually run.

    npm run build:ts
    uv run --with playwright python borch-ts/test/readme.py [--headed]

**Code in a document rots unless it is run.** A renamed name, a reordered argument, one
missing `await` — nobody says a word and the first user stops there. This repository has
already caught README install instructions that did not work, twice.

It does not ask about values — the golden does that. What is asked here is one thing:
**does it run when typed exactly as written.** So it does not refuse on a software adapter
either. Whether an example runs has nothing to do with the device.
"""

import sys

import run as runner
from launch import browser as browser_of
from verdict import verdict

PAGE = "/borch-ts/test/readme.html"
TIMEOUT_MS = 300_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "readme.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [browser] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchReadme !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchReadme")
    finally:
        stop()

    if "error" in result:
        print(f"**an example blew up** — somebody typing the document verbatim stops "
              f"here.\n"
              f"{result['error']}", file=sys.stderr)
        return 1
    print(f"adapter: {result.get('adapter', '(unknown)')}")
    print(result["text"])
    # **This line was `"그대로 돌고" in text`.** That phrase sits in the success sentence
    # of both examples, so with the first failing and only LBFGS passing it exited 0 —
    # leaving an example whose loss does not go down in the document.
    return verdict(result, "the README examples")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
