"""Whether the **runnable blocks** in the lessons and the tutorials actually run.

    npm run build:ts
    uv run --with playwright python borch-ts/test/lessons.py

## Why it exists

This repository had never once run the JS on its own site pages. `test_site.py` looks at
the words, the links and the sprites, and the golden runner looks only at its own pages
inside `borch-ts/test/`. Between the two sits **the code a user actually presses**, and
nobody was looking at it.

That place asked its price when `save`/`load` became a nested pair. Ten of them were
written `load(x).tensors` and the new `load` hands the table over directly — mending it was
a line each, but **the only way to know whether it had been mended was a person's eyes.**
The next time a public name moves, the same places break the same silent way.

## What it measures

It **presses** every `data-lang="js"` block on each page, then looks at whether output
appeared and whether an exception's wording is in it. It does not look at whether the
values are right — that is the golden's job. What is caught here is **whether it blows
up**, and blowing up is exactly what a renamed name does.

Python blocks (`data-lang="py"`) are not pressed. They want Pyodide downloaded, which is a
different kind of slow from the one this check is measuring.
"""

import pathlib
import sys

import run as runner
from launch import browser as browser_of

ROOT = runner.ROOT

# **The pages to press.** The ones with runnable JS blocks that use borch.ts.
PAGES = [
    "/site/learn/06-save-load.html",
    "/site/ko/learn/06-save-load.html",
    "/site/tutorials/01-quickstart.html",
    "/site/ko/tutorials/01-quickstart.html",
    "/site/tutorials/03-curve-fitting.html",
    "/site/ko/tutorials/03-curve-fitting.html",
    "/site/tutorials/05-adversarial.html",
    "/site/ko/tutorials/05-adversarial.html",
    "/site/learn/09-resnet.html",
    "/site/ko/learn/09-resnet.html",
    "/site/learn/10-vit.html",
    "/site/ko/learn/10-vit.html",
]

# **This list is hand-kept and reads as coverage.** Thirty-two pages under `site` carry a
# `data-lang="js"` block. These press twelve. The four lessons above were added only
# because their author went looking for the harness after a sister session found it had
# been running two of three implementations and calling it green.
#
# **A page absent from here is not reported as unwatched — it is not reported at all**,
# which on screen is indistinguishable from a page that passed. Twenty pages (about
# seventy-nine blocks) are in that state with no reason recorded: lessons 1–5, 7 and 8 in
# both languages, tutorials 02 and 06, and `python.html`. Only `04-image-classifier` below
# is left out on purpose.
#
# **The cost is why it is not simply derived from the directory.** Measured on a software
# adapter: twelve pages take 544s, of which the two `10-vit` pages are 154s — a ViT page
# is about two and a half times an ordinary one, because its last block trains. Pressing
# all thirty-two would be several times this, and `gpu.yml` carries no `push` trigger for
# it (a real adapter is needed), so **this check runs only when a person runs it.** Making
# it slower is making it run less often, which is the same road that ended in two-of-three
# above. The number is written here rather than left as a silent budget; whoever decides
# should decide against it and not against a guess.

# **`10-vit` prints a caught exception on purpose.** Its second block shows `nn.Linear`
# refusing a 3-D input, so the message `mm is 2-D by 2-D: ...` is the **right** output
# there. It survives `BAD` because none of those words are in it, and that is luck rather
# than design — a message reworded to start `Error:` would fail a page that is working.
# The `div.err` mark stays correct either way: the lesson catches, so the page never marks
# the line as an error.

# **`04-image-classifier` is left out.** It downloads CIFAR and runs convolutions for
# several epochs, which takes minutes on a software adapter — out of proportion to what
# this check measures (does a renamed name blow up). That page's `load` is the same single
# line, and the four above watch it.

# **Structure first.** The page catches an exception and writes it to the screen
# (`runnable.js`'s `write(describeError(err), "err")`), so a line that threw arrives
# carrying `class="err"`. Counting that mark rather than the words survives a change of
# wording, and survives the page being in another language.
#
# At first this looked at the words alone, and the list contained `"실패"` ("failed").
# **That pattern was dead** — what `describeError` produces is `err.name: err.message`, and
# there is nowhere in that for the word to appear. A dead pattern cannot be told from a
# rare one by looking at the screen.
ERROR_CLASS = "div.err"

# The word side is **a net, not a gate.** It is there to catch a place that runs wrongly
# without throwing, and wording that is not in this list does not make a pass — that gate
# is held by the `div.err` above.
BAD = ("is not a function", "undefined is not", "Cannot read", "TypeError",
       "ReferenceError", "Error:", "throw")

TIMEOUT_MS = 300_000


def run_page(page, path):
    """Press every JS block on one page; return (passed, what to say)."""
    page.goto(path)
    page.wait_for_selector("div.runnable button.go", timeout=TIMEOUT_MS)

    said = []
    blocks = page.query_selector_all("div.runnable")
    pressed = 0
    for i, block in enumerate(blocks):
        # Python blocks are skipped — they download Pyodide, which is not what this
        # measures.
        if (block.get_attribute("data-lang") or "js") != "js":
            continue
        go = block.query_selector("button.go")
        if go is None:
            said.append(f"block {i} has no run button")
            continue
        go.click()
        pressed += 1
        # **It is finished when the button becomes pressable again**
        # (`runnable.js`'s `runBtn.disabled`).
        #
        # At first this watched the button's **text** come back from "running". That word
        # belongs to the other file and this file would have to know its spelling —
        # **and nothing holds the two together.** Change the wording over there and this
        # cannot say it is wrong; it **waits forever** — worse than a wrong value, because
        # nothing on screen says what is being waited for. `disabled` says the same fact
        # without any wording at all.
        page.wait_for_function("el => !el.disabled", arg=go, timeout=TIMEOUT_MS)
        out = block.query_selector("pre.out, .out")
        text = (out.inner_text() if out else "").strip()
        if not text:
            said.append(f"block {i} produced nothing")
            continue
        # **The mark comes first.** A line the page wrote after catching an exception
        # catches here.
        for line in (out.query_selector_all(ERROR_CLASS) if out else []):
            said.append(f"block {i} — {line.inner_text().strip().splitlines()[0][:120]}")
        for bad in BAD:
            if bad in text:
                said.append(f"block {i} — {text.splitlines()[0][:120]}")
                break

    if pressed == 0:
        # **Running 0 of them and seeing green is the worst outcome available.**
        said.append("there was not one JS block to press — the selector may be stale")
    return not said, pressed, said


def main(argv):
    runner.require_fresh_dist()
    port, stop = runner.serve(ROOT)
    rows = []
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("pageerror",
                    lambda e: rows.append((False, 0, [f"page exception: {e}"])))
            for rel in PAGES:
                ok, pressed, said = run_page(page, f"http://127.0.0.1:{port}{rel}")
                rows.append((rel, ok, pressed, said))
    finally:
        stop()

    bad = 0
    for rel, ok, pressed, said in rows:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {rel} — {pressed} JS blocks")
        for line in said:
            print(f"      {line}", file=sys.stderr)
        if not ok:
            bad += 1
    print(f"{len(rows) - bad} of {len(rows)} pages passed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    raise SystemExit(main(sys.argv[1:]))
