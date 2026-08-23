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

# **What is pressed is a decision; what is not pressed used to be a silence.** A page
# absent from the list above is not reported as unwatched — it is not reported at all,
# which on screen is indistinguishable from a page that passed. `coverage()` below ends
# that: it counts the pages on disk that carry a JS block, subtracts what is pressed and
# what is declined, and the run prints the remainder by name every time.
#
# **The numbers are not written here, because the ones that were written here were wrong
# within a day.** This comment used to say "thirty-two pages carry a JS block" and
# "about seventy-nine blocks". The truth is thirty-four and eighty-four: the thirty-two
# was `pressed + unwatched`, silently dropping the two pages declined three lines below —
# a sentence that says *pages that carry a JS block* has to count the ones skipped on
# purpose too — and the seventy-nine was estimated rather than counted. A sister session
# found it by counting the directory instead of re-deriving from the comment.
#
# So the count is taken from the directory at run time. `declined` versus `wants
# reviewing` is the vocabulary `tests/torch_gap.py` and `tests/ts_axis.py` already use on
# their own gaps, and it is the distinction that matters: not doing a thing is a choice,
# not saying why is the defect.

# **Pages left out on purpose**, and the reason each is left out. Anything with a JS
# block that is in neither this nor `PAGES` comes out of the run as *wants reviewing*.
DECLINED = {
    "/site/tutorials/04-image-classifier.html":
        "downloads CIFAR and runs convolutions for several epochs",
    "/site/ko/tutorials/04-image-classifier.html":
        "downloads CIFAR and runs convolutions for several epochs",
}


def coverage():
    """`(pressed, declined, unwatched, problems)`, each a list of `(path, blocks)`.

    Asked of the directory, never of the comment above. The pages are found by the same
    mark the runner presses on — a `data-lang="js"` block — so a page cannot be in this
    count and out of the run's reach.
    """
    found = {}
    for path in sorted((ROOT / "site").rglob("*.html")):
        text = path.read_text(encoding="utf-8")
        n = text.count('data-lang="js"')
        if n:
            found["/" + path.relative_to(ROOT).as_posix()] = n

    problems = [f"{rel} is listed to be pressed and has no JS block on disk"
                for rel in PAGES if rel not in found]
    problems += [f"{rel} is declined and has no JS block on disk"
                 for rel in DECLINED if rel not in found]
    problems += [f"{rel} is both pressed and declined" for rel in PAGES if rel in DECLINED]

    pressed = [(r, n) for r, n in found.items() if r in PAGES]
    declined = [(r, n) for r, n in found.items() if r in DECLINED and r not in PAGES]
    unwatched = [(r, n) for r, n in found.items() if r not in PAGES and r not in DECLINED]
    return pressed, declined, unwatched, problems


def say_coverage():
    """Prints the three groups. Called before the pressing, so a long run still says it."""
    pressed, declined, unwatched, problems = coverage()
    total = len(pressed) + len(declined) + len(unwatched)
    blocks = sum(n for _, n in pressed + declined + unwatched)
    print(f"{len(pressed)} of {total} pages pressed "
          f"({sum(n for _, n in pressed)} of {blocks} JS blocks)")
    if declined:
        print(f"  declined {len(declined)}")
        for rel, n in declined:
            print(f"      {rel} — {n} blocks — {DECLINED[rel]}")
    if unwatched:
        print(f"  wants reviewing {len(unwatched)} "
              f"({sum(n for _, n in unwatched)} blocks)")
        for rel, n in unwatched:
            print(f"      {rel} — {n} blocks")
    for line in problems:
        print(f"  ! {line}", file=sys.stderr)
    return problems


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
    listing_problems = say_coverage()
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
    # **"12 of 12 passed" under "12 of 34 pressed" reads as the whole set passing.**
    # The word says which twelve, so the two lines cannot be run together by a skimmer.
    print(f"{len(rows) - bad} of {len(rows)} pressed pages passed")
    # A list naming a page that is not there is not a smaller run, it is a wrong one.
    return 0 if bad == 0 and not listing_problems else 1


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    raise SystemExit(main(sys.argv[1:]))
