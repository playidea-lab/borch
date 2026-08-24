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
    # **The two the word net was catching for working.** Added here rather than to
    # `DECLINED`, and that is the point of the exemption above: declining them would
    # take five blocks that call the library out of the check to fix a complaint about
    # wording. Pressed, `div.err` still guards them.
    #
    # The other eighteen undecided pages are being added on the branch that holds the
    # site, which is also editing this list — two branches rewriting one list is how
    # the last two rebases went. These two are here because the exemption is, and an
    # exemption for a page nobody presses is inert.
    "/site/learn/08-debugging.html",
    "/site/ko/learn/08-debugging.html",
    "/site/tutorials/07-signals-fft.html",
    "/site/ko/tutorials/07-signals-fft.html",
    "/site/tutorials/08-attention.html",
    "/site/ko/tutorials/08-attention.html",
    "/site/tutorials/09-autoencoder.html",
    "/site/ko/tutorials/09-autoencoder.html",
    "/site/tutorials/10-least-squares.html",
    "/site/ko/tutorials/10-least-squares.html",
    "/site/learn/01-tensors.html",
    "/site/ko/learn/01-tensors.html",
    "/site/learn/02-autograd.html",
    "/site/ko/learn/02-autograd.html",
    "/site/learn/03-modules.html",
    "/site/ko/learn/03-modules.html",
    "/site/learn/04-training.html",
    "/site/ko/learn/04-training.html",
    "/site/learn/05-cnn.html",
    "/site/ko/learn/05-cnn.html",
    "/site/learn/07-data.html",
    "/site/ko/learn/07-data.html",
    "/site/tutorials/02-from-scratch.html",
    "/site/ko/tutorials/02-from-scratch.html",
    "/site/tutorials/06-char-rnn.html",
    "/site/ko/tutorials/06-char-rnn.html",
    "/site/python.html",
    "/site/ko/python.html",
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
        # **`wants reviewing` was printed and then thrown away.** It went to stdout,
        # the exit code came from `bad` and `problems` alone, and a run covering a
        # third of the pages ended green — measured on `ce93871`: *12 of 34 pages
        # pressed (40 of 134 JS blocks)*, twenty pages and eighty-four blocks in this
        # group, and the command said nothing but success.
        #
        # It was the same shape three times over in this repository's own week: an
        # instrument that reports on what it looked at and is silent about what it
        # did not. A page can leave `PAGES` and nothing turns red; a page can be added
        # to `site/` and nothing turns red. **The only state that cannot arise by
        # accident is every page being decided about**, and deciding is cheap — a name
        # in `PAGES` or a sentence in `DECLINED`.
        #
        # **A count is deliberately not pinned here.** A frozen twenty has to be
        # rewritten the moment somebody adds eight pages to `PAGES`, which makes the
        # number an obstacle to the very work it is asking for, and a number rewritten
        # to make a run pass stops being evidence. The direction is what is held: this
        # group empties, and it never grows silently.
        problems = problems + [
            f"{len(unwatched)} page(s) with runnable JS are neither pressed nor "
            f"declined — put each in `PAGES` or give it a reason in `DECLINED`"]
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

# **What the list costs.** It went from twelve pages to thirty-eight, which is every page
# on the site that carries a JS block bar the four above. Measured headed on a real
# adapter, twenty of them took about four minutes; a software adapter is roughly a minute
# and a half per page, so the whole list is eight minutes against fifty. The price is paid
# by a person who typed the command rather than by a push, which is the trade that makes
# the long list affordable.
#
# **The twenty that were added had never been pressed once**, and pressing them was worth
# it for what it settled rather than for what it found: eighteen came back green, and the
# two that did not were this file's own word net firing on a page that works — the two
# `08-debugging` pages, which `WORD_NET_EXEMPT` above now waives rather than declines.
# The rename that broke `10-vit` had touched exactly one page, which nobody could say
# before, because two thirds of the site was outside the run.
#
# Tutorial 9 (the autoencoder, 600 convolution steps) is the slowest single page, and the
# first that should move to `DECLINED` if this run ever stops being run.

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

# **Pages whose subject is the error message, so the net catches them for working.**
#
# `08-debugging` is titled *Errors say what torch says*. Every one of its blocks throws
# on purpose through a `shouldFail(label, body)` helper, catches, and prints what came
# out — so a correct line reads `RuntimeError: mat1 and mat2 shapes cannot be multiplied
# (2x3 and 4x5)`, and `Error:` is in the net. Measured in the browser: `div.err` on that
# page is **zero**. The page is right and the net is wrong about it.
#
# **The comment fifteen lines up predicted this** — *a message reworded to start
# `Error:` would fail a page that is working* — and said `10-vit` survived it by luck.
# This is the page where the luck runs out, and it cannot be worded around: the wording
# is the lesson.
#
# **Exempt from the net, not from the run**, and that distinction is the whole of it.
# Declining the page instead would take five blocks that call the library out of the
# check altogether, which is a bigger hole than the one being closed. `div.err` still
# holds the gate here, and that is the check that survives a change of wording or of
# language — the net was only ever there to catch a block that runs wrongly *without*
# throwing.
#
# Two names, both attested by a browser run. `test_lessons_net_exemption_is_narrow`
# holds it to that: a set that grows is a set that has stopped being about this page.
WORD_NET_EXEMPT = {
    "/site/learn/08-debugging.html",
    "/site/ko/learn/08-debugging.html",
}

TIMEOUT_MS = 300_000


def run_page(page, url, rel=None):
    """Press every JS block on one page; return (passed, what to say).

    **`rel` is the site-relative path and `url` is where to fetch it**, and they are
    separate because `WORD_NET_EXEMPT` is keyed on the first. The first version of the
    exemption compared its set against `url` — `http://127.0.0.1:39205/site/learn/…` —
    which matches nothing, so the exemption would have been written, committed and
    silently inert. The port changes every run, so it could never have matched by
    accident either; it would simply never have fired.
    """
    rel = url if rel is None else rel
    page.goto(url)
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
        # The mark above always applies. The word net does not, on the two pages whose
        # subject is the wording itself — see `WORD_NET_EXEMPT`.
        if rel not in WORD_NET_EXEMPT:
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
                ok, pressed, said = run_page(
                    page, f"http://127.0.0.1:{port}{rel}", rel)
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
