"""**The site check has to say what it did not look at, and its exemption has to bite.**

`borch-ts/test/lessons.py` presses the run buttons on the site's runnable JS blocks. It
is the only thing here that executes what a reader executes, and it needs a browser, so
it runs when a person runs it.

Run on `ce93871` it said:

    12 of 34 pages pressed (40 of 134 JS blocks)
    wants reviewing 20 (84 blocks)
    10 of 12 pressed pages passed

— and **exited green.** Twenty pages and eighty-four blocks were in neither list, the
line naming them went to stdout, and the exit code came from the pages that were
pressed. So the one page found broken was not the one page that broke; it was the one
that broke *among the third being looked at.* (Pressed afterwards, the other twenty were
sound — the finding was about the check, not the pages.)

This file holds two properties of that check, from Python, without a browser.

## What is held

- **Every page with a runnable JS block is decided about** — pressed, or declined with
  a reason. Not a count: the site session's branch moves eight pages into `PAGES`, and
  a frozen twenty would have to be rewritten by the very work it asks for. A number
  rewritten to make a run pass has stopped being evidence.

- **The word-net exemption is narrow, and keyed on something that can match.** Two
  pages are exempt from `BAD` because their subject *is* the error message —
  `08-debugging` throws on purpose in every block, so a correct line reads
  `RuntimeError: …` and the net catches it for working. The structural gate
  (`div.err`) still applies there, which is what keeps the exemption from being a hole.

## What a green run here does not say

- **Not that the pages run.** That needs the browser and `npm run lessons:ts`.
- **Not that the exemption is right for those two pages.** It says the mechanism can
  fire and has not spread.
"""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LESSONS = ROOT / "borch-ts" / "test" / "lessons.py"


def _lessons():
    if str(LESSONS.parent) not in sys.path:
        sys.path.insert(0, str(LESSONS.parent))
    spec = importlib.util.spec_from_file_location("_lessons", LESSONS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_lessons", mod)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(not LESSONS.exists(), reason="no lessons.py")


def test_an_undecided_page_reaches_the_exit_code():
    """**Printing it was already what the check did.**

    `wants reviewing 20` went to stdout and the exit code came from the pages that were
    pressed, so the reader saw the line above a zero and read the zero. The group has to
    become a problem.

    **This asks whether the mechanism fires, not whether the pages are decided.** The
    second question belongs to `lessons.py` itself, which now exits 1 and names them —
    and to the branch that is moving eighteen of them into `PAGES`. Asserting it here
    as well would turn the whole suite red to hold somebody else's outstanding work,
    and a suite red for that reason teaches people to run it less.
    """
    mod = _lessons()
    _p, _d, unwatched, _problems = mod.coverage()
    if not unwatched:
        pytest.skip("every page is decided about — the mechanism has nothing to report")
    assert any("neither pressed nor declined" in p for p in mod.say_coverage()), (
        "`say_coverage` lists the unwatched pages and does not return them as a\n"
        "  problem, so `lessons.py` exits 0 with part of the site unexamined.")


def test_the_word_net_exemption_is_keyed_on_something_that_can_match():
    """**An exemption compared against the wrong string is inert, not wrong.**

    `run_page` is handed `http://127.0.0.1:<port>/site/...` and the exempt set holds
    `/site/...`. The first version compared the set against the URL, which the port
    makes different every run — so it could never have matched, and nothing would have
    said so. The set is checked against what `coverage()` reports, which is the same
    spelling the runner is now given.
    """
    mod = _lessons()
    pressed, declined, unwatched, _ = mod.coverage()
    known = {rel for rel, _ in pressed + declined + unwatched}
    stray = sorted(set(mod.WORD_NET_EXEMPT) - known)
    assert not stray, (
        "exempt from the word net and not a page on disk: " + ", ".join(stray) + "\n\n"
        "  Spelled as the runner receives it — `/site/learn/...`, not a URL and not a\n"
        "  filesystem path. A key that matches nothing exempts nothing and says so\n"
        "  never.")


def test_the_exemption_has_not_spread():
    """**Narrow is the whole justification**, so narrow is what is held.

    The reason those two pages are exempt is that their subject is the error message.
    That is true of one lesson. A third name appearing means either a new lesson of
    the same kind — say so — or the net being turned off wherever it is inconvenient,
    which is how a net becomes decoration.
    """
    mod = _lessons()
    assert len(mod.WORD_NET_EXEMPT) <= 2, (
        f"{len(mod.WORD_NET_EXEMPT)} pages are exempt from the word net:\n  "
        + "\n  ".join(sorted(mod.WORD_NET_EXEMPT)) + "\n\n"
        "  Two is `08-debugging` in both languages. Anything more wants a sentence\n"
        "  saying why that page's correct output contains the words too.")


def test_the_structural_gate_still_applies_where_the_net_does_not():
    """**The exemption must not reach `div.err`.**

    The net is a net; `div.err` is the gate. On the exempt pages the gate is the only
    thing left, so an exemption that dropped both would turn two pages into
    always-green — the exact shape this file exists to stop, applied to itself.

    Read from the source: the `ERROR_CLASS` loop must not sit under the exemption.

    **The guard is found by the line that branches on it**, not by the name. The first
    version searched for `WORD_NET_EXEMPT` anywhere in `run_page` and matched the
    sentence in its docstring — which sits above the `div.err` scan, so the check
    failed while the code was right. A source check that cannot tell prose from code
    reports on the prose.
    """
    src = LESSONS.read_text(encoding="utf-8")
    body = src.split("def run_page(", 1)[-1]
    guard = body.index("if rel not in WORD_NET_EXEMPT:")
    mark = body.index("query_selector_all(ERROR_CLASS)")
    assert mark < guard, (
        "the `div.err` scan now happens inside the word-net exemption, so the two\n"
        "  exempt pages have no check left at all. The mark comes first, always.")
