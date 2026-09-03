"""The census of checks that need a browser, held against the tree.

`.github/workflows/gpu.yml` counts them — *thirteen entry points that need a browser
and this file names four* — and says why the count is written rather than automated:
**so that leaving one out is a decision instead of an oversight.** Nothing held the
count, so a fourteenth would arrive unnamed and the sentence would go on saying
thirteen.

That is not hypothetical. The nine the file lists as *not here* were run by hand in one
session and **two of them were red**, each since somebody had done work that made a
check stale:

  · `parity.ts` asserted three refusals that had become implementations — `weight` and
    `posWeight` on the BCE pair, a class weight on `crossEntropy`, `dilation` on
    `MaxPool1d`, `bidirectional` and `projSize` on `RNNBase`. A check watching for a
    throw expires the day the work is done, and this one had already moved once for
    exactly that reason with the lesson written beside it.
  · `scope_escape.py` decided its exit code by matching the head line of a report, the
    report was translated to English, and the runner returned 1 with all twelve checks
    green — with a comment on that very line predicting it.

Both were invisible because nothing schedules these. This file cannot schedule them —
attaching a runner is a person's job, and `gpu.yml` argues the list should stay a list.
What it can do is refuse to let the list fall behind the tree.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "gpu.yml"

# The three that `gpu.yml` runs directly rather than through npm. They are read off
# its own `run:` steps below rather than listed here, so this is only the pattern.
DIRECT = re.compile(r"^\s*run:\s*(?:uv run .*?python )(tests/browser/\S+\.py)(.*)$")

# `thirteen` — the count is written as a word, so the words are the vocabulary.
WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty-one": 21, "twenty-two": 22, "twenty-three": 23, "twenty-four": 24,
    "twenty-five": 25, "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28,
}


def _playwright_scripts():
    """Every npm script that opens a browser. **The tree's own answer.**

    `playwright` in the command is what makes an entry point need one; a script that
    stops needing it leaves this set on its own, which is the point of deriving it
    rather than writing it down twice.
    """
    scripts = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["scripts"]
    return {name: cmd for name, cmd in scripts.items() if "playwright" in cmd}


def _direct_steps():
    """The browser entry points `gpu.yml` invokes itself, as (path, tail) pairs.

    `run.py` appears twice with different `--lib` values and they are two runs of two
    libraries, which is why the tail is part of the identity.
    """
    got = set()
    for line in WORKFLOW.read_text(encoding="utf-8").splitlines():
        found = DIRECT.match(line)
        if found:
            # `--headed` is how the run is made, not which run it is: the same entry
            # point windowed and headless is one entry point.
            got.add((found.group(1), found.group(2).replace("--headed", "").strip()))
    return got


def test_every_browser_entry_point_is_named_in_the_workflow():
    """A new one has to be named — run it, or write down that it is not run.

    Matching is on the npm name (`parity:ts`) **or** on the path the script runs
    (`borch-ts/test/run.py`), because `gpu.yml` names `golden:ts` by its path, in the
    `here` row. Either spelling is a reader finding it.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    missing = []
    for name, cmd in sorted(_playwright_scripts().items()):
        path = next((w for w in cmd.split() if w.endswith(".py")), "")
        if name not in text and (not path or path not in text):
            missing.append(f"{name}  ({cmd})")
    assert not missing, (
        "these need a browser and gpu.yml does not name them:\n  " + "\n  ".join(missing)
        + "\n\n  Nothing schedules the browser checks, so a name absent from that census "
          "is\n  run by nobody and counted by nobody. Add it to the `here` or the "
          "`not here`\n  row — leaving one out is meant to be a decision.")


def test_the_written_count_is_the_number_of_entry_points():
    """**The number in the prose against the number in the tree.**

    `gpu.yml` says the count is what a reader needs — *not a longer command but the
    number*. A number nothing checks is the one kind that goes wrong quietly, which is
    the fault this whole file is about.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    said = re.search(r"\*\*([a-z-]+) entry points that need a browser\*\*", text)
    assert said, ("gpu.yml no longer states the count in the form this reads.\n"
                  "  Expected: **<word> entry points that need a browser**")
    written = WORDS.get(said.group(1))
    assert written is not None, (
        f"'{said.group(1)}' is not a number word this knows — add it to WORDS.")

    counted = len(_playwright_scripts()) + len(_direct_steps())
    assert written == counted, (
        f"gpu.yml says {said.group(1)} ({written}) browser entry points; the tree has "
        f"{counted}\n"
        f"    {len(_playwright_scripts())} npm scripts that open a browser\n"
        f"    {len(_direct_steps())} the workflow runs itself: "
        + ", ".join(sorted(f"{p} {t}".strip() for p, t in _direct_steps()))
        + "\n\n  Update the word and the two rows under it in the same commit.")
