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
# **The other file that runs checks.** `gpu.yml` holds the census, and for a while this
# file read only that one — so `clipped.py`, which `test.yml` has run on every push, was
# counted as a check nobody runs. It was written down as unwired for a week while it was
# the one making CI red. The headline count stays gpu.yml's, because that census is about
# what the GPU boxes are asked to do; being *run at all* is a different question and this
# is the second place that answers it.
PUSH_WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"

# The three that `gpu.yml` runs directly rather than through npm. They are read off
# its own `run:` steps below rather than listed here, so this is only the pattern.
DIRECT = re.compile(r"^\s*run:\s*(?:uv run .*?python )(tests/browser/\S+\.py)(.*)$")

# `thirteen` — the count is written as a word, so the words are the vocabulary.
WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty-one": 21, "twenty-two": 22, "twenty-three": 23, "twenty-four": 24,
    "twenty-five": 25, "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28,
    "twenty-nine": 29, "thirty": 30, "thirty-one": 31, "thirty-two": 32, "thirty-three": 33,
    "thirty-four": 34, "thirty-five": 35, "thirty-six": 36, "thirty-seven": 37, "thirty-eight": 38,
    "thirty-nine": 39, "forty": 40, "forty-one": 41, "forty-two": 42, "forty-three": 43, "forty-four": 44, "forty-five": 45, "forty-six": 46, "forty-seven": 47, "forty-eight": 48,
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


# **The count is said four times and only one of them was read.** The headline was gated
# from the start; the three numbers in the rows under it were not, and they drifted apart
# — on 2026-09-10 the file said forty-three entry points, forty-one run automatically and
# twenty-nine below the four above, which cannot all be true of one list. The first of
# them (4fba4fc) was consistent: thirteen entry points, thirteen automatically, nine
# below, seven correctness — 4 + 9 = 13 and 7 + 2 = 9. So the relationships are exact and
# a test can hold them, which is cheaper than asking the next reader to notice.
_HERE = re.compile(r"^#\s+here\s+(.*?)^#\s+not here", re.M | re.S)
_CORRECTNESS = re.compile(r"\(([a-z-]+) more correctness checks\)")
_MEASUREMENTS = re.compile(r"^#\s+(\S.*?)\s+\(measurements, not checks\)", re.M)
_AUTOMATICALLY = re.compile(r"any of the ([a-z-]+) automatically")
_BELOW = re.compile(r"so the ([a-z-]+) below")


def _row_items(segment):
    """The names on a row, which is written as `a · b · c` and may wrap across lines."""
    flat = " ".join(line.lstrip("# ").strip() for line in segment.splitlines())
    return [x.strip() for x in flat.split("·") if x.strip()]


def test_the_three_counts_under_the_headline_are_the_same_list_counted():
    text = WORKFLOW.read_text(encoding="utf-8")
    total = WORDS[re.search(r"\*\*([a-z-]+) entry points that need a browser\*\*", text).group(1)]
    here = len(_row_items(_HERE.search(text).group(1)))
    correctness = WORDS[_CORRECTNESS.search(text).group(1)]
    measurements = len(_row_items(_MEASUREMENTS.search(text).group(1)))
    automatically = WORDS[_AUTOMATICALLY.search(text).group(1)]
    below = WORDS[_BELOW.search(text).group(1)]

    said = (f"    headline {total} · here {here} · correctness {correctness} · "
            f"measurements {measurements} · automatically {automatically} · below {below}")
    assert here + below == total, (
        "the row above and the row below do not add up to the headline:\n" + said)
    assert correctness + measurements == below, (
        "the two `not here` rows do not add up to the count that names them:\n" + said)
    assert automatically == total, (
        "`nothing runs any of the N automatically` is the whole list, and this N is not "
        "it:\n" + said + "\n  It went wrong by being edited with a delta rather than "
        "recounted — the same fault the book records: a number that appears twice is a "
        "number that will disagree with itself.")


# **The census counts what is wired, so what is not wired is invisible to it.** Three times
# in one week a check turned out to be run by nobody: five nightly-only probes the count did
# not reach (930d645), and `tests/browser/cost.py`, which nothing has invoked all month — and
# running it by hand on 2026-09-11 found `empty_cache()` followed by a training step
# submitting a destroyed buffer. A check nobody runs is not a check, and the file being
# present is what makes it look like one.
#
# So this reads the directory instead of the wiring: every runnable thing under
# `tests/browser/` is either wired to something that runs it, or written down here with what
# kind of thing it is. A new probe that nobody schedules fails this until somebody says why.
KIND = {
    "check": "has a verdict — being unwired is a debt, not a decision",
    "tool": "produces something; there is nothing for it to pass or fail",
    "diagnostic": "answers a question and asserts nothing",
    "on-demand": "only means anything when something else is already failing",
}

UNWIRED = {
    "bench.py": ("tool", "the timed training step behind `run.py --bench`, a measurement rather than a verdict"),
    "export_resnet18.py": ("tool", "writes the ResNet-18 weights the inference comparison shares — run when they change"),
    "features_probe.py": ("diagnostic", "prints what this adapter offers; there is no right answer to fail"),
    "readback_probe.py": ("diagnostic", "how long one value takes to come back from bare WebGPU, no borch in it"),
    "why_failing.py": ("on-demand", "groups already-failed golden cases by reason; nothing to run when they pass"),
}

# **Zero, and that is the point of the number rather than the end of it.** Nothing above is
# a check now: what is left produces something, prints something, or only means anything
# when another check has already failed. The four that were here went out one at a time —
# `cost.py` through the nightly's `run.py --cost` row, `platform_claims.py` and
# `fold_probe.py` and `sync_probe.py` as nightly rows of their own — and `clipped.py` turned
# out to have been run by `test.yml` the whole time and only looked unwired because this
# file was not reading that workflow.
#
# A check arriving here again is not a failure; writing one down without lowering this is.
UNWIRED_CHECKS = 0


def _flag_entry_points():
    """{module file: the `run.py` flags that load it} — `cost.py` is run as `run.py --cost`.

    **A module reached only through a flag has no path of its own in any command**, so the
    path match below cannot see it however it is wired. That is not hypothetical: `cost.py`
    has no `__main__` on purpose, because the page loads it, and on 2026-09-11 there was no
    way to wire it that this file would recognise.

    The pairing is already written in `run.py` and is read rather than listed here, so a new
    flag needs no edit: `if args.<flag> and not args.probe:` is followed by the
    `import <module>, importlib` that flag builds. Kept to names that are files in this
    directory — `--resnet` builds its probe from `tests/resnet.py` by path, which is outside
    it, and the same line imports the standard library's own `importlib`.
    """
    text = (ROOT / "tests" / "browser" / "run.py").read_text(encoding="utf-8")
    pairs = {}
    for hit in re.finditer(r"if args\.(\w+) and not args\.probe:", text):
        after = re.search(r"import (\w+), importlib", text[hit.end():hit.end() + 800])
        if not after:
            continue
        module = f"{after.group(1)}.py"
        if (ROOT / "tests" / "browser" / module).exists():
            pairs.setdefault(module, set()).add(hit.group(1))
    return pairs


def _module_entry_points():
    """The module files `run.py` can load — the keys of the pairing above."""
    return {name[:-3] for name in _flag_entry_points()}


def _runnable():
    """Everything under tests/browser that can be run: a `__main__`, or a module run.py loads."""
    out = set()
    for path in (ROOT / "tests" / "browser").glob("*.py"):
        if "__main__" in path.read_text(encoding="utf-8"):
            out.add(path.name)
    out |= {f"{name}.py" for name in _module_entry_points()}
    return out


def _commands():
    """Every place a check is invoked from, one command per entry.

    Line by line for the two files, because both write one invocation per line and the pair
    of tokens the flag rule looks for has to be in **the same** command — `run.py` on one
    nightly row and `--cost` on another is two commands and neither runs cost.py.
    """
    scripts = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["scripts"]
    out = list(scripts.values())
    for path in (ROOT / "tests" / "browser" / "nightly.py", WORKFLOW, PUSH_WORKFLOW):
        out += path.read_text(encoding="utf-8").splitlines()
    return out


def _is_wired(name):
    """Named by an npm script, a nightly row, or one of gpu.yml's own run steps.

    A module that only a flag reaches is wired when one command carries **both** the runner's
    path and one of that module's flags. Both, because `--bench` on its own also appears in
    `npm run bench:kernels` (`kernel_bench.py --bench=mm`), which does not run `bench.py` —
    a matcher that is politely wrong is worse here than one that is narrow.
    """
    rel = f"tests/browser/{name}"
    commands = _commands()
    if any(rel in command for command in commands):
        return True
    flags = _flag_entry_points().get(name)
    if flags:
        runner = "tests/browser/run.py"
        return any(runner in c and any(f"--{flag}" in c for flag in flags) for c in commands)
    return False


def test_every_runnable_probe_is_wired_or_written_down():
    """**A file that looks like a check and is run by nobody reads as covered.**"""
    unwired = {n for n in _runnable() if not _is_wired(n)}
    missing = sorted(unwired - set(UNWIRED))
    assert not missing, (
        "these can be run and nothing runs them, and they are not written down:\n  "
        + "\n  ".join(missing)
        + "\n\n  Wire it into package.json and the nightly, or add it to UNWIRED with the kind"
          "\n  of thing it is. Leaving a probe unwired is meant to be a sentence somebody wrote.")


def test_the_written_down_ones_are_still_there_and_still_unwired():
    """The table rots in two directions and both are silent."""
    runnable = _runnable()
    gone = sorted(n for n in UNWIRED if n not in runnable)
    assert not gone, f"UNWIRED names something that is no longer runnable: {gone}"
    now_wired = sorted(n for n in UNWIRED if _is_wired(n))
    assert not now_wired, (
        f"UNWIRED still lists {now_wired}, which something runs now — take it out, and if it "
        "was a\n  check, lower UNWIRED_CHECKS by one in the same commit.")
    wrong = sorted(n for n, (kind, why) in UNWIRED.items() if kind not in KIND or not why.strip())
    assert not wrong, f"these need a kind from {sorted(KIND)} and a reason: {wrong}"


def test_the_number_of_checks_nobody_runs_is_the_number_that_was_argued_for():
    """**Every one of these is a question somebody wrote down and nothing asks.**

    At zero this asks a narrower question — that nothing arrives quietly. The way to change
    it downwards is to wire one in; the way to change it upwards is to explain, in the
    commit, what is being given up.
    """
    checks = sorted(n for n, (kind, _) in UNWIRED.items() if kind == "check")
    assert len(checks) == UNWIRED_CHECKS, (
        f"{len(checks)} checks are run by nobody and the number written down is "
        f"{UNWIRED_CHECKS}:\n  " + "\n  ".join(checks))
