"""Groups failed golden cases **by reason.**

    uv run --with playwright python tests/browser/why_failing.py --lib borch_webgpu --headed

While laying a new implementation on, what is needed is not a list of failures but **the
kinds of failure.** Reading 617 lines by eye does not show where to start; grouped by kind it
shows "fixing this one thing opens how many".

`run.py --lib …` already does the comparison, so it is used as it is and only the results are
counted again.

**The probe words below are a contract with two other files.** `golden.py` writes the
`max diff` / `shape … vs …` / `expected …, got …` lines, and `borch_webgpu` raises the
`does not have` messages. When either side's wording moves, this bucketing goes quiet without
failing — a check that catches nothing looks exactly like one that passes. That already
happened once: the regex here was still matching the Korean error text after `borch_webgpu`
had been translated.
"""

import argparse
import collections
import re
import sys

import run as runner

# `name: kind — detail`, or `name: expected …, got …`.
KIND = re.compile(r"^(?P<name>.+?): (?P<rest>.*)$", re.S)
# The second alternative covers all three shapes `borch_webgpu` raises: "borch.ts does not
# have `x`", "borch.ts tensors do not have `x`", and "the borch.ts layer does not have `x`".
# The leading `the ` is optional and matters — the layer message carries it and the other two
# do not, and requiring adjacency to the dash silently dropped that whole message.
ERRORS = re.compile(
    r"AttributeError — (?:'(?P<obj>[^']+)' object has no attribute '(?P<attr>[^']+)'"
    r"|(?:the )?borch\.ts(?: tensors| layer)? do(?:es)? not have `(?P<js>[^`]+)`)")


def bucket(why):
    """Reduces one failure to one phrase. The same cause has to become the same phrase."""
    m = KIND.match(why)
    rest = m.group("rest") if m else why
    hit = ERRORS.search(rest)
    if hit:
        if hit.group("js"):
            return f"not in borch.ts: {hit.group('js')}"
        return f"not in the Python binding: {hit.group('obj')}.{hit.group('attr')}"
    for probe, label in (
        ("max diff", "the values diverged"),
        ("shape", "the shapes diverged"),
        ("expected", "the answers diverged (string)"),
        ("JsException", "borch.ts threw"),
        ("TypeError", "a Python-side type does not fit"),
    ):
        if probe in rest:
            return label
    return rest.split("—")[0].strip()[:40]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="borch_webgpu")
    ap.add_argument("--headed", action="store_true")
    # The other runners read `--headless` straight out of `sys.argv` in
    # `launch._headed`; argparse would refuse it here before that ever runs.
    ap.add_argument("--headless", action="store_true",
                    help="no window, and therefore a software adapter")
    ap.add_argument("--samples", type=int, default=0,
                    help="how many example lines to show per reason")
    args = ap.parse_args(argv)

    result, _ = runner.run(args.lib, args.headed)
    if result.get("error"):
        print("the runner blew up:\n" + result["error"], file=sys.stderr)
        return 1

    bad = result["bad"]
    total = result["total"]
    print(f"{args.lib} — {total - len(bad)}/{total} passed, {len(bad)} failed\n")

    groups = collections.defaultdict(list)
    for why in bad:
        groups[bucket(why)].append(why)

    print("by reason for failure:")
    for label, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(items):4d}  {label}")
        # **A few examples are shown.** Counting kinds alone stops at "RuntimeError, 84" and
        # cannot say whether those 84 are one cause or eight.
        if args.samples:
            for one in items[:args.samples]:
                print(f"          {one[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
