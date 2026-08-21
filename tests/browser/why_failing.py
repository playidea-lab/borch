"""Groups the failing golden cases **by reason.**

    uv run --with playwright python tests/browser/why_failing.py --lib borch_webgpu --headed

What is needed while standing up a new implementation is not a list of failures
but **the kinds of failure.** Reading 617 lines by eye does not show where to
start, and grouping by kind shows "how many does fixing this one open".

`run.py --lib …` already does the comparison, so it is used as it is and only the
result is counted again.
"""

import argparse
import collections
import re
import sys

import run as runner

# `name: kind — description`, or `name: expected …, got …`.
KIND = re.compile(r"^(?P<name>.+?): (?P<rest>.*)$", re.S)

# **This regex was dead and nothing said so.** Its second alternative matched the
# binding's Korean wording (`borch.ts 텐서에 \`x\``), and once those messages became
# English it stopped matching anything — the bucket fell through to the generic
# tail and the tool went on reporting, just without naming the cause. A parser
# that reads another file's wording breaks silently by construction, so both
# shapes the binding actually emits are matched here, and
# `test_the_failure_reader_still_reads` holds them together.
ERRORS = re.compile(
    r"AttributeError — (?:'(?P<obj>[^']+)' object has no attribute '(?P<attr>[^']+)'"
    r"|borch\.ts does not have `(?P<js>[^`]+)`"
    r"|`(?P<mod>[^`]+)` is in borch\.ts as a \*\*module function\*\*)")


def bucket(why):
    """Reduce one failure to one phrase. The same cause has to give the same
    phrase."""
    m = KIND.match(why)
    rest = m.group("rest") if m else why
    hit = ERRORS.search(rest)
    if hit:
        if hit.group("js"):
            return f"absent from borch.ts: {hit.group('js')}"
        if hit.group("mod"):
            return f"a module function, not bridged: {hit.group('mod')}"
        return f"absent from the Python binding: {hit.group('obj')}.{hit.group('attr')}"
    for probe, label in (
        ("max diff", "the values diverged"),
        ("shape", "the shapes diverged"),
        ("expected", "the answers diverged (as strings)"),
        ("JsException", "borch.ts threw"),
        ("TypeError", "a dtype does not fit on the Python side"),
    ):
        if probe in rest:
            return label
    return rest.split("—")[0].strip()[:40]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="borch_webgpu")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--samples", type=int, default=0,
                    help="how many sample lines to show per reason")
    args = ap.parse_args(argv)

    result, _ = runner.run(args.lib, args.headed)
    if result.get("error"):
        print("the runner blew up:\n" + result["error"], file=sys.stderr)
        return 1

    bad = result["bad"]
    total = result["total"]
    print(f"{args.lib} — {total - len(bad)}/{total} passing, {len(bad)} failing\n")

    groups = collections.defaultdict(list)
    for why in bad:
        groups[bucket(why)].append(why)

    print("by reason for failure:")
    for label, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(items):4d}  {label}")
        # **A few samples are shown.** Counting kinds alone stops at
        # "RuntimeError, 84" and leaves it unknown whether those 84 are one cause
        # or eight.
        if args.samples:
            for one in items[:args.samples]:
                print(f"          {one[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
