"""Extracts the cases that have not been asked yet.

    uv run --with playwright python borch-ts/test/missing.py

The number the runner counts (unasked cases) is **a count alone**, so knowing what they
are needs the names. Rather than skimming what is left by eye and concluding "there were
none", it comes back as a list.
"""

import collections
import json
import pathlib
import sys

import run as runner

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv):
    doc = json.loads((ROOT / "tests" / "golden.json").read_text(encoding="utf-8"))
    report = runner.run()
    if "error" in report:
        print(f"could not run: {report['error']}", file=sys.stderr)
        return 1
    # The runner knows only the registered names. Subtracting them from the whole golden
    # leaves what remains.
    asked = set(report.get("asked", []))
    missing = [n for n in doc["cases"] if n not in asked]

    by_group = collections.Counter(
        n.split("::")[0] if "::" in n else "(no prefix)" for n in missing)
    print(f"{len(missing)} not asked")
    for group, count in by_group.most_common():
        print(f"  {count:4d}  {group}")
    if "--names" in argv:
        for n in missing:
            print(f"    {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
