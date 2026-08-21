"""Counts the names the golden cases **actually call** in the library.

    uv run python tests/case_surface.py            # module functions (L.exp …)
    uv run python tests/case_surface.py --methods  # methods (x.sum() …)

The question to ask when starting a new implementation is not "how wide is
torch's surface" but **"what does this table call".** That list is the condition
for passing, and counting it decides where to start — attaching the most-called
first grows the number of passing cases quickest.

Counting by **number of cases** rather than by number of calls is the point. A
hundred calls to `mul` inside one case opens one case, and one call each across
ten cases opens ten.
"""

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# Places like `L.exp(...)` and `L.nn.functional.relu(...)`.
MODULE = re.compile(r"\bL\.((?:[a-z_][a-z0-9_]*\.)*[a-zA-Z_][a-zA-Z0-9_]*)")
# `x.sum()` and `a.masked_select(...)` — the receiver is conventionally one
# character or short.
METHOD = re.compile(r"\b[a-z][a-z0-9_]{0,3}\.([a-z_][a-z0-9_]*)\s*\(")


def main(argv):
    text = (ROOT / "cases.py").read_text(encoding="utf-8")
    want_methods = "--methods" in argv
    pattern = METHOD if want_methods else MODULE

    # A case is not read as one line — many span several, so everything from the
    # line holding a case name to the next case name is read as one block.
    seen = collections.Counter()
    for name in pattern.findall(text):
        seen[name] += 1

    kind = "methods" if want_methods else "module functions"
    print(f"{kind} the golden cases call — {len(seen)} distinct")
    for name, n in seen.most_common(60):
        print(f"  {n:4d}  {name}")
    if len(seen) > 60:
        print(f"  … and {len(seen) - 60} more below that")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
