"""**Groups golden failures by the helper that built the cases.**

A flat list of failures says how many. It does not say *how many causes*, and those are
different numbers — the binding golden once reported 94 failures of which about 64 came
from a single wrong argument seat, because every `opt::` case trains through one
`CrossEntropyLoss()`.

Reading that took a person opening `cases.py` and noticing that `trained` and `lr_trace`
step the same twenty-four optimizers and differ in exactly one thing. It worked twice
and it is not repeatable by looking harder: what made it decisive was **the helper with
no failures**, not the one with many.

## Where the grouping comes from

Nobody writes it down. A case is a closure, and the helper it calls is a *free variable*
of its code object — `trained` is in `co_freevars` for all 57 cases that use it, and
`lr_trace` for its 10. So the grouping is a property of how the cases were written, not
a table to be kept in step with them.

`co_names` is not enough on its own: it holds globals and attributes, and every helper
in `cases.py` is defined inside the function that builds its section, so it arrives as a
closure variable. The first version of this used `co_names` and grouped nothing.

## What it refuses to say

**When no helper stands out it prints nothing.** A grouping that always produces a
ranking would hand a reader a most-blamed helper on a run whose failures are unrelated,
and a plausible wrong lead costs more than no lead. The rule is a contrast between
siblings — a helper whose cases all failed, printed beside one under the same prefix
whose cases all passed — and when there is no contrast there is no output.
"""

_MIN_FAILED = 3      # below this, "all of them failed" is a coincidence
_MIN_CLEAN = 3       # a control with fewer cases than this is not a control


def _helpers(fn):
    """The machinery a case's closure mentions — **helpers, not the data they are fed.**

    Two sources, treated differently because only one can be inspected:

    - **Free variables**, kept only when the cell holds something callable. A case
      closes over its inputs as well as its helpers, and `yin`, `img` and `chans`
      grouped exactly as strongly as `trained` did — every `opt::` case shares one
      input tensor, so "all 23 cases through `yin` failed" is true and says nothing
      about a cause. The cell contents settle it: a helper is called, an input is not.
    - **`co_names`** — globals and attribute names — kept whole. These cannot be
      resolved from here, and the useful ones are attributes: the sharpest line the
      grouping has produced is `every case through CrossEntropyLoss failed`, and
      `CrossEntropyLoss` reaches the code object only as an attribute of `L.nn`.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        return frozenset()
    live = set(code.co_names)
    cells = getattr(fn, "__closure__", None) or ()
    for name, cell in zip(code.co_freevars, cells):
        try:
            if callable(cell.cell_contents):
                live.add(name)
        except ValueError:                                       # an empty cell
            continue
    return frozenset(live)


def _prefix(name):
    """`opt::SGD/손실` → `opt::`. The section a case belongs to."""
    return name.split("::", 1)[0] + "::" if "::" in name else ""


def _name_of(failure, names):
    """The case name inside a failure line, by longest match.

    The runner formats a failure as `<name>: <why>` and **a name may contain `: `** —
    `sched::ReduceLROnPlateau(threshold_mode=abs)` does not, but nothing stops one. So
    the names are matched rather than the line split, longest first.
    """
    for name in names:
        if failure == name or failure.startswith(name + ":"):
            return name
    return None


def group(failures, cases):
    """`(helper, failed, total, prefix)` rows for helpers whose cases all failed.

    `failures` is the runner's list of failure lines; `cases` is `[(name, fn), …]`.
    """
    by_name = {name: fn for name, fn in cases}
    ordered = sorted(by_name, key=len, reverse=True)
    failed = {n for n in (_name_of(f, ordered) for f in failures) if n}

    uses = {}
    for name, fn in cases:
        for helper in _helpers(fn):
            uses.setdefault(helper, []).append(name)

    rows = []
    for helper, users in uses.items():
        bad = [u for u in users if u in failed]
        if len(bad) < _MIN_FAILED or len(bad) != len(users):
            continue
        rows.append((helper, len(bad), len(users), _prefix(bad[0])))
    return sorted(rows, key=lambda r: -r[1])


def controls(rows, failures, cases):
    """Helpers under the same prefix as a blamed one whose cases **all passed**.

    This is the half that does the work. A helper with many failures is a list of
    symptoms; a sibling with none, sharing the machinery, is what turns it into one
    cause. Both were needed by hand, twice.
    """
    by_name = {name: fn for name, fn in cases}
    ordered = sorted(by_name, key=len, reverse=True)
    failed = {n for n in (_name_of(f, ordered) for f in failures) if n}
    wanted = {prefix for _, _, _, prefix in rows}
    blamed = {helper for helper, _, _, _ in rows}

    uses = {}
    for name, fn in cases:
        if _prefix(name) not in wanted:
            continue
        for helper in _helpers(fn):
            uses.setdefault(helper, []).append(name)

    out = []
    for helper, users in uses.items():
        if helper in blamed or len(users) < _MIN_CLEAN:
            continue
        if any(u in failed for u in users):
            continue
        out.append((helper, len(users), _prefix(users[0])))
    return sorted(out, key=lambda r: -r[1])


def report(failures, cases):
    """The lines to print, or `[]` when there is no contrast worth showing."""
    rows = group(failures, cases)
    if not rows:
        return []
    clean = controls(rows, failures, cases)
    if not clean:
        # A blamed helper with no clean sibling is a helper that failed, which the
        # failure list already said. **The contrast is the finding**, so without one
        # this prints nothing rather than a ranking.
        return []

    out = ["", "grouped by the helper that built the cases:"]
    for helper, bad, total, prefix in rows[:6]:
        out.append(f"  every case through `{helper}` failed — {bad}/{total} under {prefix}")
    out.append("  and, under the same prefix, sharing the machinery:")
    for helper, total, prefix in clean[:6]:
        out.append(f"    `{helper}` — {total}/{total} passed")
    out.append("  A helper that fails whole, beside one that passes whole, is one cause")
    out.append("  rather than many. That difference was read by hand twice before this.")
    return out
