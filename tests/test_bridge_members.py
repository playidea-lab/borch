"""Every borch.ts member the binding reaches for **by name** still exists.

`test_binding_arguments.py` walks *call sites* — a Python signature against the
borch.ts constructor it hands its arguments to. That is one of the two ways this
binding touches the other library. The other is a **property read**:

    if self._m.kind == "LSTM":

which asks a Pyodide proxy for a field. A field that is not there is not an error
on either side. JavaScript answers `undefined`, Python compares it to `"LSTM"`,
the comparison is false, and the layer builds the wrong number of gates.

**It happened twice in one afternoon and neither time did anything but the browser
say so.** `RNNBase` took torch's `mode` in place of `kind`, and every recurrent
case failed at once — 6/6 under one prefix, which the runner's own message calls
one cause rather than many. The signature axis cannot see it (it compares
declarations), `tsc` cannot see it (the read is in Python), and the call-site check
cannot see it (nothing is being called).

**The pool is `nn` and `optim`, and narrowing it is what gave this teeth.** Written
first against every name borch.ts declares anywhere, it passed with the defect put
back — `indexing.Slice` has a `kind`, so the renamed-away `RNNBase.kind` still looked
present. A check that asserts something weaker than its name is the shape this
repository spent a day naming, and this file nearly shipped as one; every `self._m`
here holds a layer or an optimizer, so those two modules are the honest pool.

**What this check does not do**: it does not know *which* class each `_m` holds, so a
name that moved between two `nn` classes still passes. Resolving that needs the
construction site, and the construction sites are factories. What it does catch, and
was measured catching, is a member renamed away with a reader left behind.

**What retires this line**: nothing yet — it should grow class resolution rather than
be deleted.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINDING = sorted((ROOT / "borch_webgpu").glob("*.py"))
API = ROOT / "site" / "assets" / "api.json"

#: Read off a proxy rather than declared in borch.ts. Each needs a reason.
NOT_IN_BORCH_TS: dict[str, str] = {}


def _declared() -> set[str]:
    api = json.loads(API.read_text(encoding="utf-8"))
    names: set[str] = set()
    for mod in api["modules"]:
        if mod["name"] not in ("nn", "optim"):
            continue
        for sym in mod["symbols"]:
            names.add(sym["name"])
            names.update(m["name"] for m in sym["members"])
    return names


def _read() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in BINDING:
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for name in re.findall(r"self\._m\.([A-Za-z_][A-Za-z0-9_]*)", line):
                found.setdefault(name, []).append(f"{path.name}:{line_no}")
    return found


def test_every_member_the_binding_reads_is_declared_in_borch_ts():
    declared = _declared()
    missing = {n: w for n, w in _read().items()
               if n not in declared and n not in NOT_IN_BORCH_TS}
    assert not missing, (
        "the binding reads members borch.ts does not declare:\n    "
        + "\n    ".join(f"{n} — {', '.join(w)}" for n, w in sorted(missing.items()))
        + "\n\nA missing field is `undefined` across the bridge, which raises nothing "
          "and compares false. Rename the reader, or add it to `NOT_IN_BORCH_TS` with "
          "the reason it is not declared.")


def test_the_reader_scan_finds_something():
    """**A parser that finds nothing holds no contracts while passing.**

    The sister file records producing that twice. The floor is deliberately well
    under the current count rather than at it — a number pinned to today's total
    fails on the next honest edit and teaches nobody anything.
    """
    found = _read()
    assert len(found) >= 8, (
        f"only {len(found)} `self._m.<name>` reads were found across "
        f"{len(BINDING)} binding files — the scan broke, not the binding.")


def test_no_excuse_outlives_its_member():
    stale = {n for n in NOT_IN_BORCH_TS if n in _declared()}
    assert not stale, (
        f"`NOT_IN_BORCH_TS` names members borch.ts declares after all: {sorted(stale)}")
