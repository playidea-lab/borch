"""**The committed `tests/golden.json` is checked by nothing in this suite.**

Both golden tests round-trip. `test_golden.py` dumps into `tmp_path` and checks
against what it just dumped; `test_export_json.py` exports into
`tmp_path_factory` and compares that document's manifest against a freshly
computed hash. Each is self-consistent and neither ever opens the file that is
committed.

That file is not a spare copy. `borch-ts/test/` reads it as the answers for the
TypeScript runner, and `borch-ts/test/cases.ts` states the consequence of drift:
a name there differing from a key here by one character makes that case
disappear silently. `borch-ts/test/run.py` does hold a staleness check — and it
needs a browser, so it fires in CI and not while somebody edits `cases.py`.

## What was open was the step between the two files, not the file

The committed golden is **not** unguarded in general, and saying so would be the
same overreach this file exists to catch. Measured, both ways round:

- rename in `cases.py` **and regenerate** — loud. Renaming one key in the
  committed file fails three tests in `test_case_names.py`, and
  `test_alias_rows.py` reads it too.
- rename in `cases.py` **without regenerating** — silent. Nothing looks at the
  two together, so the table and the answers drift apart in green.

A rename tried against the suite passed green, and that was the second state
rather than a refutation of the first. The second is the dangerous one: a partial
rename is exactly where somebody stops for the night. This closes that half, and
it is one line of arithmetic.

The distinction was found by another session measuring the claim after I stated
it too broadly — which is the defect this file guards against, one level up.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import cases as cases_mod  # noqa: E402

COMMITTED = ROOT / "tests" / "golden.json"


def test_the_committed_golden_covers_the_current_case_table():
    """The names in the committed file are the names the table produces now.

    A failure here means one of two things and they want different fixes:
    a case was added or renamed and the golden was not regenerated, or the
    golden was regenerated from a different table.
    """
    doc = json.loads(COMMITTED.read_text(encoding="utf-8"))
    now = cases_mod.manifest_hash(cases_mod.golden_cases())
    if doc["manifest"] == now:
        return

    have = set(doc["cases"])
    want = {name for name, _ in cases_mod.golden_cases()}
    added = sorted(want - have)
    gone = sorted(have - want)
    detail = []
    if added:
        detail.append(f"in the table and not in the golden ({len(added)}): "
                      + ", ".join(added[:8]))
    if gone:
        detail.append(f"in the golden and not in the table ({len(gone)}): "
                      + ", ".join(gone[:8]))
    if not detail:
        detail.append("the same names in a different order")
    raise AssertionError(
        "the committed tests/golden.json no longer matches tests/cases.py. "
        "The TypeScript runner reads that file, and a name it cannot find is a "
        "case that vanishes without a word.\n  " + "\n  ".join(detail))
