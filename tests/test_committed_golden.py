"""**The committed `golden.json` is what nothing was reading.**

Both existing golden tests round-trip. `test_golden` dumps into `tmp_path` and checks against
what it just dumped; `test_export_json` exports into `tmp_path_factory` and compares that
document's manifest against a freshly computed hash. Neither ever opens
`tests/golden.json` — the one artifact that is actually committed, and the only one an
implementation in another language can read.

`tests/golden.npz` cannot stand in for it: it is gitignored and absent from a fresh checkout,
so the comparison `golden.check` performs has no committed input behind it at all.

So the state this file exists for is **the case table and the committed answers disagreeing
with each other.** Renaming a case in `cases.py` and not re-exporting leaves exactly that, and
it is the natural place to stop — a rename is where someone breaks for the night. Measured:
before this file, a renamed case passed the whole suite green. Re-export as well and three
tests in `test_case_names.py` turn red, so the loud half was already covered and the quiet
half was not.

## Two halves, because they fail separately

**The manifest** covers the case *names*. It is one line of arithmetic and it catches a rename
that was not carried through.

**The values** cover what the cases *return*, which the manifest cannot see. That is the larger
risk in `cases.py`: 711 of the answers are strings, and a handful of them are Korean strings
sitting inside case bodies — `"멈췄다"`, the `"다른 문구 <…>"` family, the `흐름` words. Those
read as prose and are expected values. Anyone translating the file walks straight at them, and
until this file existed nothing local would have said a word.
"""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

_here = pathlib.Path(__file__).resolve().parent
GOLDEN_JSON = _here / "golden.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cases_mod = _load("bt_cases_committed", _here / "cases.py")
golden = _load("bt_golden_committed", _here / "golden.py")


@pytest.fixture(scope="module")
def doc():
    if not GOLDEN_JSON.exists():
        pytest.skip("tests/golden.json is missing — uv run python tests/export_json.py")
    return json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))


def _restore(entry):
    """One JSON entry back to numpy. The same rule as `test_export_json._restore`.

    Kept as its own copy on purpose. Importing that module would make this file pass or fail
    on the freshly built document that its fixture creates, which is the very thing this file
    exists not to do.
    """
    if entry["kind"] == "string":
        return entry["value"]
    values = list(entry["values"])
    if entry["kind"] == "float":
        for i, kind in entry["nonfinite"]:
            values[i] = float(kind)
        arr = np.asarray(values, dtype=np.float64)
    elif entry["kind"] == "int":
        arr = np.asarray(values, dtype=np.int64)
    else:
        arr = np.asarray(values, dtype=bool)
    return arr.reshape(entry["shape"])


def test_the_committed_manifest_matches_the_case_table(doc):
    """A renamed case that was not re-exported has to be caught **here**, not by a browser.

    The manifest is a hash over the case names, so this is one comparison covering all of
    them. It says nothing about what the cases return — that is the test below.
    """
    now = cases_mod.manifest_hash(cases_mod.golden_cases())
    if doc["manifest"] == now:
        return

    # **A hash says something moved; the names say what.** During a rename pass that is the
    # difference between re-exporting blindly and knowing whether the move was the one you
    # meant. The third branch matters as much as the other two: `manifest_hash` folds the names
    # in sequence, so reordering a list changes the hash while both sets stay identical — and
    # without saying so, the message would list nothing and read as a bug in this test.
    have = set(doc["cases"])
    want = {name for name, _ in cases_mod.golden_cases()}
    added, gone = sorted(want - have), sorted(have - want)
    if added or gone:
        detail = ""
        if added:
            detail += (f"\n  in the table and not in the golden ({len(added)}): "
                       + ", ".join(added[:8]) + ("…" if len(added) > 8 else ""))
        if gone:
            detail += (f"\n  in the golden and not in the table ({len(gone)}): "
                       + ", ".join(gone[:8]) + ("…" if len(gone) > 8 else ""))
    else:
        detail = "\n  the same names in a different order — the table was reordered, not edited"

    raise AssertionError(
        "the committed tests/golden.json does not match tests/cases.py.\n"
        f"  committed {doc['manifest'][:16]}…\n"
        f"  computed  {now[:16]}…"
        + detail +
        "\n  Re-export: uv run --with numpy --with torch python tests/golden.py dump\n"
        "             uv run --with numpy python tests/export_json.py\n"
        "  (Until then the two disagree, and every other golden test round-trips through a\n"
        "   temporary file, so none of them would notice.)")


def test_the_committed_answers_still_hold(doc):
    """**What the cases return**, against the answers as committed.

    The names are the manifest's job. This one exists for the values, and above all for the
    string answers — a returned string is an expected value that reads like prose, and there
    are 711 of them.
    """
    core = golden.load_borch()
    inp = cases_mod.golden_inputs()
    missing, bad, checked = [], [], 0
    for name, fn in cases_mod.golden_cases(inp):
        if name.startswith(cases_mod.WEBGPU_PREFIX):
            continue                            # what the core deliberately refuses
        if name not in doc["cases"]:
            missing.append(name)
            continue
        want = _restore(doc["cases"][name])
        got = fn(core)
        got = np.asarray(got) if isinstance(got, str) else cases_mod.to_numpy(got)
        checked += 1
        if isinstance(want, str):
            if want != str(got):
                bad.append(f"{name}: expected {want!r}, got {str(got)!r}")
        elif want.shape != got.shape:
            bad.append(f"{name}: shape {want.shape} vs {got.shape}")
        elif not np.allclose(want, got, atol=1e-4, rtol=1e-4, equal_nan=True):
            bad.append(f"{name}: max diff {np.nanmax(np.abs(want - got)):.2e}")

    assert not missing, (
        "cases the committed golden.json does not carry:\n  " + "\n  ".join(missing[:10]) +
        "\n\nRe-export — the table has grown since the file was written.")
    assert checked > 2000, f"only {checked} were compared — the table did not load"
    assert not bad, (
        f"the committed answers and the case table disagree ({len(bad)} of {checked}):\n  "
        + "\n  ".join(bad[:10]) +
        "\n\nIf a case body changed on purpose, re-export. If it did not, this is a\n"
        "regression in the core, and the golden answers are right.")
