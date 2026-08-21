"""Is the exported JSON **actually usable?**

That a file exists and that a comparison can be done with it are different things. Here the
core is compared using the JSON alone, **without looking at the npz** — the same procedure the
other language's side will follow.

And this file keeps one more promise: **stale JSON must not pass quietly.** A green from an
old JSON after the case table changed is not a comparison.
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_here = pathlib.Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exporter = _load("bt_export", _here / "export_json.py")
golden = _load("bt_golden", _here / "golden.py")
cases_mod = exporter.cases_mod


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    """Freezes and exports — what is looked at is **what was just made**, not the file in the repository."""
    npz = tmp_path_factory.mktemp("golden") / "golden.npz"
    out = tmp_path_factory.mktemp("golden") / "golden.json"
    golden.dump(npz)
    exporter.export(npz, out)
    return json.loads(out.read_text(encoding="utf-8"))


def _restore(entry):
    """Restores one JSON entry to numpy — imitating what the reading side will do."""
    if entry["kind"] == "string":
        return entry["value"]
    values = list(entry["values"])
    if entry["kind"] == "float":
        # **The kind is restored too.** Back when everything was restored as `nan`, `inf`
        # became `nan` and a case with an infinity in its answer came out as a max diff of 0
        # that failed.
        for i, kind in entry["nonfinite"]:
            values[i] = float(kind)
        arr = np.asarray(values, dtype=np.float64)
    elif entry["kind"] == "int":
        arr = np.asarray(values, dtype=np.int64)
    else:
        arr = np.asarray(values, dtype=bool)
    return arr.reshape(entry["shape"])


def test_json_alone_can_check_the_core(doc):
    """Compares the core with the JSON alone, **without looking at the npz.**

    This has to work for the other language's side to be able to use the file. Failing that,
    what was exported is just a large file and not an asset.
    """
    core = golden.load_borch()
    inp = cases_mod.golden_inputs()
    bad, checked = [], 0
    for name, fn in cases_mod.golden_cases(inp):
        if name.startswith(cases_mod.WEBGPU_PREFIX):
            continue                            # what the core deliberately refuses
        want = _restore(doc["cases"][name])
        got = fn(core)
        got = np.asarray(got) if isinstance(got, str) else cases_mod.to_numpy(got)
        checked += 1
        if isinstance(want, str):
            if want != str(got):
                bad.append(f"{name}: expected {want}, got {got}")
        elif want.shape != got.shape:
            bad.append(f"{name}: shape {want.shape} vs {got.shape}")
        elif not np.allclose(want, got, atol=1e-4, rtol=1e-4, equal_nan=True):
            bad.append(f"{name}: max diff {np.nanmax(np.abs(want - got)):.2e}")
    assert checked > 700, f"only {checked} were compared — the table did not load"
    assert not bad, "diverged from the JSON:\n  " + "\n  ".join(bad[:10])


def test_json_carries_every_case_name(doc):
    """One missing name and the other language's side **does not even know that case exists.**"""
    names = {name for name, _ in cases_mod.golden_cases()}
    assert set(doc["cases"]) == names


def test_json_carries_the_shared_inputs(doc):
    """The inputs the cases use have to go out too — answers with no questions cannot be worked."""
    inp = cases_mod.golden_inputs()
    assert set(doc["inputs"]) == set(inp)
    for key, arr in inp.items():
        assert np.allclose(_restore(doc["inputs"][key]), arr, equal_nan=True), key


def test_stale_json_is_detectable(doc):
    """**A changed table has to be noticeable.**

    Unlike the npz, the JSON does not go through `check`, so the manifest hash is the only
    thing guarding it when stale. Whether that really diverges is what is looked at — if it
    does not, the other language's side holds old answers against a new table and stamps a pass.
    """
    assert doc["manifest"] == cases_mod.manifest_hash(cases_mod.golden_cases())
    assert doc["manifest"] != cases_mod.manifest_hash([("the-table-changed", None)])


def test_nonfinite_survives_the_round_trip():
    """`nan` and infinity cannot be written into JSON as they are. The position and **the
    kind** are written separately.

    For a long time only the position numbers were written and the reading side restored them
    all as `nan`. Nothing caught it then because no answer held an infinity (the `nanmean`
    family has `nan` in the input only and a finite answer), and the `fmax` case was the first
    to put one in an answer — it came out as **a max diff of 0 that failed.** Because `inf` and
    `nan` were being restored as the same thing.

    What is asked is **whether the round trip carries the kind too.**
    """
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
    entry = exporter._value(arr)
    assert entry["nonfinite"] == [[1, "nan"], [2, "inf"], [3, "-inf"]]
    back = _restore(entry)
    assert back[0] == 1.0 and back[4] == 2.0
    assert np.isnan(back[1])
    assert back[2] == np.inf and back[3] == -np.inf


def test_inputs_carry_their_nan():
    """The inputs really do hold `nan` — `nanmean` and `nansum` use it.
    If that array turns into finite numbers in the round trip, the reading side **works a
    different problem.**"""
    inp = cases_mod.golden_inputs()
    withnan = [k for k, v in inp.items()
               if v.dtype.kind == "f" and not np.all(np.isfinite(v))]
    for key in withnan:
        restored = _restore(exporter._value(inp[key]))
        assert np.array_equal(np.isnan(restored), np.isnan(inp[key])), key
