"""The golden-file harness — it compares where real torch is not.

The GPU path runs in a browser alone and real torch is not in a browser. The
three cannot be called side by side in one process, so it splits in two.

    stage 1 (native)     uv run --with numpy --with torch python tests/golden.py dump
    stage 2 (anywhere)   uv run --with numpy python tests/golden.py check

Stage 2 takes the library to compare against. borch is the only one for now, and
a GPU backend goes into the same slot when there is one — the harness does not
change then.

The golden does not hold values alone. **A hash of the case list** and **a
fingerprint of the inputs** ride along, so that a changed table or changed inputs
produce a failure rather than a pass. Saying something was compared when it was
not is worse than not comparing it.
"""

import importlib.util
import pathlib
import sys

import numpy as np

_here = pathlib.Path(__file__).resolve().parent
DEFAULT_PATH = _here / "golden.npz"

_cases_spec = importlib.util.spec_from_file_location("bt_cases", _here / "cases.py")
cases_mod = importlib.util.module_from_spec(_cases_spec)
_cases_spec.loader.exec_module(cases_mod)

# The same tolerance as the wide-surface harness. Bit equivalence (T4) is this
# project's explicit non-goal.
ATOL = RTOL = 1e-4
_PREFIX = "case::"
_INPUT_PREFIX = "input::"


def load_borch():
    """Import the `borch` at the repository root.

    It used to pick up one file by path. That stopped working when it became a
    package — running `__init__.py` alone still needs a package context for the
    relative imports.
    """
    root = str(_here.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import borch
    return borch


def dump(path=DEFAULT_PATH):
    """Stage 1 — pin real torch's expected values. torch is needed here and
    nowhere else."""
    import torch as real

    inp = cases_mod.golden_inputs()
    cases = cases_mod.golden_cases(inp)
    data, broken = {}, []
    for name, fn in cases:
        try:
            got = fn(real)
            # A dtype case asks about **the dtype name** rather than a value. It
            # is pinned as the string itself.
            data[_PREFIX + name] = (np.array(got) if isinstance(got, str)
                                    else cases_mod.to_numpy(got))
        except Exception as exc:                                    # noqa: BLE001
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
    if broken:
        # What real torch cannot do is **a wrong case** rather than an expected
        # value. It must not be pinned.
        raise SystemExit("some cases failed against real torch:\n  "
                         + "\n  ".join(broken))
    # A per-key fingerprint is pinned alongside — so that a divergence can say
    # **which input** diverged.
    for key, digest in cases_mod.input_fingerprints(inp).items():
        data[_INPUT_PREFIX + key] = np.array(digest)
    np.savez(path,
             __manifest__=np.array(cases_mod.manifest_hash(cases)),
             __inputs__=np.array(cases_mod.input_fingerprint(inp)),
             **data)
    return len(cases), path


def check(lib, path=DEFAULT_PATH, faults=None):
    """Stage 2 — compare against the golden. (list of divergences, case count).

    `faults` is a function returning **how many GPU validation errors have
    occurred so far** (not given, they are not examined).

    A WebGPU validation error is not an exception. An invalid command buffer
    quietly does nothing, so **the culprit passes and a case queued behind it
    turns red instead.** That happened three times — `as_strided_`'s over-copy,
    the optimiser state's shared buffer, and an `index_select` that selects
    nothing baking a shader that divides by zero.

    Measuring the count per case **names that place.** All three began by looking
    one or two places away from the cause, and that distance is what this check is
    worth.
    """
    z = np.load(path, allow_pickle=False)
    inp = cases_mod.golden_inputs()
    cases = cases_mod.golden_cases(inp)

    if str(z["__manifest__"]) != cases_mod.manifest_hash(cases):
        raise SystemExit(
            "the golden is stale — the case table changed. Run dump again.")
    if str(z["__inputs__"]) != cases_mod.input_fingerprint(inp):
        mine = cases_mod.input_fingerprints(inp)
        drifted = [k for k, d in mine.items()
                   if _INPUT_PREFIX + k not in z or str(z[_INPUT_PREFIX + k]) != d]
        detail = ", ".join(
            f"{k}(here it is {inp[k].dtype} {inp[k].shape})" for k in drifted) or "(could not name which)"
        raise SystemExit(
            "the inputs differ from the golden's — a comparison in this state "
            "is not a comparison.\n"
            f"  diverged inputs: {detail}")

    # The two libraries' ranges part. What exists on the sister side alone is
    # compared on the sister alone — asking the core about something the core
    # refuses on purpose is a wrong answer rather than a check.
    is_webgpu = hasattr(lib, "backend")
    bad, skipped = [], 0
    seen_faults = faults() if faults else 0
    for name, fn in cases:
        if name.startswith(cases_mod.WEBGPU_PREFIX) and not is_webgpu:
            skipped += 1
            continue
        # **The divergence in the other direction.** What exists on the core
        # alone (complex numbers) is skipped by the sister — the ranges have begun
        # parting both ways, and writing one direction down makes the other half
        # look like a missing implementation.
        if is_webgpu and name.startswith(cases_mod.CORE_ONLY_PREFIXES):
            skipped += 1
            continue
        want = z[_PREFIX + name]
        try:
            raw = fn(lib)
            got = np.array(raw) if isinstance(raw, str) else cases_mod.to_numpy(raw)
        except Exception as exc:                                    # noqa: BLE001
            bad.append(f"{name}: {type(exc).__name__} — {str(exc).splitlines()[0][:60]}")
            continue
        finally:
            # **Right values with a validation error is still red.** This case
            # may pass and it moves on with an invalid command buffer behind
            # it.
            if faults:
                now = faults()
                if now > seen_faults:
                    bad.append(f"{name}: {now - seen_faults} GPU validation "
                               f"errors (raised here, whatever the values)")
                    seen_faults = now
        if want.dtype.kind == "U" or got.dtype.kind == "U":
            if str(want) != str(got):
                bad.append(f"{name}: expected {want}, got {got}")
            continue
        if want.shape != got.shape:
            bad.append(f"{name}: shape {want.shape} vs {got.shape}")
        elif not np.allclose(want, got, atol=ATOL, rtol=RTOL):
            bad.append(f"{name}: max diff {np.abs(want - got).max():.2e}")
    return bad, len(cases) - skipped


def main(argv):
    what = argv[1] if len(argv) > 1 else "check"
    if what == "dump":
        count, path = dump()
        print(f"golden pinned — {count} cases → {path}")
        return 0
    if what == "check":
        if not DEFAULT_PATH.exists():
            print(f"no golden: {DEFAULT_PATH}\n"
                  "  first: uv run --with numpy --with torch python tests/golden.py dump")
            return 1
        bad, total = check(load_borch())
        print(f"golden comparison — {total} cases")
        print(f"  matching {total - len(bad)}/{total}")
        if bad:
            print("\ndivergences:")
            for why in bad:
                print(f"  ✗ {why}")
        return 1 if bad else 0
    print("usage: golden.py [dump|check]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
