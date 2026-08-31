"""The golden-file harness — compares where real torch is not.

The GPU path runs only in a browser, and real torch is not in a browser. The three cannot be
called side by side in one process, so it is split in two.

    stage 1 (native)    uv run --with numpy --with torch python tests/golden.py dump
    stage 2 (anywhere)  uv run --with numpy python tests/golden.py check

Stage two takes the library to compare. Today that is borch only, and a GPU backend goes into
the same place when there is one — the harness is not edited then.

The golden file holds more than values. **The case list's hash** and **the inputs'
fingerprint** go in with them, so that a changed table or changed inputs produce a failure
rather than a pass. Saying something was compared when it was not is worse than not comparing.
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

# The same tolerance as the wide-surface harness. Bit equality (T4) is this project's explicit non-goal.
ATOL = RTOL = 1e-4
_PREFIX = "case::"
_INPUT_PREFIX = "input::"

# **The fewest cases this runner will call a run.**
#
# Zero cases compared prints `agreeing 0/0` and returns 0 — green, and identical on
# screen to a run that compared everything. The manifest hash normally catches an
# emptied table, but that is a comparison between the table and the file, so it stops
# catching anything the moment both are written from the same broken state: dump an
# empty table and check agrees with it perfectly.
#
# Measured that way, not imagined. A floor is the only part of this that does not
# depend on something else being right.
#
# `tests/test_committed_golden.py` has had `assert checked > 2000` since it was
# written and `borch-ts/test/verdict.py` refuses an empty `checks`. This is the third
# place in the repository to need the same guard and the first to be told about the
# other two — which is its own small lesson about where remedies live.
FLOOR = 2000


def load_borch():
    """Imports the `borch` at the repository root.

    It used to pick up one file by path. Once it became a package that stopped working —
    executing `__init__.py` alone still leaves the relative imports needing a package context.
    """
    root = str(_here.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import borch
    return borch


def dump(path=DEFAULT_PATH):
    """Stage one — freezes real torch's expected values. torch is needed here only."""
    import torch as real

    inp = cases_mod.golden_inputs()
    cases = cases_mod.golden_cases(inp)
    # **Before anything is computed, and long before anything is written.** The floor
    # was checked by the caller at first, after `dump` had already returned — so the
    # file was overwritten and the message said "Nothing was written", which was a
    # sentence about what should have happened rather than what did. Caught by looking
    # at the file the message was talking about.
    if len(cases) < FLOOR:
        raise SystemExit(
            f"only {len(cases)} cases were built, and the floor is {FLOOR} — the case "
            "table did not load.\n"
            f"  Freezing this would make {path.name} agree with a table that is not "
            "there,\n"
            "  and `check` would then pass against it. Nothing has been written.")
    data, broken = {}, []
    for name, fn in cases:
        try:
            got = fn(real)
            # A dtype case asks about **the type name**, not a value. Frozen as the string it is.
            data[_PREFIX + name] = (np.array(got) if isinstance(got, str)
                                    else cases_mod.to_numpy(got))
        except Exception as exc:                                    # noqa: BLE001
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
    if broken:
        # What real torch cannot do is not an expected value but **a wrong case.** It must not be frozen.
        raise SystemExit("some cases failed under real torch:\n  " + "\n  ".join(broken))
    # A per-key fingerprint is frozen too — so that a divergence can say **which input** diverged.
    for key, digest in cases_mod.input_fingerprints(inp).items():
        data[_INPUT_PREFIX + key] = np.array(digest)
    np.savez(path,
             __manifest__=np.array(cases_mod.manifest_hash(cases)),
             __inputs__=np.array(cases_mod.input_fingerprint(inp)),
             **data)
    return len(cases), path


def check(lib, path=DEFAULT_PATH, faults=None):
    """Stage two — compares against the golden answers. (a list of divergences, a case count).

    `faults` is a function returning **the GPU validation error count so far** (not looked at
    when absent).

    A WebGPU validation error is not an exception. An invalid command buffer quietly does
    nothing, so **the culprit passes and a case queued behind it turns red instead.** That
    happened three times — `as_strided_`'s over-copy, a shared buffer in optimizer state, and
    an `index_select` selecting nothing while compiling a shader that divided by zero.

    Reading the count per case is what makes it possible to **name that place.** All three
    started out looking one or two slots away from the cause, and that distance is what this
    check is worth.
    """
    z = np.load(path, allow_pickle=False)
    inp = cases_mod.golden_inputs()
    cases = cases_mod.golden_cases(inp)

    if str(z["__manifest__"]) != cases_mod.manifest_hash(cases):
        raise SystemExit(
            "the golden answers are stale — the case table changed. Run dump again.")
    if str(z["__inputs__"]) != cases_mod.input_fingerprint(inp):
        mine = cases_mod.input_fingerprints(inp)
        drifted = [k for k, d in mine.items()
                   if _INPUT_PREFIX + k not in z or str(z[_INPUT_PREFIX + k]) != d]
        detail = ", ".join(
            f"{k} (here {inp[k].dtype} {inp[k].shape})" for k in drifted) or "(could not name which)"
        raise SystemExit(
            "the inputs differ from the golden ones — a comparison in this state is not a comparison.\n"
            f"  diverged inputs: {detail}")

    # The two libraries' scopes diverge. What exists in the sister library alone is compared
    # there alone — asking the core about what it deliberately refuses is not a check but a
    # wrong answer.
    is_webgpu = hasattr(lib, "backend")
    bad, skipped = [], 0
    seen_faults = faults() if faults else 0
    for name, fn in cases:
        if name.startswith(cases_mod.WEBGPU_PREFIX) and not is_webgpu:
            skipped += 1
            continue
        # **The divergence in the other direction.** What exists in the core alone (complex) is
        # skipped by the sister library — the scopes have begun to diverge both ways, and
        # writing down one direction only makes the other half look like a missing implementation.
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
            # **A right value with a validation error is still red.** This case may pass while
            # the command buffer moves on to the next one invalidated.
            if faults:
                now = faults()
                if now > seen_faults:
                    bad.append(f"{name}: {now - seen_faults} GPU validation errors "
                               f"(raised here, regardless of the value)")
                    seen_faults = now
        if want.dtype.kind == "U" or got.dtype.kind == "U":
            if str(want) != str(got):
                bad.append(f"{name}: expected {want}, got {got}")
            continue
        if want.shape != got.shape:
            bad.append(f"{name}: shape {want.shape} vs {got.shape}")
        elif not np.allclose(want, got, atol=ATOL, rtol=RTOL):
            # **A boolean answer cannot be subtracted**, and this line did it: numpy
            # refuses `-` on booleans, so the first divergence in a boolean case
            # raised a `TypeError` **inside the reporting** and took the whole run
            # with it. The check had found the difference and could not say which
            # case it was in — the one thing a comparison harness must not do.
            #
            # It went years unseen because no boolean case had ever diverged. The
            # first one to try was `isclose(equal_nan)`, added the same day.
            if want.dtype.kind == "b" or got.dtype.kind == "b":
                wrong = int(np.count_nonzero(np.asarray(want) != np.asarray(got)))
                bad.append(f"{name}: {wrong} of {want.size} booleans differ")
            else:
                bad.append(f"{name}: max diff {np.abs(want - got).max():.2e}")
    return bad, len(cases) - skipped


def main(argv):
    what = argv[1] if len(argv) > 1 else "check"
    if what == "dump":
        count, path = dump()
        print(f"froze the golden answers — {count} cases → {path}")
        return 0
    if what == "check":
        if not DEFAULT_PATH.exists():
            print(f"no golden answers: {DEFAULT_PATH}\n"
                  "  first: uv run --with numpy --with torch python tests/golden.py dump")
            return 1
        bad, total = check(load_borch())
        print(f"golden comparison — {total} cases")
        if total < FLOOR:
            print(f"  and the floor is {FLOOR} — this did not compare anything.\n"
                  "  `agreeing 0/0` is what an emptied case table looks like from here,\n"
                  "  and it is the same shape on screen as having compared them all.")
            return 1
        print(f"  agreeing {total - len(bad)}/{total}")
        if bad:
            print("\nwhere it diverged:")
            for why in bad:
                print(f"  ✗ {why}")
        return 1 if bad else 0
    print("usage: golden.py [dump|check]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
