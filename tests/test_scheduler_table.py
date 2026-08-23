"""**Compiles a table that describes another module's parameter order.**

`borch_webgpu/_optim.py` holds `_SCHED_ARGS`: for each scheduler, the names of
borch.ts's constructor parameters *in borch.ts's order*, so that a Python keyword call
can be unrolled into positions. It is data, not a call. Nothing type-checks it, nothing
imports the thing it describes, and it does not raise when it is wrong — it just puts
each value one seat away from where it belongs.

This is the single worst shape found in this repository, and it has now gone wrong
twice in one file:

- `ReduceLROnPlateau` read `(factor, patience, threshold)` with a comment saying borch.ts
  had no `mode`. That was true when written. `mode` was later added as torch's second
  argument and this row was not touched, so every name shifted one seat: `factor=0.5`
  arrived as `mode`. **Fifty rows of the binding golden were failing on it.** It raised
  only because that constructor validates its own argument — the five names missing off
  the end (`threshold_mode`, `cooldown`, `min_lr`, `eps`) were unreachable in silence,
  and a caller asking for a cooldown got none and no error.
- `OneCycleLR` read six names where borch.ts had thirteen. `div_factor` would have
  landed in `epochs`' seat and produced a wrong schedule with nothing raised anywhere.

Both were found by running something, long after the edit. A positional call is a silent
bet that the callee's parameter order never moves; a *table* of positions is the same
bet written down, with the added property that no compiler in either language ever reads
it.

## Why a static check, when there is a golden that runs the binding

Because the golden did not catch either of them, and because pointing it at the right
layer used to be a thing you could get wrong in silence. `tests/browser/run.py` took a
`--lib` that **defaulted to `borch`** — the numpy core, running inside a browser, which
is the layer both other goldens already cover and is therefore almost always green.

Run bare while hunting a binding defect, it answered `3255/3255`, printed `(borch)` in a
header nobody reads when they are looking for a score, and was completely correct about
a question nobody asked. That happened here, immediately: this row was fixed, the runner
was called without `--lib borch_webgpu`, and the green was reported as proof. What caught
it was putting the defect back and running again — **the same number returned.** A run
that stays green with a confirmed defect is not evidence of a fix; it is evidence that
the run cannot see the thing.

`--lib` is required now and the library is printed on the line carrying the number, so
that particular trap is closed. This file stays anyway, for a reason the fix does not
cover: the binding golden needs a browser and several minutes, and it only catches a
wrong seat when some value happens to travel far enough to differ. This reads the table
against the source in a second, and it names which argument lands in which seat instead
of reporting a number that moved.

The measured version of that difference: with the stale row the binding golden scored
3201/3303 and with the fixed row 3209 — **eight rows**, all `mode must be 'min' or 'max',
got 0.5`, all of which only raised because that constructor happens to validate its own
argument. `threshold_mode`, `cooldown`, `min_lr` and `eps` were unreachable through the
binding at the same time and cost **zero** rows, because a default that silently replaces
what the caller asked for produces a plausible number rather than an error.

## Why this parses TypeScript instead of importing it

The obvious check imports `borch.ts` and reads the constructor. It cannot: the binding
runs inside Pyodide against a browser build, so nothing here can construct these classes
outside a page. The next idea is to check against `dist`, which is emitted JavaScript
with the parameter names still present — but `dist` is a build artefact, and a stale one
would make this file agree with a version of borch.ts nobody has.

So it reads `borch-ts/src/optim.ts`, the file a person edits. The parse is small and
deliberately fragile in the safe direction: a class it cannot find, or a constructor it
cannot read, **fails** rather than being skipped.

## The one thing this cannot see

Renames. borch.ts calls `step_size_up` just `up`, and `lr_lambda` just `fn`, and the
positions are right. Those are folded by `RENAMED` below, by hand, one line each. A fold
without a name attached would let a genuine reordering hide inside it — which is the
lesson `_camel` taught by folding `eq_` onto `eq` and reporting a name that was not there.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TS = ROOT / "borch-ts" / "src" / "optim.ts"
BINDING = ROOT / "borch_webgpu" / "_optim.py"

# borch.ts's name → the binding's (torch's) name. **One line each, by hand.** A blanket
# rule here would swallow a real reordering: `up` and `stepSizeUp` are the same
# parameter, and nothing about the strings says so.
RENAMED = {
    "up": "step_size_up",
    "fn": "lr_lambda",
}


def _table():
    """`_SCHED_ARGS`, read out of the binding rather than imported.

    Importing `borch_webgpu` needs Pyodide. Reading the literal does not, and the
    literal is what a person edits.
    """
    text = BINDING.read_text(encoding="utf-8")
    block = re.search(r"_SCHED_ARGS = \{(.*?)\n\}", text, re.S)
    assert block, (
        "`_SCHED_ARGS` is no longer a dict literal in borch_webgpu/_optim.py.\n"
        "  This file reads it as text — if it moved or changed shape, fix the reader\n"
        "  rather than deleting the check: the table is the thing that goes wrong.")
    out = {}
    for m in re.finditer(r'"(\w+)":\s*\(([^)]*)\)', block.group(1), re.S):
        out[m.group(1)] = [x.strip().strip('"')
                           for x in m.group(2).split(",") if x.strip()]
    return out


def _constructor(src, cls):
    """borch.ts's constructor parameter names for `cls`, in order, minus the optimizer.

    Returns `None` when the class or its constructor cannot be found, which the caller
    turns into a failure. **Not a skip** — a parse that silently finds nothing would
    empty this file while every test still passed.
    """
    at = src.find(f"export class {cls} ")
    if at < 0:
        at = src.find(f"export class {cls}<")
    if at < 0:
        return None
    ctor = src.find("constructor(", at)
    if ctor < 0:
        return None
    depth, i = 0, ctor + len("constructor")
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    else:
        return None
    body = re.sub(r"//[^\n]*", "", src[ctor + len("constructor("):i])

    names, depth, cur = [], 0, ""
    for ch in body:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            names.append(cur)
            cur = ""
        else:
            cur += ch
    names.append(cur)

    out = []
    for n in names:
        n = re.sub(r"^(private|public|protected)\s+", "", n.strip()).strip()
        n = re.sub(r"^readonly\s+", "", n).strip()
        if n:
            out.append(re.split(r"[:=?]", n)[0].strip())
    # The optimizer is first on both sides and is passed separately by the binding.
    return out[1:] if out else []


def _same(binding_name, ts_name):
    """`step_size` and `stepSize` are one parameter; `factor` and `mode` are not."""
    if RENAMED.get(ts_name) == binding_name:
        return True
    parts = binding_name.split("_")
    camel = parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])
    return camel.lower() == ts_name.lower()


TABLE = _table()
SRC = TS.read_text(encoding="utf-8")


def test_the_table_is_not_empty():
    """The reader is a regex over a source file, so it can quietly find nothing.

    An empty table makes every test below vacuously true — the parametrised ones do
    not even appear. That reads as a clean run, which is the failure this file exists
    to catch, one level up in the file itself.
    """
    assert len(TABLE) > 10, (
        f"only {len(TABLE)} schedulers were read out of `_SCHED_ARGS`, and there "
        f"were 13.\n  Check `_table()` before believing any result below.")


@pytest.mark.parametrize("cls", sorted(TABLE))
def test_the_binding_sends_each_argument_to_the_seat_borch_ts_keeps_for_it(cls):
    """**Every name in the table, against borch.ts's real constructor order.**

    A row that parts here does not raise in production. It puts each value one seat
    from where it belongs, and the schedule comes out wrong with nothing to read.
    """
    want = _constructor(SRC, cls)
    assert want is not None, (
        f"`{cls}` is in `_SCHED_ARGS` but no constructor for it was found in "
        f"{TS.name}.\n  Either it was renamed on the borch.ts side — in which case the "
        "binding is calling\n  something that no longer exists — or this file's parse "
        "needs fixing. Not a skip:\n  a table describing a class that is gone is the "
        "worst state of the two.")

    have = TABLE[cls]
    off = [(i, have[i], want[i]) for i in range(min(len(have), len(want)))
           if not _same(have[i], want[i])]
    assert not off, (
        f"`{cls}`: the binding's table does not match borch.ts's order.\n"
        + "".join(f"    slot {i}: the binding sends `{h}` into borch.ts's `{w}`\n"
                  for i, h, w in off)
        + "  Nothing raises for this. Each value lands one seat from where it belongs\n"
          "  and the schedule is quietly wrong — unless the receiving constructor\n"
          "  happens to validate its own argument, which is luck, not a check.")

    missing = want[len(have):]
    assert not missing, (
        f"`{cls}`: borch.ts takes {len(want)} arguments and the table names "
        f"{len(have)}.\n"
        f"    unreachable through the binding: {missing}\n"
        "  These do not fail — they are simply not passable. A caller setting one gets\n"
        "  silence and the default, which is the same shape as `cooldown` being\n"
        "  unreachable for a day.")


def test_every_renamed_row_still_names_something_real():
    """`RENAMED` is written by hand, so it can outlive what it describes.

    A fold whose left side no longer exists in borch.ts hides nothing today and hides a
    real reordering tomorrow, silently, because a fold that matches nothing simply never
    fires.
    """
    everything = set()
    for cls in TABLE:
        everything.update(_constructor(SRC, cls) or [])
    stale = sorted(n for n in RENAMED if n not in everything)
    assert not stale, (
        f"`RENAMED` folds borch.ts names that no longer appear in any constructor: "
        f"{stale}\n  Take them out. A fold that fires on nothing is indistinguishable "
        "from one that is\n  doing its job.")
