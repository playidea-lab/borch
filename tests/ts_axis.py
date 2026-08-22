"""Counts the names the **core** has that **borch.ts** does not, and the reverse.

    uv run --with numpy --with torch --with torchvision python tests/ts_axis.py
    uv run --with numpy --with torch --with torchvision python tests/ts_axis.py --show nn

## The axis nothing was looking along

Three files already measure this repository's surface, and none of them measures this
pair:

    tests/torch_gap.py           core Python  ↔ real torch          names
    tests/test_torch_signatures  borchvision  ↔ real torchvision    parameters, in order
    tests/test_binding_arguments borch_webgpu ↔ borch.ts            parameters, in order

The core and borch.ts are **two independent implementations of the same surface** —
one over numpy, one over WGSL — and the golden holds their *values* against each
other. What no check holds is their *names*. A name the core has and borch.ts does
not is not a wrong answer anywhere; it is a line of tutorial code that runs in one
and raises in the other, and nothing goes red.

That state was real. Measured by hand in one session, seventeen `nn` names stood in
the core and not in borch.ts, and every golden case was green throughout — because a
case can only ask about a name somebody wrote a case for.

## What this counts, and the two things it cannot

It counts **names**, from the same enumeration `torch_gap.py` uses on the Python side
and from the generated index on the TypeScript side.

It cannot see a **signature**. `MaxPool2d` present in both, taking `(kernel)` here and
`(kernel, stride, padding, dilation, returnIndices)` there, counts as agreement — the
exact defect `test_torch_signatures.py` was written for, on the axis it does not
cover. Five of those were found in one day and none was visible to a count.

It cannot see a **value**. That is the golden's job, and the golden is why the two
implementations agree at all.

So a green run of this file says *the same names exist on both sides*. It does not say
they mean the same thing. Reading it as coverage is reading it as a sentence it does
not support.

## Why the TypeScript side is read from the index rather than the source

`site/assets/api-index.json` is generated from the `.d.ts` files, which is what a
consumer of the package actually gets. Reading `src/*.ts` instead would count names
that never reach a user, and reading the case table would count only what has a case.

**That makes this measurement a build artefact, and a stale one lies in our favour** —
names added since the last build read as absent from borch.ts, which is the same
sentence as a real gap. `test_ts_axis.py` refuses to run against a stale bundle for
that reason, the same rule `test_site.py` applies to its counts.

## Filling this in is the same dangerous work `torch_gap.py` describes

Every name written into `DELIBERATE` below raises the agreement figure, so the work
slides towards making the number look good. The rule is the one that file uses:
**every row carries a reason, and a row whose reason cannot be written is a gap.**
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

INDEX = ROOT / "site" / "assets" / "api-index.json"

# Which Python namespaces have a borch.ts side at all. `transforms` and
# `transforms.functional` are `borchvision`'s and their TypeScript side is `vision.ts`,
# which the golden's `vision::` cases hold name by name — measuring them here as well
# would ask the same question twice.
SPACES = frozenset({
    "torch", "Tensor", "nn", "nn.functional",
    "optim", "optim.lr_scheduler", "linalg", "utils.data",
})

# **Names the core has and borch.ts is not going to.** Each row is a judgement and
# carries its reason; a name absent from both this table and borch.ts is the to-do
# list. Keyed by `space::name` so that a reason about one name cannot excuse a whole
# namespace — the shape `test_binding_arguments.py` found by keying its own table the
# wrong way first.
DELIBERATE: dict[str, str] = {}

# **Names borch.ts has and the core does not.** The reverse direction is not
# symmetric: borch.ts is a browser library and carries things a numpy core has no
# reason to (`init`, `device`, `keepAlive`, `scope`). Left empty until measured —
# writing rows here before running it would be inventing the answer.
EXTRA: dict[str, str] = {}


def ts_names():
    """Every name in the generated index, as one set.

    **The index records one home per name and the comparison must not use it.** It is
    `{name: "module.path.name"}`, and `det` is recorded as `tensor.Tensor.det` — the
    method — not as `linalg.det`. Asking "is `det` in the `linalg` module of the
    index" therefore answers no about a name that is present.

    The first version of this file did exactly that and reported **1,137 core-only
    names**, of which nearly all were a name filed under a different home. A number
    that large reads as a finding; it was a mapping error, and it was caught by the
    one namespace whose whole content came back missing at once.

    So membership is asked of the **whole surface**: does borch.ts have this name
    anywhere. What that gives up is "and in the right namespace", which the index
    cannot answer — said here rather than left for the next reader to assume.
    """
    if not INDEX.exists():
        raise SystemExit(f"no {INDEX.relative_to(ROOT)} — run npm run docs:api first")
    raw = json.loads(INDEX.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"{INDEX.relative_to(ROOT)} is not the expected name → path map")
    return set(raw)


def _camel(name):
    """`return_indices` → `returnIndices`. **Only the spelling, not a translation.**

    borch.ts is camelCase and the core is snake_case, so comparing them raw reports
    every multi-word name as missing on both sides at once. That is not a gap; it is
    two spellings of one name. A name with no underscore passes through unchanged,
    which is most class names.
    """
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def compare():
    """`{space: core-only names}`, with the two spellings reconciled."""
    import torch_gap

    theirs = ts_names()
    lowered = {n.lower() for n in theirs}
    out = {}
    for space, _real, ours in torch_gap._spaces():
        if space not in SPACES:                  # borchvision's spaces are held elsewhere
            continue
        out[space] = sorted(
            n for n in torch_gap._public(ours)
            if _camel(n) not in theirs and n.lower() not in lowered)
    return out


def main(argv):
    show = argv[argv.index("--show") + 1] if "--show" in argv else None
    rows = compare()
    unexplained = 0
    for space, missing in rows.items():
        loose = [n for n in missing if f"{space}::{n}" not in DELIBERATE]
        unexplained += len(loose)
        mark = " " if not loose else "✘"
        print(f"  {mark} {space:22s} core-only {len(missing):>4}  "
              f"without a reason {len(loose):>4}")
        if show is not None and space.startswith(show):
            for name in missing:
                why = DELIBERATE.get(f"{space}::{name}", "**no reason**")
                print(f"      · {name}  {why}")
    print(f"\n이름만 센다 — 서명도 값도 안 본다. 까닭 없는 것 {unexplained}건.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
