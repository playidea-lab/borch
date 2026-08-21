"""**Is a row that says "별칭" really an alias?**

The gap table in `borch-ts/test/run.py` groups the cases that were not ported by prefix and
writes one line of reason for each. Among them `별칭` (alias) means *"porting it would ask
the same question twice"*, and for that to hold **the name has to exist on the borch.ts side
under a different spelling.** Absent, it is not twice but zero.

Why this check exists: that claim was **wrong three rows running.**

- `bit::` 24 — "the method names of the bit operations". Those method names were not over there.
- `method2::` 60 — "a second name, as `multiply`=`mul`". Nine had no name at all.
- `top::` 50 — "top-level in-place functions". Four droppers were missing, which is why they could not be ported.

All three came out of a person opening the rows one by one, and three in a row is not
coincidence but structure. A reason is **a claim**, and an unchecked claim becomes mere
lettering with time.

## What it measures

It pulls the torch name being called out of the case name and compares it against the
declared borch.ts surface. Spelling is normalised by the same rule as
`test_binding_fills_in.py` — **the trailing underscore is kept** (the in-place edition and
the other one are different operations).

## What it does not measure

The `파이썬`/`없음`/`아직` rows are not looked at. Those three do not claim "it exists over
there". And cases whose name cannot be pulled out (ones whose title is a description rather
than a name) are **left out of the count and their number is stated** — passed over quietly,
this check would itself be a check that does not check somebody's claim.

## Why the Korean words stay

`별칭`, and the prefixes, are **keys into `run.py`'s table**, not prose. They change when
that file's wording changes, and not before.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER = ROOT / "borch-ts" / "test" / "run.py"
INDEX = ROOT / "site" / "assets" / "api-index.json"
GOLDEN = ROOT / "tests" / "golden.json"

# Places where the name cannot be pulled out. **Written down with the reason** — left empty, this check goes quiet.
NOT_A_NAME = {
    "cache::": "asks whether a global constant got dirtied — that is state, not a name",
    "grad::": "`vjp` is `backward(seed)`, so the case name is not an operation name",
}


def _flat(name):
    """The same rule as `test_binding_fills_in._flat`. Only the trailing underscore is kept."""
    tail = "_" if name.endswith("_") else ""
    return name.replace("_", "").lower() + tail


def _alias_rows():
    """The prefixes marked `별칭` in the gap table."""
    text = RUNNER.read_text(encoding="utf-8")
    rows = re.findall(r'^\s*"([a-z0-9]+::)":\s*\((\d+),\s*"([^"]*)"\)', text, re.M)
    # **Read as a leading word.** It was `"별칭" in why` at first, and that caught
    # `dtype::`'s reason, where "형 별칭" (type alias) is the same word meaning something else
    # — the marker is one word at the head of the row, which is not the same as the letters
    # appearing anywhere in the body.
    return {head for head, _, why in rows if why.startswith("별칭")}


def _declared():
    return {_flat(n.split(".")[-1])
            for n in json.loads(INDEX.read_text(encoding="utf-8"))}


# Places where an **argument name** comes first in the case title. Not an operation name, so
# not counted.
#
# Why this list is needed: a title is writing meant to be read by a person, not a grammar.
# Extraction cannot be perfect, and **passing over what cannot be pulled out quietly would
# make this check one that does not check somebody's claim** — so what is not counted is
# named here and the skipped number is put on screen.
ARGUMENT_NAMES = {
    "bias_k", "is_causal", "key_padding_mask", "need_weights", "offsets",
    "per_sample_weights", "hard",
}


def _called_name(case):
    """The torch name being called, from the case name. `None` if it cannot be pulled out."""
    rest = case.split("::", 1)[1]
    # Where there is one more segment, as in `제자리::foo_`, the last segment is the name.
    leaf = rest.split("::")[-1]
    hit = re.match(r"([A-Za-z_][A-Za-z_0-9]*)", leaf)
    if hit is None or hit.group(1) in ARGUMENT_NAMES:
        return None
    return hit.group(1)


def test_rows_calling_themselves_aliases_really_are():
    """A `별칭` row's name **has to exist over there.** Absent, it is not an alias but a gap."""
    heads = _alias_rows()
    assert heads, "not one `별칭` row was found in the gap table — this check is spinning."

    declared = _declared()
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    missing, unnamed = {}, 0
    for case in cases:
        head = case.split("::", 1)[0] + "::"
        if head not in heads or head in NOT_A_NAME:
            continue
        name = _called_name(case)
        if name is None:
            unnamed += 1
            continue
        if _flat(name) not in declared:
            missing.setdefault(head, set()).add(name)

    report = "\n".join(
        f"  {head} — {' '.join(sorted(names))}" for head, names in sorted(missing.items()))
    assert not missing, (
        "rows marked `별칭` whose name is not in borch.ts:\n" + report +
        "\n\nAlias means *porting it would ask the same question twice*, and with no name it\n"
        "is not twice but zero. Put it into borch.ts and port the cases, or correct the\n"
        "reason to something true.\n"
        f"({unnamed} cases skipped because the name could not be pulled out — that is this "
        "check's blind spot.)")
