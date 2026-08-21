"""**A prose row naming things is a claim, and counting cannot check it.**

`test_docs.py` guards the numbers in the documentation — golden counts, line
counts, and how many claims each document makes. None of that sees a sentence
that *names* things. The README's torchvision row listed five transforms while
seven existed: `Resize` and `CenterCrop` went in at `90cd0de` and the row never
moved.

Nothing caught it for the same reason nothing catches a deleted table row — the
count machinery is one level below the claim. Row totals would not have helped
either; the row was singular and its contents were wrong. Only a name-by-name
comparison speaks here.

So this file takes the lists that name a library's public surface and compares
them against the surface. Where `test_docs.py` asks "is this number current",
this asks "is this list the list".
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import borchvision  # noqa: E402

BACKTICKED = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


def _public(module):
    return {n for n in dir(module) if not n.startswith("_")}


def _row(path, marker):
    """The one line containing `marker`, or None."""
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if marker in line:
            return line
    return None


def test_the_readme_names_every_borchvision_transform():
    """The transforms row names what `borchvision.transforms` actually holds.

    A missing name reads as a smaller library than there is; a name that is not
    there reads as a promise. Both are wrong in the direction that matters — a
    reader decides whether to reach for this based on that row.
    """
    row = _row("README.md", "| what is here |")
    assert row is not None, (
        "the transforms row is gone from README.md, or its leading cell was "
        "reworded. This check keys off `| what is here |`; if the wording moved, "
        "move this with it rather than deleting the check.")

    named = set(BACKTICKED.findall(row))

    # The transforms proper, and the module-level extras beside them. They are
    # kept apart because the row states the difference — `augment_batch` is ours
    # and torchvision has no such name, so listing it among the transforms would
    # be a claim about torchvision's surface rather than about ours.
    transforms = {n for n in _public(borchvision.transforms) if n[0].isupper()}
    extras = _public(borchvision) - _public(borchvision.transforms) - {"transforms"}

    missing = sorted(transforms - named)
    invented = sorted(n for n in named
                      if n not in transforms and n not in extras)
    assert not missing and not invented, (
        "the README's transforms row does not match `borchvision`.\n"
        f"  in the library and not in the row: {missing or 'none'}\n"
        f"  in the row and in neither transforms nor the module: "
        f"{invented or 'none'}\n\n"
        "A prose row naming things is a claim. It went stale once already — the "
        "row said five while seven existed, because `Resize` and `CenterCrop` "
        "landed and nothing was watching the names.")
