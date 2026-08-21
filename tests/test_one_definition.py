"""**Two module-level `def`s of the same name mean one of them is never called.**

Python takes the later one and **says nothing.** No error, no warning. So editing the
earlier definition does nothing at all, and whoever edited it without seeing the later one
spends a long time chasing why their fix has no effect.

## What it caught

Writing a new `_dtype_name` in `borch_webgpu/_ops.py` was bitten by it. The same name
already existed further down the same file (the one the promotion table uses) and Python
took that one. The new one was written to stop on "a type that is a label with no slot", and
**not one line of it was ever called**; `dtype=torch.int` passed quietly. Nobody knew until
the browser comparison caught it as an unexpected pass.

Sweeping found three more in the core. Two (`_pair`, `matmul`) had identical bodies and were
harmless, but `vander` was **two different functions** — one where the degree grows and one
where it shrinks. And that divergence leaned on **definition order**, on the class body
taking whichever was defined first. The values were right (torch separates `vander` from
`linalg.vander` too), and to a reader it looks like the later one covering the earlier. The
names were separated to remove that dependence.

## Why not ruff

`F811` does catch this. But it also catches a local argument shadowing an imported name
(`def call(t, ...)`) and re-imports, which comes to over thirty in this repository. That many
hits makes a list, and lists do not get read. **What is dangerous is two module-level
`def`s**, so that alone is looked at.
"""

import ast
import collections
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _sources():
    for pkg in ("borch", "borch_webgpu", "tests"):
        yield from sorted((ROOT / pkg).rglob("*.py"))


def test_no_module_level_name_is_defined_twice():
    """A module-level `def` may use each name once per file."""
    twice = []
    for path in _sources():
        seen = collections.defaultdict(list)
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen[node.name].append(node.lineno)
        for name, lines in sorted(seen.items()):
            if len(lines) > 1:
                where = "·".join(str(n) for n in lines)
                twice.append(f"{path.relative_to(ROOT)}:{where} — def {name}")
    assert not twice, (
        "two or more module-level `def`s share a name:\n  " + "\n  ".join(twice) + "\n\n"
        "Python takes the later one and says nothing — editing the earlier one does\n"
        "nothing at all. Separate the names or delete one. **Even where the bodies look\n"
        "identical**, check first whether a class body is holding the earlier one\n"
        "(`vander` was)."
    )
