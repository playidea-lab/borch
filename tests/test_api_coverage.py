"""Whether the reference generator **read all of the source.**

`tests/test_site.py::test_api_reference_is_not_stale` asks whether the index it pulled
matches the declaration files as they are now. There is one thing that check cannot ask —
**which declaration files to read is decided by the generator itself.** A file left off the
list is neither pulled nor compared, so every name inside it can be missing and both sides
stay green.

That place really was empty. `index.ts` was not in `MODULES`, so `isTensor` and `setNull`,
declared only there, were in neither the reference nor the name index — names the binding
uses, and no check cried. Another session saw it **from outside**: they were reading this
index as the list of borch.ts's surface, and a name that should have been there was not.

This file only pins that outside eye in place. It is not in `test_site.py` because another
session is editing that file right now — one more file beats two checks colliding and one of
them being erased.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX_TS = ROOT / "borch-ts" / "src" / "index.ts"


def _module_list():
    """The module names the generator decided to sweep."""
    sys.path.insert(0, str(ROOT / "site"))
    try:
        import build_api
    finally:
        sys.path.pop(0)
    return {name for name, _, _ in build_api.MODULES}


def test_generator_reads_every_module_index_exports():
    """The generator has to sweep **every** module `index.ts` exports.

    **Why `index.ts` is the reference**: that file is the definition of the public surface.
    Internal business (`kernels`, `repr`) does not leave it and so falls out on its own, and a
    list wider than it is fine — only a narrower one is a problem.
    """
    # **`index` itself is added by hand.** That file does not re-export from itself, so
    # gathering re-exports alone never puts it in the list. It was written that way at first,
    # and taking `index` out of `MODULES` to see whether the check had teeth **passed** — the
    # one line this check exists because of was the one line it could not see. That is what
    # happens without measuring.
    exported = {"index"} | set(
        re.findall(r'from\s+"\./([A-Za-z_]\w*)\.js"',
                   INDEX_TS.read_text(encoding="utf-8")))
    missing = sorted(exported - _module_list())
    assert not missing, (
        f"modules `index.ts` exports and the generator does not sweep: {missing}\n"
        "  Add it to MODULES in site/build_api.py — without that, names declared only in\n"
        "  that file drop quietly out of the reference and the name index, and the check\n"
        "  asking whether the index matches the declaration files cannot see it.")
