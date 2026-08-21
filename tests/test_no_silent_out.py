"""**Nothing may swallow `out=` quietly.**

torch takes `out=` at 198 names and writes the result into a tensor made in advance. We do
not — doing it would mean computing and then putting a copy in, and **not allocating**, which
is why `out=` exists, would not happen. Imitating the saving teaches something that is not
there.

Not doing it is fine in itself. **The problem is how it is not done.**

    torch.randint(0, 5, (4,), out=buf)    # torch: writes into buf
    borch.randint(0, 5, (4,), out=buf)    # ours: buf was left at zero

Six were like this — `range`, `randperm`, `randint`, `rand_like`, `randn_like`,
`searchsorted`. With no error it goes to the next line, and the wrong value surfaces much
later. The other 190-odd have no `**kw`, so Python stops them with a `TypeError` — **safe by
accident**, and that accident ends the day one more `**kw` is added.

## What this check asks

For every name torch takes `out=` for that we also have: if that function does not take `out`
by name and does take `**kw`, its body has to contain `_no_out`. Without it, it swallows.

If `out=` is ever really supported, this check gets deleted. Until then it guards that **the
way it is not done is consistent.**

## The list comes from the docstrings — slightly wide

torch's C functions do not expose their signatures, so they are picked from `out=None` in the
docstrings. `rand_like` and `randn_like` are written there while **the actual overload does
not take it** — found while writing cases, through a `TypeError` on torch's side. So this
check catches a few extra places, and in that direction
being wide is safe: it is one more door, and it is an argument torch refuses too. Narrow,
and a place that swallows is left behind.
"""

import ast
import inspect
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _torch_names_taking_out():
    names = []
    for name in sorted(dir(torch)):
        if name.startswith("_"):
            continue
        fn = getattr(torch, name)
        if not callable(fn) or inspect.isclass(fn):
            continue
        if "out=None" in (fn.__doc__ or ""):
            names.append(name)
    return set(names)


def _functions_with_varkw(path):
    """The file's module-level functions → (the `**kw` name, whether it calls `_no_out`)."""
    found = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.args.kwarg is None or any(a.arg == "out" for a in node.args.args):
            continue
        guarded = any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_no_out"
                      for n in ast.walk(node))
        found[node.name] = guarded
    return found


def test_no_function_can_swallow_out():
    """Anywhere `out=` can arrive through `**kw` must have the door."""
    taking_out = _torch_names_taking_out()
    naked = []
    for pkg in ("borch", "borch_webgpu"):
        for path in sorted((ROOT / pkg).rglob("*.py")):
            for name, guarded in _functions_with_varkw(path).items():
                # `range_top` leaves the module as `range` — torch's name for it
                torch_name = "range" if name == "range_top" else name
                if torch_name in taking_out and not guarded:
                    naked.append(f"{path.relative_to(ROOT)} — {name}(**kw)")
    assert not naked, (
        "places that could swallow `out=` quietly:\n  " + "\n  ".join(sorted(naked)) + "\n\n"
        "Put `_no_out(kw)` on the body's first line. Swallowed, the destination is not\n"
        "written with no error at all, and that value surfaces much later somewhere\n"
        "unrelated."
    )
