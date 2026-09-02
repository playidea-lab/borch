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


# ── the same property, asked of a population that does not shrink ────────────
#
# **The check above stops seeing a function the moment it is repaired.** Its intake is
# *functions with a `**kw` bag*, and the repair for this whole class is to take the bag
# away and give `out` a seat of its own — forty in `borch_webgpu` left its view in one
# edit, and it stayed green by having less to look at. The core met the identical shape
# and answered it with `test_out_is_not_swallowed.py`, which **calls** each name.
#
# Calling is not open here: the binding runs inside a browser, and a pytest process has
# no `borch_webgpu` to import. So the property is asked of the source with the intake
# turned around — **every name torch takes `out=` for**, which is a list that does not
# move when a bag is removed.

def _binding_functions():
    """`{name: source of its definition}` across the binding, top level only."""
    out = {}
    for path in sorted((ROOT / "borch_webgpu").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for node in ast.parse(text).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            end = getattr(node, "end_lineno", node.lineno)
            out.setdefault(node.name, []).append(
                (path.relative_to(ROOT), "\n".join(lines[node.lineno - 1:end])))
    return out


def test_every_binding_name_torch_gives_out_has_a_door():
    """It must **name `out`** — as a seat or through the bag — and reach `_no_out`.

    **It does not require the argument to be honoured.** Refusing is the right answer
    here and `_no_out` is the refusal; what is forbidden is the third thing, taking it
    and going on, which leaves the caller's tensor unwritten with no error at all.

    **Writing is the other door.** `_out(value, out, name)` is what the module's
    `__getattr__` puts on every forwarded name, and a name written out by hand — the
    top-level `round`, once its `decimals` became keyword-only — can reach the same
    helper. That honours the argument, which is the opposite of swallowing it.
    """
    taking_out = _torch_names_taking_out()
    naked = []
    for name, defs in _binding_functions().items():
        torch_name = "range" if name == "range_top" else name
        if torch_name not in taking_out:
            continue
        for where, src in defs:
            head = src.split("\n", 1)[0]
            if "out" not in head and "**kw" not in head:
                continue                    # the name is not offered here at all
            if "_no_out" not in src and "_out(" not in src:
                naked.append(f"{where} — {name}")
    assert not naked, (
        "binding names torch gives an `out=` and this takes with no door:\n  "
        + "\n  ".join(sorted(naked)) + "\n\n"
        "  Either pass it to `_no_out`, which refuses and says why, or do not offer\n"
        "  the parameter. Taking it and going on writes nothing and raises nothing.")


def test_the_door_check_is_looking_at_something():
    """**A population that shrinks to nothing passes.** That is the shape this pair of
    checks exists to survive, so the size is held rather than assumed."""
    taking_out = _torch_names_taking_out()
    seen = [n for n in _binding_functions() if n in taking_out]
    assert len(seen) >= 30, (
        f"only {len(seen)} binding names overlap torch's `out=` list — the scan broke, "
        "not the binding.")
