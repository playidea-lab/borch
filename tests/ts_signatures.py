"""Counts the names that exist on **both** sides but take **different arguments.**

    uv run --with numpy --with torch --with torchvision python tests/ts_signatures.py
    uv run --with numpy --with torch --with torchvision python tests/ts_signatures.py --show nn

## The axis `ts_axis.py` says out loud that it cannot see

`tests/ts_axis.py` counts names and its own docstring names the hole:

    It cannot see a **signature**. `MaxPool2d` present in both, taking `(kernel)`
    here and `(kernel, stride, padding, dilation, returnIndices)` there, counts as
    agreement.

That is this file. A name in both with a different argument list is worse than a
missing name, because a missing name raises `is not a function` and a shifted
argument list **returns a number.** Five of those were found in this repository in
one day — `test_torch_signatures.py` exists because of them, on the
borchvision↔torchvision axis, and `test_binding_arguments.py` on the
binding↔borch.ts axis. The core↔borch.ts pair had neither.

## What it compares

Positional parameter **names**, in order:

- core side — `inspect.signature`, with `self` dropped.
- borch.ts side — the declaration string in `site/assets/api.json`, which
  `site/build_api.py` reads out of the emitted `.d.ts`.

The core is snake_case and borch.ts is camelCase, so the core's names are camelled
by `ts_axis._camel` — the same rule, imported rather than restated, so that a fix to
one spelling rule reaches both axes.

## The three things it cannot see, said before someone reads a green run as more

**Types.** `alpha: number` and `alpha: Tensor` read alike here.

**Defaults.** `padding = 0` and `padding = 1` are the same argument name, and that
exact difference is what a peer changed in a `Pad` case this week.

**Whether the values agree.** That is the golden's job, and it is the only check in
this repository that looks at an answer rather than a shape.

## The normalisers, and the claim each one makes

Any normaliser is a claim about which differences do not matter, and it usually gets
written where nobody states the claim. So each is stated here:

- **camelCase folding** — claims `return_indices` and `returnIndices` are one name.
  Established on the name axis; the trailing underscore is kept there and there is
  nothing to keep here, since no borch.ts parameter is an in-place marker.
- **an options bag ends the comparison** — borch.ts writes trailing optional
  arguments as one object (`options?: { dtype?, device? }`) where torch writes
  keyword arguments. Claiming those are equal would be inventing an answer; claiming
  they differ would flood the count with a convention. So the comparison stops at
  the bag and **what was cut is counted and printed** rather than dropped, because a
  normaliser that silently swallows is the shape this repository keeps finding.

## Why a name with two signatures is reported rather than resolved

`api.json` is keyed by name inside each module but one name can stand in several
modules, and the two declarations need not agree. Picking one silently is exactly
the key collision `test_binding_fills_in.py` went looking for on the flattening side:
a merge answers plausibly and the count still looks right. Here they are held apart
and counted as `ambiguous`.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

API = ROOT / "site" / "assets" / "api.json"

# What `kind` in `api.json` can be called with arguments. A `type` or an `interface`
# has no argument list, and asking one for its parameters would read the first
# parenthesis in an unrelated type expression.
CALLABLE_KINDS = frozenset({"function", "method", "class", "constructor"})


def _split_top(text):
    """Split a parameter list on the commas that are **not inside anything.**

    A naive `text.split(",")` cuts `options?: { parents?: readonly Tensor[];
    backwardFn?: (grad: Tensor) => readonly (Tensor | null)[] }` into six parameters
    that do not exist. Depth is tracked across all four bracket pairs and both quote
    characters — `<>` included, because a `Promise<Tensor | null>` carries no comma
    but `Record<string, number>` does.
    """
    out, depth, quote, start = [], 0, "", 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            # `=>` is an arrow, not a closing angle bracket. Counting it as one puts
            # the depth negative and every later comma reads as top level.
            if ch == ">" and i and text[i - 1] == "=":
                continue
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        out.append(tail)
    return [p.strip() for p in out]


def _arg_list(signature):
    """The text between the parameter list's parentheses, or `None`.

    **The opening parenthesis is found by depth rather than by `index("(")`.** A
    generic method can be written `gather<T>(...)` and a declaration can carry a
    parenthesis inside a type parameter's constraint; taking the first one there
    reads a type expression as an argument list.
    """
    depth = 0
    for i, ch in enumerate(signature):
        if ch == "<":
            depth += 1
        elif ch == ">" and not (i and signature[i - 1] == "="):
            depth -= 1
        elif ch == "(" and depth == 0:
            close = _matching(signature, i)
            return None if close is None else signature[i + 1:close]
    return None


def _matching(text, open_at):
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def ts_params(signature):
    """`(names, bagged)` — the parameter names in order, and how many an options bag ate.

    A destructured or object-typed trailing parameter is where borch.ts puts what
    torch spells as keyword arguments. It is not comparable name by name, so the
    list stops there and the number cut is carried out rather than dropped.
    """
    inner = _arg_list(signature)
    if inner is None:
        return None, 0
    names = []
    for raw in _split_top(inner):
        if not raw:
            continue
        head = raw.split(":", 1)[0].strip()
        if head.startswith("{"):                 # destructured — an options bag
            return names, 1
        head = head.lstrip(".").rstrip("?").strip()
        if not head:
            continue
        if head in ("options", "opts") or head.endswith("Options"):
            return names, 1
        names.append(head)
    return names, 0


def core_params(fn, receiver=False):
    """The core's positional parameter names, in order, `None` if it has no signature.

    **The receiver is dropped by position, not by name.** borch.ts writes it as
    `this` and never as a parameter, so keeping it reports every method as differing
    by one. The first version dropped it by matching `self`, and the core does not
    always spell it that way — a tally of same-position name pairs came back led by
    `t → dim` twenty-eight times and `a → other` sixteen, which is not forty-four
    renamed parameters but forty-four lists off by one.

    A function reached through the class carries its receiver first whatever it is
    called, so position is the thing that is actually true. A `staticmethod` has
    none, which is why the caller decides rather than this function guessing.

    `*args` and `**kwargs` are dropped, and that is a **loss stated rather than
    hidden**: a core function taking `*args` is comparable to nothing here, and
    counting it as either agreement or disagreement would be an invention.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    out = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out.append(name)
    if out and (receiver or out[0] in ("self", "cls")):
        out = out[1:]
    return out


# Which emitted modules a Python namespace may be paired against. **Written down
# because the first run had no such list and the mistake it made was the loud kind.**
#
# Asking the whole surface for a name, `nn.functional.normalize` found `vision.ts`'s
# image transform and reported `(x, p, dim, eps)` against `(x, mean, std)`. Two real
# functions that share a leaf name, compared to each other, and the row read exactly
# like a signature defect. `F.pad` did the same against `vision.pad`.
#
# That is the collision this file's own docstring warns about, one level up: not two
# declarations of one name, but one namespace's question answered by another
# namespace's declaration. The name axis gives up namespace precision on purpose and
# says so; here the same looseness does not merely lose precision, it **manufactures
# a finding**, so it cannot be inherited.
#
# `linalg` maps to the `linalg` module alone and not to `tensor`. When this file was
# written that pairing produced 14 rows against internal `Mat` helpers, which is what
# made the reachability problem measurable; the module has since been split, and
# keeping the mapping narrow is what will show it if the two ever merge back.
MODULES = {
    "Tensor": frozenset({"tensor", "indexing", "random", "einsum", "special",
                         "fft", "serialize", "autograd", "device", "dtype"}),
    "nn": frozenset({"nn", "rnn"}),
    "nn.functional": frozenset({"nn"}),
    "optim": frozenset({"optim"}),
    "optim.lr_scheduler": frozenset({"optim"}),
    "linalg": frozenset({"linalg"}),
    "utils.data": frozenset({"data"}),
}


def ts_signatures():
    """`{module: {name: [signature, ...]}}`, members included.

    Held as a list on purpose. One name can be declared more than once and the
    declarations need not agree; collapsing them would answer plausibly about
    whichever happened to be written last.

    A declaration may also carry a `torch` field — `build_api.py` records which torch
    name it claims to be. It is a leaf name with no namespace (checked: not one of
    the 411 is dotted), so it cannot resolve which namespace a name belongs to and is
    not used as the key. It is kept beside the signature because it is **the
    library's own claim**, and a claim is the thing worth checking.
    """
    if not API.exists():
        raise SystemExit(f"no {API.relative_to(ROOT)} — run npm run docs:api first")
    raw = json.loads(API.read_text(encoding="utf-8"))
    modules = raw.get("modules") if isinstance(raw, dict) else raw
    if not modules:
        raise SystemExit(f"{API.relative_to(ROOT)} carries no modules")
    out = {}

    def walk(into, symbols, inside=False):
        for sym in symbols:
            name, sig = sym.get("name"), sym.get("signature")
            members = sym.get("members") or []
            if inside:
                # **A member carries no `kind`.** Only top-level symbols do — checked:
                # `"kind": "method"` appears zero times in the whole file. So a rule
                # written as `kind in CALLABLE_KINDS` files no member at all, and for
                # three runs this measurement read only top-level functions while
                # reporting numbers that read like the whole surface.
                #
                # It took the `nn` row going to *all three counts zero* to show it.
                # Before that the miss was invisible, because a method that is never
                # filed produces no row of any kind — not even an unreadable one.
                # What is not filed cannot be counted as unfiled.
                if name and sig and _arg_list(sig) is not None:
                    into.setdefault(name, []).append(sig)
                walk(into, members, True)
                continue
            if sym.get("kind") == "class":
                # **A class declaration has no parentheses** — `export declare class
                # DataLoader` — and its arguments live in a `constructor` member.
                # Filed under the class's own name because that is what the Python
                # side answers: `inspect.signature(DataLoader)` gives `__init__`'s
                # parameters, not the class object's.
                #
                # Until this branch existed all 178 classes read as `no argument
                # list`, which is the wording for *could not be measured* — and the
                # layer constructors are exactly where this axis was built to look.
                # A whole namespace coming back unmeasurable is the shape that
                # caught the 1,137-name mapping error on the name axis: too clean to
                # be a finding.
                ctor = [m.get("signature") for m in members
                        if m.get("name") == "constructor" and m.get("signature")]
                # A class with no constructor member is filed with its own bare
                # declaration, which has no argument list and therefore lands in
                # `unreadable`. **Loudly unmeasured beats quietly uncompared** — the
                # first version skipped it and the whole `nn` namespace went silent.
                into.setdefault(name, []).append(ctor[0] if ctor else sig)
            elif name and sig and sym.get("kind") in CALLABLE_KINDS:
                into.setdefault(name, []).append(sig)
            walk(into, members, True)

    for module in modules:
        into = out.setdefault(module.get("name") or "?", {})
        walk(into, module.get("symbols") or [])
    return out


def _theirs(by_module, space, camel):
    """Every declaration of `camel` in the modules `space` may be paired against."""
    found = []
    for name in MODULES[space]:
        found += by_module.get(name, {}).get(camel, [])
    return found


# One parameter, two spellings. **Folded so that the real shifts become visible** —
# without it every scheduler reads as `differ` on the strength of its first argument
# and `ReduceLROnPlateau`, which genuinely drops torch's `mode` from the middle, sits
# among fifteen identical-looking rows.
#
# Kept short and each row justified, because this table is where a signature axis
# would go to die: any row makes the count better, and a row that folds two different
# concepts hides exactly the defect the file exists to find. Picked from a tally of
# same-position pairs (`--renames`) rather than from whichever rows were read first,
# and only where the two names cannot mean anything but the same thing.
#
# Deliberately **not** folded, though both stand high in that tally: `padding → p`
# (torch's `p` is a norm's order in some of those rows and a padding width in
# others), and `betas → beta1` (borch.ts splits torch's pair into two scalars, so
# it is a real arity change and belongs in `differ`).
RENAMES = {
    "optimizer": "opt",          # every scheduler and optimizer; nothing else is an optimizer
    "kernel_size": "kernel",     # every pooling and convolution layer
    "in_channels": "inC",        # convolution, beside `outC`
    "out_channels": "outC",
    "hidden_size": "hidden",     # the recurrent layers
}


def _fold_initial(names):
    """Lower the first letter of each parameter name.

    torch writes `T_max`, `T_0` and `T_mult` with a capital and borch.ts writes
    `tMax`, `t0`, `tMult`. **On parameters the initial capital carries nothing**, so
    folding it is safe here — and that is the opposite of the rule `ts_axis`'s
    `_folds_onto` applies to *names*, where an initial capital is torch's
    class/function boundary and folding it reported `nn.Embedding` as present because
    `F.embedding` exists.

    Same repository, same-looking fold, opposite verdicts, because the two are folding
    different alphabets: one a namespace's members, the other one function's
    arguments. Written out because a reader who knows the other rule will otherwise
    read this one as a mistake.
    """
    return [n[:1].lower() + n[1:] for n in names]


def _verdict(wanted, yours):
    """`agree` · `shorter` · `longer` · `differ` — **and the split is the point.**

    A tutorial line breaks in two different ways and they want different work.

    `shorter` — borch.ts takes a prefix of what torch takes. Every argument that is
    accepted means what it means in torch, and passing one too many raises. That is a
    feature not carried across, which is `ts_axis.py`'s kind of finding, and it is
    the honest reading of `ExponentialLR(opt, gamma)` beside torch's
    `(optimizer, gamma, last_epoch)`.

    `differ` — a position holds a different argument. `ReduceLROnPlateau` drops
    torch's `mode` from the middle, so `(opt, 'min', 0.1)` sets `factor` to a string
    and `patience` to `0.1` in borch.ts and neither raises. **That is the shape this
    axis was built for**: not a crash, an answer.

    Reporting both as `differ` would bury sixteen of the second kind under three
    hundred of the first, which is how a check ends up being read as noise.
    """
    wanted, yours = _fold_initial(wanted), _fold_initial(yours)
    if wanted == yours:
        return "agree"
    if len(yours) < len(wanted) and yours == wanted[:len(yours)]:
        return "shorter"
    if len(yours) > len(wanted) and yours[:len(wanted)] == wanted:
        return "longer"
    if len(yours) == len(wanted):
        # Same arity, different names. **TypeScript has no keyword arguments**, so a
        # positional caller is unaffected and this cannot silently mean something
        # else — unless the same concepts were reordered, which the names do catch.
        return "reordered" if sorted(yours) == sorted(wanted) else "renamed"
    return "differ"


def compare():
    """`{space: rows}` where each row is `(name, core, theirs, note)`.

    Only names present on both sides are looked at. What is missing on one side is
    `ts_axis.py`'s count and asking it again here would be the same question twice.
    """
    import torch_gap
    import ts_axis

    theirs = ts_signatures()
    stubs = ts_axis.refused()
    out = {}
    for space, _real, ours in torch_gap._spaces():
        if space not in ts_axis.SPACES:
            continue
        rows = []
        for name in sorted(torch_gap._public(ours)):
            if name in stubs:                    # carried only in order to refuse
                continue
            camel = ts_axis._camel(name)
            sigs = _theirs(theirs, space, camel) or _theirs(theirs, space, name)
            if not sigs:
                continue                         # a name gap — the other axis counts it
            # In the `Tensor` space every name is reached through the class, so the
            # first parameter is the receiver — except a `staticmethod`, which has
            # none. Asked of the raw attribute so that a descriptor answers as
            # itself rather than as what it returns.
            import inspect
            held = inspect.getattr_static(ours, name, None)
            mine = core_params(getattr(ours, name),
                               receiver=(space == "Tensor"
                                         and not isinstance(held, staticmethod)))
            if mine is None:
                rows.append((name, None, None, "no python signature"))
                continue
            if len({s for s in sigs}) > 1:
                rows.append((name, mine, None, f"ambiguous — {len(sigs)} declarations"))
                continue
            yours, bagged = ts_params(sigs[0])
            if yours is None:
                rows.append((name, mine, None, "no argument list"))
                continue
            wanted = [ts_axis._camel(RENAMES.get(p, p)) for p in mine]
            if bagged:
                # The bag stands where torch's keyword arguments do. Compare only as
                # far as the bag reaches and say how much was left uncompared.
                head = wanted[:len(yours)]
                if head == yours:
                    rows.append((name, mine, yours,
                                 f"agree to the bag — {len(wanted) - len(yours)} uncompared"))
                    continue
            rows.append((name, mine, yours, _verdict(wanted, yours)))
        out[space] = rows
    return out


def renames(rows):
    """How often each `(core name, borch.ts name)` pair stands at the same position.

    **Counted before anything is folded.** The differing rows mix three things that
    look alike in a list — a parameter renamed in place, a tail borch.ts does not
    take, and an argument dropped from the middle, which shifts every later one. Only
    the third is the defect this axis was built for.

    A tally separates them: a pair standing 15 times is a convention (`optimizer` is
    `opt` in every scheduler), and a pair standing once beside a length change is
    where to look. Deciding the folds first and counting afterwards would have folded
    whatever the first few rows happened to show.
    """
    tally = {}
    for found in rows.values():
        for _name, mine, yours, note in found:
            if note != "differ" or not mine or not yours:
                continue
            import ts_axis
            for a, b in zip((ts_axis._camel(p) for p in mine), yours):
                if a != b:
                    tally[(a, b)] = tally.get((a, b), 0) + 1
    return tally


def main(argv):
    show = argv[argv.index("--show") + 1] if "--show" in argv else None
    rows = compare()
    if "--renames" in argv:
        for (a, b), n in sorted(renames(rows).items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {n:>4}  {a}  →  {b}")
        return 0
    differ = bagged = unreadable = ambiguous = agreed = shorter = renamed = 0
    for space, found in rows.items():
        d = [r for r in found if r[3] in ("differ", "reordered")]
        s = [r for r in found if r[3] in ("shorter", "longer")]
        n = [r for r in found if r[3] == "renamed"]
        b = [r for r in found if r[3].startswith("agree to the bag")]
        a = [r for r in found if r[3].startswith("ambiguous")]
        u = [r for r in found if r[3].startswith("no ")]
        differ += len(d)
        shorter += len(s)
        bagged += len(b)
        ambiguous += len(a)
        unreadable += len(u)
        renamed += len(n)
        agreed += len(found) - len(d) - len(s) - len(n) - len(b) - len(a) - len(u)
        mark = " " if not d else "✘"
        print(f"  {mark} {space:22s} agree {len(found) - len(d) - len(s) - len(n):>4}   "
              f"differ {len(d):>4}   shorter {len(s):>4}   renamed {len(n):>4}   "
              f"bag {len(b):>3}   ambiguous {len(a):>3}   unreadable {len(u):>3}")
        if show is not None and space.startswith(show):
            # **The agreeing pairs print too.** A namespace reporting nothing wrong is
            # the one to distrust — `nn`'s 144 constructors went from unmeasurable to
            # all-agreeing in one edit, and a count alone cannot tell "compared and
            # matched" from "compared nothing". Reading them is what separates the two.
            for name, mine, yours, note in found:
                print(f"      · {name}({', '.join(mine or [])})")
                print(f"          borch.ts: ({', '.join(yours or [])})  — {note}")
    print("\n이름이 양쪽에 있는데 인자가 다른 자리를 센다 — 형도 기본값도 값도 안 본다.")
    print(f"맞음 {agreed}건 · **어긋남 {differ}건** · 꼬리가 짧다 {shorter}건 · "
          f"이름만 다르다 {renamed}건 · 보따리에서 멈춘 것 {bagged}건 · "
          f"두 선언 {ambiguous}건 · 못 읽음 {unreadable}건.")
    print("어긋남만이 값이 조용히 달라지는 자리다. 짧은 것은 안 옮긴 기능이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
