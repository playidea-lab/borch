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

## One bucket carried a claim it could not support

`renamed` — same arity, different names — was labelled harmless on the ground that
TypeScript has no keyword arguments. Reading the rows afterwards found
`F.gumbel_softmax(logits, tau, hard, eps, dim)` against `(logits, tau, hard, dim,
noise)`, where position three is a tolerance in one library and an axis in the other.
The label was a claim about a whole bucket drawn from the rows that happened to be
read first, which is the same error as deciding a fold before counting. It now says
what it can support: **the names cannot tell whether these are the same arguments.**

Only `shorter` is safe, and for a reason that does not depend on reading: passing an
argument borch.ts does not take raises.

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
import re
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


_BAG_MEMBER = re.compile(r"([A-Za-z_$][\w$]*)\s*\??\s*:")

_INTERFACES = {}
_CTORS = {}

_EXTENDS = re.compile(r"\bextends\s+(\w+)")


def _class_constructors():
    """`{class name: constructor signature}`, **following `extends`.**

    TypeScript gives a class that declares no constructor its parent's, so what a
    caller may write is the parent's list — and reading only the class's own members
    reported `BatchNorm2d extends BatchNormND` as *no argument list*, which is the
    wording for **could not be measured**. Eleven classes sat in that bucket with
    their arguments declared one line up the chain.

    A class with none anywhere up the chain takes **nothing**, and that is a fact
    rather than an absence of one — it is filed as `constructor()` so the row is
    compared. Left unreadable, `Hardsigmoid`, `Mish`, `ReLU6`, `LogSigmoid` and
    `Hardswish` were reported as unmeasured while `SELU` thirty lines below them in
    the same file takes torch's `inplace` and they do not. **The same family, half of
    it short, and the count said nothing.**
    """
    if not _CTORS:
        import json

        raw, parents = {}, {}
        for module in json.loads(API.read_text(encoding="utf-8"))["modules"]:
            for sym in module.get("symbols", ()):
                if sym.get("kind") != "class":
                    continue
                ctor = [m.get("signature") for m in sym.get("members", ())
                        if m.get("name") == "constructor" and m.get("signature")]
                raw[sym["name"]] = ctor[0] if ctor else None
                found = _EXTENDS.search(sym.get("signature") or "")
                parents[sym["name"]] = found.group(1) if found else None

        for name in raw:
            at, seen = name, set()
            while at is not None and at not in seen:
                seen.add(at)
                if raw.get(at):
                    _CTORS[name] = raw[at]
                    break
                at = parents.get(at)
            else:
                _CTORS[name] = "constructor()"
            _CTORS.setdefault(name, "constructor()")
    return _CTORS


def _interface_members():
    """`{interface name: [member names]}`, read once from `api.json`.

    **Following a named type used to be "a different job".** It was, while every bag in
    the library was written inline — and the day the thirteen optimizers factored their
    five shared members into `OptimizerOptions`, that sentence turned twelve measured
    rows back into `agree to the bag` and the `kw` count fell from 12 to 0. A tally
    reaching zero because the reader stopped is the failure this file has now recorded
    three times, and it is the one that reads as good news.

    `api.json` already carries every interface's members, so the reference resolves out
    of the same file the declaration came from and needs no second parser.
    """
    if not _INTERFACES:
        import json

        for module in json.loads(API.read_text(encoding="utf-8"))["modules"]:
            for sym in module.get("symbols", ()):
                if sym.get("kind") == "interface":
                    _INTERFACES[sym["name"]] = [m["name"]
                                                for m in sym.get("members", ())]
    return _INTERFACES


def _bag_members(raw):
    """The names inside an inline object type, or `None` when it is not written inline.

    **The bag's members are in the declaration and were being thrown away.** borch.ts
    writes `opts?: { maximize?: boolean }` and `.d.ts` carries that verbatim, so the one
    name inside is as readable as any positional one. Stopping at the object read as
    *nothing beyond here can be compared*, and for thirteen optimizers that was thirteen
    rows where the axis said `agree to the bag` and measured nothing — while the
    argument torch puts first in that group, `maximize`, was sitting inside and matching.

    An object written **inline** is opened here. A named one is followed through
    `_interface_members`, which reads the declaration `api.json` already carries —
    added the day `OptimizerOptions` was factored out and this function's refusal to
    follow a name silently un-measured twelve rows.
    """
    body = raw.split(":", 1)[1] if ":" in raw else raw
    body = body.strip()
    if not (body.startswith("{") and body.endswith("}")):
        named = _interface_members().get(body)
        return list(named) if named else None
    # Nested objects would need a real parser; there are none here and one would be a
    # silent under-read, so anything with a second `{` bags instead.
    inner = body[1:-1]
    if "{" in inner:
        return None
    return _BAG_MEMBER.findall(inner)


def ts_params(signature):
    """`(names, bagged)` — the parameter names in order, and how many an options bag ate.

    A destructured or object-typed trailing parameter is where borch.ts puts what
    torch spells as keyword arguments. It is not comparable **by position**, so the
    list stops there and the number cut is carried out rather than dropped.

    What is inside the bag is readable and is read by `ts_bag` below — separately,
    because it lines up against a different half of the other signature.
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


def ts_bag(signature):
    """The names inside the trailing options object — a **set** — or `None`.

    `None` means there is no bag, or there is one this cannot read. Everything else is
    what a caller can pass by keyword, and it is a set because order inside an object
    literal is not something a caller can observe. That is the same reason
    `signature_read.keyword_only` is a set on the other side, and the two are compared
    to each other.

    **This was being thrown away.** The declaration carries `opts?: { maximize?: boolean }`
    verbatim, so the name inside is as readable as any positional one — and thirteen
    optimizers were reported as `agree to the bag`, measuring nothing past `weightDecay`,
    while the argument that group is mostly about was sitting inside and matching.
    """
    inner = _arg_list(signature)
    if inner is None:
        return None
    for raw in _split_top(inner):
        head = raw.split(":", 1)[0].strip() if raw else ""
        if not (head.startswith("{") or head.rstrip("?") in ("options", "opts")
                or head.rstrip("?").endswith("Options")):
            continue
        members = _bag_members(raw)
        return None if members is None else set(members)
    return None


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

    **A signature with `*args` or `**kwargs` returns `None`**, the same as one that
    cannot be read at all. The first version dropped them and compared what was left,
    which sounds like the stated loss above and is not: nine loss constructors in the
    core are written `(*args, reduction='mean', **kw)`, so what was left was
    `[reduction]` and every one of them read as borch.ts having *inserted* arguments
    in front. borch.ts had them because torch has them; the core is the side that
    takes anything and ignores it.

    Nine rows in the dangerous bucket, pointing at the wrong library, from dropping a
    parameter and then treating the remainder as the whole list. Which is this file's
    own recurring mistake for the fifth time — **the drop was documented and the
    consequence of the drop was not.**

    **The reading itself lives in `signature_read.py` now.** The same defect turned
    out to be in `test_torch_signatures.py` at the same hour, where it did not make
    loud wrong rows but a **pass**: two Enums compared as `['kwds'] == ['kwds']`,
    because the exclusion list said `kwargs` and Python's Enum writes `kwds`. The
    filter's own incompleteness became the thing the two sides agreed on. Two files,
    one bug, neither author reading the other's — which is the argument for one
    reader rather than a shared rule about how to write three.
    """
    from signature_read import VARIADIC, parameters

    got = parameters(fn, receiver=receiver)
    return None if got is VARIADIC else got


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
    # Both of borchvision's namespaces are one file on this side: the classes and the
    # functional spellings live together in `vision.ts`, so the same module answers for
    # both rows.
    "transforms": frozenset({"vision"}),
    "transforms.functional": frozenset({"vision"}),
    # The four that were off this list as well as the name axis. Each maps to the one
    # module that answers for it — narrow on purpose, because the paragraph above is
    # about exactly this: a leaf name answered by another namespace's declaration
    # manufactures a finding rather than losing precision.
    "ops": frozenset({"ops"}),
    "transforms.v2": frozenset({"vision_v2_twins"}),
    "transforms.v2.functional": frozenset({"vision_v2"}),
    "datasets": frozenset({"datasets"}),
}


def _declarations(sym):
    """Every declaration of one name, not only the first.

    **`build_api.py` keeps the first in `signature` and pushes the rest into
    `overloads`**, and this file read only the first. TypeScript declares the narrow
    overload first — resolution walks them in order — so an overloaded name arrived
    here as its *smallest* shape. `unique` carries `sorted`, `returnInverse`,
    `returnCounts` and `dim`, and was counted as short of the last three because the
    reference showed `unique(sorted?: boolean)`.

    That is the second blind spot of this kind. `InstanceNorm1d` was the first, from a
    different cause: an empty subclass body emits no argument list at all. Both read as
    a row this axis skips rather than a row it disagrees with, which is the quietest
    way for a measurement to be wrong.

    **The widest one, not all of them.** Filing every declaration was tried and moved
    `unique` from `shorter` to `ambiguous` — less wrong and still not compared.
    `ambiguous` is this file's answer to *one name in two modules*, where the two can
    mean different things; overloads are several declarations of **one** callable, so
    there is nothing to be ambiguous between. The widest is the closest single answer
    to what a caller may pass, and TypeScript's narrow-first order is a resolution
    detail rather than a statement about the surface.
    """
    sigs = [sym.get("signature")] + list(sym.get("overloads") or ())
    sigs = [one for one in sigs if one and _arg_list(one) is not None]
    if not sigs:
        return []
    return [max(sigs, key=lambda one: len(_split_top(_arg_list(one) or "")))]


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
                    for one in _declarations(sym):
                        into.setdefault(name, []).append(one)
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
                # **A class with no constructor of its own is not unreadable.**
                # TypeScript hands it the parent's, and with none anywhere up the
                # chain the implicit one takes nothing — both are facts, and
                # `_class_constructors` works out which. Filed as the bare
                # declaration instead, twenty-eight classes landed in `unreadable`,
                # and *loudly unmeasured* turned out to be quiet after all: half the
                # activation family is short of torch's `inplace` and the count read
                # `unread` rather than `shorter` for every one of them.
                into.setdefault(name, []).append(
                    ctor[0] if ctor else _class_constructors().get(name, sig))
            elif name and sig and sym.get("kind") in CALLABLE_KINDS:
                for one in _declarations(sym):
                    into.setdefault(name, []).append(one)
            walk(into, members, True)

    for module in modules:
        into = out.setdefault(module.get("name") or "?", {})
        walk(into, module.get("symbols") or [])
    return out


def ts_members():
    """`{module: {name}}` — the names declared **inside a class**, not beside one.

    `ts_signatures` collapses both into one table, and for every namespace but
    `Tensor` that is right: `nn.Linear` and `optim.SGD` are classes either way, and
    where a thing lives is not what that axis asks.

    **`Tensor` is different, because a method has a receiver and a function does
    not.** borch.ts writes five of torch's tensor methods as module-level functions
    only — `stft`, `istft`, `polygamma`, `igamma`, `igammac` — and the axis paired
    torch's `x.stft(n_fft, …)` against `stft(input, nFft, options?)`. One list has
    the tensor in it and the other does not, so they cannot line up, and the verdicts
    said so in the wrong words: `stft` and `istft` landed in `unaligned` (*these
    cannot be lined up*) and `polygamma` in `longer` (*we take an argument torch does
    not*). All three are false. Line up `(n, x)` against `(n)` after dropping the
    receiver and they agree.

    Dropping borch.ts's first parameter would fix two of the three and break the
    third: `polygamma(n, x)` carries the receiver **second**, because the core
    reverses it too. The receiver's position is not a rule, so the honest move is not
    to guess it — it is to say the method is not there.
    """
    if not API.exists():
        raise SystemExit(f"no {API.relative_to(ROOT)} — run npm run docs:api first")
    raw = json.loads(API.read_text(encoding="utf-8"))
    modules = raw.get("modules") if isinstance(raw, dict) else raw
    out = {}
    for module in modules or []:
        held = out.setdefault(module.get("name") or "?", set())
        for sym in module.get("symbols") or []:
            for member in sym.get("members") or []:
                if member.get("name") and member.get("signature"):
                    held.add(member["name"])
    return out


def _theirs(by_module, space, camel):
    """Every declaration of `camel` in the modules `space` may be paired against."""
    found = []
    for name in MODULES[space]:
        found += by_module.get(name, {}).get(camel, [])
    return found


def _is_member(members, space, camel):
    """Is `camel` declared inside a class anywhere `space` maps to?"""
    for name in MODULES[space]:
        if camel in members.get(name, ()):
            return True
    return False


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
# **A row must also not be the only place a gap is recorded, and that rule cost this
# table its most obvious entry.** `t → a` is true — the core calls its matrix `t` and
# borch.ts calls it `a`, one file and one author, attested. Folding it moves six
# `linalg` rows out of `unaligned` and into `shorter`, which reads as *safe*; and
# those six rows are the only visible record that `linalg.norm` takes no `p`, `pinv`
# no `rcond`, `matrix_rank` no `tol`. A slightly wrong label on a countable row beats
# a correct label on a row nobody can find, so the entry is left out on purpose.
#
# Deliberately **not** folded, though both stand high in that tally: `padding → p`
# (torch's `p` is a norm's order in some of those rows and a padding width in
# others), and `betas → beta1` (borch.ts splits torch's pair into two scalars, so
# it is a real arity change and belongs in `differ`).
RENAMES = {
    "optimizer": "opt",          # every scheduler and optimizer; nothing else is an optimizer
    # **`kernel_size`, `in_channels` and `out_channels` were folded here and are
    # gone.** borch.ts spells all three torch's way now. They came out because a fold
    # is global and a name is not: `LazyConv*` had taken torch's `outChannels` and
    # `kernelSize`, and the folds rewrote the *core's* names into borch.ts's older
    # spellings, so six rows that matched exactly were reported as unalignable.
    #
    # **A fold that is right about one row can be wrong about another**, and nothing
    # says which — the table has no place to write "except here". Closing the
    # difference is the only shape of fix that does not need one.
    "hidden_size": "hidden",     # the recurrent layers
    # **The core cannot spell this one.** torch's `uniform_` and `random_` take a
    # parameter literally named `from`, which is a Python keyword, so the core writes
    # `from_` and no other spelling is available to it. borch.ts writes `from`, which
    # is torch's own — measured: `x.uniform_(from=0., to=1.)` is accepted and
    # `from_=` is a `TypeError`.
    #
    # It is here rather than fixed because there is nothing to fix: the two rows were
    # a fact about Python's grammar being read as a divergence between two libraries.
    "from_": "from",             # `uniform_` and `random_`
    # **The fold cannot lower an acronym.** torch spells these `LU_data` and
    # `LU_pivots`; `_camel` capitalises after each underscore and leaves what is
    # already capital alone, so they come out `LUData`/`LUPivots`, and `_fold_initial`
    # lowers one letter to `lUData`. borch.ts writes `luData`/`luPivots`, which is what
    # anybody writing camelCase from `LU_data` writes.
    #
    # So the two rows were a weakness in the fold being read as a divergence between
    # two libraries — the same shape as `from_` above. Fixing `_camel` to lower a
    # leading run of capitals would be the deeper repair and it is global: it would
    # also rewrite `T_max`, `T_0` and `T_mult`, which `_fold_initial` already handles
    # its own way. Two mechanisms for one job is how the last set of folds here went
    # wrong.
    #
    # Narrow because it can be: `LU_data` and `LU_pivots` are on `lu_solve` and
    # nowhere else in torch.
    "LU_data": "lu_data",        # `lu_solve`, the method form
    "LU_pivots": "lu_pivots",
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


# **Trailing arguments borch.ts does not take, on purpose, with the reason.**
#
# Without this the table has one bucket for two different things: *a feature not carried
# across* and *a layering decision already made and implemented one layer up*. Both come
# out as `shorter`, and then the number is a mixture nobody can act on — 47 rows of which
# 23 were the same settled decision counted 23 times.
#
# That is `torch_gap.py`'s `SKIPPED` in a different file, and it took the same argument to
# get here: a count is only a to-do list if everything in it is to do.
#
# **The claim in each reason has to be true of every row it covers**, so `out` was measured
# before it was written: all 29 `linalg` names that torch gives an `out=` have one on the
# Python side, and none of them has one in borch.ts. Whoever adds a row here should call
# the thing before writing the sentence — the first attempt at *checking* this one found
# seven names that seemed to break under `out=` and every one was the probe, which asked
# `hasattr(result, "_fields")` on results that `_named` builds with `__slots__`. The
# comment in `borch/_ops.py:_out` warns about exactly that question.
TAIL_NOT_IN_TS = {
    "out": ("borch.ts has no `out=` anywhere — the Python surfaces add it, by table: "
            "`_TAKES_OUT` and `_LINALG_TAKES_OUT` in `borch/__init__.py`, which "
            "`borch_webgpu` reads rather than restating. Measured: 29 of 29 `linalg` "
            "names that torch gives an `out=` have one in Python and none in borch.ts."),
}


def _strip_declined(wanted, yours):
    """`wanted` without the trailing arguments `TAIL_NOT_IN_TS` names, and the reason.

    **Only from the tail, and only while they are missing on our side.** An argument gone
    from the middle shifts everything after it, which is the one shape this axis exists to
    catch, and it must not become invisible because the name happens to be in this table.
    """
    cut, why = list(wanted), None
    while cut and cut[-1] in TAIL_NOT_IN_TS and cut[-1] not in yours:
        why = TAIL_NOT_IN_TS[cut[-1]]
        cut.pop()
    return cut, why


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
    # **A settled decision is taken out before the tail is measured, not after.**
    # `TAIL_NOT_IN_TS` names arguments borch.ts does not carry on purpose; with those
    # gone the row either agrees — and is a decision rather than a debt — or it is still
    # short, and what is left is the part somebody has to do.
    trimmed, why = _strip_declined(wanted, yours)
    if why is not None and trimmed == yours:
        return "declined"
    if len(yours) < len(wanted) and yours == wanted[:len(yours)]:
        return "shorter"
    if len(yours) > len(wanted) and yours[:len(wanted)] == wanted:
        return "longer"
    # **An argument gone from the middle, or one put in front of the list.** Both
    # shift every argument after them by a place, and neither raises: the call still
    # has an acceptable number of arguments and each lands on the wrong parameter.
    # `ReduceLROnPlateau` is the first and `CosineEmbeddingLoss` the second.
    #
    # Told apart from a short tail **by alignment rather than by length**, which is
    # the correction a peer's reading forced. `shorter` was tested as an exact
    # prefix, so a list that was both renamed and short — `linalg.norm(t, p, dim,
    # dtype)` against `(a)` — matched neither `shorter` nor `renamed` and fell into
    # `differ`, where it read as six dangerous rows that were nothing of the kind.
    if _subsequence(yours, wanted):
        return "dropped"
    if _subsequence(wanted, yours):
        return "inserted"
    if len(yours) == len(wanted):
        # Same arity, different names.
        #
        # **This bucket was called harmless and that was wrong.** The reasoning was
        # that TypeScript has no keyword arguments, so a positional caller cannot be
        # hurt by a spelling. But a position that holds a *different argument* has
        # the same shape as a position that holds the same one under another name,
        # and the names cannot separate them:
        #
        #   F.gumbel_softmax(logits, tau, hard, eps, dim)
        #   borch.ts        (logits, tau, hard, dim, noise)
        #
        # Position three is torch's `eps` and borch.ts's `dim`. `gumbelSoftmax(x, 1,
        # false, -1)` sets a tolerance in one library and an axis in the other, and
        # nothing raises. Found by reading the rows after the bucket had already been
        # labelled safe — the label was a claim about the whole bucket made from the
        # rows that happened to be read first.
        #
        # So `renamed` means *the same number of arguments and no way to tell from
        # the names whether they are the same arguments.* It is `unaligned`'s
        # equal-length sibling, not a clean bill. A name difference that IS just a
        # spelling belongs in `RENAMES`, attested by a person, where it becomes
        # `agree` — that is the only route out of here.
        return "reordered" if sorted(yours) == sorted(wanted) else "renamed"
    # Renamed **and** a different length: the names give nothing to align on, so this
    # says so instead of guessing. `unaligned` is a request to read the row, not a
    # verdict — calling it `differ` claimed a shift nobody had shown.
    return "unaligned"


def _subsequence(small, big):
    """Whether `small` appears inside `big` in order, with at least one gap.

    An exact prefix is excluded: that is `shorter`, and it is the safe case. What is
    left is a name removed from somewhere other than the end, which moves everything
    behind it.
    """
    if len(small) >= len(big) or not small:
        return False
    at = 0
    for name in big:
        if at < len(small) and small[at] == name:
            at += 1
    return at == len(small) and small != big[:len(small)]


def compare():
    """`{space: rows}` where each row is `(name, core, theirs, note)`.

    Only names present on both sides are looked at. What is missing on one side is
    `ts_axis.py`'s count and asking it again here would be the same question twice.
    """
    import torch_gap
    import ts_axis

    theirs = ts_signatures()
    members = ts_members()
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
            # **A method's counterpart has to be a method.** In the `Tensor` space a
            # module-level function is a different callable with a different arity —
            # `x.stft(n_fft, …)` against `stft(input, nFft, …)` — and comparing the
            # two lists produces a verdict about arguments where the difference is
            # about the receiver. Three rows read falsely that way; see `ts_members`.
            #
            # It is filed rather than skipped. The name axis accepts the free function
            # as satisfying the name (it gives up namespace precision on purpose and
            # says so), so skipping here would leave **nobody asking whether
            # `x.stft(…)` works** — and it does not.
            if space == "Tensor" and not _is_member(members, space, camel) \
                    and not _is_member(members, space, name):
                rows.append((name, None, None,
                             "only a free function — no method of this name"))
                continue
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
                # **The bag and torch's keyword-only group are the same thing**, and
                # both are unordered — nobody can observe the order of either, so
                # comparing them by position describes a call that cannot be written.
                # `signature_read.positional` already says where positions stop.
                bag = ts_bag(sigs[0])
                kw = _keyword_only_of(ours, name, space)
                if bag is not None and kw is not None:
                    rows.append(_bagged_row(name, mine, yours, wanted, bag, kw,
                                            ours, space))
                    continue
                # Unreadable bag — as before: compare as far as it reaches and say how
                # much was left. An unread tail is not an agreeing one.
                head = wanted[:len(yours)]
                if head == yours:
                    rows.append((name, mine, yours,
                                 f"agree to the bag — {len(wanted) - len(yours)} uncompared"))
                    continue
            rows.append((name, mine, yours, _verdict(wanted, yours)))
        out[space] = rows
    return out


def _keyword_only_of(namespace, name, space):
    """The core's keyword-only names for one member, or `None` when unreadable."""
    import inspect

    from signature_read import VARIADIC, keyword_only

    held = inspect.getattr_static(namespace, name, None)
    del held, space                                   # the receiver never lands here
    got = keyword_only(getattr(namespace, name))
    return None if got is VARIADIC else got


def _bagged_row(name, mine, yours, wanted, bag, kw, ours, space):
    """One row when borch.ts writes an options object and the core has keyword-onlys.

    The two halves are judged apart, which is the whole point:

    - **positions** against `signature_read.positional` — the part where a shift is a
      real hazard, because a positional call lands somewhere;
    - **the bag** against `keyword_only` as sets — where a shift is not expressible, so
      what is left is presence and absence.

    Reading them together as one ordered list is what produced ten `dropped` rows on
    optimizers whose only difference was that `maximize` sits inside an object. Ten
    entries in the sharpest bucket there is, every one describing a call nobody can
    write.
    """
    import ts_axis
    from signature_read import VARIADIC, positional

    pos = positional(getattr(ours, name), receiver=(space == "Tensor"))
    if pos is None or pos is VARIADIC:
        return (name, mine, yours, f"agree to the bag — {len(wanted) - len(yours)} uncompared")
    want_pos = [ts_axis._camel(RENAMES.get(p, p)) for p in pos]
    verdict = _verdict(want_pos, yours)
    # A name the core reaches positionally and borch.ts only by keyword. Not a shift —
    # the object sits in that seat, so a positional call lands on the object and not on
    # a wrong parameter — but it is a difference and it is named rather than absorbed.
    moved = sorted(bag & {ts_axis._camel(p) for p in pos})
    absent = sorted({ts_axis._camel(p) for p in kw} - bag)
    if verdict == "agree" and not absent and not moved:
        return (name, mine, yours + sorted(bag), "agree")
    notes = []
    if verdict != "agree":
        notes.append(verdict)
    if absent:
        notes.append(f"keyword-only absent: {', '.join(absent)}")
    if moved:
        notes.append(f"by keyword here, positional in torch: {', '.join(moved)}")
    return (name, mine, yours + sorted(bag), " · ".join(notes))


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

    **The tally may propose an entry for `RENAMES`. It must never install one.** A
    peer tried the automatic version — anything standing three times becomes a
    convention — and it moved two real defects out of danger: `reduction → margin`
    stands in four loss constructors and `betas → beta1` in four optimizers, and both
    are arguments genuinely added or split rather than renamed. **A defect that
    repeats is indistinguishable from a convention by frequency alone**, and the more
    instances there are the more certainly it disappears. That is why `_verdict`
    classifies by alignment instead, and why every row of `RENAMES` is hand-written.
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
    unaligned = freefn = declined = kwgap = 0
    for space, found in rows.items():
        d = [r for r in found if r[3] in ("dropped", "inserted", "reordered")]
        s = [r for r in found if r[3] in ("shorter", "longer")]
        n = [r for r in found if r[3] == "renamed"]
        # **Its own column, not folded in with the renames.** `unaligned` means the
        # names give nothing to align on — `Adam(params, lr, betas, eps, weightDecay)`
        # against `(params, lr, beta1, beta2, eps, weightDecay)`, where borch.ts
        # splits torch's pair into two positions. That is a real arity change, and
        # counting it beside the harmless spelling differences would bury it.
        x = [r for r in found if r[3] == "unaligned"]
        b = [r for r in found if r[3].startswith("agree to the bag")]
        # **The keyword-only gap, and it needed its own column immediately.** The rows
        # `_bagged_row` writes carry a compound note — a positional verdict, then what
        # is missing from the options object — and until this line existed the twelve
        # optimizers with `foreach`/`fused`/`capturable` absent fell through every
        # branch into the residual and were counted as **agreement**. That is the
        # failure written up beside `FREE_FUNCTION` two paragraphs down, happening
        # again in the same file, on the same day, to the person who read it.
        k = [r for r in found if "keyword-only absent" in r[3] or "by keyword here" in r[3]]
        a = [r for r in found if r[3].startswith("ambiguous")]
        u = [r for r in found if r[3].startswith("no ")]
        # **Its own column too, and it had to be.** The verdict reads *only a free
        # function — no method of this name*, which starts with neither "no " nor
        # anything else already caught, so without this line it fell through to the
        # residual and was counted as **agreement** — the one outcome worse than the
        # false `unaligned` it replaced. A new verdict that nothing subtracts is a new
        # verdict that reads as green.
        f = [r for r in found if r[3].startswith("only a free function")]
        # **Its own column, subtracted from the residual like every other verdict.**
        # A new verdict that nothing subtracts is a new verdict that reads as green —
        # written down when `only a free function` did exactly that.
        c = [r for r in found if r[3] == "declined"]
        differ += len(d)
        shorter += len(s)
        declined += len(c)
        bagged += len(b)
        ambiguous += len(a)
        unreadable += len(u)
        renamed += len(n)
        unaligned += len(x)
        freefn += len(f)
        kwgap += len(k)
        counted = (len(d) + len(s) + len(n) + len(x) + len(b) + len(a) + len(u)
                   + len(f) + len(c) + len(k))
        agreed += len(found) - counted
        mark = " " if not (d or x or n or f) else "✘"
        print(f"  {mark} {space:22s} "
              f"agree {len(found) - counted:>4}   "
              f"shifted {len(d):>3}   unaligned {len(x):>3}   shorter {len(s):>4}   "
              f"renamed {len(n):>4}   bag {len(b):>3}   two {len(a):>3}   "
              f"free {len(f):>3}   declined {len(c):>3}   kw {len(k):>3}   "
              f"unread {len(u):>3}")
        if show is not None and space.startswith(show):
            # **The agreeing pairs print too.** A namespace reporting nothing wrong is
            # the one to distrust — `nn`'s 144 constructors went from unmeasurable to
            # all-agreeing in one edit, and a count alone cannot tell "compared and
            # matched" from "compared nothing". Reading them is what separates the two.
            for name, mine, yours, note in found:
                print(f"      · {name}({', '.join(mine or [])})")
                print(f"          borch.ts: ({', '.join(yours or [])})  — {note}")
    print("\n이름이 양쪽에 있는데 인자가 다른 자리를 센다 — 형도 기본값도 값도 안 본다.")
    print(f"맞음 {agreed}건 · **밀림 {differ}건** · **못 맞춤 {unaligned}건** · "
          f"꼬리가 짧다 {shorter}건 · 이름만 다르다 {renamed}건 · "
          f"보따리에서 멈춘 것 {bagged}건 · 두 선언 {ambiguous}건 · "
          f"**메서드가 없고 자유 함수만 {freefn}건** · 사유가 적힌 것 {declined}건 · "
          f"키워드 전용이 빈 것 {kwgap}건 · 못 읽음 {unreadable}건.")
    if kwgap:
        print(f"키워드 전용 {kwgap}건은 위치로 못 미는 자리다 — 옵션 객체와 torch 의 "
              "`*` 뒤는 둘 다 순서가 관측되지 않으므로 집합으로 비교한다.")
    if declined:
        print(f"사유가 적힌 {declined}건은 부채가 아니라 이미 내려진 결정이다 "
              "— `TAIL_NOT_IN_TS` 에 이유가 있다:")
        for arg, why in sorted(TAIL_NOT_IN_TS.items()):
            print(f"  {arg}: {why}")
    print("밀림은 이름으로 밀린 것이 보이는 자리다. 못 맞춤과 이름만 다름은 "
          "이름으로 가를 수 없어 사람이 읽어야 하는 자리다 —")
    print("  `Optimizer` 는 둘째 자리가 torch 에선 defaults 사전이고 borch.ts 에선 "
          "defaultLr 숫자다.")
    # **이 칸은 `shorter` 와 `longer` 를 함께 센다** — 한쪽 목록이 다른 쪽의 앞부분인
    # 경우 전부다. 꼬리가 어느 쪽에 붙었는지만 다르다.
    print("이 칸은 한쪽이 다른 쪽의 앞부분인 경우다 — 자리는 안 밀린다. "
          "다만 그것이 안전하다는 뜻은 아니다:")
    print("  없는 꼬리는 없는 기능이고, **JavaScript 는 남는 인자를 말없이 버린다** "
          "— raise 하지 않는다.")
    print("  그 침묵을 잡는 것이 tests/test_binding_arguments.py 다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
