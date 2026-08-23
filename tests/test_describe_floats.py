"""`describe()` prints a float where torch prints a float.

JavaScript drops the decimal point on an integral number: `` `${1.0}` `` is `"1"`.
torch's `repr` keeps it, because the value is a Python `float`. So every layer in
`borch-ts/src/nn.ts` whose `describe()` interpolates a float argument is one line
away from a frozen golden string it cannot match — and each author meets that alone.

**Three copies of the workaround were written before anyone noticed it was a rule.**
`LocalResponseNorm` had `const py = (v) => Number.isInteger(v) ? `${v}.0` : String(v)`
inline, the padding layers had the same expression again, and a third was about to be
typed. Two of the three carried a comment explaining JavaScript's behaviour — the
knowledge was written down twice and findable neither time.

**And the layer that invented the workaround still had the defect.** `beta` and `k`
went through its local helper; `alpha`, two lines above, did not. It passes today
only because **torch's default `alpha` is 1e-4, which is not integral**, so the one
golden case that exists can never see it. That case is not testing `alpha` — it is
testing that torch picked a fractional default, and it would turn red on a change
upstream that nobody here made.

That is the rule this file exists to enforce mechanically, and the reason it asks
torch rather than reading the code: **which fields are floats is not an opinion.**
torch's own signature says so, and a check that classified fields by name or by
looking plausible would be one more thing written down and not findable.
"""

import inspect
import pathlib
import re

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "borch-ts" / "src" / "nn.ts"

# `class Name extends Anything {` … up to the next class. Read from `src` and not
# `dist`: a stale build would make this agree with a borch.ts nobody has.
_CLASS = re.compile(r"^export class (\w+)\b", re.M)
# **The body is read by counting braces, not by matching a closing pattern.** The
# first version ended at `\n  }`, which is how a multi-line method closes — and 27 of
# the 41 `describe()` methods in this file are one-liners (`{ return "Softmax2d()"; }`)
# that close on their own line. It read 14 of 41 and reported no findings for the
# other 27, which is the same shape as everything else in this repository: a parse
# that matches less than it looks like it does says nothing about what it missed.
#
# It was caught by this file's own floor, not by review — see
# `test_the_sweep_still_reads_layers_and_finds_floats`, which was written before the
# defect and failed on the first run.
_DESCRIBE = re.compile(r"describe\(\)\s*:\s*string\s*\{")
# `key=${expr}` inside a template literal. The expression runs to the closing brace,
# which is safe here because none of these interpolations nests one.
_PAIR = re.compile(r"(\w+)=\$\{([^}]+)\}")
# An expression that hands the value to something before printing it.
_CALL = re.compile(r"^\s*\w+\(")

# Layers whose `describe()` this cannot judge, each with the reason. **Not a skip
# list** — every entry is a name torch does not have under `torch.nn`, so there is no
# authority to ask. A layer torch *does* have never lands here.
NO_TORCH_CLASS = (
    # Shared bases. torch splits these by dimension — `BatchNorm1d`, `Conv2d` — and
    # the per-dimension subclasses here are judged in their place, so nothing goes
    # unlooked-at: these carry no `describe()` of their own that a subclass does not
    # inherit and override.
    "ConvND", "ConvTransposeND", "BatchNormND", "InstanceNormND",
    "PoolND", "PadND", "PadNd", "Recurrent", "RNNBase", "Sequential",
    # The lazy layers' base. Its `describe()` prints the target's name and nothing
    # else, so it interpolates no argument of any kind.
    "LazyModule",
)

# Floors. **A parse that matches nothing passes every assertion below**, which is the
# failure this file would be least able to report about itself: no violations found
# and no layers read look identical from outside.
LEAST_CLASSES = 30
LEAST_FLOAT_FIELDS = 5


def _balanced(text, at):
    """From the `{` at `at`, the body up to its matching `}`."""
    depth = 0
    for i in range(at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[at + 1:i]
    raise AssertionError(f"unbalanced braces from offset {at} in {SOURCE.name}")


def _bodies():
    """`{class name: describe() body}` for every class in `nn.ts` that has one."""
    text = SOURCE.read_text(encoding="utf-8")
    spots = [(m.group(1), m.start()) for m in _CLASS.finditer(text)]
    out = {}
    for i, (name, at) in enumerate(spots):
        end = spots[i + 1][1] if i + 1 < len(spots) else len(text)
        found = _DESCRIBE.search(text, at, end)
        if found:
            out[name] = _balanced(text, found.end() - 1)
    return out


def _kind(param):
    """`True` float, `False` not a float, `None` when torch does not say.

    Two declarations are asked and the default is asked first. `type(...) is float`
    rather than `isinstance` because `bool` is an `int` subclass and prints without
    a point either way; a default of `None` says nothing about the value's type, so
    it falls through to the annotation.

    **The annotation is asked because a parameter can be a float with no default.**
    `ConstantPad2d(padding, value)` is one: torch prints `value=0.0` and nothing in
    its default says so, because there is no default. Judging on the default alone
    would have read those as *not floats* — a silent pass, not a skip.
    """
    if type(param.default) is float:
        return True
    if param.annotation is float:
        return True
    if param.default is not inspect.Parameter.empty and param.default is not None:
        return False
    if param.annotation is not inspect.Parameter.empty:
        # `int`, or a union of int and tuple — neither prints a decimal point. A
        # union that *contains* float is not one of these and is left unjudged.
        return False if float not in getattr(param.annotation, "__args__",
                                             (param.annotation,)) else True
    return None


def _float_arguments(name):
    """`{argument: is a float}` from torch's own signature. `None` if torch has no
    such class, which is the only reason a layer goes unjudged."""
    cls = getattr(torch.nn, name, None)
    if cls is None or not inspect.isclass(cls):
        return None
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return None
    return {p.name: _kind(p) for p in params.values()}


def _findings():
    """`(class, argument, expression)` for each float printed without a helper."""
    bad, seen_classes, seen_floats = [], 0, 0
    for name, body in sorted(_bodies().items()):
        floats = _float_arguments(name)
        if floats is None:
            continue
        seen_classes += 1
        for key, expr in _PAIR.findall(body):
            if not floats.get(key):
                continue
            seen_floats += 1
            if not _CALL.match(expr):
                bad.append((name, key, expr.strip()))
    return bad, seen_classes, seen_floats


def _unjudged_arguments():
    """`(class, argument)` printed by name where torch's signature says neither.

    **Counted rather than skipped.** A parameter torch declares with no default and
    no annotation cannot be classified here, and a check that quietly passed those
    would report a clean sweep over a set it never looked at.
    """
    out = []
    for name, body in sorted(_bodies().items()):
        floats = _float_arguments(name)
        if floats is None:
            continue
        out += [(name, key) for key, _e in _PAIR.findall(body)
                if key in floats and floats[key] is None]
    return out


def test_a_float_argument_is_printed_with_its_decimal_point():
    bad, _classes, _floats = _findings()
    assert not bad, (
        "these print a float the way JavaScript prints it, and torch prints it "
        "with a decimal point:\n  "
        + "\n  ".join(f"{c}.describe(): {k}=${{{e}}}" for c, k, e in bad)
        + "\n\n  `${1.0}` is \"1\" in JavaScript and `1.0` in Python. Hand the value "
          "to the shared helper.\n"
          "  Each of these passes today only if every golden case for it happens to "
          "use a non-integral value.")


def test_the_sweep_still_reads_layers_and_finds_floats():
    """**The instrument's own absence.** Three regexes and a torch lookup stand
    between this file and its subject, and every one of them fails silently: a
    changed `describe()` signature, a renamed export, a `torch.nn` that moved. Each
    would leave the check passing with nothing read.
    """
    _bad, classes, floats = _findings()
    assert classes >= LEAST_CLASSES, (
        f"only {classes} layers matched a torch class, {LEAST_CLASSES} expected — "
        "the parse or the lookup stopped working, not the code")
    assert floats >= LEAST_FLOAT_FIELDS, (
        f"only {floats} float arguments were examined, {LEAST_FLOAT_FIELDS} "
        "expected — the `key=${…}` pattern stopped matching")


def test_the_helper_test_can_tell_a_wrapped_value_from_a_bare_one():
    """The distinction the check rests on, asked directly.

    Without this, a `_CALL` pattern that matched everything would report zero
    findings and read exactly like a clean sweep.
    """
    assert _CALL.match("pyFloat(this.beta)")
    assert _CALL.match(" py(this.k)")
    assert not _CALL.match("this.alpha")
    assert not _CALL.match("  this.eps")


def test_every_unjudged_layer_is_a_name_torch_does_not_have():
    """A layer skipped for want of an authority has to actually lack one.

    The list is written down so that a layer torch gains — or one of ours renamed
    to match torch — leaves the list rather than sitting in it unexamined.
    """
    unjudged = [n for n in _bodies() if _float_arguments(n) is None]
    assert set(unjudged) <= set(NO_TORCH_CLASS), (
        "these have a `describe()` and are not being judged, and torch has no class "
        f"of that name to ask: {sorted(set(unjudged) - set(NO_TORCH_CLASS))}\n"
        "  Add them to NO_TORCH_CLASS with the reason, or give them torch's name.")


# Arguments torch declares with neither a float default nor a type — nothing to ask.
# Empty today, and pinned so that it failing is how a new one announces itself rather
# than joining a set nobody counts.
UNJUDGED_ARGUMENTS = 0


def test_every_named_argument_gets_an_answer_from_torch():
    found = _unjudged_arguments()
    assert len(found) == UNJUDGED_ARGUMENTS, (
        f"{len(found)} arguments are printed by name and torch's signature says "
        "neither float nor not-float, so this file passes over them:\n  "
        + "\n  ".join(f"{c}.{k}" for c, k in found))
