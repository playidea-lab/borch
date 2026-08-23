"""**The binding accepts arguments and JavaScript throws the extras away.**

`borch_webgpu` is Python calling borch.ts. Every optimizer there is a Python function
whose signature imitates torch's, forwarding to a borch.ts constructor:

    def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        return _Opt(_ts.optim.Adam.new(_params(params), lr, betas[0], betas[1],
                                       eps, weight_decay))

Two things go wrong there and **neither raises.** A surplus argument is discarded
silently, because that is what JavaScript does with arguments a function did not
declare. And an argument the Python signature accepts can simply never appear in the
call — the user passes `weight_decay=0.1`, training runs, converges slightly
differently, and nothing anywhere says the number was not used.

Seven optimizers were in that state when this was written. `Adam` passed a sixth
argument into a constructor taking five; `RMSprop` a fifth into four; `Adagrad`,
`Adadelta`, `Adamax`, `NAdam` and `RAdam` accepted `weight_decay` and never passed it.

## Why a check rather than a fix

The fixes are not all reachable from this side — five of them need `weightDecay` on
the borch.ts constructor first, which is another library and another session's work.
A repair that lands in pieces needs something to hold the pieces that have not landed,
or the remainder starts to look deliberate. So the six outstanding are **written down
as owed**, and this file fails on a seventh.

## Why the comparison is positional

`NAdam` is why. Its Python signature is
`(params, lr, betas, eps, weight_decay, momentum_decay)` and its call passes six
arguments into a constructor taking six — **the arity agrees perfectly** and the sixth
is `momentumDecay`, while `weight_decay` appears nowhere in the call. Anything counting
arguments reads that as correct. Anything comparing sets reads it as correct too. Only
walking the parameters by position, name by name, reaches it.

## What this check is for, stated after a day of moving argument orders

**A positional call is a silent bet that the callee's parameter order never moves.**
Two sessions moved a dozen of them in one afternoon — losses, convolutions, pooling,
optimizers, all onto torch's own lists — and the bet came due four times. What caught
each one divides cleanly, and not by how bad the defect was:

    boolean into a number slot      `new Conv2d(cin, cout, 3, s, 1, false)`
                                    tsc named all six call sites, free
    number into a number slot       `new SGD(p, lr, 0.9, 5e-4)`
                                    tsc silent — both are `number`. A grep found it.
    Python, no compiler at all      `F.cross_entropy` → `CrossEntropyLoss(reduction)`
                                    six tests red, with a class-weight refusal none
                                    of them was asking for
    across the language boundary    the binding building borch.ts's optimizers
                                    **this file, and nothing else in three languages**

The gradient runs from *caught free* to *caught by one purpose-built check*, and the
deciding factor is **whether the two types happen to differ** — which is not a
property of the defect. `Tensor.sum(dtype)` was safe by accident because `DType` is a
string union; `Tensor.std(correction)` was not, because a correction and an axis are
both numbers, and the two signatures are the same shape side by side.

So the coverage this repository appears to have on this class is mostly luck about
type widths. The one place it is not luck is the check written for the question, and
that is the argument for treating this file as the model rather than the exception:
**it would have caught its defect whatever the types happened to be.**
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# **Two pairs, one parser.** The pattern is not the optimizers' — it is every place
# Python calls borch.ts positionally, and `nn` is the larger of the two. A second copy
# of the argument reader would be a second place for one bug, which this repository
# paid for once already the day three of them had the same variadic defect.
#
# `namespace` is what the call looks like on the Python side; a class arrives as
# `_ts.nn.Linear.new(` and a free function as `_ts.nn.softmax(`, and both are read.
SOURCES = (
    ("optim", ROOT / "borch_webgpu" / "_optim.py", ROOT / "borch-ts" / "src" / "optim.ts"),
    ("nn", ROOT / "borch_webgpu" / "_nn.py", ROOT / "borch-ts" / "src" / "nn.ts"),
)

# What is known to be wrong and not yet fixable here. Each row is removed by the fix,
# and `test_no_owed_row_describes_something_that_now_works` fails if one is left behind.
# **Keyed by the argument, not the optimizer.** Keyed by optimizer, a row for
# `Adagrad`'s `weight_decay` would excuse `Adagrad` — every other argument it drops
# would go unlooked-at behind a reason written about one of them. That is the
# one-prose-reason problem this repository keeps finding, and it very nearly went in
# here: the first version of this table was keyed by name and hid exactly that.
OWED = {
    # ── nn ───────────────────────────────────────────────────────────────────
    #
    # **Keyed by the pair as well.** A reason about `_optim.py` must not excuse
    # anything in `_nn.py`; that is the one-prose-reason problem this file already
    # records one level down, and two pairs sharing one table is the same mistake
    # one level up.
    ("nn", "Bilinear", "bias"):
        "borch.ts's `Bilinear` takes `(in1, in2, out)` and always builds a bias, so "
        "`Bilinear(..., bias=False)` gets one anyway. The argument is held to keep "
        "torch's position; the fix is a flag on the borch.ts constructor.",
    ("nn", "_gumbel_softmax", "eps"):
        "borch.ts's `gumbelSoftmax` has no `eps` — the floor under the log of a "
        "uniform draw. Ours uses its own, so a caller's value is ignored rather than "
        "refused.",
    ("nn", "_mha_forward", "dropout_p"):
        "borch.ts's `multiHeadAttentionForward` performs no dropout at all, so there "
        "is nothing to hand it to.",
    ("nn", "_mha_forward", "training"):
        "only meaningful with dropout, which that function does not do. It goes with "
        "`dropout_p`.",
    ("nn", "_mha_forward", "embed_dim_to_check"):
        "torch uses it as an assertion and nothing else. Ours does not assert it, "
        "which is a missing check rather than a dropped computation — and it is here "
        "rather than nowhere so that the difference is written down.",
    ("nn", "_mha_forward", "q_proj_weight"):
        "reachable only under `use_separate_proj_weight`, which this function refuses "
        "loudly. **The refusal is on a different argument**, so nothing above sees "
        "that these three are unreachable — which is exactly why the row exists.",
    ("nn", "_mha_forward", "k_proj_weight"): "as `q_proj_weight`.",
    ("nn", "_mha_forward", "v_proj_weight"): "as `q_proj_weight`.",
    # ── optim ────────────────────────────────────────────────────────────────
    # **`Adagrad.maximize` used to be here and the check can see it now.** The row read
    # "refused rather than dropped — which is the one shape this check cannot see: it
    # reads the call site, and a refusal happens before the call". Widening the second
    # branch from *does it appear in the call* to *does the body mention it at all* made
    # a refusal visible, and `test_no_owed_row_describes_something_that_now_works` took
    # the row out on the spot.
    #
    # Worth the four lines it costs: **an exemption disappeared because the instrument
    # improved**, not because anything was fixed. That is the direction this table is
    # supposed to shrink in and the only time it has.
    # **It was six before that.** Every row came out with the fix that made it false:
    # borch.ts gained `weightDecay` on eight optimizers (`414af4d`) and the five call
    # sites here now pass it. The table stays because the next dropped argument needs
    # somewhere to be written down, and an empty one says "nothing is owed" where a
    # deleted one says nothing at all.
}

# Call sites with no fixed argument list to compare. **Not a skip — a reason.**
NOT_COMPARABLE = {
    ("optim", "_sched"):
        "a factory: it takes the borch.ts name at runtime and forwards *args, so there "
        "is no call site with a fixed shape to walk",
}


def _balanced(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def _split(txt):
    """Top-level commas only — `betas[0], betas[1]` and `(0.9, 0.999)` stay whole."""
    out, depth, cur = [], 0, ""
    for ch in txt:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _declarations(path):
    """`{name: [parameters]}` for the borch.ts side — **classes and free functions.**

    A class arrives at the binding as `X.new(...)` and a free function as `x(...)`, and
    the same comparison serves both. Reading only the classes was enough while this file
    watched `optim.ts`, where everything is a class; `nn.ts` is nine free functions and a
    hundred and twenty-nine classes, and taking only one kind would have left the other
    unlooked-at with nothing saying so.
    """
    src = path.read_text()
    found = {}
    # **`extends` is optional.** Requiring it read `SequentialLR` and
    # `TripletMarginWithDistanceLoss` as absent — two classes that extend nothing — and
    # a declaration the parser cannot find is a call site with no rule. They were the
    # only two, and the check that they are readable is the assertion below, not this
    # comment.
    for m in re.finditer(r"export class (\w+)(?: extends \w+)? \{", src):
        after = re.search(r"constructor\(", src[m.end():])
        if not after:
            continue
        args = _split(_balanced(src, m.end() + after.end() - 1)[1:-1])
        found[("class", m.group(1))] = [
            re.sub(r"^(private |readonly |public )+", "", a).split(":")[0].split("=")[0].strip()
            for a in args]
    for m in re.finditer(r"export function (\w+)\(", src):
        args = _split(_balanced(src, m.end() - 1)[1:-1])
        found[("function", m.group(1))] = [
            a.split(":")[0].split("=")[0].strip() for a in args]
    return found


def _camel(name):
    head, *rest = name.split("_")
    return head + "".join(w.capitalize() for w in rest)


def _reachable(body, params):
    """Which parameters a call can still be carrying, **through the locals.**

    `_ctc_loss` builds `rows` and `lens` out of `targets` and `target_lengths` and hands
    those over, so asking whether the argument expression names the parameter answers no
    about a parameter that is plainly there. Walking `name = expr` assignments closes
    that, and without it the check manufactures a finding — which is the fault this file
    exists to catch, arriving inside the instrument for the third time.

    Deliberately shallow: it follows assignments and nothing else. Anything cleverer
    would start agreeing with things it has not read.
    """
    carries = {p: {p} for p in params}
    for _ in range(3):                                          # a short fixed point
        for m in re.finditer(r"^\s*(\w+)\s*=\s*(.+)$", body, re.M):
            local, expr = m.group(1), m.group(2)
            for p in params:
                if any(n in carries and p in carries[n] for n in _mentions(expr)):
                    carries.setdefault(local, set()).add(p)
    return carries


def _call_sites(binding, namespace):
    """`(python name, ts key, python parameters, arguments passed, body)` per call."""
    src = binding.read_text()
    for m in re.finditer(r"^def (\w+)\(", src, re.M):
        # **The bare `*` is a marker, not a parameter.** It arrived the day the
        # optimizers took torch's argument lists, where `maximize` is keyword-only,
        # and this read it as an argument named `*` that the call never passes —
        # a finding manufactured by the instrument, which is the shape this file
        # exists to catch one level down.
        params = [a.split(":")[0].split("=")[0].strip()
                  for a in _split(_balanced(src, m.end() - 1)[1:-1])]
        # **`*`, `*args` and `**kw` are markers, not parameters.** The bare `*` arrived
        # the day the optimizers took torch's lists, where `maximize` is keyword-only,
        # and this read it as an argument named `*` that the call never passes. `**kw`
        # is the same shape and arrived with `nn`: a catch-all that by definition is not
        # forwarded, reported once per function as a dropped argument. Two manufactured
        # findings from one habit of reading a signature as a list of names.
        params = [p for p in params if p and not p.startswith("*")]
        end = src.find("\ndef ", m.end())
        # **The body starts after the signature, not at it.** Slicing from the opening
        # parenthesis leaves the parameter list inside `body`, so every parameter is
        # "mentioned" and the branch below credits all of them. Written that way for one
        # measurement it reported **zero findings across both pairs**, including three
        # already established by hand — a check that passes everything, produced while
        # weakening a rule that had been reporting too much.
        signature = _balanced(src, m.end() - 1)
        body = src[m.end() - 1 + len(signature): end if end > 0 else len(src)]
        call = re.search(rf"_ts\.{namespace}\.(\w+)(\.new)?\(", body)
        if not call:
            continue
        passed = _split(_balanced(body, call.end() - 1)[1:-1])
        kind = "class" if call.group(2) else "function"
        yield m.group(1), (kind, call.group(1)), params, passed, body


_NAMES = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _mentions(expr):
    """The identifiers in an argument expression — `betas[0]` mentions `betas`.

    **Whole names, not substrings.** Asking whether `"lr" in "lr_decay"` says yes, so a
    call handing `lr_decay` to the slot `lr` belongs in would have read as correct. That
    is the same failure this file exists to catch, one level up in the instrument: a
    comparison loose enough to agree with the thing it is checking.
    """
    return set(_NAMES.findall(expr))


def _dropped(ts_params, py_params, passed, body=""):
    """`{argument: what is wrong with it}` — per argument, so a reason about one of
    them cannot excuse the rest."""
    lost = {}
    carries = _reachable(body, py_params)

    def reaches(expr, name):
        """Whether this argument expression is carrying that parameter, directly or
        through a local built out of it."""
        return any(name in carries.get(n, {n}) for n in _mentions(expr))

    if len(passed) > len(ts_params):
        # The surplus is charged to the last argument passed, which is the one falling
        # off the end into nothing.
        lost[py_params[-1]] = (f"the call passes {len(passed)} arguments into "
                               f"{len(ts_params)}, so JavaScript discards the last")
    for name in py_params[1:]:                                  # `params` is positional
        want = _camel(name)
        if want in ts_params:
            at = ts_params.index(want)
            got = passed[at] if at < len(passed) else "(nothing)"
            if not reaches(got, name):
                lost[name] = f"belongs at position {at} and the call has {got}"
        elif not any(reaches(expr, name) for expr in passed):
            # **An argument can be honoured without being forwarded**, and three shapes
            # of that live in `_nn.py`: refused before the call (`bias_k` raises),
            # acted on by branching (`is_causal` builds the mask), and handled after it
            # (`need_weights` chooses what to return; `batch_first` goes to the Python
            # wrapper). Demanding that every parameter appear *in the call* reports all
            # three as dropped, which is thirteen manufactured findings in one function.
            #
            # So this branch asks the weaker question — is it touched at all — while the
            # positional branch above stays strict. An argument the body never mentions
            # is the defect this file was written for: accepted, discarded, silent.
            if name not in _mentions(body):
                lost[name] = "is accepted and the body never mentions it"
    return lost


def _live():
    """`{(pair, name, argument): what is wrong}` across every source pair."""
    found = {}
    for pair, binding, ts in SOURCES:
        declared = _declarations(ts)
        for name, key, py_params, passed, body in _call_sites(binding, pair):
            if (pair, name) in NOT_COMPARABLE:
                continue
            ts_params = declared.get(key)
            assert ts_params is not None, (
                f"{binding.name}'s `{name}` forwards to borch.ts's {key[1]} and this "
                f"file could not read that {key[0]}. **A call site with nothing to "
                "compare against is a call site with no rule** — fix the parsing rather "
                "than letting it pass.")
            for argument, wrong in _dropped(ts_params, py_params, passed, body).items():
                found[(pair, name, argument)] = f"-> {key[1]}: `{argument}` {wrong}"
    return found


def test_no_call_site_drops_an_argument_it_accepts():
    """Anything not written down as owed is a defect nobody has recorded."""
    surprises = [f"{pair}.{name} {wrong}" for (pair, name, _arg), wrong
                 in sorted(_live().items()) if (pair, name, _arg) not in OWED]
    assert not surprises, (
        "call sites accepting arguments that never reach borch.ts:\n    "
        + "\n    ".join(surprises) +
        "\n\nJavaScript discards surplus arguments without a word, so this runs with "
        "the argument ignored and raises nothing. Fix it, or add it to `OWED` with the "
        "reason it cannot be fixed here yet.")


def test_no_owed_row_describes_something_that_now_works():
    """**A row outliving its defect is worse than no row.**

    It reads to the next person as "known, accepted", which is exactly how six wrong
    reasons survived in `tests/torch_gap.py` until they were re-measured. When borch.ts
    gains the argument and the call passes it, the row has to go with the fix.
    """
    live = _live()
    stale = [f"{pair}.{name}.{argument}" for (pair, name, argument) in OWED
             if (pair, name, argument) not in live]
    assert not stale, (
        f"`OWED` names arguments that now reach borch.ts: {sorted(stale)}\n"
        "  Take the rows out — the defect is fixed and the row now describes nothing. A "
        "row outliving its defect reads as \"known, accepted\" to the next person, which "
        "is how six wrong reasons survived in `tests/torch_gap.py`.")


def test_every_call_site_is_either_compared_or_explained():
    """**Nothing is skipped quietly.** A name absent from both tables and from the
    comparison is the shape this repository has lost to repeatedly — what is off the
    list has no rule."""
    seen = {(pair, name) for pair, binding, _ts in SOURCES
            for name, _k, _p, _a, _b in _call_sites(binding, pair)}
    assert len(seen) > 20, (
        f"only {len(seen)} call sites were found across both pairs — the parsing broke, "
        "not the binding. **A parser that finds nothing holds no contracts while "
        "passing**, and this file has produced that twice: once reading `**kw` as a "
        "parameter, once slicing the body from the signature so every parameter counted "
        "as mentioned, which reported zero findings across both pairs.")
    unexplained = {n for n in NOT_COMPARABLE if n not in seen}
    assert not unexplained, (
        f"`NOT_COMPARABLE` names call sites that no longer exist: {sorted(unexplained)}")
