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
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TS = ROOT / "borch-ts" / "src" / "optim.ts"
BINDING = ROOT / "borch_webgpu" / "_optim.py"

# What is known to be wrong and not yet fixable here. Each row is removed by the fix,
# and `test_no_owed_row_describes_something_that_now_works` fails if one is left behind.
# **Keyed by the argument, not the optimizer.** Keyed by optimizer, a row for
# `Adagrad`'s `weight_decay` would excuse `Adagrad` — every other argument it drops
# would go unlooked-at behind a reason written about one of them. That is the
# one-prose-reason problem this repository keeps finding, and it very nearly went in
# here: the first version of this table was keyed by name and hid exactly that.
OWED = {
    ("RMSprop", "weight_decay"):
        "the call passes it and borch.ts's constructor takes four arguments, so "
        "JavaScript discards it — fixable in borch.ts alone, like Adam was",
    ("Adagrad", "weight_decay"):
        "accepted and never passed; borch.ts has no weightDecay on this optimizer yet",
    ("Adadelta", "weight_decay"): "as above",
    ("Adamax", "weight_decay"): "as above",
    ("NAdam", "weight_decay"):
        "as above — and this is the one an arity check cannot see, since the call "
        "passes six into six and the sixth is momentumDecay",
    ("RAdam", "weight_decay"): "as above",
}

# Call sites with no fixed argument list to compare. **Not a skip — a reason.**
NOT_COMPARABLE = {
    "_sched": "a factory: it takes the borch.ts name at runtime and forwards *args, so "
              "there is no call site with a fixed shape to walk",
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


def _constructors():
    src = TS.read_text()
    found = {}
    for m in re.finditer(r"export class (\w+) extends", src):
        after = re.search(r"constructor\(", src[m.end():])
        if not after:
            continue
        args = _split(_balanced(src, m.end() + after.end() - 1)[1:-1])
        found[m.group(1)] = [
            re.sub(r"^(private |readonly |public )+", "", a).split(":")[0].split("=")[0].strip()
            for a in args]
    return found


def _camel(name):
    head, *rest = name.split("_")
    return head + "".join(w.capitalize() for w in rest)


def _call_sites():
    """`(python name, ts name, python parameters, arguments passed)` per optimizer."""
    src = BINDING.read_text()
    for m in re.finditer(r"^def (\w+)\(", src, re.M):
        params = [a.split(":")[0].split("=")[0].strip()
                  for a in _split(_balanced(src, m.end() - 1)[1:-1])]
        end = src.find("\ndef ", m.end())
        body = src[m.end(): end if end > 0 else len(src)]
        call = re.search(r"_ts\.optim\.(\w+)\.new\(", body)
        if not call:
            continue
        passed = _split(_balanced(body, call.end() - 1)[1:-1])
        yield m.group(1), call.group(1), params, passed


_NAMES = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _mentions(expr):
    """The identifiers in an argument expression — `betas[0]` mentions `betas`.

    **Whole names, not substrings.** Asking whether `"lr" in "lr_decay"` says yes, so a
    call handing `lr_decay` to the slot `lr` belongs in would have read as correct. That
    is the same failure this file exists to catch, one level up in the instrument: a
    comparison loose enough to agree with the thing it is checking.
    """
    return set(_NAMES.findall(expr))


def _dropped(ts_params, py_params, passed):
    """`{argument: what is wrong with it}` — per argument, so a reason about one of
    them cannot excuse the rest."""
    lost = {}
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
            if name not in _mentions(got):
                lost[name] = f"belongs at position {at} and the call has {got}"
        elif not any(name in _mentions(expr) for expr in passed):
            lost[name] = "is accepted and never passed"
    return lost


def test_no_optimizer_drops_an_argument_it_accepts():
    """Anything not written down as owed is a defect nobody has recorded."""
    ctors = _constructors()
    surprises = []
    for name, ts_name, py_params, passed in _call_sites():
        if name in NOT_COMPARABLE:
            continue
        ts_params = ctors.get(ts_name)
        assert ts_params is not None, (
            f"{name} forwards to borch.ts's {ts_name} and this file could not read that "
            "constructor. A call site with nothing to compare against is a call site with "
            "no rule — fix the parsing rather than letting it pass.")
        for argument, wrong in _dropped(ts_params, py_params, passed).items():
            if (name, argument) in OWED:
                continue
            surprises.append(f"{name} -> {ts_name}: `{argument}` {wrong}")
    assert not surprises, (
        "optimizers accepting arguments that never reach borch.ts:\n    "
        + "\n    ".join(surprises) +
        "\n\nJavaScript discards surplus arguments without a word, so this trains with "
        "the argument ignored and raises nothing. Fix it, or add it to `OWED` with the "
        "reason it cannot be fixed here yet.")


def test_no_owed_row_describes_something_that_now_works():
    """**A row outliving its defect is worse than no row.**

    It reads to the next person as "known, accepted", which is exactly how six wrong
    reasons survived in `tests/torch_gap.py` until they were re-measured. When borch.ts
    gains the argument and the call passes it, the row has to go with the fix.
    """
    ctors = _constructors()
    live = {name: _dropped(ctors[ts_name], py_params, passed)
            for name, ts_name, py_params, passed in _call_sites()
            if name not in NOT_COMPARABLE and ts_name in ctors}
    stale = [f"{name}.{argument}" for (name, argument) in OWED
             if argument not in live.get(name, {})]
    assert not stale, (
        f"`OWED` names arguments that now reach borch.ts: {sorted(stale)}\n"
        "  Take the rows out — the defect is fixed and the row now describes nothing. A "
        "row outliving its defect reads as \"known, accepted\" to the next person, which "
        "is how six wrong reasons survived in `tests/torch_gap.py`.")


def test_every_call_site_is_either_compared_or_explained():
    """**Nothing is skipped quietly.** A name absent from both tables and from the
    comparison is the shape this repository has lost to repeatedly — what is off the
    list has no rule."""
    seen = {name for name, _ts, _p, _a in _call_sites()}
    assert seen, "no call sites were found at all — the parsing broke, not the binding"
    unexplained = {n for n in NOT_COMPARABLE if n not in seen}
    assert not unexplained, (
        f"`NOT_COMPARABLE` names call sites that no longer exist: {sorted(unexplained)}")
