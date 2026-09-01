"""Measures whether the two ways of carrying a tensor out of a scope behave
**differently.**

The refusing side is checked alongside — something made without `keep` staying
alive means the scope is not doing its job, and then a training loop collapses on
memory.
"""

import json as _json

import borch_webgpu as torch

_lines = []


def _say(ok, name, note=""):
    _lines.append(f"  {'○' if ok else '×'} {name}{' — ' + note if note else ''}")
    return ok


def _usable(t):
    """Is this tensor still usable. Dead, the other side stops."""
    try:
        t.sum().item()
        return True
    except Exception:
        return False


def run():
    ok = []

    # 1) With nothing done it has to die. The rest means nothing unless this is
    #    true.
    with torch.scope():
        loose = torch.randn(4)
    ok.append(_say(not _usable(loose),
                   "something made without keep dies after the scope"))

    # 2) scope.keep — it leaves this scope.
    with torch.scope() as s:
        carried = s.keep(torch.randn(4))
    ok.append(_say(_usable(carried), "scope.keep survives outside the scope"))

    # 3) keep_alive — permanent.
    with torch.scope():
        forever = torch.keep_alive(torch.randn(4))
    ok.append(_say(_usable(forever), "keep_alive survives outside the scope"))

    # 4) Does it hand back what it was given. The sister library does, and
    #    otherwise `x = keep_alive(x)` ends up pointing at a different tensor.
    t = torch.randn(4)
    ok.append(_say(torch.keep_alive(t) is t,
                   "keep_alive hands back what it was given"))
    with torch.scope() as s:
        u = torch.randn(4)
        ok.append(_say(s.keep(u) is u,
                       "scope.keep hands back what it was given too"))
        s.keep(u)

    # 5) **The difference between the two.** What happens to something kept
    #    inside when the outer scope closes.
    with torch.scope():
        with torch.scope() as inner:
            promoted = inner.keep(torch.randn(4))
            permanent = torch.keep_alive(torch.randn(4))
        both = _usable(promoted) and _usable(permanent)
        ok.append(_say(both,
                       "nested: just after leaving the inner one, both survive"))
    ok.append(_say(not _usable(promoted),
                   "when the outer closes, what scope.keep held is released"))
    ok.append(_say(_usable(permanent),
                   "even when the outer closes, what keep_alive held survives"))

    # 6) A tensor on the host is not refused — there is simply nothing to keep.
    try:
        with torch.scope() as s:
            s.keep(torch.randn(4).cpu())
        ok.append(_say(True, "a CPU tensor does not stop it"))
    except Exception as e:
        ok.append(_say(False, "a CPU tensor does not stop it", str(e)))

    # 7) Anything that is not a tensor is refused.
    try:
        torch.keep_alive(3)
        ok.append(_say(False, "a non-tensor is refused", "it did not refuse"))
    except TypeError:
        ok.append(_say(True, "a non-tensor is refused"))

    # 8) **Is the wording true.** It separates a name that does not exist from a
    #    name that exists in borch.ts as a module function.
    try:
        torch.definitely_not_a_kernel
        ok.append(_say(False, "an absent name is reported as absent",
                       "it did not stop"))
    except AttributeError as e:
        ok.append(_say("does not have" in str(e) and "definitelyNotAKernel" in str(e),
                       "an absent name is reported as absent", str(e)))
    # `gradMode` is exported by `index.ts`, is absent from `Tensor.prototype`, and
    # has not been bridged by this binding yet — a name satisfying all three, so it
    # points exactly at where this wording parts.
    #
    # `make_node` was chosen first and it was wrong. That one is in `tensor.ts` and
    # `index.ts` does not export it, so it is invisible from `js.borch`, and then
    # "absent" is **the true statement.** The borch.ts the binding sees is the
    # published surface rather than the whole source.
    try:
        torch.grad_mode
        ok.append(_say(False, "a module function is reported as one",
                       "it did not stop"))
    except AttributeError as e:
        ok.append(_say("module function" in str(e),
                       "a module function is reported as one", str(e)))

    head = "scope escape works" if all(ok) else "**something does not work**"
    # **The verdict crosses as data, not as a sentence.** `scope_escape.py` used to
    # decide its exit code by matching the head line, and the head line is prose:
    # translating this file to English left every check passing and the runner
    # returning 1, because the literal it matched was still the Korean one. The
    # comment beside that line had predicted exactly this. A boolean cannot be
    # translated.
    return _json.dumps({"ok": all(ok), "text": "\n".join([head, *_lines])})
