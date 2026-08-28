"""Where `torch.optim` sits. It calls borch.ts's optimisers directly.

Parameters are **unwrapped to their handles** on the way through — our `Tensor`
is a Python shell that the JavaScript side does not know about.
"""

import numpy as _np

import js as _js

from ._base import handle, tensor, wrap

_ts = _js.borch


class _Opt:
    __slots__ = ("_o",)

    def __init__(self, o):
        self._o = o

    def zero_grad(self):
        self._o.zeroGrad()

    def step(self):
        self._o.step()

    def add_param_group(self, param_group):
        """torch's — attach another group after the optimizer was built.

        **It fell between two checks and neither was wrong.** The name axis counts a
        namespace's top-level names and this is a method; the signature axis compares
        constructors. `Optimizer`'s methods were read by nothing, so a name torch has,
        borch.ts has as `addParamGroup`, and neither Python surface had was invisible
        to every instrument here — `tests/test_class_methods.py` is the axis that asks
        now.

        The two refusals are torch's, measured: a dict is required, and a parameter
        already in a group is refused because two groups would step it twice.
        borch.ts refuses the second one; the first is a Python question and stops here.
        """
        if not isinstance(param_group, dict):
            raise TypeError(
                f"param_group must be a dict, but got {type(param_group)}")
        group = dict(param_group)
        params = group.get("params")
        # A single tensor rather than a list — torch wraps it, so one parameter needs
        # no brackets.
        params = [params] if not isinstance(params, (list, tuple)) else list(params)
        init = _js.JSON.parse("{}")
        arr = _js.Array.new()
        for p in params:
            arr.push(handle(p))
        init.params = arr
        if "lr" in group:
            init.lr = group["lr"]
        if "weight_decay" in group:
            init.weightDecay = group["weight_decay"]
        self._o.addParamGroup(init)

    def state_dict(self):
        """**Resuming a run hangs on this.**

        This was missing entirely — `__slots__`, so an `AttributeError`. Model
        weights save through `Module.state_dict` while the optimiser had no way
        to be saved at all, so training resumed on the GPU path **without its
        momentum or second moments.** Nothing raises; the loss curve jumps once.

        **The shape differs from torch's.** torch has
        `{"state": …, "param_groups": …}`; borch.ts has banks,
        `{tensors, numbers}`. Splitting that per parameter and attaching torch's
        names (`exp_avg` and the rest) would mean redesigning the other side,
        and all it buys is **reading somebody else's torch checkpoint** — a path
        already closed at the kernels. Saving ours and restoring ours works in
        this shape, and the golden holds it by value.

        `state_dict` hands back **the live slots, not a copy**, which is also
        what torch does. Keep stepping after saving one and the saved one moves
        too.
        """
        got = self._o.stateDict()
        return {
            "tensors": {str(k): wrap(getattr(got.tensors, k))
                        for k in _js.Object.keys(got.tensors)},
            "numbers": {str(k): getattr(got.numbers, k)
                        for k in _js.Object.keys(got.numbers)},
        }

    def load_state_dict(self, state):
        """Restore what `state_dict()` gave. Build the optimiser **with the
        same arguments first**, then call this."""
        obj = _js.Object.new()
        tensors = _js.Object.new()
        for key, value in state["tensors"].items():
            setattr(tensors, key, handle(value))
        numbers = _js.Object.new()
        for key, value in state["numbers"].items():
            setattr(numbers, key, value)
        obj.tensors = tensors
        obj.numbers = numbers
        self._o.loadStateDict(obj)
        return self

    @property
    def param_groups(self):
        """**Read as `opt.param_groups[0]["lr"]`.** That is how torch code
        writes it.

        `to_py()` alone is not enough — the JavaScript object stays a proxy
        rather than becoming a Python dict, and `["lr"]` does not reach it. And
        the learning rate has to be **the current value every time it is read**,
        so copying it once hides whatever the scheduler changed.
        """
        return [_Group(g) for g in self._o.paramGroups]


class _Group:
    """One parameter group. Reads like a dict, but each value is fetched from
    the JavaScript side as it is asked for."""

    __slots__ = ("_g",)

    def __init__(self, g):
        self._g = g

    def __getitem__(self, key):
        return getattr(self._g, key)

    def __setitem__(self, key, value):
        setattr(self._g, key, value)

    def get(self, key, default=None):
        got = getattr(self._g, key, None)
        return default if got is None else got


def _params(ps):
    # Parameters can arrive as a JS array — this is where `model.parameters()`
    # gets passed straight through.
    if hasattr(ps, "to_py"):
        ps = ps.to_py()
    return _js.Array.new(*[handle(p) for p in ps])


def _pair(two):
    """A two-element tuple, as a JavaScript array.

    **torch packs these pairs and borch.ts used to split them** — `betas`, `etas`,
    `step_sizes` and Adafactor's `eps` each became two positions over there, so this
    file wrote `betas[0], betas[1]` and every later argument sat one seat early on
    that side. borch.ts takes the pair now, and a Python tuple crossing as-is becomes
    a proxy that is neither an array nor a list, so it is laid out by hand — the same
    reason the padding layers convert theirs.
    """
    from js import Array as _Array

    out = _Array.new()
    for v in two:
        out.push(v)
    return out


def _opts(maximize, foreach, fused, capturable, differentiable):
    """borch.ts's trailing options object.

    **Built through `JSON.parse` rather than as a dict.** A Python dict crossing the
    FFI arrives as a `Map`-like proxy, and `{ maximize = false } = {}` destructuring
    reads `undefined` off it — the flag is accepted on this side, delivered to the far
    side, and silently dropped there. That failure looks exactly like the feature not
    being implemented, which is what it was mistaken for for a long time.

    **The four after `maximize` are carried, not judged here.** `capturable` and
    `differentiable` are refused — by borch.ts, one step further on, which is where the
    core's wording already lives. Deciding it twice would let the two answers drift, and
    carrying it proves the bridge actually delivers the word: a flag dropped on the way
    across looks exactly like a flag the far side chose to ignore.
    """
    flags = {"maximize": maximize, "foreach": foreach, "fused": fused,
             "capturable": capturable, "differentiable": differentiable}
    body = ",".join(f'"{k}":{str(bool(v)).lower()}' for k, v in flags.items())
    return _js.JSON.parse("{" + body + "}")


def SGD(params, lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0,
        nesterov=False, *, maximize=False,
        foreach=False, fused=False, capturable=False, differentiable=False):
    """torch's order. **`weight_decay` moved from fourth to fifth** and this call
    moved with it — a positional bridge is a bet that the far side's parameter order
    never changes, and `test_binding_arguments.py` is what collects on it."""
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.SGD.new(_params(params), lr, momentum, dampening,
                                  weight_decay, nesterov, bag))


def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
         amsgrad=False, *, maximize=False, decoupled_weight_decay=False,
         foreach=False, fused=False, capturable=False, differentiable=False):
    """**These two used to be accepted and refused.** `maximize` raised
    `NotImplementedError` and `amsgrad` was not there at all, on the reasoning that
    holding the position was better than dropping the flag — which was right while it
    lasted, and is a reason with nothing left to hold now that borch.ts carries both.

    `decoupled_weight_decay` is the third: torch reaches `AdamW`'s placement through
    this flag as well as through the other name, and absent from the far side the word
    was accepted and the **coupled** answer came back — 0.781 against 0.800 on the
    second step. A training curve that is merely slightly wrong.
    """
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.Adam.new(_params(params), lr, _pair(betas), eps,
                                   weight_decay, amsgrad,
                                   decoupled_weight_decay, bag))


def AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
          amsgrad=False, *, maximize=False,
          foreach=False, fused=False, capturable=False, differentiable=False):
    """**`weight_decay` defaults to 0.01 here and to 0 in `Adam`**, which is torch's
    split and most of the reason the two are separate names.

    borch.ts has had this class all along; only this file had no line for it, so
    `opt::AdamW` came back as `module 'borch_webgpu._optim' has no attribute 'AdamW'`.
    **An absence says so.** The defect a few lines below was a table sending arguments
    into the wrong seats, which said nothing at all for four of its eight.
    """
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.AdamW.new(_params(params), lr, _pair(betas),
                                    eps, weight_decay, amsgrad, bag))


def RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0,
            momentum=0.0, centered=False, *, maximize=False,
            foreach=False, fused=False, capturable=False, differentiable=False):
    """`momentum` and `centered` sit sixth and seventh — torch's seats. They were
    absent, so `RMSprop(p, 0.01, 0.99, 1e-8, 0, 0.9)` raised rather than adding a
    momentum buffer."""
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.RMSprop.new(_params(params), lr, alpha, eps,
                                      weight_decay, momentum, centered,
 bag))


def Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0,
            initial_accumulator_value=0.0, eps=1e-10, *, maximize=False,
            foreach=False, fused=False, capturable=False, differentiable=False):
    """`initial_accumulator_value` sits fifth, before `eps` — torch's order, and
    borch.ts moved with the core."""
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.Adagrad.new(_params(params), lr, lr_decay, weight_decay,
                                      initial_accumulator_value, eps,
 bag))


def Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0, *,
             maximize=False,
             foreach=False, fused=False, capturable=False, differentiable=False):
    # `maximize` is the core's now and is not carried across here yet. **Accepted so
    # the position is held, refused so it cannot be believed** — the shape `Adagrad`
    # above already uses. Dropping it instead would train in the wrong direction and
    # say nothing.
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.Adadelta.new(_params(params), lr, rho, eps, weight_decay,
 bag))


def Adamax(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, *,
           maximize=False,
           foreach=False, fused=False, capturable=False, differentiable=False):
    # `maximize` is the core's now and is not carried across here yet. **Accepted so
    # the position is held, refused so it cannot be believed** — the shape `Adagrad`
    # above already uses. Dropping it instead would train in the wrong direction and
    # say nothing.
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.Adamax.new(_params(params), lr, _pair(betas), eps,
                                     weight_decay, bag))


def NAdam(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
          momentum_decay=4e-3, decoupled_weight_decay=False, *,
          maximize=False,
          foreach=False, fused=False, capturable=False, differentiable=False):
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.NAdam.new(_params(params), lr, _pair(betas), eps,
                                    weight_decay, momentum_decay,
                                    decoupled_weight_decay, bag))


def RAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
          decoupled_weight_decay=False, *,
          maximize=False,
          foreach=False, fused=False, capturable=False, differentiable=False):
    # `maximize` is the core's now and is not carried across here yet. **Accepted so
    # the position is held, refused so it cannot be believed** — the shape `Adagrad`
    # above already uses. Dropping it instead would train in the wrong direction and
    # say nothing.
    #
    # **`decoupled_weight_decay` sits sixth and positionally**, which is where torch
    # puts it on this class alone — `Adam` has it keyword-only. Given the `*` a
    # position early it would take `weight_decay`'s place.
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.RAdam.new(_params(params), lr, _pair(betas), eps,
                                    weight_decay, decoupled_weight_decay, bag))


def ASGD(params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0, *,
         maximize=False,
         foreach=False, fused=False, capturable=False, differentiable=False):
    # `maximize` is the core's now and is not carried across here yet. **Accepted so
    # the position is held, refused so it cannot be believed** — the shape `Adagrad`
    # above already uses. Dropping it instead would train in the wrong direction and
    # say nothing.
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.ASGD.new(_params(params), lr, lambd, alpha, t0,
                                   weight_decay, bag))


def Rprop(params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50), *,
          maximize=False,
          foreach=False, fused=False, capturable=False, differentiable=False):
    # `maximize` is the core's now and is not carried across here yet. **Accepted so
    # the position is held, refused so it cannot be believed** — the shape `Adagrad`
    # above already uses. Dropping it instead would train in the wrong direction and
    # say nothing.
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.Rprop.new(_params(params), lr, _pair(etas),
                                    _pair(step_sizes), bag))


def Adafactor(params, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0,
              weight_decay=0.0, *,
              maximize=False,
              foreach=False, fused=False, capturable=False, differentiable=False):
    # `maximize` is the core's now and is not carried across here yet. **Accepted so
    # the position is held, refused so it cannot be believed** — the shape `Adagrad`
    # above already uses. Dropping it instead would train in the wrong direction and
    # say nothing.
    bag = _opts(maximize, foreach, fused, capturable, differentiable)
    return _Opt(_ts.optim.Adafactor.new(_params(params), lr, beta2_decay,
                                        _pair(eps), d, weight_decay,
 bag))


class LBFGS:
    """Quasi-Newton. **`step` takes a closure** because one step re-measures
    the loss several times.

    ## Why it is here and not in borch.ts

    This algorithm's **control flow depends on values** — whether `ys > 1e-10`,
    whether the gradient is under a threshold, whether the loss stopped
    falling. borch.ts has no synchronous read, so it cannot look at a number on
    the GPU where it stands. Here `run_sync` exists, so here is the place.

    Parameters stay on the GPU and only **the flattened gradient vector**
    crosses. LBFGS is for small problems anyway, so that cost stays small.

    **The core carries a second copy of this algorithm.** That follows from the
    two packages deliberately not importing each other, and the golden asks the
    same three cases of both so a divergence is caught.

    No line search yet — `line_search_fn` is refused loudly.
    """

    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None,
                 tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
                 line_search_fn=None):
        got = params.to_py() if hasattr(params, "to_py") else list(params)
        self._ps = [wrap(handle(p)) for p in got]
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        self.param_groups = [dict(
            lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=history_size, line_search_fn=line_search_fn)]
        self._state = {}

    def zero_grad(self, set_to_none=True):
        for p in self._ps:
            p.grad = None

    def _flat_grad(self):
        parts = []
        for p in self._ps:
            g = p.grad
            parts.append(_np.zeros(int(handle(p).size), dtype=_np.float32)
                         if g is None else
                         _np.asarray(g.numpy(), dtype=_np.float32).reshape(-1))
        return _np.concatenate(parts)

    def _add_step(self, size, direction):
        from ._ops import no_grad as _no_grad
        at = 0
        # **Edited inside `no_grad`.** Parameters are leaves with gradients on,
        # and changing one in place is refused everywhere else — an optimiser is
        # the one thing allowed to, so the door opens here.
        with _no_grad():
            for p in self._ps:
                h = handle(p)
                n = int(h.size)
                shape = [int(v) for v in h.shape]
                moved = (_np.asarray(p.numpy(), dtype=_np.float32).reshape(-1)
                         + size * direction[at:at + n])
                h.copyFrom(handle(tensor(moved.reshape(shape))))
                at += n

    def step(self, closure):
        group = self.param_groups[0]
        if group["line_search_fn"] is not None:
            raise RuntimeError(
                f"LBFGS(line_search_fn={group['line_search_fn']!r}) is not here yet.")
        lr, max_iter = group["lr"], group["max_iter"]
        max_eval = group["max_eval"]
        tol_grad, tol_change = group["tolerance_grad"], group["tolerance_change"]
        history = group["history_size"]
        st = self._state
        st.setdefault("n_iter", 0)

        orig = closure()
        loss = float(orig.item() if hasattr(orig, "item") else orig)
        evals = 1
        flat = self._flat_grad()
        if _np.abs(flat).max() <= tol_grad:
            return orig

        d, t = st.get("d"), st.get("t")
        old_dirs = st.get("old_dirs", [])
        old_stps = st.get("old_stps", [])
        ro = st.get("ro", [])
        h_diag = st.get("h_diag", 1.0)
        prev_flat, prev_loss = st.get("prev_flat"), st.get("prev_loss")

        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            st["n_iter"] += 1
            if st["n_iter"] == 1:
                d = -flat
                old_dirs, old_stps, ro, h_diag = [], [], [], 1.0
            else:
                y = flat - prev_flat
                s = d * t
                ys = float(y @ s)
                if ys > 1e-10:
                    if len(old_dirs) == history:
                        old_dirs.pop(0), old_stps.pop(0), ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    h_diag = ys / float(y @ y)
                al = [0.0] * len(old_dirs)
                q = -flat
                for i in range(len(old_dirs) - 1, -1, -1):
                    al[i] = float(old_stps[i] @ q) * ro[i]
                    q = q - al[i] * old_dirs[i]
                r = q * h_diag
                for i in range(len(old_dirs)):
                    be = float(old_dirs[i] @ r) * ro[i]
                    r = r + old_stps[i] * (al[i] - be)
                d = r

            prev_flat, prev_loss = flat.copy(), loss
            t = min(1.0, 1.0 / _np.abs(flat).sum()) * lr if st["n_iter"] == 1 else lr
            if float(flat @ d) > -tol_change:
                break

            self._add_step(t, d)
            if n_iter != max_iter:
                got = closure()
                loss = float(got.item() if hasattr(got, "item") else got)
                flat = self._flat_grad()
                evals += 1
                if _np.abs(flat).max() <= tol_grad:
                    break
            if n_iter == max_iter or evals >= max_eval:
                break
            if _np.abs(d * t).max() <= tol_change:
                break
            if abs(loss - prev_loss) < tol_change:
                break

        st.update(d=d, t=t, old_dirs=old_dirs, old_stps=old_stps, ro=ro,
                  h_diag=h_diag, prev_flat=prev_flat, prev_loss=prev_loss)
        return orig


class _Sched:
    # `_keep` holds on to the proxy for a Python function handed to JavaScript.
    # Unheld, it is destroyed the moment the call returns, and the scheduler
    # calling it **later** dies with "borrowed proxy was automatically
    # destroyed" — the same thing the sister library hit in `tf.grad`.
    __slots__ = ("_s", "_keep")

    def __init__(self, s, keep=None):
        self._s = s
        self._keep = keep

    def step(self, *args):
        self._s.step(*args)

    def get_last_lr(self):
        return list(self._s.getLastLr())

    def state_dict(self):
        """**Restoring only the optimiser puts the learning rate back to its
        starting value.**

        This was missing too. A run cooled halfway heats back up the moment it
        resumes, and nothing errors — the loss curve simply climbs once and
        comes back down.

        The other side keeps a dict of numbers, so it passes straight through.
        The shape differs from torch's (`last_epoch`, `_step_count`), but saving
        ours and restoring ours works with this.
        """
        got = self._s.stateDict()
        return {str(k): getattr(got, k) for k in _js.Object.keys(got)}

    def load_state_dict(self, state):
        obj = _js.Object.new()
        for key, value in state.items():
            setattr(obj, key, value)
        self._s.loadStateDict(obj)
        return self


# Schedulers are called with keyword arguments too —
# `StepLR(o, step_size=2, gamma=0.5)`. The JavaScript side takes positions only,
# so they are unrolled here.
_SCHED_ARGS = {
    "StepLR": ("step_size", "gamma", "last_epoch"),
    "MultiStepLR": ("milestones", "gamma", "last_epoch"),
    "ExponentialLR": ("gamma", "last_epoch"),
    "CosineAnnealingLR": ("T_max", "eta_min", "last_epoch"),
    "LambdaLR": ("lr_lambda", "last_epoch"),
    # **The comment here used to read "borch.ts has no `mode`", and it was true when it
    # was written.** `mode` was then added to borch.ts as torch's second argument, and
    # this row was not touched — so every name below shifted one seat left. `factor=0.5`
    # arrived as `mode`, which is the error the binding golden was reporting fifty times
    # over: `mode must be 'min' or 'max', got 0.5`.
    #
    # It raised only because that constructor checks its own argument. The five names
    # missing from the end did not raise at all: `threshold_mode`, `cooldown`, `min_lr`
    # and `eps` were simply unreachable through the binding, and a caller setting a
    # cooldown got silence and no cooldown.
    #
    # **A comment stating a fact about another module is the same hazard as the table
    # it explains** — it goes stale the same way, is read as current, and here it was
    # the reason nobody re-checked the row. The measurement that found this compares
    # every row against `borch-ts/src/optim.ts` directly; a sentence cannot.
    "ReduceLROnPlateau": ("mode", "factor", "patience", "threshold",
                          "threshold_mode", "cooldown", "min_lr", "eps"),
    "ConstantLR": ("factor", "total_iters", "last_epoch"),
    "LinearLR": ("start_factor", "end_factor", "total_iters", "last_epoch"),
    "PolynomialLR": ("total_iters", "power", "last_epoch"),
    "MultiplicativeLR": ("lr_lambda", "last_epoch"),
    "CosineAnnealingWarmRestarts": ("T_0", "T_mult", "eta_min", "last_epoch"),
    # **This table describes another module's parameter order**, which is the one kind
    # of entry that stays plausible after it goes wrong — it is data, not a call, so
    # nothing type-checks it and nothing raises. `OneCycleLR` grew from six names to
    # thirteen when the core took torch's list; had this stayed at six, `div_factor`
    # would have gone into `epochs`' seat and the schedule would have been wrong with
    # no error anywhere.
    "OneCycleLR": ("max_lr", "total_steps", "epochs", "steps_per_epoch", "pct_start",
                   "anneal_strategy", "cycle_momentum", "base_momentum",
                   "max_momentum", "div_factor", "final_div_factor", "three_phase",
                   "last_epoch"),
    # The middle used to be missing and `last_epoch` was left out with it — appending
    # it alone would have put it in `scale_fn`'s seat over there, which is the shape
    # this table's own comments have been wrong about twice. borch.ts carries all six
    # now, so the row is torch's whole list.
    "CyclicLR": ("base_lr", "max_lr", "step_size_up", "step_size_down", "mode",
                 "gamma", "scale_fn", "scale_mode", "cycle_momentum",
                 "base_momentum", "max_momentum", "last_epoch"),
}


# A slot nobody filled, kept apart from a slot somebody filled with `None`. torch has
# arguments whose default *is* `None` — `steps_per_epoch`, `step_size_down` — and the
# two have to cross the boundary differently.
_DEFAULT = object()


def _undefined():
    """JavaScript's `undefined`, which is what makes a TypeScript default apply."""
    import js
    return js.undefined


def _sched(js_name):
    def make(opt, *args, **kw):
        from ._ops import _arg
        out = list(args)
        order = _SCHED_ARGS.get(js_name, ())
        # **A slot given twice is a refusal.** `StepLR(opt, 2, step_size=3)` used the
        # keyword and answered; torch raises `TypeError: got multiple values`.
        clash = [key for i, key in enumerate(order) if i < len(args) and key in kw]
        if clash:
            raise TypeError(
                f"{js_name}() got multiple values for argument '{clash[0]}'")
        for i, key in enumerate(order):
            if key in kw:
                while len(out) <= i:
                    out.append(_DEFAULT)
                out[i] = kw[key]
        while out and out[-1] is _DEFAULT:
            out.pop()
        # **This distinction is belt-and-braces, and the comment that used to be here
        # named it as the cause of a defect it did not cause.**
        #
        # It said: a TypeScript default applies for `undefined` and not for `null`, so
        # a skipped middle slot arriving as `None` would hand `pctStart` a null and
        # every rate on the curve would come back `NaN`. The first half is true about
        # JavaScript. The second half is not what happens here — **measured in the
        # browser: Pyodide hands a Python `None` across as `undefined`, and the
        # TypeScript default does apply to it.** So both branches below produce the
        # same thing, and this line changes no behaviour.
        #
        # The `NaN` was real and came from the other half of the same commit: the table
        # above listed five names where borch.ts takes thirteen, so `div_factor` landed
        # in `stepsPerEpoch`'s seat and `pct_start` in `epochs`'. Widening the row is
        # what fixed it. **A true sentence standing where the cause belongs is the
        # defect this repository spent a day removing**, and it reached the remedy
        # itself here.
        #
        # Kept because it costs nothing and Pyodide's conversion is not ours to
        # guarantee across versions — but kept as a guard, not as an explanation. If it
        # ever becomes load-bearing, something changed underneath and this comment is
        # the place that says so.
        out = [None if a is None else (_undefined() if a is _DEFAULT else a)
               for a in out]
        # Python functions become proxies and are held — `LambdaLR` calls one later.
        keep = None
        ready = []
        for a in out:
            if callable(a) and not isinstance(a, (int, float, str)):
                from pyodide.ffi import create_proxy
                keep = create_proxy(a)
                ready.append(keep)
            else:
                ready.append(_arg(a))
        s = getattr(_ts.optim, js_name).new(opt._o, *ready)
        # **Epoch zero is applied right after construction.** torch does this in
        # the constructor; on the TypeScript side subclass fields are filled in
        # after `super()`, so it cannot happen there. `ReduceLROnPlateau` is
        # outside this lineage and has no such point — it judges by the value it
        # is handed, so it has no notion of an epoch at all.
        if hasattr(s, "start"):
            s.start()
        return _Sched(s, keep)
    return make


# **`torch.optim.lr_scheduler` is a namespace.** The golden calls through it.
class _LRScheduler:
    StepLR = staticmethod(_sched("StepLR"))
    MultiStepLR = staticmethod(_sched("MultiStepLR"))
    ExponentialLR = staticmethod(_sched("ExponentialLR"))
    CosineAnnealingLR = staticmethod(_sched("CosineAnnealingLR"))
    LambdaLR = staticmethod(_sched("LambdaLR"))
    ReduceLROnPlateau = staticmethod(_sched("ReduceLROnPlateau"))
    ConstantLR = staticmethod(_sched("ConstantLR"))
    LinearLR = staticmethod(_sched("LinearLR"))
    PolynomialLR = staticmethod(_sched("PolynomialLR"))
    MultiplicativeLR = staticmethod(_sched("MultiplicativeLR"))
    CosineAnnealingWarmRestarts = staticmethod(_sched("CosineAnnealingWarmRestarts"))
    OneCycleLR = staticmethod(_sched("OneCycleLR"))
    CyclicLR = staticmethod(_sched("CyclicLR"))
    LRScheduler = _Sched

    @staticmethod
    def SequentialLR(optimizer, schedulers, milestones, last_epoch=-1):
        """Hand the scheduler list and the milestones over as JS arrays.

        **Not with `Array.new`.** Given a single number that becomes `Array(3)`,
        which is **an empty array of length 3** rather than `[3]`. Milestones are
        usually a single value, so it landed exactly there, and the symptom was
        a learning rate that never changed at the milestone — the second time
        this repository fell into that trap.
        """
        return _Sched(_ts.optim.SequentialLR.new(
            optimizer._o, _js.Array.of(*[s._s for s in schedulers]),
            _js.Array.of(*[int(m) for m in milestones])))

    @staticmethod
    def ChainedScheduler(schedulers, optimizer=None):
        return _Sched(_ts.optim.ChainedScheduler.new(
            _js.Array.of(*[s._s for s in schedulers])))


lr_scheduler = _LRScheduler()

StepLR = _sched("StepLR")
MultiStepLR = _sched("MultiStepLR")
ExponentialLR = _sched("ExponentialLR")
CosineAnnealingLR = _sched("CosineAnnealingLR")
LambdaLR = _sched("LambdaLR")
