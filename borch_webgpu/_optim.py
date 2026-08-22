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


def SGD(params, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0,
        nesterov=False, *, maximize=False):
    """torch's order. **`weight_decay` moved from fourth to fifth** and this call
    moved with it — a positional bridge is a bet that the far side's parameter order
    never changes, and `test_binding_arguments.py` is what collects on it."""
    return _Opt(_ts.optim.SGD.new(_params(params), lr, momentum, dampening,
                                  weight_decay, nesterov,
                                  _js.JSON.parse(f'{{"maximize":{str(bool(maximize)).lower()}}}')))


def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.Adam.new(_params(params), lr, betas[0], betas[1],
                                   eps, weight_decay))


def RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.RMSprop.new(_params(params), lr, alpha, eps,
                                      weight_decay))


def Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0,
            initial_accumulator_value=0.0, eps=1e-10, *, maximize=False):
    """`initial_accumulator_value` sits fifth, before `eps` — torch's order, and
    borch.ts moved with the core. `maximize` is not carried across yet; it is
    accepted so the position is held and refused so it cannot be believed."""
    if maximize:
        raise NotImplementedError(
            "Adagrad(maximize=True) is not carried into the browser yet — "
            "the argument is here so it cannot take another's place.")
    return _Opt(_ts.optim.Adagrad.new(_params(params), lr, lr_decay, weight_decay,
                                      initial_accumulator_value, eps))


def Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
    return _Opt(_ts.optim.Adadelta.new(_params(params), lr, rho, eps, weight_decay))


def Adamax(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.Adamax.new(_params(params), lr, betas[0], betas[1], eps,
                                     weight_decay))


def NAdam(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
          momentum_decay=4e-3):
    return _Opt(_ts.optim.NAdam.new(_params(params), lr, betas[0], betas[1], eps,
                                    weight_decay, momentum_decay))


def RAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.RAdam.new(_params(params), lr, betas[0], betas[1], eps,
                                    weight_decay))


def ASGD(params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0):
    return _Opt(_ts.optim.ASGD.new(_params(params), lr, lambd, alpha, t0,
                                   weight_decay))


def Rprop(params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
    return _Opt(_ts.optim.Rprop.new(_params(params), lr, etas[0], etas[1],
                                    step_sizes[0], step_sizes[1]))


def Adafactor(params, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0,
              weight_decay=0.0):
    return _Opt(_ts.optim.Adafactor.new(_params(params), lr, beta2_decay,
                                        eps[0], eps[1], d, weight_decay))


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
    "StepLR": ("step_size", "gamma"),
    "MultiStepLR": ("milestones", "gamma"),
    "ExponentialLR": ("gamma",),
    "CosineAnnealingLR": ("T_max", "eta_min"),
    "LambdaLR": ("lr_lambda",),
    # borch.ts has no `mode` — there is only `rel`, so there is no slot for it.
    "ReduceLROnPlateau": ("factor", "patience", "threshold"),
    "ConstantLR": ("factor", "total_iters"),
    "LinearLR": ("start_factor", "end_factor", "total_iters"),
    "PolynomialLR": ("total_iters", "power"),
    "MultiplicativeLR": ("lr_lambda",),
    "CosineAnnealingWarmRestarts": ("T_0", "T_mult", "eta_min"),
    "OneCycleLR": ("max_lr", "total_steps", "pct_start", "div_factor",
                   "final_div_factor"),
    "CyclicLR": ("base_lr", "max_lr", "step_size_up", "step_size_down", "mode",
                 "gamma"),
}


def _sched(js_name):
    def make(opt, *args, **kw):
        from ._ops import _arg
        out = list(args)
        for i, key in enumerate(_SCHED_ARGS.get(js_name, ())):
            if key in kw:
                while len(out) <= i:
                    out.append(None)
                out[i] = kw[key]
        while out and out[-1] is None:
            out.pop()
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
