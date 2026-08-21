"""A piece of borch, split out. __init__ gathers the public names."""

import math as _math

import numpy as _np

from ._tensor import (
    Tensor,
)
from ._ops import (
    _Namespace, _unsupported,
)
from ._base import (
    _math, _np,
)

# ================================================================ optim

class Optimizer:
    """A torch optimiser carries a list called `param_groups`.

    The standard path for reading and writing the learning rate is
    `opt.param_groups[0]["lr"]`, and the schedulers change it there too. Keeping
    it as `opt.lr` is shorter and **stops other people's code from running and
    makes the schedulers unusable.**
    """

    def __init__(self, params, defaults):
        params = list(params)
        if params and isinstance(params[0], dict):
            self.param_groups = [dict(defaults, **g, ) for g in params]
            for g in self.param_groups:
                g["params"] = list(g["params"])
        else:
            self.param_groups = [dict(defaults, params=params)]
        self.state = {}
        self.defaults = defaults

    @property
    def params(self):
        return [p for g in self.param_groups for p in g["params"]]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def _state(self, p):
        return self.state.setdefault(id(p), {})

    def state_dict(self):
        """torch's shape — the "resume training" chapter 6 teaches hangs on
        this.

        Adam remembers a step size per parameter. Discarding that memory and
        resuming makes the loss jump once and then come back down — no error, and
        a curve that looks odd.
        """
        order = {id(p): i for i, p in enumerate(self.params)}
        return {
            "state": {order[k]: {n: (Tensor(v.copy()) if isinstance(v, _np.ndarray) else v)
                                 for n, v in st.items()}
                      for k, st in self.state.items() if k in order},
            "param_groups": [{k: v for k, v in g.items() if k != "params"}
                             | {"params": [order[id(p)] for p in g["params"]]}
                             for g in self.param_groups],
        }

    def load_state_dict(self, state):
        params = self.params
        self.state = {}
        for index, st in state.get("state", {}).items():
            self.state[id(params[int(index)])] = {
                n: (v.data.copy() if isinstance(v, Tensor) else v) for n, v in st.items()}
        for group, saved in zip(self.param_groups, state.get("param_groups", [])):
            for key, value in saved.items():
                if key != "params":
                    group[key] = value
        return self

    def step(self):
        raise NotImplementedError

    def __repr__(self):
        lr = self.param_groups[0].get("lr")
        return f"{type(self).__name__} (lr: {lr})"


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                if group["momentum"]:
                    st = self._state(p)
                    buf = st.get("momentum_buffer")
                    buf = g if buf is None else group["momentum"] * buf + g
                    st["momentum_buffer"] = buf
                    g = buf
                p._array = p.data - group["lr"] * g


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    decoupled = False          # True for AdamW — the decay applies to the weights directly rather than to the gradient

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_avg_sq", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"] and not self.decoupled:
                    g = g + group["weight_decay"] * p.data
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * (g * g)
                mh = st["exp_avg"] / (1 - b1 ** st["step"])
                vh = st["exp_avg_sq"] / (1 - b2 ** st["step"])
                new = p.data - group["lr"] * mh / (_np.sqrt(vh) + group["eps"])
                if group["weight_decay"] and self.decoupled:
                    new = new - group["lr"] * group["weight_decay"] * p.data
                p._array = new


class AdamW(Adam):
    """The same as Adam with the weight decay applied **to the weights directly
    rather than to the gradient.**"""

    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("square_avg", _np.zeros_like(p.data))
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["square_avg"] = (group["alpha"] * st["square_avg"]
                                    + (1 - group["alpha"]) * g * g)
                p._array = p.data - group["lr"] * g / (_np.sqrt(st["square_avg"]) + group["eps"])




class Adagrad(Optimizer):
    """**Keeps adding** the squared gradients — it only shrinks, never grows.

    `RMSprop` puts an exponential moving average in the same place and forgets
    the old ones; this one does not forget. So run long enough the step size
    converges to zero — that is this optimiser's nature and not a defect.
    """

    def __init__(self, params, lr=0.01, lr_decay=0.0, weight_decay=0.0, eps=1e-10):
        super().__init__(params, dict(lr=lr, lr_decay=lr_decay,
                                      weight_decay=weight_decay, eps=eps))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("sum", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["sum"] = st["sum"] + g * g
                lr = group["lr"] / (1 + (st["step"] - 1) * group["lr_decay"])
                p._array = p.data - lr * g / (_np.sqrt(st["sum"]) + group["eps"])


class Adadelta(Optimizer):
    """**The learning rate is barely used.** The step size is built from the
    history of the updates themselves.

    Which is why the default `lr` is 1.0 — given the value another optimiser
    takes, it is wildly large.
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, rho=rho, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            rho, eps = group["rho"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("square_avg", _np.zeros_like(p.data))
                st.setdefault("acc_delta", _np.zeros_like(p.data))
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["square_avg"] = rho * st["square_avg"] + (1 - rho) * g * g
                delta = (_np.sqrt(st["acc_delta"] + eps)
                         / _np.sqrt(st["square_avg"] + eps)) * g
                st["acc_delta"] = rho * st["acc_delta"] + (1 - rho) * delta * delta
                p._array = p.data - group["lr"] * delta


class Adamax(Optimizer):
    """Adam's second moment kept as **a maximum rather than a mean of squares.**
    The infinity-norm version."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_inf", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_inf"] = _np.maximum(b2 * st["exp_inf"],
                                            _np.abs(g) + group["eps"])
                bias = 1 - b1 ** st["step"]
                p._array = p.data - (group["lr"] / bias) * st["exp_avg"] / st["exp_inf"]


class NAdam(Optimizer):
    """Adam with Nesterov's look-ahead attached.

    **The momentum coefficient changes every step** — a sequence that grows
    slowly by `momentum_decay`, and **the running product** of that sequence has
    to be carried along. Kept as a constant, the first few steps quietly
    diverge.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum_decay=4e-3):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      momentum_decay=momentum_decay))

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            psi = group["momentum_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("mu_product", 1.0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_avg_sq", _np.zeros_like(p.data))
                st["step"] += 1
                t = st["step"]
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                mu = b1 * (1 - 0.5 * 0.96 ** (t * psi))
                mu_next = b1 * (1 - 0.5 * 0.96 ** ((t + 1) * psi))
                st["mu_product"] = st["mu_product"] * mu
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * g * g
                denom = _np.sqrt(st["exp_avg_sq"] / (1 - b2 ** t)) + group["eps"]
                new = p.data - group["lr"] * (1 - mu) / (1 - st["mu_product"]) * g / denom
                new = new - group["lr"] * mu_next / (
                    1 - st["mu_product"] * mu_next) * st["exp_avg"] / denom
                p._array = new


class RAdam(Optimizer):
    """Adam that **does not use the adaptive step size** early on.

    With few samples of the second moment the variance is large and the first few
    steps jump, which is a known property of Adam, and this one crosses that
    stretch like SGD. Leaving the boundary out (`rho > 5`) makes the values equal
    Adam's, so the golden walks five steps.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            rho_inf = 2.0 / (1 - b2) - 1
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_avg_sq", _np.zeros_like(p.data))
                st["step"] += 1
                t = st["step"]
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * g * g
                mh = st["exp_avg"] / (1 - b1 ** t)
                rho = rho_inf - 2 * t * b2 ** t / (1 - b2 ** t)
                if rho > 5.0:
                    rect = _math.sqrt(((rho - 4) * (rho - 2) * rho_inf)
                                      / ((rho_inf - 4) * (rho_inf - 2) * rho))
                    denom = _np.sqrt(st["exp_avg_sq"] / (1 - b2 ** t)) + group["eps"]
                    p._array = p.data - group["lr"] * mh * rect / denom
                else:
                    # No adaptive step size — this is the stretch that runs like
                    # SGD.
                    p._array = p.data - group["lr"] * mh


class ASGD(Optimizer):
    """Averaged SGD. **The learning rate shrinks every step**, and from some
    point the parameters are averaged.

    `eta` shrinks on its own as `lr / (1 + lambd·lr·step)^alpha`, and `mu` is the
    weight of the average. The default `t0` is a million, so in ordinary training
    `mu` is 1 and `ax` is simply a copy of the parameters — seeing the averaging
    actually run means lowering `t0`.

    **The decay is multiplicative.** `param *= (1 - lambd·eta)` comes first and
    the gradient is subtracted after. A different place from the additive form
    (`weight_decay`), and with both present both apply.
    """

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                                      weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            lr, lambd = group["lr"], group["lambd"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("eta", lr)
                st.setdefault("mu", 1.0)
                st.setdefault("ax", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                eta, mu = st["eta"], st["mu"]
                p._array = p.data * (1 - lambd * eta) - eta * g
                # `mu` of 1 makes it **a copy** rather than an average — adding
                # would double it.
                st["ax"] = (p.data.copy() if mu == 1
                            else st["ax"] + (p.data - st["ax"]) * mu)
                step = st["step"]
                st["eta"] = lr / ((1 + lambd * lr * step) ** group["alpha"])
                st["mu"] = 1.0 / max(1.0, step - group["t0"])


class Rprop(Optimizer):
    """Looks at **the sign only.** The magnitude goes unused and the step width
    grows and shrinks per element.

    A sign that holds grows the width by `etas[1]`, and a sign that flips shrinks
    it by `etas[0]`. **An element whose sign flipped does not take that step at
    all** — its gradient is set to zero, and so the next step's "previous
    gradient" is zero too. Without those two lines the values are plausibly
    different, and an input whose signs never flip never catches it.
    """

    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        super().__init__(params, dict(lr=lr, etas=etas, step_sizes=step_sizes))

    def step(self):
        for group in self.param_groups:
            eta_minus, eta_plus = group["etas"]
            low, high = group["step_sizes"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("prev", _np.zeros_like(p.data))
                st.setdefault("step_size", _np.full_like(p.data, group["lr"]))
                st["step"] += 1
                g = _np.array(p.grad.data, copy=True)
                sign = _np.sign(g * st["prev"])
                factor = _np.where(sign > 0, eta_plus,
                                   _np.where(sign < 0, eta_minus, 1.0))
                st["step_size"] = _np.clip(st["step_size"] * factor, low, high)
                g[sign < 0] = 0.0
                p._array = p.data - _np.sign(g) * st["step_size"]
                st["prev"] = g


class Adafactor(Optimizer):
    """Adam that carries the second moment **split into rows and columns.**

    Adam carries one variance per parameter, so the memory costs as much as the
    weights. Here only a row mean and a column mean are carried and the rest is
    revived from their outer product — `R + C` where `(R, C)` would go. The
    method exists to fit large language models into memory.

    **A 1-D parameter is not split** — there is only one axis to split, so the
    variance is carried plainly. The divergence starts at the state keys
    (`variance` versus `row_var` and `col_var`). Asked in 1-D only, the whole
    point of this optimisation never runs.
    """

    def __init__(self, params, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3),
                 d=1.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, beta2_decay=beta2_decay, eps=eps,
                                      d=d, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            lr, d = group["lr"], group["d"]
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                one = eps1 if eps1 is not None else _np.finfo(p.data.dtype).eps
                st = self._state(p)
                st.setdefault("step", 0)
                st["step"] += 1
                step = float(st["step"])
                blend = step ** group["beta2_decay"]
                rho = min(lr, 1.0 / _math.sqrt(step))
                alpha = max(eps2, _np.linalg.norm(p.data.reshape(-1))
                            / _math.sqrt(p.data.size)) * rho
                if group["weight_decay"]:
                    p._array = p.data * (1 - lr * group["weight_decay"])

                if g.ndim > 1:
                    st.setdefault("row_var", _np.zeros(g.shape[:-1] + (1,),
                                                       dtype=g.dtype))
                    st.setdefault("col_var", _np.zeros(g.shape[:-2] + (1, g.shape[-1]),
                                                       dtype=g.dtype))
                    row_mean = (g * g).mean(axis=-1, keepdims=True)
                    col_mean = (g * g).mean(axis=-2, keepdims=True)
                    st["row_var"] += (row_mean - st["row_var"]) * blend
                    st["col_var"] += (col_mean - st["col_var"]) * blend
                    var = st["row_var"] @ st["col_var"]
                    var = var / _np.maximum(st["row_var"].mean(axis=-2, keepdims=True),
                                            one)
                else:
                    st.setdefault("variance", _np.zeros_like(g))
                    st["variance"] += (g * g - st["variance"]) * blend
                    var = st["variance"].copy()

                update = g / _np.sqrt(_np.maximum(var, one * one))
                denom = max(1.0, _np.linalg.norm(update.reshape(-1))
                            / (_math.sqrt(update.size) * d))
                p._array = p.data - (alpha / denom) * update


class LBFGS(Optimizer):
    """A quasi-Newton method. **`step` takes a closure** — because it re-measures
    the loss several times within one call.

    Other optimisers take one step per set of gradients; this one loops
    `max_iter` times inside and asks for the loss and the gradients again each
    time. So the training loop has a different shape, and without a closure it
    can do nothing.

    **There is no line search yet.** `line_search_fn="strong_wolfe"` is refused
    loudly — going quietly with a fixed step size makes the convergence come out
    differently, and that difference shows in the curve rather than in a value.
    """

    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None,
                 tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100,
                 line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        super().__init__(params, dict(
            lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=history_size, line_search_fn=line_search_fn))
        if len(self.param_groups) != 1:
            raise ValueError("LBFGS takes a single parameter group.")
        self._global = {}

    def _flat_grad(self):
        return _np.concatenate([
            (_np.zeros(p.data.size, dtype=p.data.dtype) if p.grad is None
             else p.grad.data.reshape(-1)) for p in self.params])

    def _add_step(self, size, direction):
        at = 0
        for p in self.params:
            n = p.data.size
            p._array = p.data + size * direction[at:at + n].reshape(p.data.shape)
            at += n

    def step(self, closure):                                    # noqa: D102
        group = self.param_groups[0]
        if group["line_search_fn"] is not None:
            _unsupported(f"LBFGS(line_search_fn={group['line_search_fn']!r})")
        lr = group["lr"]
        max_iter, max_eval = group["max_iter"], group["max_eval"]
        tol_grad, tol_change = group["tolerance_grad"], group["tolerance_change"]
        history = group["history_size"]
        st = self._global
        st.setdefault("n_iter", 0)

        orig = closure()
        loss = float(orig)
        evals = 1
        flat = self._flat_grad()
        if _np.abs(flat).max() <= tol_grad:
            return orig

        d = st.get("d")
        t = st.get("t")
        old_dirs = st.get("old_dirs", [])
        old_stps = st.get("old_stps", [])
        ro = st.get("ro", [])
        h_diag = st.get("h_diag", 1.0)
        prev_flat = st.get("prev_flat")
        prev_loss = st.get("prev_loss")

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
                # The two-loop recursion — it produces a direction without
                # building the inverse Hessian.
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

            prev_flat = flat.copy()
            prev_loss = loss
            t = min(1.0, 1.0 / _np.abs(flat).sum()) * lr if st["n_iter"] == 1 else lr
            gtd = float(flat @ d)
            if gtd > -tol_change:
                break

            self._add_step(t, d)
            if n_iter != max_iter:
                # No re-measure on the last iteration — torch is the same.
                loss = float(closure())
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


class _Scheduler:
    """A scheduler changes the lr in `optimizer.param_groups`. Called once per
    epoch."""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        # **The baseline is `initial_lr`, not the current lr.** It is stamped
        # onto the optimiser once (`setdefault`), and schedulers built afterwards
        # see **the same** baseline.
        #
        # At first the lr at construction time was the baseline. Used alone the
        # two are equal and nothing catches it, and chaining schedulers makes the
        # second take as its baseline what the first had already cut —
        # `SequentialLR` continued at 0.05 where it should have returned to 0.2
        # at the milestone, a max diff of 1.5e-01.
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        """**The optimiser is not carried** — torch is the same. Carried, the two
        hold each other in a cycle.

        In resumed training, restoring the optimiser alone and building a fresh
        scheduler sends the learning rate **back to its first value.** Training
        that had been half cooled goes hot again, and a loss that was descending
        rises once and comes back down — with no error. The golden holds this
        place as `opt::StepLR/이어서 학습하기`.

        A subclass carrying more state of its own needs nothing written here.
        Everything in `__dict__` except the optimiser is carried, so things like
        `T_cur` follow along on their own — listing the names instead means
        updating that list for every scheduler added, and forgetting leaves
        **that one scheduler quietly unable to resume.**
        """
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state):
        self.__dict__.update(state)
        return self


class StepLR(_Scheduler):
    """Multiply by gamma every step_size epochs. Chapter 4's "big strides far
    out, careful up close".

    **Recursive** — the reason `ExponentialLR` below writes down. This was the
    one place with a closed form,
    `base * gamma ** (last_epoch // step_size)`.

    Run alone from the start the two produce **the same sequence.** Which is why
    `StepLR/자취` was green for a long time. They diverge when a fresh scheduler
    is built on an optimiser whose lr has already moved — that is, **when
    resuming.** At that moment the closed form puts the learning rate back to its
    first value (0.05 to 0.2). Half-cooled training goes hot again, with no
    error. `opt::StepLR/이어서 학습하기` holds this place.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        now = [g["lr"] for g in self.optimizer.param_groups]
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return now
        return [lr * self.gamma for lr in now]


class MultiStepLR(_Scheduler):
    """Cuts at the milestones only. **Recursive** — fixed for `StepLR`'s reason.

    A milestone written twice (`[3, 3]`) multiplies twice at that point — torch
    does that, and so did the closed form, so the count is taken here too.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones, self.gamma = sorted(milestones), gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        now = [g["lr"] for g in self.optimizer.param_groups]
        hits = self.milestones.count(self.last_epoch)
        if hits == 0:
            return now
        return [lr * self.gamma ** hits for lr in now]


class ExponentialLR(_Scheduler):
    """**Recursive** — it multiplies the current learning rate. It does not
    recompute from the original one.

    Used alone the two produce the same sequence. They diverge **when another
    scheduler touches the same lr** — overlapped through `ChainedScheduler`, the
    recursive form stacks on the other's result and the closed form overwrites
    what the other did. torch is recursive, and so is this.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [g["lr"] for g in self.optimizer.param_groups]
        return [g["lr"] * self.gamma for g in self.optimizer.param_groups]


class CosineAnnealingLR(_Scheduler):
    """Descends along a cosine over T_max epochs. It settles smoothly at the
    end."""

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base - self.eta_min)
                * (1 + _math.cos(_math.pi * self.last_epoch / self.T_max)) / 2
                for base in self.base_lrs]


class LambdaLR(_Scheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.lr_lambda(self.last_epoch) for base in self.base_lrs]


class ConstantLR(_Scheduler):
    """**Held down until `total_iters` and then back to normal.** The simplest
    form of a warm-up."""

    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
        self.factor, self.total_iters = factor, total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Recursive. Both the cut and the restore happen **once each, at that
        # moment.**
        groups = self.optimizer.param_groups
        if self.last_epoch == 0:
            return [g["lr"] * self.factor for g in groups]
        if self.last_epoch != self.total_iters:
            return [g["lr"] for g in groups]
        return [g["lr"] / self.factor for g in groups]


class LinearLR(_Scheduler):
    """Moves **in a straight line** from the starting factor to the ending one.

    It meets `ConstantLR` at the end — past `total_iters` both are the original
    learning rate. So the last value alone cannot tell them apart, and the golden
    asks for the whole trace.
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0,
                 total_iters=5, last_epoch=-1):
        self.start_factor, self.end_factor = start_factor, end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(self.last_epoch, self.total_iters)
        scale = self.start_factor + (self.end_factor - self.start_factor) * (
            t / self.total_iters if self.total_iters else 1.0)
        return [base * scale for base in self.base_lrs]


class PolynomialLR(_Scheduler):
    """Descends as `(1 - t/T)^power`. `power=1` is a straight line."""

    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1):
        self.total_iters, self.power = total_iters, power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        groups = self.optimizer.param_groups
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [g["lr"] for g in groups]
        # Being recursive it multiplies **one step's ratio.** It reaches 0 at
        # `t == total_iters`.
        decay = ((1.0 - self.last_epoch / self.total_iters)
                 / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [g["lr"] * decay for g in groups]


class MultiplicativeLR(_Scheduler):
    """**It multiplies through** — it takes a factor like `LambdaLR` and its
    baseline is the current learning rate rather than the original one. That
    difference makes the same lambda give a different result."""

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return list(self.base_lrs)
        return [g["lr"] * self.lr_lambda(self.last_epoch)
                for g in self.optimizer.param_groups]


class CosineAnnealingWarmRestarts(_Scheduler):
    """Descends along a cosine and then **restarts.** Each period is `T_mult`
    times longer."""

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        self.T_0, self.T_mult, self.eta_min = T_0, T_mult, eta_min
        self.T_i, self.T_cur = T_0, last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base - self.eta_min)
                * (1 + _math.cos(_math.pi * self.T_cur / self.T_i)) / 2
                for base in self.base_lrs]

    def step(self):
        self.last_epoch += 1
        self.T_cur = self.T_cur + 1
        # At the end of a period it restarts and lengthens the next one.
        while self.T_cur >= self.T_i:
            self.T_cur -= self.T_i
            self.T_i *= self.T_mult
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr


class OneCycleLR(_Scheduler):
    """Up and then down. **Close to the default in a modern training recipe.**

    torch's default is a cosine shape with the rising stretch at 30% of the
    whole. The initial learning rate is `max_lr/div_factor` and the end is
    `initial/final_div_factor`, so **the learning rate given to the optimiser
    goes entirely unused** — it is overwritten at construction.
    """

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25.0, final_div_factor=1e4, last_epoch=-1):
        self.max_lr, self.total_steps, self.pct_start = max_lr, total_steps, pct_start
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        # **torch's arithmetic, as written.** It is
        # `pct_start × total_steps − 1`, not `pct_start × (total_steps − 1)` —
        # written the latter way the peak shifted by about half a step and gave a
        # max diff of 7.6e-02.
        self.up = float(pct_start * total_steps) - 1
        self.down = total_steps - self.up - 1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(self.last_epoch, self.total_steps - 1)
        if t <= self.up:
            frac = t / self.up if self.up else 1.0
            lo, hi = self.initial_lr, self.max_lr
        else:
            frac = (t - self.up) / max(1e-12, self.down)
            lo, hi = self.max_lr, self.min_lr
        # Cosine interpolation — the slope is zero at both ends.
        scale = (1 - _math.cos(_math.pi * frac)) / 2
        return [lo + (hi - lo) * scale for _ in self.base_lrs]


class CyclicLR(_Scheduler):
    """Makes the learning rate **rise and fall.** A deliberate shake to get out
    of a saddle point.

    It rises for `step_size_up` and falls for `step_size_down`. Given nothing it
    falls as far as it rose — **with the rise and fall equal, that argument's
    existence is invisible.**

    Three `mode`s:
      `triangular`  — the peaks are always the same height
      `triangular2` — the height halves every period
      `exp_range`   — the height is multiplied by `gamma^step` (by **step**, not
                      by period)

    That the last one counts by step is where they diverge. A `scale_mode` of
    `cycle` supplies the period number and `iterations` supplies the step count,
    and `exp_range` alone uses the latter.
    """

    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000,
                 step_size_down=None, mode="triangular", gamma=1.0,
                 scale_fn=None, scale_mode="cycle", cycle_momentum=True,
                 base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        self.base_lr, self.max_lr = base_lr, max_lr
        self.up = step_size_up
        self.down = step_size_up if step_size_down is None else step_size_down
        self.mode, self.gamma = mode, gamma
        self.cycle_momentum = cycle_momentum
        self.base_momentum, self.max_momentum = base_momentum, max_momentum
        if scale_fn is None:
            if mode == "triangular":
                scale_fn, scale_mode = (lambda _c: 1.0), "cycle"
            elif mode == "triangular2":
                scale_fn, scale_mode = (lambda c: 1 / (2.0 ** (c - 1))), "cycle"
            elif mode == "exp_range":
                scale_fn, scale_mode = (lambda i: gamma ** i), "iterations"
            else:
                raise ValueError(f"CyclicLR: unknown mode {mode!r}")
        self.scale_fn, self.scale_mode = scale_fn, scale_mode
        super().__init__(optimizer, last_epoch)

    def _shape(self):
        total = self.up + self.down
        ratio = self.up / total
        cycle = _math.floor(1 + self.last_epoch / total)
        x = 1 + self.last_epoch / total - cycle
        # The rising and falling stretches have different slopes — the trap
        # described above.
        rise = x / ratio if x <= ratio else (x - 1) / (ratio - 1)
        scale = self.scale_fn(cycle if self.scale_mode == "cycle"
                              else self.last_epoch)
        return rise * scale

    def get_lr(self):
        height = (self.max_lr - self.base_lr) * self._shape()
        lr = self.base_lr + height
        if self.cycle_momentum:
            # **The momentum goes the other way** — low where the learning rate
            # is high.
            span = (self.max_momentum - self.base_momentum) * self._shape()
            for group in self.optimizer.param_groups:
                if "momentum" in group:
                    group["momentum"] = self.max_momentum - span
        return [lr for _ in self.optimizer.param_groups]


class SequentialLR:
    """**Chains schedulers.** At a milestone it hands over to the next one.

    It does not inherit `_Scheduler` — it has no `get_lr` of its own and its job
    is choosing whose to call, so inheriting means the constructor calls `step()`
    once and the order goes wrong.
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        self.optimizer = optimizer
        self.schedulers = list(schedulers)
        self.milestones = list(milestones)
        self.last_epoch = 0
        # **At construction it returns to the first scheduler's value.** Each
        # scheduler changed the lr once as it was built, so left alone it starts
        # from the last one's value.
        for group, base in zip(optimizer.param_groups, self.schedulers[0].base_lrs):
            group["lr"] = base
        self.schedulers[0].last_epoch = -1
        self.schedulers[0].step()

    def _which(self):
        idx = sum(1 for m in self.milestones if m <= self.last_epoch)
        return idx, (self.milestones[idx - 1] if idx else 0)

    def step(self):
        self.last_epoch += 1
        idx, start = self._which()
        sch = self.schedulers[min(idx, len(self.schedulers) - 1)]
        if self.last_epoch == start:
            # **At the handover it returns to the baseline learning rate** and
            # walks the new scheduler from the start. It does not continue from
            # what the previous scheduler had cut — torch uses a closed form at
            # that point, and a closed form's baseline is `initial_lr`.
            for group, base in zip(self.optimizer.param_groups, sch.base_lrs):
                group["lr"] = base
            sch.last_epoch = -1
        sch.step()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class ChainedScheduler:
    """Applies several **at once.** Their factors multiply together."""

    def __init__(self, schedulers):
        self.schedulers = list(schedulers)
        self.optimizer = self.schedulers[0].optimizer

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class ReduceLROnPlateau:
    """Descends **when things stop improving.** Unlike the others it takes a
    value, as `step(metric)` — the same idea as chapter 6's early stopping,
    shrinking the step instead of stopping."""

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10,
                 threshold=1e-4, min_lr=0.0):
        self.optimizer = optimizer
        self.mode, self.factor, self.patience = mode, factor, patience
        self.threshold, self.min_lr = threshold, min_lr
        self.best = None
        self.num_bad_epochs = 0

    def _better(self, value):
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best * (1 - self.threshold)
        return value > self.best * (1 + self.threshold)

    def step(self, metric):
        metric = float(metric.item() if isinstance(metric, Tensor) else metric)
        if self._better(metric):
            self.best, self.num_bad_epochs = metric, 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                for group in self.optimizer.param_groups:
                    group["lr"] = max(group["lr"] * self.factor, self.min_lr)
                self.num_bad_epochs = 0

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class _LRScheduler(_Namespace):
    StepLR = StepLR
    MultiStepLR = MultiStepLR
    ExponentialLR = ExponentialLR
    CosineAnnealingLR = CosineAnnealingLR
    LambdaLR = LambdaLR
    ReduceLROnPlateau = ReduceLROnPlateau
    ConstantLR = ConstantLR
    LinearLR = LinearLR
    PolynomialLR = PolynomialLR
    MultiplicativeLR = MultiplicativeLR
    CosineAnnealingWarmRestarts = CosineAnnealingWarmRestarts
    OneCycleLR = OneCycleLR
    CyclicLR = CyclicLR
    SequentialLR = SequentialLR
    ChainedScheduler = ChainedScheduler
    # torch exposes the base class under this name. Code that subclasses it to
    # build its own scheduler calls this.
    LRScheduler = _Scheduler


class _Optim(_Namespace):
    SGD = SGD
    Adam = Adam
    AdamW = AdamW
    RMSprop = RMSprop
    Adagrad = Adagrad
    Adadelta = Adadelta
    Adamax = Adamax
    NAdam = NAdam
    RAdam = RAdam
    ASGD = ASGD
    Rprop = Rprop
    Adafactor = Adafactor
    LBFGS = LBFGS
    Optimizer = Optimizer
    lr_scheduler = _LRScheduler()


optim = _Optim()


