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

def _execution_switches(who, *, foreach=None, fused=None, capturable=False,
                        differentiable=False):
    """torch's four per-optimizer switches, which say *how* a step is computed
    rather than *what* it computes.

    **`foreach` and `fused` are accepted and ignored, and that is not the "accepted
    and unused becomes a lie" shape this file refuses elsewhere.** Measured across
    every optimizer torch offers them on — sixteen settings, four steps each —
    **all sixteen reproduce the default answer exactly.** They choose a multi-tensor
    or a fused kernel; there is one kernel here and it is the same arithmetic. An
    argument that cannot change the answer is not a capability being faked.

    **`capturable` and `differentiable` are refused, because torch refuses them
    too.** On CPU `capturable=True` asserts the parameters are on a CUDA-like device
    and `differentiable=True` walks into the in-place guard — measured, on ten
    optimizers each. Matching torch means matching that: a caller gets the refusal
    here they would get there, rather than a silence torch would not have given.

    **Their order and their kind are torch's, per optimizer, and torch is not
    consistent with itself.** `SGD` has `maximize` before `foreach`; `Adam` the other
    way. `Adamax` puts `differentiable` before `capturable` and `Rprop` puts
    `capturable` first. `ASGD` and `RMSprop` take theirs *positionally* where the
    rest take them keyword-only. Tidying that would be measuring a preference.
    """
    if capturable:
        raise RuntimeError(
            f"{who}(capturable=True) needs a CUDA-like device — torch asserts the "
            "same thing, and there is no such device here.")
    if differentiable:
        raise RuntimeError(
            f"{who}(differentiable=True) is not here — the step edits its parameters "
            "in place, which is what torch refuses for it too.")
    del foreach, fused          # performance only; measured identical, see above


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

    def add_param_group(self, param_group):
        """torch's — attach another group after the optimizer was built.

        **It fell between two checks and neither was wrong.** The name axis counts a
        namespace's top-level names, and this is a method; the signature axis compares
        the *constructors* of the classes in `optim`. `Optimizer`'s methods were read
        by nothing, so a name torch has, borch.ts has as `addParamGroup`, and this side
        did not have was invisible to every instrument in the repository.

        It is what fine-tuning is written with — a fresh head at one rate bolted onto a
        backbone at another — and `opt.add_param_group({"params": head.parameters(),
        "lr": 1e-3})` is the line that does it.

        The four behaviours are torch's, measured rather than assumed:

            not a dict            TypeError, naming the type it got
            `params` one tensor   wrapped, so a single parameter needs no brackets
            missing keys          filled from `defaults`, the constructor's arguments
            a parameter twice     ValueError — two groups would step it twice

        The state bank needs no growing: `_state` allocates on first use, so a
        parameter added here gets its slot the first time it is stepped.
        """
        if not isinstance(param_group, dict):
            raise TypeError(
                f"param_group must be a dict, but got {type(param_group)}")
        group = dict(param_group)
        params = group.get("params")
        # A single tensor rather than a list — torch accepts it and wraps it, so a
        # caller adding one parameter writes no brackets.
        group["params"] = ([params] if isinstance(params, Tensor)
                           else list(params))
        seen = {id(p) for p in self.params}
        if any(id(p) in seen for p in group["params"]):
            raise ValueError(
                "some parameters appear in more than one parameter group")
        for key, value in self.defaults.items():
            group.setdefault(key, value)
        self.param_groups.append(group)

    def zero_grad(self, set_to_none=True):
        """**`set_to_none` was in the signature and the body never read it.**

        It always set `None`, which is torch's default, so the common call agreed
        and `zero_grad(set_to_none=False)` — the older behaviour, and what code
        written before torch 2.0 asks for — silently did the other thing. Nobody
        sees it until a line reads `p.grad` between the two calls: `None` there has
        no shape and no `.zero_()`, and `optimizer.zero_grad(set_to_none=False)`
        exists precisely so that it does.

        torch's rule is measured: `False` fills the existing gradient with zeros
        and leaves a tensor behind; `True` drops it.
        """
        for p in self.params:
            if set_to_none or p.grad is None:
                p.grad = None
            else:
                p.grad = Tensor(_np.zeros_like(p.grad.data))

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
    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None):
        """torch's order, with `dampening` third and `nesterov` sixth.

        **The default learning rate was `0.01` and torch's is `1e-3`** — ten times
        too large, on the one optimizer a tutorial is most likely to build without
        arguments. `SGD(model.parameters())` trained, and trained ten times faster
        than the same line does in torch, which does not fail: it converges to a
        different place, or diverges on a problem torch handles.
        
        Nothing had noticed because every golden case names its own rate. A default
        is the one value a case cannot check by using it — using it is what makes
        the case agree with whatever the default happens to be. It surfaced from a
        harness written for `maximize`, comparing the two libraries built with no
        arguments at all.

        **This read `(params, lr, momentum, weight_decay)`**, so `SGD(p, 0.1, 0.9,
        1e-4)` — a line anybody transcribes out of a torch tutorial — set the
        dampening to the weight decay and left the decay at zero. The two do
        different things and both are plausible small numbers, so the run trains and
        trains slightly wrong.

        `foreach`, `differentiable` and `fused` are torch's own execution switches
        and change no value; they are absent here rather than refused, which is the
        one case where absence is right — there is nothing they could mean.
        """
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        _execution_switches("SGD", foreach=foreach, differentiable=differentiable, fused=fused)
        super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening,
                                      weight_decay=weight_decay, nesterov=nesterov,
                                      maximize=maximize))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = -p.grad.data if group["maximize"] else p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                if group["momentum"]:
                    st = self._state(p)
                    buf = st.get("momentum_buffer")
                    # **The first step is the raw gradient, undamped.** torch seeds the
                    # buffer with `g` and only then starts damping, so a dampening of
                    # 0.9 does not shrink the very first move.
                    buf = g if buf is None else (group["momentum"] * buf
                                                 + (1 - group["dampening"]) * g)
                    st["momentum_buffer"] = buf
                    # Nesterov looks ahead: the step is the gradient plus the
                    # momentum, not the momentum alone.
                    g = g + group["momentum"] * buf if group["nesterov"] else buf
                p._array = p.data - group["lr"] * g


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None, decoupled_weight_decay=False):
        """`decoupled_weight_decay=True` **is `AdamW`**, and torch says so — the two
        agree to the last digit (measured, three steps).

        The switch already existed here as a class attribute, because `AdamW`
        subclasses this and flips it. What was missing was the argument. A capability
        present in the file and unreachable from outside is the cheapest kind of
        absence and the one nothing notices, because every internal use works.
        """
        _execution_switches("Adam", foreach=foreach, capturable=capturable, differentiable=differentiable, fused=fused)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, amsgrad=amsgrad,
                                      maximize=maximize))
        if decoupled_weight_decay:
            self.decoupled = True

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
                g = -p.grad.data if group["maximize"] else p.grad.data
                if group["weight_decay"] and not self.decoupled:
                    g = g + group["weight_decay"] * p.data
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * (g * g)
                mh = st["exp_avg"] / (1 - b1 ** st["step"])
                second = st["exp_avg_sq"]
                if group["amsgrad"]:
                    # **The running maximum, not the running average.** `amsgrad`
                    # keeps the largest second moment seen so far, so the step size
                    # can only shrink — the fix for Adam's non-convergence proof.
                    # It was absent, and torch takes it *positionally*, so it was not
                    # on the keyword-only list of what this file owed.
                    #
                    # **The maximum is over the raw moment and the bias correction
                    # comes after.** Taking it over the corrected value instead is
                    # the reading that looks equivalent — the correction is a
                    # positive scalar, so it commutes with `max` at a fixed step —
                    # and it is not, because the scalar changes every step: an early
                    # entry corrected by a small `1 − β₂ᵗ` can beat a later, larger
                    # raw moment. Measured against torch at 6.6e-04 after six steps.
                    st.setdefault("max_exp_avg_sq", _np.zeros_like(p.data))
                    st["max_exp_avg_sq"] = _np.maximum(st["max_exp_avg_sq"], second)
                    second = st["max_exp_avg_sq"]
                vh = second / (1 - b2 ** st["step"])
                new = p.data - group["lr"] * mh / (_np.sqrt(vh) + group["eps"])
                if group["weight_decay"] and self.decoupled:
                    new = new - group["lr"] * group["weight_decay"] * p.data
                p._array = new


class AdamW(Adam):
    """The same as Adam with the weight decay applied **to the weights directly
    rather than to the gradient.**"""

    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None):
        _execution_switches("AdamW", foreach=foreach, capturable=capturable, differentiable=differentiable, fused=fused)
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad,
                         maximize=maximize)


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0,
                 momentum=0.0, centered=False, capturable=False, foreach=None, maximize=False, differentiable=False):
        """**`momentum` and `centered` were missing and both are the algorithm.**

        `centered` subtracts a running mean of the gradient from the running mean of
        its square, which makes the denominator an estimate of the *variance* rather
        than of the second moment — that is the version Graves proposed and the one
        several recipes ask for by name. `momentum` gives it a velocity buffer, as
        `SGD` has.

        Neither was on `KEYWORD_ONLY_ABSENCES`, which had been the list of things
        owed here, **because torch takes both positionally.** The count was measuring
        one axis while two real features sat on the other, and `RMSprop(0.01, 0.99,
        1e-8, 0, 0.9)` — a recipe asking for momentum — raised a `TypeError` that
        named nothing.
        """
        _execution_switches("RMSprop", capturable=capturable, foreach=foreach, differentiable=differentiable)
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps,
                                      weight_decay=weight_decay, momentum=momentum,
                                      centered=centered, maximize=maximize))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("square_avg", _np.zeros_like(p.data))
                g = -p.grad.data if group["maximize"] else p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["square_avg"] = (group["alpha"] * st["square_avg"]
                                    + (1 - group["alpha"]) * g * g)
                avg = st["square_avg"]
                if group["centered"]:
                    st.setdefault("grad_avg", _np.zeros_like(p.data))
                    st["grad_avg"] = (group["alpha"] * st["grad_avg"]
                                      + (1 - group["alpha"]) * g)
                    avg = avg - st["grad_avg"] * st["grad_avg"]
                step = g / (_np.sqrt(avg) + group["eps"])
                if group["momentum"]:
                    st.setdefault("momentum_buffer", _np.zeros_like(p.data))
                    st["momentum_buffer"] = group["momentum"] * st["momentum_buffer"] + step
                    step = st["momentum_buffer"]
                p._array = p.data - group["lr"] * step




class Adagrad(Optimizer):
    """**Keeps adding** the squared gradients — it only shrinks, never grows.

    `RMSprop` puts an exponential moving average in the same place and forgets
    the old ones; this one does not forget. So run long enough the step size
    converges to zero — that is this optimiser's nature and not a defect.
    """

    def __init__(self, params, lr=0.01, lr_decay=0.0, weight_decay=0.0,
                 initial_accumulator_value=0.0, eps=1e-10, foreach=None, *, maximize=False, differentiable=False, fused=None):
        """torch's order — `initial_accumulator_value` sits **before** `eps`.

        Without it `Adagrad(p, 0.01, 0, 0, 1e-8)` set the accumulator's start where
        the caller meant the epsilon, and an accumulator seeded at 1e-8 rather than 0
        is a different first step from the one they asked for.
        """
        _execution_switches("Adagrad", foreach=foreach, differentiable=differentiable, fused=fused)
        super().__init__(params, dict(
            lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value, eps=eps,
            maximize=maximize))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("sum", _np.full_like(
                    p.data, group["initial_accumulator_value"]))
                st["step"] += 1
                g = -p.grad.data if group["maximize"] else p.grad.data
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

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0, foreach=None, *, capturable=False, maximize=False, differentiable=False):
        _execution_switches("Adadelta", foreach=foreach, capturable=capturable, differentiable=differentiable)
        super().__init__(params, dict(lr=lr, rho=rho, eps=eps,
                                      weight_decay=weight_decay, maximize=maximize))

    def step(self):
        for group in self.param_groups:
            rho, eps = group["rho"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("square_avg", _np.zeros_like(p.data))
                st.setdefault("acc_delta", _np.zeros_like(p.data))
                g = -p.grad.data if group["maximize"] else p.grad.data
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

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=None, *, maximize=False, differentiable=False, capturable=False):
        _execution_switches("Adamax", foreach=foreach, differentiable=differentiable, capturable=capturable)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, maximize=maximize))

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
                g = -p.grad.data if group["maximize"] else p.grad.data
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
                 weight_decay=0.0, momentum_decay=4e-3,
                 decoupled_weight_decay=False, *, foreach=None, maximize=False, capturable=False, differentiable=False):
        """`decoupled_weight_decay` applies the decay to the weights rather than to
        the gradient — `AdamW`'s split, offered here as an argument. torch takes it
        positionally, so it was never on the keyword-only list of what was owed."""
        _execution_switches("NAdam", foreach=foreach, capturable=capturable, differentiable=differentiable)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      momentum_decay=momentum_decay,
                                      decoupled_weight_decay=decoupled_weight_decay,
                                      maximize=maximize))

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
                g = -p.grad.data if group["maximize"] else p.grad.data
                decoupled = group["decoupled_weight_decay"]
                if group["weight_decay"] and not decoupled:
                    g = g + group["weight_decay"] * p.data
                if decoupled and group["weight_decay"]:
                    p._array = p.data * (1 - group["lr"] * group["weight_decay"])
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled_weight_decay=False, *, foreach=None, maximize=False, capturable=False, differentiable=False):
        """See `NAdam` on `decoupled_weight_decay`."""
        _execution_switches("RAdam", foreach=foreach, capturable=capturable, differentiable=differentiable)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      decoupled_weight_decay=decoupled_weight_decay,
                                      maximize=maximize))

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
                g = -p.grad.data if group["maximize"] else p.grad.data
                decoupled = group["decoupled_weight_decay"]
                if group["weight_decay"] and not decoupled:
                    g = g + group["weight_decay"] * p.data
                if decoupled and group["weight_decay"]:
                    p._array = p.data * (1 - group["lr"] * group["weight_decay"])
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
                 weight_decay=0.0, foreach=None, maximize=False, differentiable=False, capturable=False):
        _execution_switches("ASGD", foreach=foreach, differentiable=differentiable, capturable=capturable)
        super().__init__(params, dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                                      weight_decay=weight_decay, maximize=maximize))

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
                g = -p.grad.data if group["maximize"] else p.grad.data
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

    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50), *, capturable=False, foreach=None, maximize=False, differentiable=False):
        _execution_switches("Rprop", capturable=capturable, foreach=foreach, differentiable=differentiable)
        super().__init__(params, dict(lr=lr, etas=etas, step_sizes=step_sizes, maximize=maximize))

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
                if group["maximize"]:
                    g = -g
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
                 d=1.0, weight_decay=0.0, *, foreach=None, maximize=False):
        _execution_switches("Adafactor", foreach=foreach)
        super().__init__(params, dict(lr=lr, beta2_decay=beta2_decay, eps=eps,
                                      d=d, weight_decay=weight_decay, maximize=maximize))

    def step(self):
        for group in self.param_groups:
            lr, d = group["lr"], group["d"]
            eps1, eps2 = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = -p.grad.data if group["maximize"] else p.grad.data
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


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    """The minimum of the cubic through two points with values **and slopes.**

    Two points and two derivatives determine a cubic, and its stationary point is
    where the next trial step goes. Where the discriminant is negative there is no
    real stationary point inside and the bisection is taken instead — which is what
    keeps the line search from wandering off a flat or concave stretch.

    torch's own arithmetic, one line at a time. It is a port of `polyinterp.lua` and
    the branch on `x1 <= x2` is not symmetry-for-its-own-sake: the formula is written
    for the farther point, so which of the two that is has to be decided first.
    """
    if bounds is not None:
        lo, hi = bounds
    else:
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    square = d1 * d1 - g1 * g2
    if square >= 0:
        # **`np.sqrt` and not `math.sqrt`.** torch's slopes are float32 tensors here
        # and the whole interpolation runs at that width; `math.sqrt` returns a Python
        # float and widens everything after it. Measured, this line alone changes
        # nothing the golden can see — the case that catches the widening catches it
        # through `gtd`, and even there at 1.65e-04, under the table's threshold.
        # It is here for fidelity to torch's arithmetic, not on the strength of a
        # measurement, and saying so is the point.
        d2 = _np.sqrt(square)
        if x1 <= x2:
            at = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            at = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(at, lo), hi)
    return (lo + hi) / 2.0


def _strong_wolfe(obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9,
                  tolerance_change=1e-9, max_ls=25):
    """A step length satisfying the strong Wolfe conditions — torch's `lswolfe`.

    **Two phases.** The first brackets: it walks outwards from the initial step until
    it finds either a point that satisfies both conditions or an interval that must
    contain one. The second zooms: it interpolates inside that interval, moving
    whichever end is worse, until the conditions hold.

    Armijo (`c1`) says the loss fell by enough for the distance travelled; the
    curvature condition (`c2`) says the slope flattened by enough. Either alone is
    satisfied by steps that are useless — Armijo by a step of nothing, curvature by a
    step past the minimum — which is why both are checked at every candidate.

    **The `insuf_progress` flag is not tidiness.** Cubic interpolation can land
    arbitrarily close to a bracket end and then keep landing there, and the loop would
    spend its whole budget shrinking the bracket by nothing. Twice in a row near the
    edge and the trial is moved a tenth of the bracket in from it.

    Written from torch's own source and then measured against it: a `_both_stop` case
    is no use here because torch does not stop, and the trajectory is the answer.
    """
    d_norm = float(_np.abs(d).max())
    g = _np.array(g, copy=True)
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    # **The slopes stay float32.** torch's are 0-d tensors, so every comparison and
    # every interpolation below is done in float32 — and numpy's scalars follow the
    # same weak-promotion rule, so leaving them as `np.float32` reproduces it.
    #
    # Widened to Python floats the search takes a slightly different step and lands
    # 1.65e-04 from torch on a coupled quadratic. **That is under the golden's
    # threshold there** (1e-4 + 1e-4·1.13 ≈ 2.1e-04), so no case defends this — it is
    # fidelity to torch's arithmetic, and the reason is written down rather than
    # implied by a green check. What the cases do defend is the structure: skipping
    # the zoom, or keeping Armijo without the curvature test, moves them by units.
    gtd_new = g_new @ d

    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    bracket = bracket_f = bracket_g = bracket_gtd = None
    while ls_iter < max_ls:
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, _np.array(g_new, copy=True)]
            bracket_gtd = [gtd_prev, gtd_new]
            break
        if abs(gtd_new) <= -c2 * gtd:
            bracket, bracket_f, bracket_g = [t], [f_new], [g_new]
            bracket_gtd = [gtd_new]
            done = True
            break
        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, _np.array(g_new, copy=True)]
            bracket_gtd = [gtd_prev, gtd_new]
            break
        # Outwards: at least a hundredth past the last step and at most ten times it.
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(t_prev, f_prev, gtd_prev, t, f_new, gtd_new,
                               bounds=(min_step, max_step))
        t_prev, f_prev = tmp, f_new
        g_prev, gtd_prev = _np.array(g_new, copy=True), gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new @ d
        ls_iter += 1

    if ls_iter == max_ls:
        # **torch leaves `bracket_gtd` unset on this path** and the zoom loop below
        # reads it — a latent unbound name its own source marks with a type-ignore.
        # Reached only when the bracketing spends the whole budget without deciding,
        # which needs a very small `max_ls`. The two slopes at hand are the ones that
        # belong there, so this fills them in rather than carrying torch's crash.
        bracket = [0.0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]
        bracket_gtd = [gtd, gtd_new]

    insuf_progress = False
    low, high = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new @ d
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low]:
            bracket[high] = t
            bracket_f[high] = f_new
            bracket_g[high] = _np.array(g_new, copy=True)
            bracket_gtd[high] = gtd_new
            low, high = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high] - bracket[low]) >= 0:
                bracket[high] = bracket[low]
                bracket_f[high] = bracket_f[low]
                bracket_g[high] = bracket_g[low]
                bracket_gtd[high] = bracket_gtd[low]
            bracket[low] = t
            bracket_f[low] = f_new
            bracket_g[low] = _np.array(g_new, copy=True)
            bracket_gtd[low] = gtd_new

    return bracket_f[low], bracket_g[low], bracket[low], ls_func_evals


class LBFGS(Optimizer):
    """A quasi-Newton method. **`step` takes a closure** — because it re-measures
    the loss several times within one call.

    Other optimisers take one step per set of gradients; this one loops
    `max_iter` times inside and asks for the loss and the gradients again each
    time. So the training loop has a different shape, and without a closure it
    can do nothing.

    **`line_search_fn="strong_wolfe"` was refused** on the ground that going quietly
    with a fixed step size makes the convergence come out differently, and that
    difference shows in the curve rather than in a value. The reason was right and the
    way out was to write the line search: `_strong_wolfe` above is torch's own, and
    the fixed step is what happens when the argument is left at `None`.
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

    def _clone_param(self):
        return [p.data.copy() for p in self.params]

    def _set_param(self, saved):
        for p, value in zip(self.params, saved):
            p._array = value.copy()

    def _directional_evaluate(self, closure, x, t, d):
        """The loss and gradient at `x + t·d`, **with the parameters put back.**

        The line search asks about several step lengths from one place, so every
        probe has to start from the same `x`; leaving the last probe's position in
        the parameters would make the next one a step from wherever it happened to
        land. torch relies on the caller having left them at `x` and only restores
        afterwards — this sets them first as well, which is the same thing and does
        not depend on that.
        """
        self._set_param(x)
        self._add_step(t, d)
        loss = float(closure())
        flat = self._flat_grad()
        self._set_param(x)
        return loss, flat

    def step(self, closure):                                    # noqa: D102
        group = self.param_groups[0]
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
                ys = y @ s
                if ys > 1e-10:
                    if len(old_dirs) == history:
                        old_dirs.pop(0), old_stps.pop(0), ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    h_diag = ys / (y @ y)
                # The two-loop recursion — it produces a direction without
                # building the inverse Hessian.
                al = [0.0] * len(old_dirs)
                q = -flat
                for i in range(len(old_dirs) - 1, -1, -1):
                    al[i] = (old_stps[i] @ q) * ro[i]
                    q = q - al[i] * old_dirs[i]
                r = q * h_diag
                for i in range(len(old_dirs)):
                    be = (old_dirs[i] @ r) * ro[i]
                    r = r + old_stps[i] * (al[i] - be)
                d = r

            prev_flat = flat.copy()
            prev_loss = loss
            t = min(1.0, 1.0 / _np.abs(flat).sum()) * lr if st["n_iter"] == 1 else lr
            gtd = flat @ d
            if gtd > -tol_change:
                break

            ls_evals = 0
            opt_cond = False
            if group["line_search_fn"] is not None:
                # **torch's wording, its kind, and its position.** Checked here rather
                # than at the top of `step`, because that is where torch checks and
                # the difference is visible: a call whose gradient is already inside
                # `tolerance_grad` returns before the loop, and torch never looks at
                # the name at all. Refusing early makes a line that torch accepts stop.
                if group["line_search_fn"] != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                # **The budget the line search gets is what is left of `max_eval`**,
                # not `max_ls`'s default — torch passes it, and a search allowed more
                # probes than the caller's evaluation budget spends it inside one
                # iteration.
                start = self._clone_param()
                loss, flat, t, ls_evals = _strong_wolfe(
                    lambda x, tt, dd: self._directional_evaluate(closure, x, tt, dd),
                    start, t, d, loss, flat, gtd, max_ls=max_eval - evals)
                self._add_step(t, d)
                opt_cond = _np.abs(flat).max() <= tol_grad
            else:
                self._add_step(t, d)
                if n_iter != max_iter:
                    # No re-measure on the last iteration — torch is the same.
                    loss = float(closure())
                    flat = self._flat_grad()
                    opt_cond = _np.abs(flat).max() <= tol_grad
                    ls_evals = 1
            evals += ls_evals
            # **torch's five checks in torch's order.** This used to break out of the
            # re-measure block the moment the gradient was small, before the
            # iteration and evaluation counts were compared — the same stopping
            # place by luck, and not the same one to read.
            if n_iter == max_iter:
                break
            if evals >= max_eval:
                break
            if opt_cond:
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
    `sched::StepLR`, which walks the whole trajectory, was green for a long time.
    (That is the case to name: StepLR is not in the `_SCHEDULERS` trace list, so
    the `opt::` trace family has no entry for it and `sched::` is where its
    trajectory is asked.) They diverge when a fresh scheduler
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

    def __init__(self, optimizer, max_lr, total_steps=None, epochs=None,
                 steps_per_epoch=None, pct_start=0.3, anneal_strategy="cos",
                 cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                 div_factor=25.0, final_div_factor=1e4, three_phase=False,
                 last_epoch=-1):
        """**torch's list, in torch's order.** It used to be

            (optimizer, max_lr, total_steps, pct_start, div_factor,
             final_div_factor, last_epoch)

        which agrees with torch for three arguments and then parts for eleven
        consecutive positions. `OneCycleLR(opt, 0.1, None, 10, 100)` — a line a torch
        recipe writes to mean ten epochs of a hundred steps — set `pct_start` to 10
        and `div_factor` to 100 here. `pct_start` is a fraction of the cycle; given
        the value 10, the rising stretch runs ten times past the end, so the rate
        climbs for the whole run and never comes down. Nothing raises, and the shape
        of the curve is the one thing this scheduler exists to control.
        """
        if total_steps is None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError(
                    "OneCycleLR needs total_steps, or epochs and steps_per_epoch.")
            total_steps = int(epochs) * int(steps_per_epoch)
        elif epochs is not None or steps_per_epoch is not None:
            # torch takes `total_steps` and ignores the other two here. Ours refuses,
            # because the two answers can disagree and silently keeping one of them is
            # how the argument order above went unnoticed in the first place.
            raise ValueError(
                "OneCycleLR: give total_steps, or epochs and steps_per_epoch — "
                "not both.")
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(
                f"OneCycleLR: anneal_strategy is 'cos' or 'linear', not {anneal_strategy!r}.")
        self.max_lr, self.total_steps, self.pct_start = max_lr, total_steps, pct_start
        self.anneal_strategy = anneal_strategy
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.three_phase = bool(three_phase)
        # **torch's arithmetic, as written.** It is
        # `pct_start × total_steps − 1`, not `pct_start × (total_steps − 1)` —
        # written the latter way the peak shifted by about half a step and gave a
        # max diff of 7.6e-02.
        #
        # `_phases` is `[(end_step, from, to), ...]`. Two of them normally; three
        # under `three_phase`, where torch climbs, comes back down to the *initial*
        # rate over the same span, and only then anneals to the floor.
        rise = float(pct_start * total_steps) - 1
        if self.three_phase:
            # The second boundary is `2 × pct_start × total_steps − 2`, which is
            # `2 × rise` and not `2 × rise + 1`. Written the second way the middle
            # phase runs one step long and every value after it slides: max diff
            # 7.4e-02 against torch, measured, on a curve that still looked like a
            # one-cycle curve.
            self._phases = [
                (rise, self.initial_lr, max_lr),
                (2 * rise, max_lr, self.initial_lr),
                (total_steps - 1, self.initial_lr, self.min_lr)]
        else:
            self._phases = [
                (rise, self.initial_lr, max_lr),
                (total_steps - 1, max_lr, self.min_lr)]

        self.cycle_momentum = bool(cycle_momentum)
        if self.cycle_momentum:
            # torch refuses an optimiser with nowhere to put it rather than cycling
            # nothing. `betas` is Adam's spelling and the first of the pair is the
            # one that moves.
            groups = optimizer.param_groups
            self._momentum_key = next(
                (k for k in ("momentum", "betas") if any(k in g for g in groups)), None)
            if self._momentum_key is None:
                raise ValueError(
                    "OneCycleLR(cycle_momentum=True) needs an optimizer with "
                    "momentum or betas; pass cycle_momentum=False.")
            # **Momentum runs the other way from the rate.** High while the rate is
            # low, lowest at the peak — that is the whole of the one-cycle idea and it
            # was simply absent here, so `cycle_momentum` had nothing to be true about.
            #
            # Written out rather than derived from the rate's direction. The obvious
            # rule — *rate rising means momentum falling* — gets the two-phase table
            # exactly right and the third phase wrong: there the rate anneals from the
            # initial value down to the floor while torch holds momentum **flat at
            # `max_momentum`**. Deriving it gave a rising momentum instead, and it
            # agreed with torch everywhere the rule was checked before that phase
            # existed.
            ends = [end for end, _lo, _hi in self._phases]
            pairs = ([(max_momentum, base_momentum), (base_momentum, max_momentum),
                      (max_momentum, max_momentum)] if self.three_phase else
                     [(max_momentum, base_momentum), (base_momentum, max_momentum)])
            self._momentum_phases = [(e, a, b) for e, (a, b) in zip(ends, pairs)]
        super().__init__(optimizer, last_epoch)

    def _interpolate(self, t, phases):
        """Where `t` falls in `phases`, and how far through that phase it is."""
        start = 0.0
        for end, lo, hi in phases:
            if t <= end or (end, lo, hi) is phases[-1]:
                span = end - start
                frac = (t - start) / span if span > 0 else 1.0
                return lo, hi, min(1.0, max(0.0, frac))
            start = end
        raise AssertionError("unreachable — the last phase always answers")

    def _anneal(self, lo, hi, frac):
        if self.anneal_strategy == "linear":
            return lo + (hi - lo) * frac
        # Cosine interpolation — the slope is zero at both ends.
        return lo + (hi - lo) * (1 - _math.cos(_math.pi * frac)) / 2

    def get_lr(self):
        t = min(self.last_epoch, self.total_steps - 1)
        lo, hi, frac = self._interpolate(t, self._phases)
        return [self._anneal(lo, hi, frac) for _ in self.base_lrs]

    def step(self):
        super().step()
        if not self.cycle_momentum:
            return
        t = min(self.last_epoch, self.total_steps - 1)
        lo, hi, frac = self._interpolate(t, self._momentum_phases)
        value = self._anneal(lo, hi, frac)
        for group in self.optimizer.param_groups:
            if self._momentum_key == "betas":
                group["betas"] = (value, group["betas"][1])
            else:
                group["momentum"] = value


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


class _Saves:
    """`state_dict` and `load_state_dict` for the schedulers outside `_Scheduler`.

    **Three of them had neither**, so a run that saved everything it could still
    resumed with `ReduceLROnPlateau`'s patience, best value and cooldown all back at
    their starting points — the cut arriving epochs late, or early, with nothing on
    screen to say so. torch's `ReduceLROnPlateau` carries fifteen keys.

    The rule is `_Scheduler`'s and is copied here rather than restated: **everything
    except the optimizer**, which is not carried because the two would hold each other
    in a cycle. A scheduler holding other schedulers carries theirs by recursion.

    It was found by a peer's check that steps every scheduler, watches which
    attributes move, and asserts each one is in `state_dict()` — the general form of
    the cooldown counter I had just added to borch.ts and not to the core. **I checked
    that the counter reached the save and not that there was a save.**
    """

    def state_dict(self):
        out = {k: v for k, v in self.__dict__.items() if k != "optimizer"}
        if "schedulers" in out:
            out["schedulers"] = [s.state_dict() for s in self.schedulers]
        return out

    def load_state_dict(self, state):
        state = dict(state)
        inner = state.pop("schedulers", None)
        if inner is not None:
            for sch, saved in zip(self.schedulers, inner):
                sch.load_state_dict(saved)
        self.__dict__.update(state)
        return self


def _one_optimizer(who, optimizer, schedulers):
    """**Every scheduler in a chain has to be stepping the same optimizer.**

    torch checks it and this did not, so `SequentialLR(a, [ConstantLR(a),
    ConstantLR(b)])` was built without complaint and then, at the milestone, stepped
    `b`'s learning rate while reporting `a`'s. The rate that trains and the rate that
    is printed part company, and nothing raises.

    torch's message embeds the whole optimizer `repr`, which is a paragraph. The
    stable opening clause is what a search finds and what is matched here.
    """
    for at, sch in enumerate(schedulers):
        if getattr(sch, "optimizer", optimizer) is not optimizer:
            raise ValueError(
                f"{who} expects all schedulers to belong to the same optimizer, "
                f"but got scheduler {type(sch).__name__} at index {at} has "
                f"{type(sch.optimizer).__name__}, which is different from "
                f"{type(optimizer).__name__}.")


class SequentialLR(_Saves):
    """**Chains schedulers.** At a milestone it hands over to the next one.

    It does not inherit `_Scheduler` — it has no `get_lr` of its own and its job
    is choosing whose to call, so inheriting means the constructor calls `step()`
    once and the order goes wrong.
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        self.optimizer = optimizer
        self.schedulers = list(schedulers)
        self.milestones = list(milestones)
        _one_optimizer("SequentialLR", optimizer, self.schedulers)
        # **torch counts them.** One scheduler per interval and one interval more
        # than there are milestones; given two of each, the last scheduler is never
        # reached and `step` silently walks the wrong one forever.
        if len(self.schedulers) != len(self.milestones) + 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be "
                "one more than the number of milestone points, but got number of "
                f"schedulers {len(self.schedulers)} and the number of milestones to "
                f"be equal to {len(self.milestones)}")
        # **`last_epoch` was a seat the body never read**, so resuming a run put the
        # chain back at its first interval however far it had got. Measured against
        # torch: the attribute becomes `last_epoch + 1` and the whole trace shifts by
        # that much, which is what resuming means.
        self.last_epoch = last_epoch + 1
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


class ChainedScheduler(_Saves):
    """Applies several **at once.** Their factors multiply together."""

    def __init__(self, schedulers, optimizer=None):
        """**`optimizer` sits second, where torch has it**, and is optional there
        too: given nothing, torch takes the first scheduler's. Passing one that is
        not theirs is refused rather than believed — `get_last_lr` reads the rates
        off it, so a mismatched optimizer reports rates nobody set."""
        self.schedulers = list(schedulers)
        found = self.schedulers[0].optimizer
        if optimizer is not None and optimizer is not found:
            raise ValueError(
                "ChainedScheduler: the optimizer given is not the one the schedulers "
                "are stepping.")
        # **The `optimizer=` seat was checked and the schedulers were not.** Given
        # none — which is the ordinary call — any mismatch among them went through,
        # and `get_last_lr` then reads the first one's rates while the others step
        # somebody else's. `SequentialLR` had the same hole.
        _one_optimizer("ChainedScheduler", found, self.schedulers)
        self.optimizer = found

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class ReduceLROnPlateau(_Saves):
    """Descends **when things stop improving.** Unlike the others it takes a
    value, as `step(metric)` — the same idea as chapter 6's early stopping,
    shrinking the step instead of stopping."""

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0,
                 eps=1e-8):
        """torch's order. **`threshold_mode`, `cooldown` and `eps` were missing from
        the middle**, so `ReduceLROnPlateau(opt, "min", 0.5, 5, 1e-3, 0, 1e-4)` — a
        call written from torch's documentation — put the cooldown where
        `threshold_mode` goes and the minimum rate where `cooldown` does.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold mode {threshold_mode} is unknown!")
        self.optimizer = optimizer
        self.mode, self.factor, self.patience = mode, factor, patience
        self.threshold, self.threshold_mode = threshold, threshold_mode
        self.cooldown, self.min_lr, self.eps = cooldown, min_lr, eps
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def _better(self, value):
        """torch's four comparisons — two directions crossed with two thresholds.

        **`rel` scales the threshold by the best value and `abs` subtracts it.** At a
        loss near 1 the two are almost the same number, which is why a case built on
        the default cannot tell them apart, and at a loss near 100 they are not close
        at all.
        """
        if self.best is None:
            return True
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return value < self.best * (1 - self.threshold)
            return value < self.best - self.threshold
        if self.threshold_mode == "rel":
            return value > self.best * (1 + self.threshold)
        return value > self.best + self.threshold

    def step(self, metric):
        metric = float(metric.item() if isinstance(metric, Tensor) else metric)
        if self._better(metric):
            self.best, self.num_bad_epochs = metric, 0
        else:
            self.num_bad_epochs += 1
        # **The cooldown counter runs down before the patience is looked at, and the
        # bad epochs are cleared while it does.** torch ignores the epochs inside a
        # cooldown entirely rather than counting them towards the next cut, so a
        # counter that merely blocked the cut would fire the moment it expired.
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        elif self.num_bad_epochs > self.patience:
            for group in self.optimizer.param_groups:
                new = max(group["lr"] * self.factor, self.min_lr)
                # **A cut smaller than `eps` is not made at all.** Without it the
                # rate keeps shrinking by amounts that do not change the training
                # and do change the number a checkpoint carries.
                if group["lr"] - new > self.eps:
                    group["lr"] = new
            self.cooldown_counter = self.cooldown
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


