"""browsertorch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._tensor import (
    Tensor,
)
from ._ops import (
    _Namespace,
)
from ._base import (
    _math, _np,
)

# ================================================================ optim

class Optimizer:
    """torch 의 옵티마이저는 `param_groups` 라는 목록을 들고 있다.

    학습률을 읽고 쓰는 표준 경로가 `opt.param_groups[0]["lr"]` 이고, 스케줄러도 그것을
    고친다. `opt.lr` 로 두면 짧지만 **남의 코드가 안 돌고, 스케줄러를 직접 못 쓴다.**
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
        """torch 와 같은 모양 — 6장이 가르치는 "이어서 학습하기"가 이것에 걸려 있다.

        Adam 은 파라미터마다 보폭을 기억한다. 그 기억을 버리고 이어 학습하면
        손실이 한 번 튀었다가 다시 내려간다 — 오류는 안 나고 곡선만 이상해진다.
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

    decoupled = False          # AdamW 는 True — 감쇠를 기울기가 아니라 가중치에 직접 건다

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
    """Adam 과 같은데 가중치 감쇠를 **기울기가 아니라 가중치에 직접** 건다."""

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




class _Scheduler:
    """스케줄러는 `optimizer.param_groups` 의 lr 을 고친다. 에폭마다 한 번 부른다."""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
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


class StepLR(_Scheduler):
    """step_size 에폭마다 gamma 를 곱한다. 4장의 "멀리서는 성큼, 가까이서는 조심"."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.gamma ** (self.last_epoch // self.step_size)
                for base in self.base_lrs]


class MultiStepLR(_Scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones, self.gamma = sorted(milestones), gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        passed = sum(1 for m in self.milestones if m <= self.last_epoch)
        return [base * self.gamma ** passed for base in self.base_lrs]


class ExponentialLR(_Scheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.gamma ** self.last_epoch for base in self.base_lrs]


class CosineAnnealingLR(_Scheduler):
    """T_max 에폭에 걸쳐 코사인 곡선으로 내린다. 끝에서 부드럽게 멎는다."""

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


class ReduceLROnPlateau:
    """**좋아지지 않을 때** 내린다. 다른 것들과 달리 `step(metric)` 으로 값을 받는다 —
    6장의 조기 종료와 같은 발상이고, 멈추는 대신 보폭을 줄인다."""

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


class _Optim(_Namespace):
    SGD = SGD
    Adam = Adam
    AdamW = AdamW
    RMSprop = RMSprop
    Optimizer = Optimizer
    lr_scheduler = _LRScheduler()


optim = _Optim()


