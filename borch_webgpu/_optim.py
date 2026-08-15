"""borch_webgpu 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import numpy as _np

try:
    import js as _js
    from pyodide.ffi import create_proxy as _create_proxy
    from pyodide.ffi import to_js as _to_js
except ImportError as _exc:                                          # pragma: no cover
    raise ImportError(
        "borch_webgpu 는 브라우저(Pyodide) 안에서만 돕니다. "
        "네이티브에서는 `borch` 를 쓰세요 — 이쪽을 CPU 로 흉내 내면 "
        "GPU 로 돌렸다고 착각하게 됩니다."
    ) from _exc

_tf = getattr(_js, "tf", None)
if _tf is None:                                                      # pragma: no cover
    raise ImportError("TF.js 가 페이지에 없습니다. tf.min.js 를 먼저 실으세요.")

from ._tensor import (
    Tensor, _grad_mode,
)
from ._base import (
    _keep,
)

# ---------------------------------------------------------------- optim
#
# 갱신은 **GPU 에서** 한다. 파라미터를 읽어와 numpy 로 고치면 매 스텝 전량 왕복이
# 생기고, 그 순간 GPU 를 쓰는 의미가 사라진다(WEBGPU-DESIGN.md 8절 S3).

def _replace(state, key, handle):
    """옵티마이저 상태를 갈아끼우고 옛 버퍼를 놓는다.

    상태는 `Tensor` 가 아니라 손잡이라 파이썬 GC 가 안 봐준다 — 여기서 직접 놓지 않으면
    모멘텀·Adam 상태가 스텝마다 쌓인다.
    """
    old = state.get(key)
    state[key] = _keep(handle)
    if old is not None and old is not handle:
        try:
            old.dispose()
        except Exception:                                            # noqa: BLE001
            pass
    return handle


class Optimizer:
    def __init__(self, params, defaults):
        params = list(params)
        if params and isinstance(params[0], dict):
            self.param_groups = [dict(defaults, **g) for g in params]
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

    def _assign(self, p, handle):
        """새 값으로 갈아끼우고 **옛 버퍼를 놓는다.** TF.js 는 수동 해제라
        안 놓으면 스텝마다 GPU 메모리가 는다."""
        old = p._h
        p._h = _keep(handle)          # 스코프를 나가도 파라미터는 살아야 한다
        old.dispose()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad._h
                if group["weight_decay"]:
                    g = _tf.add(g, _tf.mul(float(group["weight_decay"]), p._h))
                if group["momentum"]:
                    st = self._state(p)
                    buf = st.get("momentum_buffer")
                    # 첫 스텝에서 **복제해야 한다.** 그대로 두면 버퍼가 `p.grad` 의
                    # 손잡이를 물고, 다음 zero_grad 에서 grad 가 사라질 때 __del__ 이
                    # 그 버퍼까지 놓는다 — 두 번째 스텝이 죽은 손잡이를 읽는다.
                    g = (_tf.clone(g) if buf is None
                         else _tf.add(_tf.mul(float(group["momentum"]), buf), g))
                    _replace(st, "momentum_buffer", g)
                self._assign(p, _tf.sub(p._h, _tf.mul(float(group["lr"]), g)))


class Adam(Optimizer):
    decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _tf.zerosLike(p._h))
                st.setdefault("exp_avg_sq", _tf.zerosLike(p._h))
                st["step"] += 1
                g = p.grad._h
                if group["weight_decay"] and not self.decoupled:
                    g = _tf.add(g, _tf.mul(float(group["weight_decay"]), p._h))
                # 상태를 갈아끼울 때 **옛 버퍼를 놓는다.** 파라미터와 달리 이쪽은
                # 텐서가 아니라 손잡이라 파이썬 GC 가 안 봐준다.
                st["exp_avg"] = _replace(st, "exp_avg",
                                         _tf.add(_tf.mul(float(b1), st["exp_avg"]),
                                                 _tf.mul(1.0 - float(b1), g)))
                st["exp_avg_sq"] = _replace(st, "exp_avg_sq",
                                            _tf.add(_tf.mul(float(b2), st["exp_avg_sq"]),
                                                    _tf.mul(1.0 - float(b2), _tf.mul(g, g))))
                mh = _tf.div(st["exp_avg"], 1.0 - float(b1) ** st["step"])
                vh = _tf.div(st["exp_avg_sq"], 1.0 - float(b2) ** st["step"])
                new = _tf.sub(p._h, _tf.mul(float(group["lr"]),
                                            _tf.div(mh, _tf.add(_tf.sqrt(vh),
                                                                float(group["eps"])))))
                if group["weight_decay"] and self.decoupled:
                    new = _tf.sub(new, _tf.mul(float(group["lr"]) * float(group["weight_decay"]),
                                               p._h))
                self._assign(p, new)


class AdamW(Adam):
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
                g = p.grad._h
                if group["weight_decay"]:
                    g = _tf.add(g, _tf.mul(float(group["weight_decay"]), p._h))
                prev = st.get("square_avg")
                sq = _tf.mul(g, g)
                new_avg = (_tf.mul(1.0 - float(group["alpha"]), sq) if prev is None
                           else _tf.add(_tf.mul(float(group["alpha"]), prev),
                                        _tf.mul(1.0 - float(group["alpha"]), sq)))
                _replace(st, "square_avg", _keep(new_avg))
                self._assign(p, _tf.sub(p._h, _tf.div(
                    _tf.mul(float(group["lr"]), g),
                    _tf.add(_tf.sqrt(st["square_avg"]), float(group["eps"])))))


# ---------------------------------------------------------------- 스케줄러
#
# 코어에서 그대로 옮겼다. **파이썬 실수 연산뿐이라 텐서를 안 건드린다** — 옮기면서
# 바꿀 것이 없었고, 두 벌로 두면 갈릴 이유도 없다.
#
# 스케줄러는 `optimizer.param_groups` 의 lr 을 고친다. `opt.lr` 로 두면 짧지만
# 남의 코드가 안 돌고 남의 스케줄러를 못 쓴다.

class _Scheduler:
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
    """step_size 에폭마다 gamma 를 곱한다 — 멀리서는 성큼, 가까이서는 조심."""

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
        import math
        return [self.eta_min + (base - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base in self.base_lrs]


class LambdaLR(_Scheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.lr_lambda(self.last_epoch) for base in self.base_lrs]


class ReduceLROnPlateau:
    """**좋아지지 않을 때** 내린다. 다른 것들과 달리 `step(metric)` 으로 값을 받는다."""

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


class _LRScheduler:
    StepLR = StepLR
    MultiStepLR = MultiStepLR
    ExponentialLR = ExponentialLR
    CosineAnnealingLR = CosineAnnealingLR
    LambdaLR = LambdaLR
    ReduceLROnPlateau = ReduceLROnPlateau


class _Optim:
    Optimizer = Optimizer
    SGD = SGD
    Adam = Adam
    AdamW = AdamW
    RMSprop = RMSprop
    lr_scheduler = _LRScheduler


optim = _Optim()


class scope:
    """한 스텝 동안 만들어진 GPU 버퍼를 통째로 놓는다.

    파이썬 GC 는 `Tensor` 가 든 손잡이만 놓아준다. 그런데 **역전파 클로저가 붙들고 있는
    중간 버퍼**(gelu 의 `ope`, gather 의 `onehot` 같은 것)는 `Tensor` 가 아니라
    아무도 안 놓는다 — 실측으로 학습 스텝당 92.7개가 남았다.

    설계 문서 7절은 "backward() 시점에 묶으면 사용자 API 에 스코프를 노출하지 않아도
    된다"고 적었는데, **그 전제가 틀렸다.** 클로저가 든 것은 그래프를 훑어서 찾을 수
    없다. 그래서 노출한다 — 코어와 다른 한 줄이 생기지만, 새는 것보다 낫다.

        with torch.scope():
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

    파라미터와 옵티마이저 상태는 `tf.keep` 으로 살려두므로 스코프를 나가도 남는다.
    """

    def __enter__(self):
        _tf.engine().startScope()
        return self

    def __exit__(self, *exc):
        _tf.engine().endScope()
        return False


def memory():
    """지금 잡고 있는 것 — `{"tensors": n, "bytes": n}`.

    **벤치가 누수를 재는 자리다.** 여태 `js.tf.memory()` 를 직접 불렀는데, 그러면
    계측이 TF.js 에 묶여서 같은 벤치를 다른 구현으로 못 돌린다 — 실제로 borch_ts 를
    같은 잣대로 재려다 거기서 막혔다. 라이브러리에 물으면 누가 밑에 있든 답한다.
    """
    got = _tf.memory()
    return {"tensors": int(got.numTensors), "bytes": int(got.numBytes)}


class no_grad:
    def __enter__(self):
        self._prev = _grad_mode.enabled
        _grad_mode.enabled = False
        return self

    def __exit__(self, *exc):
        _grad_mode.enabled = self._prev
        return False


