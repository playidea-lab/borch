"""`torch.optim` 자리. borch.ts 의 옵티마이저를 그대로 부른다.

파라미터를 넘길 때 **손잡이로 바꿔서** 넘긴다 — 우리 `Tensor` 는 파이썬 껍데기라
JS 쪽이 그것을 모른다.
"""

import js as _js

from ._base import handle

_ts = _js.borch


class _Opt:
    __slots__ = ("_o",)

    def __init__(self, o):
        self._o = o

    def zero_grad(self):
        self._o.zeroGrad()

    def step(self):
        self._o.step()

    @property
    def param_groups(self):
        """**`opt.param_groups[0]["lr"]` 로 읽는다.** torch 코드가 그렇게 쓴다.

        `to_py()` 만으로는 안 된다 — JS 객체가 파이썬 딕셔너리가 아니라 프록시로
        남아서 `["lr"]` 이 안 먹는다. 그리고 학습률은 **읽을 때마다 지금 값**이어야
        하므로 한 번 베껴 두면 스케줄러가 바꾼 것을 못 본다.
        """
        return [_Group(g) for g in self._o.paramGroups]


class _Group:
    """파라미터 묶음 하나. 딕셔너리처럼 읽히되 값은 그때그때 JS 쪽에서 가져온다."""

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
    # 파라미터가 JS 배열로 올 수도 있다 — `model.parameters()` 를 그대로 넘기는 자리다.
    if hasattr(ps, "to_py"):
        ps = ps.to_py()
    return _js.Array.new(*[handle(p) for p in ps])


def SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0):
    return _Opt(_ts.optim.SGD.new(_params(params), lr, momentum, weight_decay))


def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.Adam.new(_params(params), lr, betas[0], betas[1],
                                   eps, weight_decay))


def RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.RMSprop.new(_params(params), lr, alpha, eps,
                                      weight_decay))


def Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, eps=1e-10):
    return _Opt(_ts.optim.Adagrad.new(_params(params), lr, lr_decay, eps))


def Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
    return _Opt(_ts.optim.Adadelta.new(_params(params), lr, rho, eps))


def Adamax(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.Adamax.new(_params(params), lr, betas[0], betas[1], eps))


def NAdam(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
          momentum_decay=4e-3):
    return _Opt(_ts.optim.NAdam.new(_params(params), lr, betas[0], betas[1], eps,
                                    momentum_decay))


def RAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.RAdam.new(_params(params), lr, betas[0], betas[1], eps))


class _Sched:
    # `_keep` 은 JS 로 넘긴 파이썬 함수의 프록시를 붙잡아 둔다. 안 잡으면 호출이
    # 끝나는 순간 파괴되고, 스케줄러가 **나중에** 부를 때 "borrowed proxy was
    # automatically destroyed" 로 터진다 — 자매가 `tf.grad` 에서 겪은 것과 같다.
    __slots__ = ("_s", "_keep")

    def __init__(self, s, keep=None):
        self._s = s
        self._keep = keep

    def step(self, *args):
        self._s.step(*args)

    def get_last_lr(self):
        return list(self._s.getLastLr())


# 스케줄러도 이름 붙은 인자로 부른다 — `StepLR(o, step_size=2, gamma=0.5)`.
# JS 쪽은 자리로만 받으므로 여기서 편다.
_SCHED_ARGS = {
    "StepLR": ("step_size", "gamma"),
    "MultiStepLR": ("milestones", "gamma"),
    "ExponentialLR": ("gamma",),
    "CosineAnnealingLR": ("T_max", "eta_min"),
    "LambdaLR": ("lr_lambda",),
    # borch.ts 에는 `mode` 가 없다 — `rel` 하나뿐이라 자리도 없다.
    "ReduceLROnPlateau": ("factor", "patience", "threshold"),
    "ConstantLR": ("factor", "total_iters"),
    "LinearLR": ("start_factor", "end_factor", "total_iters"),
    "PolynomialLR": ("total_iters", "power"),
    "MultiplicativeLR": ("lr_lambda",),
    "CosineAnnealingWarmRestarts": ("T_0", "T_mult", "eta_min"),
    "OneCycleLR": ("max_lr", "total_steps", "pct_start", "div_factor",
                   "final_div_factor"),
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
        # 파이썬 함수는 프록시로 만들어 붙잡아 둔다 — `LambdaLR` 이 그것을 나중에 부른다.
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
        # **세운 직후 0 번째 에폭을 적용한다.** torch 는 생성자가 하는 일인데
        # TypeScript 쪽은 하위 클래스 필드가 `super()` 뒤에 채워져서 거기서 못 한다.
        # `ReduceLROnPlateau` 만 이 계보 밖이라 그 자리가 없다 — 값을 받아 판단하는
        # 것이라 에폭이라는 개념 자체가 없다.
        if hasattr(s, "start"):
            s.start()
        return _Sched(s, keep)
    return make


# **`torch.optim.lr_scheduler` 는 이름 공간이다.** 골든이 그 경로로 부른다.
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
    LRScheduler = _Sched

    @staticmethod
    def SequentialLR(optimizer, schedulers, milestones, last_epoch=-1):
        """스케줄러 목록과 이정표를 JS 배열로 넘긴다.

        **`Array.new` 를 쓰면 안 된다.** 인자가 수 하나면 `Array(3)` 이 되어 `[3]` 이
        아니라 **길이 3 짜리 빈 배열**이 나온다. 이정표가 보통 하나라 정확히 그 자리에
        걸렸고, 증상은 이정표를 지나도 학습률이 안 바뀌는 것이었다 — 이 저장소가 같은
        함정에 두 번째로 빠진 자리다.
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
