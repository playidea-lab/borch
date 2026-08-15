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
        # JS 배열은 파이썬에서 바로 못 돈다 — 목록으로 받는다.
        return self._o.paramGroups.to_py()


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


class _Sched:
    __slots__ = ("_s",)

    def __init__(self, s):
        self._s = s

    def step(self, *args):
        self._s.step(*args)

    def get_last_lr(self):
        return list(self._s.getLastLr())


def _sched(js_name):
    def make(opt, *args, **kw):
        from ._ops import _arg
        return _Sched(getattr(_ts.optim, js_name).new(
            opt._o, *[_arg(a) for a in args]))
    return make


# **`torch.optim.lr_scheduler` 는 이름 공간이다.** 골든이 그 경로로 부른다.
class _LRScheduler:
    StepLR = staticmethod(_sched("StepLR"))
    MultiStepLR = staticmethod(_sched("MultiStepLR"))
    ExponentialLR = staticmethod(_sched("ExponentialLR"))
    CosineAnnealingLR = staticmethod(_sched("CosineAnnealingLR"))
    LambdaLR = staticmethod(_sched("LambdaLR"))
    ReduceLROnPlateau = staticmethod(_sched("ReduceLROnPlateau"))


lr_scheduler = _LRScheduler()

StepLR = _sched("StepLR")
MultiStepLR = _sched("MultiStepLR")
ExponentialLR = _sched("ExponentialLR")
CosineAnnealingLR = _sched("CosineAnnealingLR")
LambdaLR = _sched("LambdaLR")
