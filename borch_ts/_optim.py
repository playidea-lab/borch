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
        return self._o.paramGroups


def _params(ps):
    return _js.Array.new(*[handle(p) for p in ps])


def SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0):
    return _Opt(_ts.optim.SGD.new(_params(params), lr, momentum, weight_decay))


def Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.Adam.new(_params(params), lr, betas[0], betas[1],
                                   eps, weight_decay))


def RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
    return _Opt(_ts.optim.RMSprop.new(_params(params), lr, alpha, eps,
                                      weight_decay))


def _sched(js_name):
    def make(opt, *args, **kw):
        return _Opt(getattr(_ts.optim, js_name).new(opt._o, *args))
    return make


StepLR = _sched("StepLR")
MultiStepLR = _sched("MultiStepLR")
ExponentialLR = _sched("ExponentialLR")
CosineAnnealingLR = _sched("CosineAnnealingLR")
LambdaLR = _sched("LambdaLR")
