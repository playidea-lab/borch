"""`torch.nn` 자리. **여기도 이름을 옮겨 적는 것이 대부분이다.**

두 갈래다.

`nn.functional.relu(x)` 는 borch.ts 에서 `x.relu()` 다 — torch 자신이 그 둘을 같은
것으로 두므로 첫 인자를 받아 메서드로 넘기면 끝난다. `__getattr__` 하나가 그 일을
전부 한다.

`nn.Linear(6, 8)` 은 borch.ts 의 클래스다. 이쪽은 이름이 갈리는 자리가 있어서
(`BatchNorm2d` 는 저쪽에서 `BatchNormND`) 표를 둔다. 그리고 torch 에 있고 borch.ts 에
층으로는 없는 것들(`Softmax`·`ELU`·`L1Loss` …)은 **텐서 메서드를 한 줄 감싼 층**으로
만든다 — 없는 것을 근사하는 것이 아니라, 있는 것에 이름을 붙이는 것이다.
"""

import js as _js

from ._base import Tensor, guarded, handle, settle, wrap
from ._ops import _arg, camel, positional

_ts = _js.borch


class _Functional:
    """`nn.functional`. 첫 인자의 메서드로 넘긴다 — torch 의 규칙 그대로다."""

    def __getattr__(self, name):
        js_name = camel(name)

        def call(x, *args, **kw):
            h = handle(x)
            fn = getattr(h, js_name, None)
            if fn is None:
                raise AttributeError(
                    f"borch.ts 텐서에 `{js_name}` 이 없다 (F.{name})")
            return guarded(fn, *positional(name, args, kw))

        call.__name__ = name
        return call


functional = _Functional()


class _Rnn:
    """`nn.utils.rnn`. 지금 여기 있는 것은 `pad_sequence` 하나다."""

    @staticmethod
    def pad_sequence(parts, batch_first=False, padding_value=0.0):
        return wrap(_ts.Tensor.padSequence(
            _js.Array.new(*[handle(p) for p in parts]), batch_first, padding_value))


class _Utils:
    rnn = _Rnn()


utils = _Utils()


class Module:
    """borch.ts 의 층 하나를 감싼다.

    파이썬 쪽에서 필요한 것은 셋이다 — 부를 수 있을 것, 파라미터를 줄 것,
    `state_dict` 로 값을 넣고 뺄 수 있을 것.
    """

    __slots__ = ("_m",)

    def __init__(self, module):
        self._m = module

    def __call__(self, *args):
        return guarded(self._m.call, *[_arg(a) for a in args])

    def forward(self, *args):
        return self(*args)

    def parameters(self):
        # JS 배열은 파이썬에서 바로 못 돈다 — `to_py` 로 목록을 받아야 한다.
        return [wrap(p) for p in self._m.parameters().to_py()]

    def state_dict(self):
        got = self._m.stateDict()
        return {str(k): wrap(getattr(got, k)) for k in _js.Object.keys(got)}

    def load_state_dict(self, values, strict=True):
        from pyodide.ffi import to_js
        obj = _js.Object.new()
        for k, v in values.items():
            setattr(obj, k, handle(v))
        self._m.loadStateDict(obj, strict)

    def train(self, mode=True):
        self._m.train(mode)
        return self

    def eval(self):
        self._m.eval()
        return self

    def __getattr__(self, name):
        """`bn.weight` 처럼 층이 들고 있는 것을 그대로 넘긴다."""
        got = getattr(self._m, camel(name), None)
        if got is None:
            raise AttributeError(f"borch.ts 층에 `{name}` 이 없다")
        if _ts.isTensor(got):
            return wrap(got)
        return got


def _layer(js_name, *args):
    return Module(getattr(_ts.nn, js_name).new(*args))


class _Wrap(Module):
    """borch.ts 에 층으로는 없고 **텐서 메서드로는 있는** 것들.

    `nn.Softmax(dim)` 은 `x.softmax(dim)` 이다. 없는 것을 근사하는 것이 아니라
    있는 것에 torch 의 이름을 붙이는 것이므로, 값은 같은 자리에서 나온다.
    """

    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args):
        return self._fn(*args)

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, values, strict=True):
        pass

    def train(self, mode=True):
        return self

    def eval(self):
        return self


def Sequential(*layers):
    flat = []
    for l in layers:
        flat.extend(l if isinstance(l, (list, tuple)) else [l])
    return Module(_ts.nn.Sequential.new(*[m._m for m in flat]))


def Linear(inf, outf, bias=True):
    return _layer("Linear", inf, outf, bias)


def Conv1d(cin, cout, k, stride=1, padding=0, bias=True):
    return _layer("Conv1d", cin, cout, k, stride, padding, bias)


def Conv2d(cin, cout, k, stride=1, padding=0, bias=True):
    return _layer("Conv2d", cin, cout, k, stride, padding, bias)


def Conv3d(cin, cout, k, stride=1, padding=0, bias=True):
    return _layer("Conv3d", cin, cout, k, stride, padding, bias)


def _batchnorm(n, eps=1e-5, momentum=0.1):
    return _layer("BatchNormND", n, eps, momentum)


BatchNorm1d = BatchNorm2d = BatchNorm3d = _batchnorm


def ReLU():
    return _layer("ReLU")


def MaxPool2d(k=2, stride=None):
    return _Wrap(lambda x: wrap(handle(x).maxPool2d(k, stride)))


def MaxPool1d(k=2, stride=None):
    return _Wrap(lambda x: wrap(handle(x).maxPool1d(k, stride)))


def MaxPool3d(k=2, stride=None):
    return _Wrap(lambda x: wrap(handle(x).maxPool3d(k, stride)))


def Flatten(start=1):
    return _Wrap(lambda x: wrap(handle(x).flatten(start)))


def Identity():
    return _Wrap(lambda x: x)


def Softmax(dim=-1):
    return _Wrap(lambda x: wrap(handle(x).softmax(dim)))


def LogSoftmax(dim=-1):
    return _Wrap(lambda x: wrap(handle(x).logSoftmax(dim)))


def LeakyReLU(slope=0.01):
    return _Wrap(lambda x: wrap(handle(x).leakyRelu(slope)))


def ELU():
    return _Wrap(lambda x: wrap(handle(x).unary("elu")))


def SiLU():
    return _Wrap(lambda x: wrap(handle(x).unary("silu")))


def GELU():
    return _Wrap(lambda x: wrap(handle(x).unary("gelu")))


def Sigmoid():
    return _Wrap(lambda x: wrap(handle(x).unary("sigmoid")))


def Tanh():
    return _Wrap(lambda x: wrap(handle(x).unary("tanh")))


def LayerNorm(shape, eps=1e-5):
    return _Wrap(lambda x: wrap(handle(x).layerNorm(-1, eps)))


def Unflatten(dim, sizes):
    return _Wrap(lambda x: wrap(handle(x).unflatten(dim, _arg(list(sizes)))))


def Upsample(scale_factor=2, mode="nearest"):
    return _Wrap(lambda x: wrap(handle(x).upsample(scale_factor)))


def L1Loss():
    return _Wrap(lambda a, b: wrap(handle(a).l1Loss(handle(b))))


def MSELoss():
    return _Wrap(lambda a, b: wrap(handle(a).mseLoss(handle(b))))


def SmoothL1Loss(beta=1.0):
    return _Wrap(lambda a, b: wrap(handle(a).smoothL1Loss(handle(b), beta)))


def NLLLoss():
    return _Wrap(lambda a, b: wrap(handle(a).nllLoss(handle(b))))


def BCEWithLogitsLoss():
    return _Wrap(lambda a, b: wrap(handle(a).bceWithLogits(handle(b))))


def CrossEntropyLoss():
    return _Wrap(lambda a, b: wrap(handle(a).crossEntropy(handle(b))))


class _Recurrent(Module):
    """**torch 의 순환망은 튜플을 준다** — `(출력, 마지막상태)`.

    borch.ts 의 `forward` 는 출력만 주고 `run()` 이 셋을 함께 준다. LSTM 은 상태가
    둘(`h`, `c`)이라 `(출력, (h, c))` 이고, 나머지는 `(출력, h)` 다. 모양까지 맞춘다 —
    torch 의 마지막 상태는 `(층수, 배치, 은닉)` 이라 축이 하나 더 있다.
    """

    def __call__(self, x, *rest):
        got = self._m.run(handle(x))
        out, h = wrap(got.output), wrap(got.hidden)
        h = wrap(h._h.unsqueeze(0))
        if self._m.kind == "LSTM":
            c = wrap(wrap(got.cell)._h.unsqueeze(0))
            return out, (h, c)
        return out, h


def _recurrent(kind):
    def make(inp, hid, **kw):
        return _Recurrent(_ts.nn.Recurrent.new(inp, hid, kind))
    return make


RNN, LSTM, GRU = _recurrent("RNN"), _recurrent("LSTM"), _recurrent("GRU")


class _Attention(Module):
    """torch 의 어텐션은 `(질의, 키, 값)` 셋을 받고 `(출력, 가중치)` 를 준다."""

    def __call__(self, q, k=None, v=None, attn_mask=None, **kw):
        got = self._m.call(handle(q), handle(k if k is not None else q),
                           handle(v if v is not None else q),
                           handle(attn_mask) if attn_mask is not None else None)
        if _ts.isTensor(got):
            return wrap(got), None
        return wrap(got.output), wrap(got.weights)


def MultiheadAttention(embed, heads, batch_first=False):
    return _Attention(_ts.nn.MultiheadAttention.new(embed, heads))
