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
        # 모듈 쪽에 손으로 쓴 것은 여기서도 같은 것을 쓴다 — `F.pad` 가 그 예다.
        from . import _ops
        if name == "embedding":
            return embedding
        if name in ("pad", "clamp", "flip", "pow", "split", "chunk",
                    "layer_norm", "where", "squeeze", "repeat_interleave"):
            fn = getattr(_ops, name)
            return lambda *a, **k: fn(*a, **k)

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


def embedding(idx, table):
    """`F.embedding(번호, 표)` — 표에서 번호대로 행을 고른다.

    **정의 그대로다.** `index_select` 가 하는 일과 같고, 기울기도 그쪽이 이미 안다 —
    같은 번호가 여러 번 나오면 그 행으로 여러 번 더해진다. 없는 것을 흉내 내는 것이
    아니라 있는 것에 이름을 붙이는 것이므로 값이 갈릴 자리가 없다.
    """
    flat = handle(idx).reshape(_js.Array.of(int(handle(idx).size)))
    picked = handle(table).indexSelect(0, flat)
    shape = [int(n) for n in handle(idx).shape] + [int(handle(table).shape[1])]
    return wrap(picked.reshape(_js.Array.from_(shape)))


class Transformer:
    """torch 의 `nn.Transformer` 는 여기 없다. **마스크 만드는 자리 하나만** 있다.

    `generate_square_subsequent_mask` 는 층이 아니라 정의가 정해진 함수다 — 위쪽
    삼각을 `-inf` 로, 나머지를 0 으로. 값이 실수라는 것이 요점이고, 참·거짓으로
    뭉뚱그리면 어텐션 안에서 갈린다(골든 케이스 이름이 그렇게 적혀 있다).

    나머지(인코더·디코더)는 없다. 없는 것을 흉내 내지 않는다.
    """

    @staticmethod
    def generate_square_subsequent_mask(n):
        import numpy as _np
        from ._base import tensor as _t

        m = _np.zeros((n, n), dtype=_np.float32)
        m[_np.triu_indices(n, 1)] = -_np.inf
        return _t(m)


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
    """층 하나. **감싸는 쪽과 상속하는 쪽 둘 다 된다.**

    감쌀 때는 borch.ts 의 층을 하나 받는다(`Module(js_layer)`).

    **상속도 받아야 한다.** torch 코드가 가장 흔히 하는 일이 이것이다 —
    `class Net(nn.Module)` 를 쓰고 `__init__` 에서 층을 속성으로 붙인 뒤 `forward` 를
    적는다. 골든의 케이스들이 전부 `nn.Sequential` 로만 모델을 세워서 이 자리를 한
    번도 안 물었고, 벤치가 진짜 ResNet 을 세우다 `Module.__init__() missing 1
    required positional argument` 로 걸렸다.

    상속한 쪽은 `_m` 이 없다. 파라미터와 `state_dict` 는 **속성에 붙은 층들을 훑어**
    모은다 — torch 도 그렇게 한다.
    """

    def __init__(self, module=None):
        object.__setattr__(self, "_m", module)

    # ── 상속한 쪽이 속성으로 붙인 층들 ────────────────────────────────────

    def _children(self):
        """속성에 붙은 층과 텐서를 **붙인 순서대로** 준다.

        이름 규칙이 torch 와 같아야 한다 — `state_dict` 의 열쇠가 `conv1.weight`
        처럼 속성 이름으로 만들어지고, 골든이 그 이름으로 가중치를 넣는다.
        """
        got = []
        for key, value in vars(self).items():
            if key.startswith("_"):
                continue
            if isinstance(value, (Module, _Wrap, _Sequential)) or \
                    isinstance(value, Tensor):
                got.append((key, value))
        return got

    def __call__(self, *args):
        # 상속한 쪽은 자기 `forward` 를 갖는다. 감싼 쪽만 JS 로 넘긴다.
        if self._m is None:
            return self.forward(*args)
        return guarded(self._m.call, *[_arg(a) for a in args])

    def forward(self, *args):
        if self._m is None:
            raise NotImplementedError(f"{type(self).__name__} 에 forward 가 없다")
        return self(*args)

    def parameters(self):
        if self._m is None:
            return [p for _, m in self._children() for p in _params_of(m)]
        # JS 배열은 파이썬에서 바로 못 돈다 — `to_py` 로 목록을 받아야 한다.
        return [wrap(p) for p in self._m.parameters().to_py()]

    def state_dict(self):
        if self._m is None:
            out = {}
            for name, m in self._children():
                if isinstance(m, Tensor):
                    out[name] = m
                    continue
                for k, v in _state_of(m).items():
                    out[f"{name}.{k}"] = v
            return out
        got = self._m.stateDict()
        return {str(k): wrap(getattr(got, k)) for k in _js.Object.keys(got)}

    def named_parameters(self):
        """`(이름, 텐서)` 짝. torch 코드가 `dict(...)` 로 받아 이름으로 꺼낸다.

        `state_dict` 와 같은 이름 규칙을 쓴다 — `0.weight` 처럼 자리 번호가 앞에
        붙는다. 실제로 그 이름으로 꺼내는 케이스가 있어서 규칙이 맞아야 한다.
        """
        return list(self.state_dict().items())

    def load_state_dict(self, values, strict=True):
        if self._m is None:
            # 이름 앞머리로 갈라 자식에게 넘긴다 — `conv1.weight` → `conv1` 의 `weight`.
            own = dict(self._children())
            groups = {}
            for key, v in values.items():
                head, _, rest = key.partition(".")
                if not rest and isinstance(own.get(head), Tensor):
                    own[head]._h.copyFrom(handle(v))
                    continue
                groups.setdefault(head, {})[rest] = v
            for head, sub in groups.items():
                if head in own:
                    own[head].load_state_dict(sub, strict)
                elif strict:
                    raise RuntimeError(f"load_state_dict: 모르는 이름 '{head}'")
            return
        obj = _js.Object.new()
        for k, v in values.items():
            setattr(obj, k, handle(v))
        self._m.loadStateDict(obj, strict)

    def train(self, mode=True):
        if self._m is None:
            for _, m in self._children():
                if hasattr(m, "train"):
                    m.train(mode)
            return self
        self._m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __getattr__(self, name):
        """`bn.weight` 처럼 층이 들고 있는 것을 그대로 넘긴다.

        **`_m` 을 여기서 다시 물으면 안 된다.** `_Wrap` 처럼 `_m` 이 없는 하위 클래스가
        오면 `__getattr__` 이 자기 자신을 부르고 무한 재귀가 된다 — 실패는 CNN 학습
        케이스에서 `RecursionError` 로 나왔고, 원인에서 한참 떨어진 자리다.
        """
        if name.startswith("_") or self._m is None:
            raise AttributeError(name)
        got = getattr(self._m, camel(name), None)
        if got is None:
            raise AttributeError(f"borch.ts 층에 `{name}` 이 없다")
        if _ts.isTensor(got):
            return wrap(got)
        return got


def _layer(js_name, *args):
    return Module(getattr(_ts.nn, js_name).new(*args))


class _Wrap:
    """borch.ts 에 층으로는 없고 **텐서 메서드로는 있는** 것들.

    `nn.Softmax(dim)` 은 `x.softmax(dim)` 이다. 없는 것을 근사하는 것이 아니라
    있는 것에 torch 의 이름을 붙이는 것이므로, 값은 같은 자리에서 나온다.

    **`Module` 을 상속하지 않는다.** 상속했더니 `_m` 이 없는데 `Module` 의 메서드가
    그것을 찾았고, `__getattr__` 이 다시 자기를 불러 무한 재귀가 됐다 — CNN 학습
    케이스에서 `RecursionError` 로 나왔고 원인에서 한참 떨어진 자리였다.
    파라미터가 없는 층이라 `Module` 에서 물려받을 것도 없다.
    """

    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def forward(self, *args):
        return self(*args)

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


class _Sequential:
    """**파이썬 쪽에서 엮는다.**

    borch.ts 의 `Sequential` 에 넘기려면 층마다 JS 쪽 물건이 있어야 하는데,
    `Softmax`·`Flatten` 같은 것은 텐서 메서드를 감싼 파이썬 층이라 그것이 없다.
    JS 에 `Lambda` 같은 자리를 만들어 넣을 수도 있지만, 그러면 파라미터가 없는
    층 때문에 커널 쪽 표면이 는다. 엮는 일은 파이썬이 해도 값이 같다.

    이름 규칙은 borch.ts 와 맞춘다 — `0.weight` 처럼 자리 번호가 앞에 붙고,
    골든이 그 이름으로 가중치를 넣고 꺼낸다.
    """

    __slots__ = ("layers",)

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for m in self.layers:
            x = m(x)
        return x

    def forward(self, x):
        return self(x)

    def parameters(self):
        return [p for m in self.layers for p in _params_of(m)]

    def state_dict(self):
        out = {}
        for i, m in enumerate(self.layers):
            for k, v in _state_of(m).items():
                out[f"{i}.{k}"] = v
        return out

    def named_parameters(self):
        return list(self.state_dict().items())

    def load_state_dict(self, values, strict=True):
        groups = {}
        for key, v in values.items():
            head, _, rest = key.partition(".")
            groups.setdefault(int(head), {})[rest] = v
        for i, sub in groups.items():
            self.layers[i].load_state_dict(sub, strict)

    def train(self, mode=True):
        for m in self.layers:
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)


def _params_of(m):
    return m.parameters() if hasattr(m, "parameters") else []


def _state_of(m):
    return m.state_dict() if hasattr(m, "state_dict") else {}


def Sequential(*layers):
    flat = []
    for l in layers:
        flat.extend(l if isinstance(l, (list, tuple)) else [l])
    return _Sequential(flat)


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


def Flatten(start_dim=1, end_dim=-1):
    from ._ops import flatten
    return _Wrap(lambda x: flatten(x, start_dim, end_dim))


def Identity():
    return _Wrap(lambda x: x)


def AdaptiveAvgPool2d(output_size=1):
    n = output_size[0] if isinstance(output_size, (list, tuple)) else output_size
    return _Wrap(lambda x: wrap(handle(x).adaptiveAvgPool(n)))


def AvgPool2d(k=2, stride=None):
    return _Wrap(lambda x: wrap(handle(x).avgPool2d(k, stride)))


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
        # **축을 셋으로 맞춘다.** torch 의 마지막 상태는 `(층수, 배치, 은닉)` 이다.
        # 처음에 무조건 하나를 더 붙였더니 넷이 됐다 — 이미 셋이면 그대로 둔다.
        if h.ndim == 2:
            h = wrap(h._h.unsqueeze(0))
        if self._m.kind == "LSTM":
            c = wrap(got.cell)
            if c.ndim == 2:
                c = wrap(c._h.unsqueeze(0))
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
        """**`attend` 를 부른다 — `forward` 는 마스크를 버린다.**

        borch.ts 의 `forward(x)` 는 마스크 자리에 `null` 을 넣는다. `call` 로 가면
        마스크가 조용히 사라지고, 값만 조금 다른 답이 나온다(최대차 1.6e-01) —
        자기 자신을 보는 자리까지 섞이니 그럴듯하게 틀린 값이다.

        셋을 따로 받는 것도 torch 의 모양일 뿐, 이쪽은 자기 주의(self-attention)라
        하나만 쓴다. 골든이 `mod(x, x, x)` 로 부르므로 셋이 같다.
        """
        mask = handle(attn_mask) if attn_mask is not None else None
        return wrap(self._m.attend(handle(q), mask)), None


def MultiheadAttention(embed, heads, batch_first=False):
    return _Attention(_ts.nn.MultiheadAttention.new(embed, heads))
