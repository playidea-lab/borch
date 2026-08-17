"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import collections as _collections
import math as _math

import numpy as _np

from ._tensor import (
    Tensor,
)
from ._base import (
    _DEFAULT_DTYPE, _like_torch, _math, _np, _unsupported,
)
from ._ops import (
    _Namespace, _gelu, _pool_all, _reduce, _rng, _spread, _wrap, adaptive_avg_pool1d,
    adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool1d,
    adaptive_max_pool2d, adaptive_max_pool3d, avg_pool1d, avg_pool2d, avg_pool3d,
    celu, lp_pool1d, lp_pool2d, lp_pool3d,
    conv1d,
    conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d,
    cosine_similarity, dropout, elu, embedding, gelu, glu, group_norm, hardshrink,
    hardsigmoid, hardswish, hardtanh, instance_norm, interpolate, l1_loss, layer_norm,
    leaky_relu, log_softmax, logsigmoid, max_pool1d, max_pool2d, max_pool3d, mish,
    nll_loss, no_grad, norm, normalize, pad, prelu, relu, relu6, rms_norm, selu,
    sigmoid, silu, smooth_l1_loss, softmax, softmin, softplus, softshrink, softsign,
    cat, stack, tanh, tanhshrink, zeros,
    # 손실과 거리.
    cosine_embedding_loss, gaussian_nll_loss, hinge_embedding_loss, huber_loss,
    kl_div, margin_ranking_loss, multi_margin_loss, multilabel_margin_loss,
    multilabel_soft_margin_loss, pairwise_distance, pdist, poisson_nll_loss,
    soft_margin_loss, triplet_margin_loss, triplet_margin_with_distance_loss,
    # 자리 옮기기와 채널째 dropout.
    alpha_dropout, channel_shuffle, dropout1d, dropout2d, dropout3d,
    feature_alpha_dropout, pixel_shuffle, pixel_unshuffle,
    # 창 펴기와 나머지. **`unfold_im2col` 은 이름이 갈려 있다** — `Tensor.unfold` 와
    # 하는 일이 달라서 모듈 자리를 나눠 쓸 수가 없다(그쪽 주석 참고).
    amax, bilinear, fold, local_response_norm, rrelu, unfold_im2col,
    # 이긴 자리를 함께 내는 풀링과, 그 자리로 되돌리는 짝.
    adaptive_max_pool1d_with_indices, adaptive_max_pool2d_with_indices,
    adaptive_max_pool3d_with_indices, max_pool1d_with_indices,
    max_pool2d_with_indices, max_pool3d_with_indices,
    max_unpool1d, max_unpool2d, max_unpool3d,
    fractional_max_pool2d, fractional_max_pool2d_with_indices,
    fractional_max_pool3d, fractional_max_pool3d_with_indices,
    ctc_loss,
    # 제자리 활성과 `interpolate` 의 옛 이름들.
    celu_, elu_, hardtanh_, leaky_relu_, relu_, rrelu_, selu_, threshold_,
    upsample, upsample_bilinear, upsample_nearest,
    # 층이 얹혀 있는 함수들. 식을 한 벌만 둔다.
    batch_norm, embedding_bag, gumbel_softmax,
    # 공간 변환기의 짝.
    affine_grid, grid_sample,
)
# **`_wrap` 을 함수 안에서 들여오면 안 된다.** 한 번 그렇게 두었더니
# `tests/test_alias.py` 가 `sys.modules` 에서 `borch.*` 를 지운 뒤 그 임포트가 다시
# 돌면서 `_ops` 의 **두 번째 사본**을 만들었고, `Tensor` 클래스가 둘이 되어
# `isinstance` 가 어긋났다. 값이 object 배열로 나왔는데 그 케이스만 따로 돌리면
# 통과해서 원인이 한참 멀어 보였다 — 늦은 임포트는 이 저장소에서 그 값을 한다.
#
# **`threshold` 만 이름을 바꿔 들여온다.** `Threshold` 층이 `self.threshold` 라는
# 속성을 갖는데(torch 가 그 이름을 쓴다), 그러면 `forward` 안에서 같은 이름이 함수와
# 속성 둘을 가리키게 된다. 이름을 갈라 두면 그 자리가 아예 안 생긴다.
from ._ops import threshold as threshold_fn

# ================================================================ nn

class _NN(_Namespace):
    pass


nn = _NN()


class Parameter(Tensor):
    """학습 대상. `nn.Linear` 가 만드는 가중치는 처음부터 requires_grad 다."""

    def __init__(self, data):
        super().__init__(data.data if isinstance(data, Tensor) else _np.asarray(data), True)


class Module:
    def __init__(self):
        self._modules = {}
        self._params = {}
        self._buffers = {}          # 학습은 안 하지만 저장·복원되는 값 (running_mean 등)
        self.training = True

    def register_buffer(self, name, value):
        """torch 의 `register_buffer`. `state_dict` 에 들어가고 학습 대상은 아니다.

        BatchNorm 의 running_mean 이 여기 들어간다 — 빠뜨리면 저장했다 불러왔을 때
        **평가 모드가 초기값으로 돌아가고**, 학습은 멀쩡해 보이는데 추론만 틀린다.
        """
        self.__dict__.setdefault("_buffers", {})[name] = value
        object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_params", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        elif name in self.__dict__.get("_buffers", {}):
            self._buffers[name] = value
        object.__setattr__(self, name, value)

    def named_buffers(self, prefix=""):
        for n, b in self.__dict__.get("_buffers", {}).items():
            yield (f"{prefix}{n}", b)
        for n, m in self._modules.items():
            yield from m.named_buffers(f"{prefix}{n}.")

    def parameters(self):
        for p in self._params.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix=""):
        for n, p in self._params.items():
            yield (f"{prefix}{n}", p)
        for n, m in self._modules.items():
            yield from m.named_parameters(f"{prefix}{n}.")

    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *a, **k):
        for x in list(a) + list(k.values()):
            if isinstance(x, str) and x != "cpu":
                _unsupported(f"장치 '{x}'")
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self):
        out = {name: Tensor(p.data.copy()) for name, p in self.named_parameters()}
        for name, buf in self.named_buffers():
            out[name] = Tensor(_np.array(buf, copy=True))
        return out

    def load_state_dict(self, state, strict=True):
        own = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        missing = [k for k in list(own) + list(buffers) if k not in state]
        unexpected = [k for k in state if k not in own and k not in buffers]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"state_dict 가 안 맞습니다. 빠진 것: {missing}, 남는 것: {unexpected}"
            )
        for name, value in state.items():
            if name in buffers:
                data = value.data if isinstance(value, Tensor) else _np.asarray(value)
                holder = self
                *path, leaf = name.split(".")
                for part in path:
                    holder = holder._modules[part]
                holder.register_buffer(leaf, data.copy() if data.ndim else data.item())
                continue
            if name in own:
                target = own[name]
                incoming = value.data if isinstance(value, Tensor) else _np.asarray(value)
                if incoming.shape != target.data.shape:
                    raise RuntimeError(
                        f"{name} 의 모양이 다릅니다: {incoming.shape} vs {tuple(target.data.shape)}"
                    )
                target._array = incoming.astype(target._array.dtype).copy()
        return self

    def forward(self, *a, **k):
        raise NotImplementedError("forward 를 구현하세요.")

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def __repr__(self):
        # **자식이 여러 줄이면 그 줄들도 들여쓴다.** 한 줄만 들여썼더니 컨테이너가
        # 컨테이너를 담을 때(`ModuleList` 안의 `Sequential`) 안쪽이 왼쪽 끝에 붙어
        # torch 와 글자가 갈렸다 — 값은 멀쩡하고 그림만 틀리는 종류다.
        parts = []
        for name, mod in self._modules.items():
            head, *rest = repr(mod).splitlines()
            parts.append(f"  ({name}): {head}")
            parts.extend(f"  {line}" for line in rest)
        inner = "\n".join(parts)
        return f"{type(self).__name__}(\n{inner}\n)" if inner else f"{type(self).__name__}()"


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 진짜 torch 와 같은 초기화 (Kaiming uniform 계열): U(-1/√fan_in, 1/√fan_in)
        # **입력이 0 일 수 있다.** `AdaptiveLogSoftmaxWithLoss` 의 꼬리 차원이
        # `in_features // div_value**(i+1)` 이라 0 으로 떨어진다 — torch 는 빈 텐서를
        # 만들고 "초기화할 것이 없다" 며 넘어간다. 여기서는 √0 으로 나눠 멈췄다.
        bound = 1.0 / _math.sqrt(in_features) if in_features else 0.0
        self.weight = Parameter(_rng.uniform(-bound, bound, (out_features, in_features)).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(_rng.uniform(-bound, bound, (out_features,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        out = x @ self.weight.transpose(-2, -1)
        return out + self.bias if self.bias is not None else out

    def __repr__(self):
        # `bias=` 까지 찍는다 — torch 가 그렇고, 게으른 층이 굳은 뒤의 글자를 골든이
        # 묻기 시작하면서 드러났다. 편향이 있는지는 `state_dict` 열쇠를 바꾸는 정보다.
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={getattr(self, 'bias', None) is not None})")


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return tanh(x)


class Flatten(Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        return x.flatten(self.start_dim)


class Identity(Module):
    """**아무 인자나 받는다.** torch 가 그렇다(실측) — `Identity(64, unused=True)`.

    자리 채우개로 쓰는 층이라, 쓰는 쪽은 갈아 끼운 층의 인자를 그대로 둔 채 이름만
    바꾼다. 인자를 거절하면 그 줄에서 멈춘다.
    """

    def __init__(self, *args, **kw):
        super().__init__()

    def forward(self, x):
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (_rng.random(x.data.shape) > self.p).astype(_DEFAULT_DTYPE) / (1 - self.p)
        return x * Tensor(mask)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self._layers = list(layers)
        for i, m in enumerate(layers):
            self._modules[str(i)] = m

    def forward(self, x):
        for m in self._layers:
            x = m(x)
        return x

    def __getitem__(self, i):
        return self._layers[i]

    def __len__(self):
        return len(self._layers)


class ModuleList(Module):
    """층 목록. **번호가 곧 이름이다** — `layers.0.weight` 처럼.

    `append` 가 없으면 층 수가 정해지지 않은 모델을 쓸 방법이 없다. torch 코드에서
    가장 흔한 모양이 그것이다:

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(Block())
    """

    def __init__(self, mods=()):
        super().__init__()
        self._layers = list(mods)
        for i, m in enumerate(self._layers):
            self._modules[str(i)] = m

    def _renumber(self):
        """`_modules` 를 목록 순서대로 다시 매긴다.

        `insert` 가 있으면 번호가 밀리므로 통째로 다시 쓴다. 밀린 자리만 고치면
        중간에 옛 이름이 남고, 그것은 `state_dict` 열쇠로 새어 나온다.
        """
        self._modules.clear()
        for i, m in enumerate(self._layers):
            self._modules[str(i)] = m

    def append(self, module):
        self._layers.append(module)
        self._modules[str(len(self._layers) - 1)] = module
        return self

    def extend(self, mods):
        for m in mods:
            self.append(m)
        return self

    def insert(self, index, module):
        self._layers.insert(index, module)
        self._renumber()
        return self

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def __setitem__(self, i, module):
        self._layers[i] = module
        self._renumber()

    def __iadd__(self, mods):
        return self.extend(mods)

    def __len__(self):
        return len(self._layers)


def _ordered(mapping):
    """torch 의 순서 규칙. **평범한 dict 는 열쇠를 정렬해서 넣는다.**

    `OrderedDict` 나 같은 종류의 컨테이너로 주면 넣은 순서를 지키고, 그냥 `dict`
    로 주면 정렬한다. torch 가 그렇게 하는데(옛 파이썬의 dict 가 순서를 안 지켰다),
    안 맞추면 `named_parameters` 의 순서가 갈리고 그것이 곧 `state_dict` 의 순서다.

    골든이 이 자리를 잡았다 — `{"w": …, "b": …}` 를 넣었더니 torch 는 `ws.b ws.w`
    를 내고 우리는 `ws.w ws.b` 를 냈다. 두 열쇠를 알파벳 순으로 골랐으면 우연히
    통과했을 자리다.
    """
    items = dict(mapping or {})
    if isinstance(mapping, (ModuleDict, ParameterDict, _collections.OrderedDict)):
        return list(items.items())
    return sorted(items.items(), key=lambda kv: str(kv[0]))


class ModuleDict(Module):
    """이름 붙은 층 묶음. 갈래를 이름으로 고르는 모델이 쓴다.

    번호 대신 준 이름이 그대로 `state_dict` 열쇠가 된다 — `blocks.down.weight`.
    """

    def __init__(self, mods=None):
        super().__init__()
        for name, m in _ordered(mods):
            self._modules[str(name)] = m

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self._modules[str(key)] = module

    def __contains__(self, key):
        return key in self._modules

    def __iter__(self):
        return iter(self._modules)

    def __len__(self):
        return len(self._modules)

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, mods):
        for name, m in _ordered(mods):
            self._modules[str(name)] = m
        return self


class ParameterList(Module):
    """`Parameter` 목록. **이것이 없으면 대신할 방법이 없다.**

    맨 리스트에 `Parameter` 를 담아 속성으로 붙이면 `Module.__setattr__` 이 그것을
    `Parameter` 로도 `Module` 로도 못 알아본다. 어느 목록에도 안 들어가고,
    `parameters()` 가 안 내놓고, 옵티마이저가 못 본다 — 그런데 **손실은 내려간다.**
    남은 파라미터가 대신 맞추기 때문이다. 예외도 경고도 없다.

    torch 도 똑같이 못 알아보고, 그래서 torch 에 이 클래스가 있다.
    """

    def __init__(self, params=()):
        super().__init__()
        for i, p in enumerate(params):
            self._params[str(i)] = p

    def _at(self, i):
        keys = list(self._params)
        return keys[i]

    def append(self, param):
        self._params[str(len(self._params))] = param
        return self

    def extend(self, params):
        for p in params:
            self.append(p)
        return self

    def __getitem__(self, i):
        return self._params[self._at(i)]

    def __setitem__(self, i, param):
        self._params[self._at(i)] = param

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self):
        return len(self._params)


class ParameterDict(Module):
    """이름 붙은 `Parameter` 묶음. `ParameterList` 와 같은 이유로 있다."""

    def __init__(self, params=None):
        super().__init__()
        for name, p in _ordered(params):
            self._params[str(name)] = p

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, param):
        self._params[str(key)] = param

    def __contains__(self, key):
        return key in self._params

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    def items(self):
        return self._params.items()

    def update(self, params):
        for name, p in _ordered(params):
            self._params[str(name)] = p
        return self


# **접는 방식은 손실의 일부다.** `_ops._reduce` 가 그 규칙을 한 자리에 두고 열셋이
# 그것을 쓰는데, **제일 흔한 다섯이 안 쓰고 있었다** — 여기 넷과 `NLLLoss` 다.
# `.mean()` 이 박혀 있어서 `reduction=` 을 주면 `TypeError` 로 멈췄다.
#
# 뒤집혀 있다는 것이 실마리였다. `cosine_embedding`·`multi_margin`·`triplet` 처럼
# **드문 손실은 전부** `reduction` 을 받는데 `MSELoss`·`CrossEntropyLoss` 가 안 받았다.
# 나중에 쓴 것이 torch 서명을 따랐고 처음 쓴 것이 안 고쳐진 것이다. 골든이 못 본
# 이유는 튜토리얼이 기본값 `mean` 만 쓰기 때문이다.
class _Loss(Module):
    """`reduction` 을 한 번만 받아 둔다. 손실마다 적으면 자리마다 어긋난다."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class MSELoss(_Loss):
    def forward(self, pred, target):
        return _reduce((pred - target) ** 2, self.reduction)


class BCEWithLogitsLoss(_Loss):
    def forward(self, logits, target):
        # log(1+e^-|x|) + max(x,0) - x*t  — 큰 값에서도 안전한 형태
        x, t = logits, target
        return _reduce(relu(x) - x * t + (1 + (-(x.abs())).exp()).log(),
                       self.reduction)


class BCELoss(_Loss):
    def forward(self, p, t):
        eps = 1e-12
        return _reduce(-(t * (p + eps).log() + (1 - t) * (1 - p + eps).log()),
                       self.reduction)


class CrossEntropyLoss(_Loss):
    def forward(self, logits, target):
        n = logits.data.shape[0]
        sm = softmax(logits, dim=-1)
        idx = target.data.astype(int)
        picked = sm[_np.arange(n), idx]
        return _reduce(-(picked + 1e-12).log(), self.reduction)


def _nn_unsupported(name):
    def factory(*a, **k):
        _unsupported(f"nn.{name}")
    return factory


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        bound = 1.0 / _math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = Parameter(_rng.uniform(
            -bound, bound, (out_channels, in_channels, kernel_size, kernel_size)).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(_rng.uniform(-bound, bound, (out_channels,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class MaxPool2d(Module):
    """**`return_indices` 를 켜면 답이 둘이 된다** — 값과 이긴 자리.

    `MaxUnpool2d` 에 그 자리표를 넘겨 되돌린다. 자동 부호기(autoencoder)에서 흔한
    짝이고, 자리표 없이는 되돌릴 수가 없다 — 어느 칸이 이겼는지가 값 안에 없다.
    """

    def __init__(self, kernel_size, stride=None, return_indices=False):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.return_indices = return_indices

    def forward(self, x):
        return max_pool2d(x, self.kernel_size, self.stride,
                          return_indices=self.return_indices)


class Embedding(Module):
    """번호를 벡터로 바꾸는 학습 가능한 표. 8장에서 '번호를 매기면 안 된다'고 한 그 대안."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(_rng.standard_normal(
            (num_embeddings, embedding_dim)).astype(_DEFAULT_DTYPE))

    def forward(self, idx):
        ids = idx.data.astype(int)
        out = self.weight.data[ids]

        def back(g):
            gw = _np.zeros_like(self.weight.data)
            _np.add.at(gw, ids.reshape(-1), _np.asarray(g).reshape(-1, self.embedding_dim))
            return (gw,)

        return self.weight._make(out, (self.weight,), back)

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class LayerNorm(Module):
    """**`normalized_shape` 는 접는 축의 개수를 정한다** — 마지막 축 하나가 아니다.

    `LayerNorm(4)` 로만 재면 이 갈림이 안 보인다. 축이 하나일 때는 "마지막 축을
    접는다" 와 같은 답이기 때문이고, 실제로 그렇게 적혀 있었다. `LayerNorm((3, 4))`
    는 뒤 두 축을 **한 덩어리로** 접는다 — 평균과 분산이 12 칸에서 나온다.

    `elementwise_affine` 도 받는다. 끄면 파라미터가 없어지고, 그러면 `state_dict`
    열쇠가 통째로 사라진다 — 그것은 값이 아니라 배선의 이야기다.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 bias=True):
        super().__init__()
        shape = ((normalized_shape,) if isinstance(normalized_shape, int)
                 else tuple(normalized_shape))
        self.normalized_shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(_np.ones(shape, dtype=_DEFAULT_DTYPE))
            if bias:
                self.bias = Parameter(_np.zeros(shape, dtype=_DEFAULT_DTYPE))

    def forward(self, x):
        dims = len(self.normalized_shape)
        # 뒤 `dims` 개 축을 하나로 접어 평균·분산을 한 번에 낸다. 축마다 따로 접으면
        # 같은 수가 안 나온다 — 평균의 평균은 평균이 아니다.
        shape = tuple(int(n) for n in x.shape)
        # **모양이 안 맞으면 멈춘다.** 관대하면 잘못된 축을 조용히 접는다.
        if shape[len(shape) - dims:] != self.normalized_shape:
            raise RuntimeError(_like_torch(
                f"normalized_shape={list(self.normalized_shape)} 인데 입력이 "
                f"{list(shape)} 입니다.",
                f"Given normalized_shape={list(self.normalized_shape)}, expected "
                f"input with shape [*, "
                f"{', '.join(str(n) for n in self.normalized_shape)}]"))
        lead = shape[:len(shape) - dims]
        flat = x.reshape(*lead, -1) if dims > 1 else x
        mean = flat.mean(dim=-1, keepdim=True)
        centered = flat - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        normed = centered / (var + self.eps) ** 0.5
        if dims > 1:
            normed = normed.reshape(*shape)
        if not self.elementwise_affine:
            return normed
        out = normed * self.weight
        return out + self.bias if hasattr(self, "bias") else out


class BatchNorm2d(Module):
    """학습 중에는 이번 배치로, 평가 때는 모아둔 값으로 — 6장에서 eval() 이 바꾸는 그 층."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = Parameter(_np.ones(num_features, dtype=_DEFAULT_DTYPE))
        self.bias = Parameter(_np.zeros(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("running_mean", _np.zeros(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("running_var", _np.ones(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("num_batches_tracked", 0)

    def forward(self, x):
        # **계산은 `F.batch_norm` 이 한다.** 층과 함수가 각자 적으면 언젠가 갈리고,
        # 갈리는 자리가 running 통계라 학습은 멀쩡하고 평가만 틀린다.
        #
        # 랭크를 안 따진다 — 채널 축만 남기고 나머지를 전부 줄이므로 (N,C,H,W) 도
        # (N,C,D,H,W) 도 같은 코드로 돈다. `BatchNorm3d` 가 그대로 물려받는다.
        out = batch_norm(x, self.running_mean, self.running_var,
                         self.weight, self.bias, self.training,
                         self.momentum, self.eps)
        if self.training:
            with no_grad():
                self.num_batches_tracked = self.num_batches_tracked + 1
        return out



class _RNNBase(Module):
    """RNN·LSTM·GRU 의 공통 부분 — 파라미터 만들기와 층·시간 루프.

    파라미터 이름을 torch 와 같게 둔다(`weight_ih_l0` …). 이름이 맞아야 `state_dict` 키가
    맞고, 체크포인트가 양쪽을 오간다.

    시간 방향은 파이썬 반복문이다. 순환은 앞을 끝내야 뒤를 볼 수 있어서(30장) 병렬화가 안 되고,
    그 느림이 곧 트랜스포머가 나온 이유다. 다만 **입력 쪽 곱은 h 에 안 걸리므로**
    시간 전체를 한 번에 계산해 둔다 — 반복문 안에는 은닉 쪽 곱만 남는다.
    """

    gates = 1

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.has_bias = bias

        bound = 1.0 / _math.sqrt(hidden_size)
        g = self.gates
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            setattr(self, f"weight_ih_l{layer}", Parameter(
                _rng.uniform(-bound, bound, (g * hidden_size, in_size)).astype(_DEFAULT_DTYPE)))
            setattr(self, f"weight_hh_l{layer}", Parameter(
                _rng.uniform(-bound, bound, (g * hidden_size, hidden_size)).astype(_DEFAULT_DTYPE)))
            if bias:
                setattr(self, f"bias_ih_l{layer}", Parameter(
                    _rng.uniform(-bound, bound, g * hidden_size).astype(_DEFAULT_DTYPE)))
                setattr(self, f"bias_hh_l{layer}", Parameter(
                    _rng.uniform(-bound, bound, g * hidden_size).astype(_DEFAULT_DTYPE)))

    def _weights(self, layer):
        return (getattr(self, f"weight_ih_l{layer}"), getattr(self, f"weight_hh_l{layer}"),
                getattr(self, f"bias_ih_l{layer}", None), getattr(self, f"bias_hh_l{layer}", None))

    def _run(self, x, init):
        """(출력, 층별 마지막 상태 목록). init 은 층별 초기 상태를 주는 함수."""
        if self.batch_first:
            x = x.transpose(0, 1)                       # (N,T,I) → (T,N,I)
        T, N = x.data.shape[0], x.data.shape[1]

        layer_input = x
        finals = []
        for layer in range(self.num_layers):
            w_ih, w_hh, b_ih, b_hh = self._weights(layer)
            pre = layer_input @ w_ih.transpose(0, 1)     # (T, N, gates*H) — h 와 무관
            if self.has_bias:
                pre = pre + b_ih
            state = init(layer, N)
            steps = []
            for t in range(T):
                state, out = self._step(pre[t], state, w_hh, b_hh)
                steps.append(out)
            layer_input = stack(steps)
            finals.append(state)

        out = layer_input
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, finals

    def _step(self, pre_t, state, w_hh, b_hh):
        raise NotImplementedError

    def __repr__(self):
        return (f"{type(self).__name__}({self.input_size}, {self.hidden_size}"
                + (f", num_layers={self.num_layers}" if self.num_layers > 1 else "")
                + (", batch_first=True" if self.batch_first else "") + ")")


class RNN(_RNNBase):
    """h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh) — 29장이 가르치는 그것."""

    def __init__(self, *a, nonlinearity="tanh", **k):
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity 는 'tanh' 나 'relu' 여야 합니다.")
        self.nonlinearity = nonlinearity
        super().__init__(*a, **k)

    def _step(self, pre_t, h, w_hh, b_hh):
        act = tanh if self.nonlinearity == "tanh" else relu
        z = pre_t + h @ w_hh.transpose(0, 1)
        if self.has_bias:
            z = z + b_hh
        h = act(z)
        return h, h

    def forward(self, x, hx=None):
        if hx is None:
            hx = zeros(self.num_layers, x.data.shape[1 if not self.batch_first else 0],
                       self.hidden_size)
        out, finals = self._run(x, lambda layer, n: hx[layer])
        return out, stack(finals)


class LSTM(_RNNBase):
    """게이트 넷으로 **무엇을 잊고 무엇을 남길지** 배운다.

        i = σ(...)  잊지 말고 넣을 것      f = σ(...)  버릴 것
        g = tanh(...) 넣을 값             o = σ(...)  내보낼 것
        c' = f·c + i·g                   h' = o·tanh(c')

    `weight_ih_l0` 는 (4H, I) 이고 행 순서가 **i, f, g, o** 다. torch 와 같게 둬야
    체크포인트가 오간다 — 순서를 바꾸면 값은 그럴듯한데 학습이 안 된다.
    """

    gates = 4

    def _step(self, pre_t, state, w_hh, b_hh):
        h, c = state
        z = pre_t + h @ w_hh.transpose(0, 1)
        if self.has_bias:
            z = z + b_hh
        H = self.hidden_size
        i = sigmoid(z[:, 0 * H:1 * H])
        f = sigmoid(z[:, 1 * H:2 * H])
        g = tanh(z[:, 2 * H:3 * H])
        o = sigmoid(z[:, 3 * H:4 * H])
        c = f * c + i * g
        h = o * tanh(c)
        return (h, c), h

    def forward(self, x, hx=None):
        batch = x.data.shape[0 if self.batch_first else 1]
        if hx is None:
            zero = zeros(self.num_layers, batch, self.hidden_size)
            hx = (zero, zeros(self.num_layers, batch, self.hidden_size))
        h0, c0 = hx
        out, finals = self._run(x, lambda layer, n: (h0[layer], c0[layer]))
        return out, (stack([h for h, _ in finals]), stack([c for _, c in finals]))


class GRU(_RNNBase):
    """게이트 셋. LSTM 보다 단순하고 대개 비슷하게 동작한다.

        r = σ(...)  과거를 얼마나 볼까      z = σ(...)  얼마나 갈아탈까
        n = tanh(W_in·x + b_in + r·(W_hn·h + b_hn))
        h' = (1-z)·n + z·h

    n 게이트에서 **r 이 편향까지 포함한 은닉 항에 곱해진다** — 편향을 밖에 두면
    값이 미세하게 어긋나고, 그건 눈에 안 띈다.
    """

    gates = 3

    def _step(self, pre_t, h, w_hh, b_hh):
        H = self.hidden_size
        hh = h @ w_hh.transpose(0, 1)
        if self.has_bias:
            hh = hh + b_hh
        r = sigmoid(pre_t[:, 0 * H:1 * H] + hh[:, 0 * H:1 * H])
        z = sigmoid(pre_t[:, 1 * H:2 * H] + hh[:, 1 * H:2 * H])
        n = tanh(pre_t[:, 2 * H:3 * H] + r * hh[:, 2 * H:3 * H])
        h = (1 - z) * n + z * h
        return h, h

    def forward(self, x, hx=None):
        if hx is None:
            hx = zeros(self.num_layers, x.data.shape[0 if self.batch_first else 1],
                       self.hidden_size)
        out, finals = self._run(x, lambda layer, n: hx[layer])
        return out, stack(finals)




class RNNCellBase(Module):
    """되풀이의 **한 걸음.** 시간 루프를 손으로 적고 싶은 코드가 이것을 부른다.

    **이름이 층 쪽과 다르다.** 층은 `weight_ih_l0` 처럼 층 번호를 붙이고 셀은
    `weight_ih` 다 — 셀에는 층이 없기 때문이다. `state_dict` 열쇠가 그 이름이므로
    틀리면 체크포인트가 안 맞는다.

    게이트 수만 다르고 나머지는 같다. 한 걸음 식은 `_RNNBase` 쪽 것을 그대로 쓴다 —
    두 벌로 적으면 게이트 순서가 갈리는 날이 오고, 그때 모양은 같고 값만 틀린다.
    """

    gates = 1

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.has_bias = bias
        bound = 1.0 / _math.sqrt(hidden_size)
        g = self.gates
        self.weight_ih = Parameter(_rng.uniform(
            -bound, bound, (g * hidden_size, input_size)).astype(_DEFAULT_DTYPE))
        self.weight_hh = Parameter(_rng.uniform(
            -bound, bound, (g * hidden_size, hidden_size)).astype(_DEFAULT_DTYPE))
        if bias:
            self.bias_ih = Parameter(
                _rng.uniform(-bound, bound, g * hidden_size).astype(_DEFAULT_DTYPE))
            self.bias_hh = Parameter(
                _rng.uniform(-bound, bound, g * hidden_size).astype(_DEFAULT_DTYPE))

    def _pre(self, x):
        out = x @ self.weight_ih.transpose(0, 1)
        return out + self.bias_ih if self.has_bias else out

    def _zeros(self, x):
        return zeros(x.data.shape[0], self.hidden_size)

    def __repr__(self):
        tail = "" if self.has_bias else ", bias=False"
        return f"{type(self).__name__}({self.input_size}, {self.hidden_size}{tail})"


class RNNCell(RNNCellBase):
    gates = 1

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity 는 'tanh' 나 'relu' 여야 합니다.")
        self.nonlinearity = nonlinearity
        super().__init__(input_size, hidden_size, bias)

    def forward(self, x, hx=None):
        h = self._zeros(x) if hx is None else hx
        z = self._pre(x) + h @ self.weight_hh.transpose(0, 1)
        if self.has_bias:
            z = z + self.bias_hh
        return (tanh if self.nonlinearity == "tanh" else relu)(z)

    def __repr__(self):
        parts = f"{self.input_size}, {self.hidden_size}"
        if not self.has_bias:
            parts += ", bias=False"
        if self.nonlinearity != "tanh":
            parts += f", nonlinearity={self.nonlinearity}"
        return f"RNNCell({parts})"


class GRUCell(RNNCellBase):
    """게이트 셋. **`weight_ih` 안의 순서가 `r, z, n` 이다** — 바꾸면 값만 갈린다."""

    gates = 3

    def forward(self, x, hx=None):
        h = self._zeros(x) if hx is None else hx
        H = self.hidden_size
        pre = self._pre(x)
        hh = h @ self.weight_hh.transpose(0, 1)
        if self.has_bias:
            hh = hh + self.bias_hh
        r = sigmoid(pre[:, 0:H] + hh[:, 0:H])
        z = sigmoid(pre[:, H:2 * H] + hh[:, H:2 * H])
        n = tanh(pre[:, 2 * H:3 * H] + r * hh[:, 2 * H:3 * H])
        return (1 - z) * n + z * h


class LSTMCell(RNNCellBase):
    """게이트 넷. **혼자 둘을 돌려준다** — `(h, c)` 다.

    셋을 한 모양으로 두면 기억 칸이 사라지고, 그러면 값은 나오는데 학습이 안 된다.
    `weight_ih` 안의 순서는 `i, f, g, o` 다.
    """

    gates = 4

    def forward(self, x, hx=None):
        h, c = (self._zeros(x), self._zeros(x)) if hx is None else hx
        H = self.hidden_size
        z = self._pre(x) + h @ self.weight_hh.transpose(0, 1)
        if self.has_bias:
            z = z + self.bias_hh
        i = sigmoid(z[:, 0:H])
        f = sigmoid(z[:, H:2 * H])
        g = tanh(z[:, 2 * H:3 * H])
        o = sigmoid(z[:, 3 * H:4 * H])
        c = f * c + i * g
        return o * tanh(c), c


nn.RNNCellBase = RNNCellBase
nn.RNNCell = RNNCell
nn.GRUCell = GRUCell
nn.LSTMCell = LSTMCell
# `RNNBase` 는 `RNN`·`LSTM`·`GRU` 의 부모다. torch 에서도 직접 못 만든다(ValueError).
nn.RNNBase = _RNNBase


# ---------------------------------------------------------------- 최상위 순환
#
# **층과 같은 계산인데 가중치를 목록으로 받는다.** `torch.lstm(x, (h,c), params, …)`
# 이 그 꼴이고, 층이 안에서 부르는 것이 이것이다.
#
# **층을 지어 가중치를 갈아 끼운다.** 되풀이 식을 여기서 다시 적으면 게이트 순서가
# 갈리는 날이 오고, 그때 모양은 같고 값만 틀린다 — 셀 쪽 docstring 이 이미 같은
# 이유로 층의 식을 빌려 쓴다. 지어 두면 파라미터가 한 벌 낭비되지만, 그것은 값이
# 아니라 시간이다.

def _install_weights(mod, params, num_layers, has_biases):
    """평평한 가중치 목록을 층의 이름표 자리에 꽂는다.

    차례는 **층마다 `[w_ih, w_hh, b_ih, b_hh]`** 다(실측). 편향이 없으면 둘씩이다.
    `Parameter` 로 감싸지 않는다 — 부르는 쪽이 준 텐서로 기울기가 그대로 가야 한다.
    """
    per = 4 if has_biases else 2
    want = per * num_layers
    if len(params) != want:
        raise RuntimeError(_like_torch(
            f"가중치가 {want} 개여야 하는데 {len(params)} 개입니다 "
            f"(층 {num_layers} × {per}).",
            "expected a tuple of tensors of the right length"))
    for layer in range(num_layers):
        chunk = params[layer * per:(layer + 1) * per]
        setattr(mod, f"weight_ih_l{layer}", chunk[0])
        setattr(mod, f"weight_hh_l{layer}", chunk[1])
        if has_biases:
            setattr(mod, f"bias_ih_l{layer}", chunk[2])
            setattr(mod, f"bias_hh_l{layer}", chunk[3])


def _rnn_top(cls, x, hx, params, has_biases, num_layers, dropout, train,
             bidirectional, batch_first, **kw):
    """최상위 순환 넷의 공통 몸통.

    **양방향과 층간 드롭아웃은 거절한다.** 우리 층에 그 둘이 없어서다 — 여기서
    한 방향만 돌려주면 모양이 절반이라 시끄럽게 걸리겠지만, 드롭아웃 쪽은 값이
    그럴듯한 채로 갈린다(정칙화가 안 걸린 학습). 둘 다 여기서 멈춘다.
    """
    if bidirectional:
        _unsupported("양방향 순환(bidirectional=True)")
    if train and dropout:
        _unsupported(f"층간 드롭아웃(dropout={dropout})")
    first = params[0]
    hidden = first.data.shape[0] // cls.gates
    mod = cls(first.data.shape[1], hidden, num_layers, bias=bool(has_biases),
              batch_first=bool(batch_first), **kw)
    _install_weights(mod, params, num_layers, bool(has_biases))
    return mod(x, hx)


def lstm(input, hx, params, has_biases, num_layers, dropout, train,      # noqa: A002
         bidirectional, batch_first=False):
    """`(출력, h_n, c_n)` — **셋을 편다.** 층 쪽은 `(출력, (h, c))` 로 묶는데
    최상위는 안 묶는다(실측). 묶은 채로 주면 받는 쪽의 풀기가 한 칸 어긋난다."""
    out, (h, c) = _rnn_top(LSTM, input, hx, params, has_biases, num_layers,
                           dropout, train, bidirectional, batch_first)
    return out, h, c


def gru(input, hx, params, has_biases, num_layers, dropout, train,      # noqa: A002
        bidirectional, batch_first=False):
    return _rnn_top(GRU, input, hx, params, has_biases, num_layers, dropout,
                    train, bidirectional, batch_first)


def rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train,  # noqa: A002
             bidirectional, batch_first=False):
    return _rnn_top(RNN, input, hx, params, has_biases, num_layers, dropout,
                    train, bidirectional, batch_first, nonlinearity="tanh")


def rnn_relu(input, hx, params, has_biases, num_layers, dropout, train,  # noqa: A002
             bidirectional, batch_first=False):
    return _rnn_top(RNN, input, hx, params, has_biases, num_layers, dropout,
                    train, bidirectional, batch_first, nonlinearity="relu")


def _cell_top(cls, x, hx, w_ih, w_hh, b_ih, b_hh, **kw):
    """칸 넷의 공통 몸통. 층 쪽과 달리 이름에 층 번호가 없다."""
    hidden = w_ih.data.shape[0] // cls.gates
    cell = cls(w_ih.data.shape[1], hidden, bias=b_ih is not None, **kw)
    cell.weight_ih, cell.weight_hh = w_ih, w_hh
    if b_ih is not None:
        cell.bias_ih, cell.bias_hh = b_ih, b_hh
    return cell(x, hx)


def lstm_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):              # noqa: A002
    """한 걸음. `nn.LSTMCell` 과 **같은 값**이다(실측)."""
    return _cell_top(LSTMCell, input, tuple(hx), w_ih, w_hh, b_ih, b_hh)


def gru_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):              # noqa: A002
    return _cell_top(GRUCell, input, hx, w_ih, w_hh, b_ih, b_hh)


def rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):         # noqa: A002
    return _cell_top(RNNCell, input, hx, w_ih, w_hh, b_ih, b_hh,
                     nonlinearity="tanh")


def rnn_relu_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):         # noqa: A002
    return _cell_top(RNNCell, input, hx, w_ih, w_hh, b_ih, b_hh,
                     nonlinearity="relu")


class MultiheadAttention(Module):
    """45일차에 짠 어텐션을 여러 관점으로 나눈 것.

    torch 는 Q·K·V 의 가중치를 **하나로 묶어** `in_proj_weight` (3E, E) 에 담는다 —
    행렬곱을 세 번이 아니라 한 번 하려는 것이고, 그래서 체크포인트도 그 모양이다.
    나눠 들면 값은 같아도 `state_dict` 가 안 맞는다.
    """

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=False):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim({embed_dim}) 이 num_heads({num_heads}) 로 안 나뉩니다.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        bound = _math.sqrt(1.0 / embed_dim)
        self.in_proj_weight = Parameter(
            _rng.uniform(-bound, bound, (3 * embed_dim, embed_dim)).astype(_DEFAULT_DTYPE))
        self.in_proj_bias = Parameter(_np.zeros(3 * embed_dim, dtype=_DEFAULT_DTYPE)) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=True,
                key_padding_mask=None, average_attn_weights=True):
        key = query if key is None else key
        value = query if value is None else value
        # **함수 꼴은 길이가 앞이다.** `batch_first` 면 여기서 뒤집어 넘긴다 —
        # 계산은 `multi_head_attention_forward` 한 벌만 둔다.
        if self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        out, weights = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, None, None, False, 0.0,
            self.out_proj.weight, self.out_proj.bias, self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, weights

    def __repr__(self):
        return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None):
    """어텐션의 알맹이. **`MultiheadAttention` 이 안에서 하던 계산을 이름으로 낸다.**

    층은 있는데 이 함수가 없었다. 층을 안 쓰고 어텐션을 손으로 짜는 코드가 이 이름을
    부르고, 요즘 트랜스포머 코드의 기본형이 그것이다.

    **가림막은 곱하는 것이 아니라 더하는 것이다.** `-inf` 를 더해 softmax 가 0 을
    내게 하는 것이지 0 을 곱하는 것이 아니다 — 곱하면 softmax 가 이미 정규화한 뒤라
    남은 자리가 1 로 안 돌아간다.
    """
    query, key, value = _wrap(query), _wrap(key), _wrap(value)
    dim = query.data.shape[-1]
    factor = (1.0 / _math.sqrt(dim)) if scale is None else scale
    scores = (query @ key.transpose(-2, -1)) * factor
    if is_causal:
        # 위 삼각을 막는다. 가림막을 같이 주면 torch 는 둘 다 적용한다.
        length = query.data.shape[-2]
        upper = _np.triu(_np.ones((length, key.data.shape[-2]), dtype=bool), k=1)
        scores = scores.masked_fill(Tensor(upper), float("-inf"))
    if attn_mask is not None:
        scores = _apply_mask(scores, attn_mask)
    weights = softmax(scores, dim=-1)
    if dropout_p:
        weights = dropout(weights, dropout_p, training=True)
    return weights @ value


def multi_head_attention_forward(
        query, key, value, embed_dim_to_check, num_heads,
        in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn,
        dropout_p, out_proj_weight, out_proj_bias, training=True,
        key_padding_mask=None, need_weights=True, attn_mask=None,
        use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
        v_proj_weight=None, static_k=None, static_v=None,
        average_attn_weights=True, is_causal=False):
    """`MultiheadAttention` 이 안에서 하는 계산을 이름으로 낸다. **층이 이것을 부른다.**

    **입력이 `(L, N, E)` 다** — 길이가 앞이다. 층은 `batch_first` 를 받지만 이 함수는
    torch 에서도 늘 길이가 앞이라, 배치를 앞에 두고 부르면 조용히 다른 축을 섞는다.

    가중치를 밖에서 받으므로 층을 안 쓰고 어텐션을 손으로 조립하는 코드가 이 이름을
    부른다 — torch 의 `MultiheadAttention` 자신도 이것을 부른다.

    **안 하는 것은 시끄럽게 거절한다.** `bias_k`·`add_zero_attn`·`static_k` 처럼
    드물게 쓰는 갈래를 조용히 무시하면 값이 그럴듯하게 다르다.
    """
    for name, given in (("bias_k", bias_k), ("bias_v", bias_v),
                        ("static_k", static_k), ("static_v", static_v)):
        if given is not None:
            _unsupported(f"multi_head_attention_forward({name}=…)")
    if add_zero_attn:
        _unsupported("multi_head_attention_forward(add_zero_attn=True)")
    if use_separate_proj_weight:
        _unsupported("multi_head_attention_forward(use_separate_proj_weight=True)")

    query, key, value = _wrap(query), _wrap(key), _wrap(value)
    # 안에서는 배치를 앞에 두고 센다. 들어올 때와 나갈 때만 뒤집는다.
    query, key, value = (t.transpose(0, 1) for t in (query, key, value))
    B, T, E = query.data.shape
    S = key.data.shape[1]
    if embed_dim_to_check is not None and int(embed_dim_to_check) != E:
        raise AssertionError(
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {E}")
    head_dim = E // num_heads

    w, b = _wrap(in_proj_weight), None if in_proj_bias is None else _wrap(in_proj_bias)

    def project(t, index):
        piece = w[index * E:(index + 1) * E]
        out = t @ piece.transpose(0, 1)
        return out + b[index * E:(index + 1) * E] if b is not None else out

    q = _split_heads(project(query, 0), B, T, num_heads, head_dim)
    k = _split_heads(project(key, 1), B, S, num_heads, head_dim)
    v = _split_heads(project(value, 2), B, S, num_heads, head_dim)

    scores = (q @ k.transpose(-2, -1)) / _math.sqrt(head_dim)
    if is_causal and attn_mask is None:
        attn_mask = _np.triu(_np.ones((T, S), dtype=bool), k=1)
    if attn_mask is not None:
        scores = _apply_mask(scores, attn_mask)
    if key_padding_mask is not None:
        # `(N, S)` 를 `(N, 1, 1, S)` 로 펴 머리마다 같은 자리를 가린다.
        pad = _wrap(key_padding_mask)
        scores = _apply_mask(scores, pad.reshape(B, 1, 1, S))
    weights = softmax(scores, dim=-1)
    if training and dropout_p:
        weights = dropout(weights, dropout_p, True)

    merged = (weights @ v).transpose(1, 2).reshape(B, T, E)
    out = merged @ _wrap(out_proj_weight).transpose(0, 1)
    if out_proj_bias is not None:
        out = out + _wrap(out_proj_bias)
    out = out.transpose(0, 1)                       # 다시 길이가 앞으로
    if not need_weights:
        return out, None
    return out, (weights.mean(dim=1) if average_attn_weights else weights)


def _split_heads(t, B, T, heads, head_dim):
    return t.reshape(B, T, heads, head_dim).transpose(1, 2)      # (B, heads, T, head_dim)


def _apply_mask(scores, mask):
    """torch 는 마스크를 두 가지로 받는다.

      불리언 — True 인 자리를 가린다(-inf 로 채운다)
      실수   — 점수에 **더한다.** `generate_square_subsequent_mask` 가 주는 0/-inf 가 그것이다

    실수 마스크를 "0 이 아니면 가림"으로 뭉뚱그리면 인과 마스크는 우연히 맞지만
    가중치를 조절하는 마스크에서 어긋난다.
    """
    m = mask if isinstance(mask, Tensor) else Tensor(_np.asarray(mask))
    if m.data.dtype.kind == "b":
        return scores.masked_fill(m, float("-inf"))
    return scores + m


class TransformerEncoderLayer(Module):
    """어텐션 + 피드포워드, 각각에 잔차와 정규화. 10장의 Block 그대로다."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, norm_first=False, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.norm_first = norm_first
        self.activation = relu if activation == "relu" else (
            _gelu if activation == "gelu" else activation)

    def _sa(self, x, mask):
        return self.dropout(self.self_attn(x, attn_mask=mask, need_weights=False)[0])

    def _ff(self, x):
        return self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src, src_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa(self.norm1(x), src_mask)
            x = x + self._ff(self.norm2(x))
        else:
            x = self.norm1(x + self._sa(x, src_mask))
            x = self.norm2(x + self._ff(x))
        return x


class TransformerEncoder(Module):
    """같은 층을 여러 겹. torch 와 같이 `layers.N.…` 로 이름이 붙는다."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        import copy as _copy
        self.layers = ModuleList([_copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self._modules["layers"] = self.layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x) if self.norm is not None else x



class TransformerDecoderLayer(Module):
    """자기 어텐션 → **인코더를 보는 어텐션** → 피드포워드.

    인코더 층과 다른 점은 가운데 하나다 — `multihead_attn` 이 자기 자신이 아니라
    인코더의 출력(memory)을 본다. 번역에서 "지금까지 쓴 문장"과 "원문"을 함께 보는 자리다.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=False, norm_first=False, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.norm_first = norm_first
        self.activation = relu if activation == "relu" else (
            _gelu if activation == "gelu" else activation)

    def _sa(self, x, mask):
        return self.dropout(self.self_attn(x, attn_mask=mask, need_weights=False)[0])

    def _mha(self, x, memory, mask):
        return self.dropout(
            self.multihead_attn(x, memory, memory, attn_mask=mask, need_weights=False)[0])

    def _ff(self, x):
        return self.dropout(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        if self.norm_first:
            x = x + self._sa(self.norm1(x), tgt_mask)
            x = x + self._mha(self.norm2(x), memory, memory_mask)
            x = x + self._ff(self.norm3(x))
        else:
            x = self.norm1(x + self._sa(x, tgt_mask))
            x = self.norm2(x + self._mha(x, memory, memory_mask))
            x = self.norm3(x + self._ff(x))
        return x


class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        import copy as _copy
        self.layers = ModuleList([_copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self._modules["layers"] = self.layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x) if self.norm is not None else x


class Transformer(Module):
    """인코더와 디코더를 묶은 것. 「Attention Is All You Need」의 그림 전체다."""

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 batch_first=False, norm_first=False, layer_norm_eps=1e-5):
        super().__init__()
        common = dict(dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                      batch_first=batch_first, norm_first=norm_first, layer_norm_eps=layer_norm_eps)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, **common), num_encoder_layers,
            LayerNorm(d_model, eps=layer_norm_eps))
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, **common), num_decoder_layers,
            LayerNorm(d_model, eps=layer_norm_eps))
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, mask=src_mask)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """윗삼각을 -inf 로 채운 **실수** 마스크. 더해서 쓴다.

        32장의 "미래를 보지 못하게" 가 이 한 줄이다.
        """
        m = _np.zeros((sz, sz), dtype=_DEFAULT_DTYPE)
        m[_np.triu_indices(sz, 1)] = -_np.inf
        return Tensor(m)


nn.Module = Module
nn.Parameter = Parameter
nn.Linear = Linear
nn.ReLU = ReLU
nn.Sigmoid = Sigmoid
nn.Tanh = Tanh
nn.Flatten = Flatten
nn.Identity = Identity
nn.Dropout = Dropout
nn.Conv2d = Conv2d
nn.Embedding = Embedding
nn.LayerNorm = LayerNorm
nn.BatchNorm2d = BatchNorm2d
class _Activation(Module):
    """활성함수 층 — 상태가 없으니 함수 하나를 감싼다."""

    fn = staticmethod(relu)

    def forward(self, x):
        return type(self).fn(x)


class GELU(Module):
    """**인자를 받는다** — `approximate='tanh'` 는 다른 식이고 값이 다르다.

    `_Activation` 껍데기로 두면 `nn.GELU('tanh')` 가 `Module.__init__() takes 1
    positional argument` 로 멈춘다. 없는 인자를 조용히 버리는 것보다는 낫지만,
    torch 에 있는 것을 없다고 하는 것은 그대로 갈림이다.
    """

    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return gelu(x, self.approximate)


class SiLU(_Activation):
    fn = staticmethod(silu)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return leaky_relu(x, self.negative_slope)


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return elu(x, self.alpha)


# ── 상태 없는 활성함수 층. 전부 함수 하나를 감싼다. ─────────────────────────
#
# 감싸개가 **다른 함수를 부르는** 실수가 이 부류의 유일한 실패 방식이고, 그것은
# 눈으로 안 보이고 값으로만 갈린다 — 그래서 골든이 함수 꼴과 층 꼴을 따로 묻는다.

class Hardsigmoid(_Activation):
    fn = staticmethod(hardsigmoid)


class Hardswish(_Activation):
    fn = staticmethod(hardswish)


class LogSigmoid(_Activation):
    fn = staticmethod(logsigmoid)


class Mish(_Activation):
    fn = staticmethod(mish)


class ReLU6(_Activation):
    fn = staticmethod(relu6)


class SELU(_Activation):
    fn = staticmethod(selu)


class Softsign(_Activation):
    fn = staticmethod(softsign)


class Tanhshrink(_Activation):
    fn = staticmethod(tanhshrink)


class CELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return celu(x, self.alpha)


class Hardshrink(Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return hardshrink(x, self.lambd)


class Softshrink(Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return softshrink(x, self.lambd)


class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.min_val, self.max_val = min_val, max_val

    def forward(self, x):
        return hardtanh(x, self.min_val, self.max_val)


class Softplus(Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.beta, self.threshold = beta, threshold

    def forward(self, x):
        return softplus(x, self.beta, self.threshold)


class Threshold(Module):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold, self.value = threshold, value

    def forward(self, x):
        return threshold_fn(x, self.threshold, self.value)


class Softmin(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return softmin(x, dim=self.dim)


class GLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return glu(x, dim=self.dim)


class PReLU(Module):
    """음수 쪽 기울기를 **학습한다.** 이 부류에서 유일하게 파라미터가 있다.

    `weight` 라는 이름이 `state_dict` 열쇠가 되므로 torch 와 같아야 한다 — 이름이
    갈리면 남의 체크포인트를 못 읽는다.
    """

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(_np.full(num_parameters, init, dtype=_DEFAULT_DTYPE))

    def forward(self, x):
        return prelu(x, self.weight)


class GroupNorm(Module):
    """채널을 그룹으로 묶어 정규화. **배치가 작을 때 BatchNorm 대신 쓴다.**

    BatchNorm 은 배치 통계를 쓰므로 배치가 1~2 면 통계가 못 미덥다. 이쪽은 표본
    하나 안에서 묶으므로 배치 크기와 무관하다.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
        if affine:
            self.weight = Parameter(_np.ones(num_channels, dtype=_DEFAULT_DTYPE))
            self.bias = Parameter(_np.zeros(num_channels, dtype=_DEFAULT_DTYPE))
        else:
            self.weight = self.bias = None

    def forward(self, x):
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class _InstanceNorm(Module):
    """표본마다·채널마다 따로. **기본이 `affine=False` 다** — torch 가 그렇다.

    `BatchNorm` 과 반대라 헷갈리는 자리이고, 기본값을 뒤집으면 파라미터가 있는 층과
    없는 층이 바뀌어서 `state_dict` 열쇠가 통째로 갈린다.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super().__init__()
        self.num_features, self.eps = num_features, eps
        if affine:
            self.weight = Parameter(_np.ones(num_features, dtype=_DEFAULT_DTYPE))
            self.bias = Parameter(_np.zeros(num_features, dtype=_DEFAULT_DTYPE))
        else:
            self.weight = self.bias = None

    def forward(self, x):
        return instance_norm(x, self.weight, self.bias, self.eps)


class InstanceNorm1d(_InstanceNorm):
    pass


class InstanceNorm2d(_InstanceNorm):
    pass


class InstanceNorm3d(_InstanceNorm):
    pass


class RMSNorm(Module):
    """**평균을 안 뺀다.** 그것이 `LayerNorm` 과의 유일한 차이다."""

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = (Parameter(_np.ones(self.normalized_shape, dtype=_DEFAULT_DTYPE))
                       if elementwise_affine else None)

    def forward(self, x):
        return rms_norm(x, self.normalized_shape, self.weight, self.eps)


class _ConvTransposeND(Module):
    """전치 합성곱. **가중치가 `(입력, 출력, …)` 이다** — `Conv2d` 와 뒤집혀 있다.

    정사각 커널이면 뒤집어 놓아도 모양이 맞아서 값으로만 갈린다. `state_dict` 열쇠는
    `weight`·`bias` 로 `Conv2d` 와 같으므로, 모양만 보고 넣으면 조용히 틀린다.
    """

    nd = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        shape = (in_channels, out_channels) + (kernel_size,) * type(self).nd
        bound = 1.0 / _math.sqrt(out_channels * kernel_size ** type(self).nd)
        self.weight = Parameter(_rng.uniform(-bound, bound, shape).astype(_DEFAULT_DTYPE))
        self.bias = (Parameter(_rng.uniform(-bound, bound, out_channels)
                               .astype(_DEFAULT_DTYPE)) if bias else None)

    def forward(self, x):
        fn = {1: conv_transpose1d, 2: conv_transpose2d, 3: conv_transpose3d}[type(self).nd]
        return fn(x, self.weight, self.bias, self.stride, self.padding)


class ConvTranspose1d(_ConvTransposeND):
    nd = 1


class ConvTranspose2d(_ConvTransposeND):
    nd = 2


class ConvTranspose3d(_ConvTransposeND):
    nd = 3


def _default_softmax_dim(ndim):
    """`dim` 을 안 주었을 때 torch 가 고르는 축.

    **`-1` 이 아니다.** 랭크에 따라 0 이나 1 이고, torch 는 그 자리에서 경고까지
    낸다("Implicit dimension choice for softmax has been deprecated"). 규칙은
    실측했다 — 랭크 1 → 0, 2 → 1, 3 → **0**, 4 → 1.

    **랭크 2 로만 물으면 이 결함이 안 보인다.** 거기서는 `dim=1` 과 `dim=-1` 이
    같은 축이라 `-1` 을 기본값으로 두어도 답이 같다. 실제로 그렇게 두고 있었고,
    랭크 3 에서 조용히 다른 축을 접고 있었다.
    """
    return 0 if ndim in (0, 1, 3) else 1


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dim = _default_softmax_dim(x.dim()) if self.dim is None else self.dim
        return softmax(x, dim=dim)


class LogSoftmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dim = _default_softmax_dim(x.dim()) if self.dim is None else self.dim
        return log_softmax(x, dim=dim)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return avg_pool2d(x, self.kernel_size, self.stride)


class _PoolND(Module):
    """창을 정해 접는 층들의 몸통. **어느 함수를 부르느냐만 다르다.**

    층마다 `forward` 를 따로 적으면 그중 하나가 다른 함수를 부르는 날이 오고, 그것은
    값으로만 갈린다 — 활성함수 쪽에서 같은 이유로 함수 꼴과 층 꼴을 따로 물었다.
    """

    fn = staticmethod(avg_pool2d)
    adaptive = False

    def __init__(self, size, stride=None):
        super().__init__()
        self.size, self.stride = size, stride

    def forward(self, x):
        fn = type(self).fn
        return fn(x, self.size) if type(self).adaptive else fn(x, self.size, self.stride)


class AvgPool1d(_PoolND):
    fn = staticmethod(avg_pool1d)


class AvgPool3d(_PoolND):
    fn = staticmethod(avg_pool3d)


class AdaptiveAvgPool1d(_PoolND):
    fn = staticmethod(adaptive_avg_pool1d)
    adaptive = True


class AdaptiveAvgPool3d(_PoolND):
    fn = staticmethod(adaptive_avg_pool3d)
    adaptive = True


class _AdaptiveMaxPoolND(_PoolND):
    """적응형 최대 풀링. **`return_indices` 를 받는다** — 평균 쪽에는 없는 인자다."""

    adaptive = True

    def __init__(self, size, return_indices=False):
        super().__init__(size)
        self.return_indices = return_indices

    def forward(self, x):
        return type(self).fn(x, self.size, return_indices=self.return_indices)


class AdaptiveMaxPool1d(_AdaptiveMaxPoolND):
    fn = staticmethod(adaptive_max_pool1d)


class AdaptiveMaxPool2d(_AdaptiveMaxPoolND):
    fn = staticmethod(adaptive_max_pool2d)


class AdaptiveMaxPool3d(_AdaptiveMaxPoolND):
    fn = staticmethod(adaptive_max_pool3d)


class LPPool1d(Module):
    def __init__(self, norm_type, kernel_size, stride=None):
        super().__init__()
        self.norm_type, self.kernel_size, self.stride = norm_type, kernel_size, stride

    def forward(self, x):
        return lp_pool1d(x, self.norm_type, self.kernel_size, self.stride)


class LPPool2d(LPPool1d):
    def forward(self, x):
        return lp_pool2d(x, self.norm_type, self.kernel_size, self.stride)


class LPPool3d(LPPool1d):
    def forward(self, x):
        return lp_pool3d(x, self.norm_type, self.kernel_size, self.stride)


ASMoutput = _collections.namedtuple("ASMoutput", ["output", "loss"])


class AdaptiveLogSoftmaxWithLoss(Module):
    """글자가 아주 많을 때의 softmax. **자주 나오는 것을 싸게 낸다.**

    어휘가 수십만이면 마지막 선형층 하나가 모델보다 크다. 여기서는 글자를 빈도순으로
    묶어 두고, 앞쪽 뭉치(`shortlist`)는 머리에서 바로 내고 뒤쪽 뭉치들은 **머리에서
    그 뭉치를 고른 확률 × 뭉치 안의 확률**로 낸다. 뒤쪽일수록 중간 차원을 `div_value`
    로 나눠 좁힌다 — 드물게 나오는 글자에 자리를 덜 쓴다.

    ## 재보고 적은 것들

    - **기본값이 `div_value=4.0`·`head_bias=False`** 다. 2.0 으로 알고 물으면 꼬리
      층의 모양이 통째로 달라진다(`tests/probe_asm.py`).
    - 중간 차원은 `in_features // div_value**(i+1)` 이고 **0 이 될 수 있다.**
      torch 도 그 자리에서 빈 층을 만든다 — 막지 않는다.
    - `forward` 는 이름 붙은 튜플 `(output, loss)` 를 낸다. `output` 은 정답 자리의
      로그확률이고 `loss` 는 그것의 평균에 음수를 붙인 것이다.
    """

    def __init__(self, in_features, n_classes, cutoffs, div_value=4.0,
                 head_bias=False):
        super().__init__()
        cutoffs = list(cutoffs)
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(in_features, self.head_size, bias=head_bias)
        tail = []
        for i in range(self.n_clusters):
            hidden = int(in_features // (div_value ** (i + 1)))
            out = self.cutoffs[i + 1] - self.cutoffs[i]
            tail.append(Sequential(Linear(in_features, hidden, bias=False),
                                   Linear(hidden, out, bias=False)))
        self.tail = ModuleList(tail)

    def log_prob(self, x):
        """모든 글자의 로그확률 `(N, n_classes)`.

        뒤쪽 뭉치의 확률에 **머리가 그 뭉치를 고른 로그확률을 더한다** — 곱셈이
        로그에서 덧셈이고, 그래서 행 전체의 합이 1 로 남는다.
        """
        head = log_softmax(self.head(x), dim=1)
        parts = [head[:, :self.shortlist_size]]
        for i in range(self.n_clusters):
            inside = log_softmax(self.tail[i](x), dim=1)
            picked = head[:, self.shortlist_size + i:self.shortlist_size + i + 1]
            parts.append(inside + picked)
        return cat(parts, 1)

    def forward(self, x, target):
        lp = self.log_prob(x)
        idx = _np.asarray(target.data if isinstance(target, Tensor) else target)
        picked = lp.gather(1, Tensor(idx.reshape(-1, 1).astype(_np.int64)))
        out = picked.reshape(-1)
        return ASMoutput(out, -out.mean())

    def predict(self, x):
        return self.log_prob(x).argmax(dim=1)


class CTCLoss(Module):
    """소리와 글자를 **자리를 맞추지 않고** 잇는 손실.

    `forward` 가 인자를 넷 받는다 — 로그확률, 표적, 입력 길이, 표적 길이. 길이가
    표본마다 다르고 그것이 이 손실의 요점이라, 다른 손실처럼 둘로는 못 받는다.

    **`repr` 이 비어 있다.** torch 의 `extra_repr` 가 아무것도 안 낸다(재봤다).
    """

    def __init__(self, blank=0, reduction="mean", zero_infinity=False):
        super().__init__()
        self.blank, self.reduction = blank, reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return ctc_loss(log_probs, targets, input_lengths, target_lengths,
                        self.blank, self.reduction, self.zero_infinity)

    def __repr__(self):
        return "CTCLoss()"


class _FractionalMaxPoolND(Module):
    """창 자리를 무작위로 흔드는 최대 풀링.

    고정 창은 격자가 늘 같은 자리에 놓여서 그 격자에 맞는 무늬만 잘 본다. 여기서는
    창 시작이 표본에 따라 흔들려서, 같은 층을 여러 번 지나면 다른 격자를 본다 —
    학습에서 규제로 쓰인다.

    **`repr` 이 비어 있다.** torch 의 `extra_repr` 가 아무것도 안 내서 `()` 로만
    찍힌다(재봤다). 흉내가 아니라 그쪽이 그렇다.
    """

    fn = None
    dim = 0

    def __init__(self, kernel_size, output_size=None, output_ratio=None,
                 return_indices=False, _random_samples=None):
        super().__init__()
        # **둘 중 하나만 받는다** — torch 는 생성자에서 멈춘다. 관대하면 둘 다
        # 주었을 때 어느 쪽이 이겼는지가 값으로만 드러나고, 하나도 안 주었을 때는
        # 크기가 어디서 왔는지 아무도 못 읽는다.
        if (output_size is None) == (output_ratio is None):
            raise ValueError(
                "FractionalMaxPool 은 output_size 나 output_ratio 중 하나만 받습니다."
                "\n(torch: FractionalMaxPool2d requires specifying either "
                "an output size, or a pooling ratio)")
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices
        self._random_samples = _random_samples

    def forward(self, x):
        return type(self).fn(x, self.kernel_size, self.output_size,
                             self.output_ratio, self.return_indices,
                             self._random_samples)

    def __repr__(self):
        return f"FractionalMaxPool{type(self).dim}d()"


class FractionalMaxPool2d(_FractionalMaxPoolND):
    fn = staticmethod(fractional_max_pool2d)
    dim = 2


class FractionalMaxPool3d(_FractionalMaxPoolND):
    fn = staticmethod(fractional_max_pool3d)
    dim = 3


class AdaptiveAvgPool2d(_PoolND):
    """**출력 크기가 1 이 아니어도 된다.**

    예전에는 1 만 받고 나머지를 거절했다. 그때는 `_pool_all` 로 전부 평균 내는 것이
    전부여서 그랬는데, 이제 창을 자리마다 달리 잡는 기계가 생겼으므로 거절할 이유가
    없다 — 거절은 흉내의 한 방식이 아니라 다른 규칙이었다.
    """

    fn = staticmethod(adaptive_avg_pool2d)
    adaptive = True


class Unflatten(Module):
    """축 하나를 여러 축으로 편다. **뒤에 오는 축은 그대로 남는다.**

    `shape[:dim] + sizes` 로만 적혀 있었다 — 펴는 축이 마지막일 때는 맞는 답이라
    오래 안 보였고, 가운데 축을 펴면 뒤쪽이 통째로 사라진다. 원소 수가 안 맞아
    `reshape` 이 멈추므로 조용히 틀리지는 않지만, 멈추는 자리가 원인에서 멀다.
    """

    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim, self.unflattened_size = dim, tuple(unflattened_size)

    def forward(self, x):
        shape = tuple(x.data.shape)
        dim = self.dim if self.dim >= 0 else self.dim + len(shape)
        return x.reshape(shape[:dim] + self.unflattened_size + shape[dim + 1:])


class L1Loss(_Loss):
    def forward(self, pred, target):
        return l1_loss(pred, target, self.reduction)


class SmoothL1Loss(_Loss):
    def __init__(self, beta=1.0, reduction="mean"):
        super().__init__(reduction)
        self.beta = beta

    def forward(self, pred, target):
        return smooth_l1_loss(pred, target, self.beta, self.reduction)


class NLLLoss(_Loss):
    def forward(self, log_probs, target):
        return nll_loss(log_probs, target, reduction=self.reduction)


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = Parameter(_np.ones(num_features, dtype=_DEFAULT_DTYPE))
        self.bias = Parameter(_np.zeros(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("running_mean", _np.zeros(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("running_var", _np.ones(num_features, dtype=_DEFAULT_DTYPE))
        self.register_buffer("num_batches_tracked", 0)

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            centered = x - mean
            var = (centered * centered).mean(dim=0)
            with no_grad():
                self.running_mean = ((1 - self.momentum) * self.running_mean
                                     + self.momentum * mean.data)
                self.running_var = ((1 - self.momentum) * self.running_var
                                    + self.momentum * x.data.var(axis=0, ddof=1))
                self.num_batches_tracked = self.num_batches_tracked + 1
            normed = centered / (var + self.eps) ** 0.5
        else:
            normed = ((x - Tensor(self.running_mean))
                      / Tensor(_np.sqrt(self.running_var + self.eps)))
        return normed * self.weight + self.bias


# ---- 1차원·3차원 계열
#
# **거절 stub 이었다.** 자매(webgpu)에는 실물이 있는데 코어에는 없어서, `import` 하나
# 바꾸면 코드가 깨지는 방향이 열려 있었다 — 이 프로젝트의 약속이 정확히 그 반대다.
#
# 산수는 `conv2d`·`max_pool2d` 가 한다. 새 im2col 을 쓰지 않는다: 같은 계산을 두 벌로
# 두면 한쪽만 고쳐진 채로 갈리는 날이 온다.

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        bound = 1.0 / _math.sqrt(in_channels * kernel_size)
        self.weight = Parameter(_rng.uniform(
            -bound, bound, (out_channels, in_channels, kernel_size)).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(
            _rng.uniform(-bound, bound, (out_channels,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        return conv1d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv1d({self.in_channels}, {self.out_channels}, "
                f"kernel_size=({self.kernel_size},), stride=({self.stride},))")


class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        bound = 1.0 / _math.sqrt(in_channels * k * k * k)
        shape = (out_channels, in_channels) + ((k, k, k) if isinstance(kernel_size, int)
                                               else tuple(kernel_size))
        self.weight = Parameter(_rng.uniform(-bound, bound, shape).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(
            _rng.uniform(-bound, bound, (out_channels,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        return conv3d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv3d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride})")


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, return_indices=False):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.return_indices = return_indices

    def forward(self, x):
        return max_pool1d(x, self.kernel_size, self.stride,
                          return_indices=self.return_indices)

    def __repr__(self):
        return f"MaxPool1d(kernel_size={self.kernel_size}, stride={self.stride})"


class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, return_indices=False):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.return_indices = return_indices

    def forward(self, x):
        return max_pool3d(x, self.kernel_size, self.stride,
                          return_indices=self.return_indices)

    def __repr__(self):
        return f"MaxPool3d(kernel_size={self.kernel_size}, stride={self.stride})"


class _MaxUnpoolND(Module):
    """`MaxPool` 이 고른 자리로 값을 되돌린다. 차원마다 함수만 다르다.

    **`forward` 가 인자를 둘 받는다.** 다른 층과 다른 모양이라 `Sequential` 에 그냥
    못 넣는데, torch 도 같다 — 자리표는 값과 함께 흘러야 하고 그것을 층 안에 숨기면
    같은 층을 두 번 쓸 때 남의 자리표를 쓰게 된다.
    """

    fn = None
    dim = 0

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        # **축마다 펴서 든다.** torch 가 그렇게 들고, `repr` 에 그 튜플이 그대로
        # 나온다 — 수 하나로 들고 있으면 `kernel_size=2` 로 찍혀 갈린다.
        n = type(self).dim
        self.kernel_size = tuple(_spread(kernel_size, n))
        self.stride = tuple(_spread(kernel_size if stride is None else stride, n))
        self.padding = tuple(_spread(padding, n))

    def forward(self, x, indices, output_size=None):
        return type(self).fn(x, indices, self.kernel_size, self.stride,
                             self.padding, output_size)

    def __repr__(self):
        return (f"MaxUnpool{type(self).dim}d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")


class MaxUnpool1d(_MaxUnpoolND):
    fn = staticmethod(max_unpool1d)
    dim = 1


class MaxUnpool2d(_MaxUnpoolND):
    fn = staticmethod(max_unpool2d)
    dim = 2


class MaxUnpool3d(_MaxUnpoolND):
    fn = staticmethod(max_unpool3d)
    dim = 3


class BatchNorm3d(BatchNorm2d):
    """`BatchNorm2d` 와 **같은 코드다** — 위에서 랭크를 안 따지게 고쳤으므로
    (N,C,D,H,W) 도 그대로 통한다. 자매도 같은 구조다."""


# ---------------------------------------------------------------- 나머지 층

class Unfold(Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.kernel_size, self.dilation = kernel_size, dilation
        self.padding, self.stride = padding, stride

    def forward(self, x):
        return unfold_im2col(x, self.kernel_size, self.dilation, self.padding,
                             self.stride)

    def __repr__(self):
        return (f"Unfold(kernel_size={self.kernel_size}, "
                f"dilation={self.dilation}, padding={self.padding}, "
                f"stride={self.stride})")


class Fold(Module):
    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.output_size, self.kernel_size = output_size, kernel_size
        self.dilation, self.padding, self.stride = dilation, padding, stride

    def forward(self, x):
        return fold(x, self.output_size, self.kernel_size, self.dilation,
                    self.padding, self.stride)

    def __repr__(self):
        return (f"Fold(output_size={self.output_size}, "
                f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
                f"padding={self.padding}, stride={self.stride})")


class Bilinear(Module):
    """두 입력을 **한꺼번에** 섞는다. 가중치가 세 축이라 `(out, in1, in2)` 다."""

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.in1_features, self.in2_features = in1_features, in2_features
        self.out_features = out_features
        bound = 1.0 / _math.sqrt(in1_features)
        self.weight = Parameter(_rng.uniform(
            -bound, bound, (out_features, in1_features, in2_features)
        ).astype(_DEFAULT_DTYPE))
        if bias:
            self.bias = Parameter(
                _rng.uniform(-bound, bound, out_features).astype(_DEFAULT_DTYPE))

    def forward(self, x1, x2):
        return bilinear(x1, x2, self.weight, getattr(self, "bias", None))

    def __repr__(self):
        return (f"Bilinear(in1_features={self.in1_features}, "
                f"in2_features={self.in2_features}, "
                f"out_features={self.out_features}, "
                f"bias={getattr(self, 'bias', None) is not None})")


class LocalResponseNorm(Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1.0):
        super().__init__()
        self.size, self.alpha, self.beta, self.k = size, alpha, beta, k

    def forward(self, x):
        return local_response_norm(x, self.size, self.alpha, self.beta, self.k)

    def __repr__(self):
        return (f"LocalResponseNorm({self.size}, alpha={self.alpha}, "
                f"beta={self.beta}, k={self.k})")


class Softmax2d(Module):
    """`(N, C, H, W)` 의 **채널 방향** softmax. `softmax(dim=1)` 과 같다."""

    def forward(self, x):
        return softmax(x, dim=1)

    def __repr__(self):
        return "Softmax2d()"


class RReLU(Module):
    def __init__(self, lower=1.0 / 8, upper=1.0 / 3, inplace=False):
        super().__init__()
        self.lower, self.upper, self.inplace = lower, upper, inplace

    def forward(self, x):
        return rrelu(x, self.lower, self.upper, self.training)

    def __repr__(self):
        return f"RReLU(lower={self.lower}, upper={self.upper})"


class _Upsampling(Module):
    """옛 이름 둘. **`UpsamplingBilinear2d` 는 `align_corners=True` 다** —
    `Upsample(mode='bilinear')` 의 기본값과 다르다."""

    _mode = "nearest"
    _corners = None

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return interpolate(x, size=self.size, scale_factor=self.scale_factor,
                           mode=self._mode, align_corners=self._corners)

    def __repr__(self):
        return (f"{type(self).__name__}(scale_factor="
                f"{float(self.scale_factor)}, mode={self._mode!r})")


class UpsamplingNearest2d(_Upsampling):
    _mode = "nearest"


class UpsamplingBilinear2d(_Upsampling):
    _mode = "bilinear"
    _corners = True


class EmbeddingBag(Module):
    """가방마다 한 줄. 표에서 골라 **합치는 것**까지가 한 층이다.

    `offsets` 를 주면 1 차원 번호 줄을 가방으로 자른다 — 가방 길이가 제각각인
    자리가 그 모양이다.
    """

    def __init__(self, num_embeddings, embedding_dim, mode="mean"):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.mode = mode
        self.weight = Parameter(
            _rng.standard_normal((num_embeddings, embedding_dim)).astype(_DEFAULT_DTYPE))

    def forward(self, idx, offsets=None, per_sample_weights=None):
        # 계산은 `F.embedding_bag` 이 한다 — 층과 함수를 두 벌로 두지 않는다.
        return embedding_bag(idx, self.weight, offsets, self.mode,
                             per_sample_weights)

    def __repr__(self):
        return (f"EmbeddingBag({self.num_embeddings}, {self.embedding_dim}, "
                f"mode={self.mode!r})")


for _cls in (Unfold, Fold, Bilinear, LocalResponseNorm, Softmax2d, RReLU,
             UpsamplingNearest2d, UpsamplingBilinear2d, EmbeddingBag):
    setattr(nn, _cls.__name__, _cls)


# ------------------------------------------------- 자리 옮기기·채널째 dropout
#
# 여덟 층이 전부 함수 하나를 부른다. 갈리는 것은 넘길 인자와 찍는 글자뿐이다.

class _Rearrange(Module):
    _fn = None
    _arg = "factor"

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return type(self)._fn(x, self.value)

    def __repr__(self):
        return f"{type(self).__name__}({self._arg}={self.value})"


class PixelShuffle(_Rearrange):
    _fn = staticmethod(pixel_shuffle)
    _arg = "upscale_factor"


class PixelUnshuffle(_Rearrange):
    _fn = staticmethod(pixel_unshuffle)
    _arg = "downscale_factor"


class ChannelShuffle(_Rearrange):
    _fn = staticmethod(channel_shuffle)
    _arg = "groups"


class _FeatureDropout(Module):
    """채널째 떨구는 것들. **`inplace` 까지 찍는다** — torch 가 그렇다."""

    _fn = None

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p, self.inplace = p, inplace

    def forward(self, x):
        return type(self)._fn(x, self.p, self.training)

    def __repr__(self):
        return f"{type(self).__name__}(p={self.p}, inplace={self.inplace})"


def _make_feature_dropouts():
    table = (("Dropout1d", dropout1d), ("Dropout2d", dropout2d),
             ("Dropout3d", dropout3d), ("AlphaDropout", alpha_dropout),
             ("FeatureAlphaDropout", feature_alpha_dropout))
    return {name: type(name, (_FeatureDropout,), {"_fn": staticmethod(fn)})
            for name, fn in table}


for _name, _cls in {"PixelShuffle": PixelShuffle,
                    "PixelUnshuffle": PixelUnshuffle,
                    "ChannelShuffle": ChannelShuffle,
                    **_make_feature_dropouts()}.items():
    globals()[_name] = _cls
    setattr(nn, _name, _cls)


# ---------------------------------------------------------------- 게으른 층
#
# **모양을 첫 forward 에서 알아낸다.** `nn.LazyLinear(3)` 은 `in_features` 를 안 받고
# 처음 지나가는 값을 보고 정한다 — 합성곱 뒤에 몇 채널이 나오는지를 손으로 세는 일이
# 사라지므로 실제 코드가 자주 쓴다.
#
# **굳으면 클래스가 바뀐다.** 이것이 규약의 핵심이고 짐작으로는 안 나온다. 첫 forward
# 뒤에 그 물건은 더 이상 `LazyLinear` 가 아니라 `Linear` 다 — 이름도 `isinstance` 도
# 바뀌고 `has_uninitialized_params` 라는 메서드 자체가 사라진다(진짜 torch 에 물어
# 확인했다). 깃발 하나로 처리하면 이름이 안 바뀌고, 그러면 `repr` 이 갈린다.

class UninitializedParameter(Parameter):
    """아직 모양을 모르는 파라미터.

    **있기는 있다.** `parameters()` 가 내놓고 `state_dict` 에 열쇠가 있다 — 그래서
    굳기 전에 옵티마이저에 넣는 것이 된다(torch 가 허용하고 그 순서로 짜는 코드가
    있다). 다만 모양을 묻거나 셈을 하면 던진다.
    """

    def __init__(self):
        super().__init__(_np.zeros((0,), dtype=_DEFAULT_DTYPE))

    @property
    def shape(self):
        raise RuntimeError(
            "Can't access the shape of an uninitialized parameter or buffer. "
            "This error usually happens in `load_state_dict` when the parameter "
            "has not been materialized — 먼저 한 번 지나가게 하세요.")

    def __repr__(self):
        return "<UninitializedParameter>"

    def _refuse(self, *_a, **_k):
        raise ValueError(
            "Attempted to use an uninitialized parameter — 먼저 한 번 지나가게 하세요.")

    __add__ = __radd__ = __mul__ = __rmul__ = _refuse
    __sub__ = __rsub__ = __truediv__ = __rtruediv__ = _refuse
    __matmul__ = __rmatmul__ = __pow__ = _refuse


class UninitializedBuffer(UninitializedParameter):
    def __repr__(self):
        return "<UninitializedBuffer>"


def Buffer(data):
    """`nn.Buffer(t)` — 학습 안 하지만 저장되는 값이라는 표시.

    torch 에서 이것은 텐서 자신이다(`isinstance(nn.Buffer(t), Tensor)` 가 참).
    표시를 실제로 읽는 것은 `register_buffer` 쪽이라, 여기서는 그대로 돌려준다.
    """
    return data


class _Lazy(Module):
    """게으른 층의 뿌리. `_becomes` 가 굳은 뒤 될 클래스이고 `_infer` 가 모양을 읽는다.

    굳는 자리를 `forward` 에 둔다 — 그래야 `Sequential` 안에서든 손으로 부르든 같은
    자리를 지난다. 굳은 뒤에는 **자기 자신을 진짜 층으로 바꿔치고** 다시 부른다.
    """

    _becomes = None
    _names = ()

    def __init__(self, *args, **kw):
        super().__init__()
        self._lazy_args, self._lazy_kw = list(args), dict(kw)
        self.weight = UninitializedParameter()
        if kw.get("bias", True):
            self.bias = UninitializedParameter()

    def has_uninitialized_params(self):
        return True

    def _infer(self, x):
        """모양에서 읽어낸 첫 인자. 층마다 다르므로 여기서 갈린다."""
        raise NotImplementedError

    def forward(self, x):
        cls = type(self)._becomes
        real = cls(*self._infer(x), *self._lazy_args, **self._lazy_kw)
        # 속을 통째로 갈아 끼운다. 새 물건을 돌려주면 이미 옵티마이저에 들어간
        # 파라미터와 딴 것이 되어, 학습이 도는데 가중치가 안 움직인다.
        self.__dict__.clear()
        self.__dict__.update(real.__dict__)
        self.__class__ = cls
        return self(x)

    def __repr__(self):
        inner = ", ".join(f"{n}={v}" for n, v in
                          zip(self._names, [0] + self._lazy_args))
        tail = ", bias=True" if self._lazy_kw.get("bias", True) else ", bias=False"
        return f"{type(self).__name__}({inner}{tail})"


class LazyLinear(_Lazy):
    _becomes = Linear
    _names = ("in_features", "out_features")

    def _infer(self, x):
        return (x.shape[-1],)


def _lazy_channels(x):
    """(N, C, …) 에서 채널 수. 합성곱과 정규화가 같이 쓴다."""
    return (x.shape[1],)


def _make_lazies():
    """열두 개를 여기서 찍는다. 무엇이 될지와 무엇을 읽을지만 다르다."""
    table = (
        ("LazyConv1d", Conv1d, ("in_channels", "out_channels")),
        ("LazyConv2d", Conv2d, ("in_channels", "out_channels")),
        ("LazyConv3d", Conv3d, ("in_channels", "out_channels")),
        ("LazyConvTranspose1d", ConvTranspose1d, ("in_channels", "out_channels")),
        ("LazyConvTranspose2d", ConvTranspose2d, ("in_channels", "out_channels")),
        ("LazyConvTranspose3d", ConvTranspose3d, ("in_channels", "out_channels")),
        ("LazyBatchNorm1d", BatchNorm1d, ("num_features",)),
        ("LazyBatchNorm2d", BatchNorm2d, ("num_features",)),
        ("LazyBatchNorm3d", BatchNorm3d, ("num_features",)),
        ("LazyInstanceNorm1d", InstanceNorm1d, ("num_features",)),
        ("LazyInstanceNorm2d", InstanceNorm2d, ("num_features",)),
        ("LazyInstanceNorm3d", InstanceNorm3d, ("num_features",)),
    )
    return {name: type(name, (_Lazy,), {
        "_becomes": becomes, "_names": names,
        "_infer": staticmethod(_lazy_channels),
    }) for name, becomes, names in table}


for _name, _lazy_cls in {"LazyLinear": LazyLinear, **_make_lazies()}.items():
    globals()[_name] = _lazy_cls
    setattr(nn, _name, _lazy_cls)

nn.UninitializedParameter = UninitializedParameter
nn.UninitializedBuffer = UninitializedBuffer
nn.Buffer = Buffer


# ---------------------------------------------------------------- 손실 층
#
# **전부 같은 모양이다** — 만들 때 인자를 받아 두고 부를 때 함수로 넘긴다. torch 의
# 손실 층이 하는 일이 그것뿐이라, 층마다 `forward` 를 적으면 같은 두 줄을 열세 번
# 적는 것이 된다. 인자 이름만 표로 두고 나머지는 여기서 찍는다.

class _Loss(Module):
    """손실 층의 뿌리. `_fn` 이 함수를, `_keys` 가 넘길 인자 이름을 정한다."""

    _fn = None
    _keys = ()

    def __init__(self, *args, reduction="mean", **kw):
        super().__init__()
        self.reduction = reduction
        for key, value in zip(self._keys, args):
            kw.setdefault(key, value)
        self._opts = kw

    def forward(self, *inputs):
        return type(self)._fn(*inputs, reduction=self.reduction, **self._opts)

    def __repr__(self):
        # torch 는 손실 층을 **인자 없이** 찍는다 — `HuberLoss(delta=0.5)` 도
        # `HuberLoss()` 로 나온다(실측). 글자가 답의 일부라 그대로 따른다.
        return f"{type(self).__name__}()"


def _make_losses():
    table = (
        ("HuberLoss", huber_loss, ("delta",)),
        ("KLDivLoss", kl_div, ("log_target",)),
        ("PoissonNLLLoss", poisson_nll_loss, ("log_input", "full", "eps")),
        ("GaussianNLLLoss", gaussian_nll_loss, ("full", "eps")),
        ("MarginRankingLoss", margin_ranking_loss, ("margin",)),
        ("CosineEmbeddingLoss", cosine_embedding_loss, ("margin",)),
        ("HingeEmbeddingLoss", hinge_embedding_loss, ("margin",)),
        ("SoftMarginLoss", soft_margin_loss, ()),
        ("TripletMarginLoss", triplet_margin_loss, ("margin", "p", "eps", "swap")),
        ("TripletMarginWithDistanceLoss", triplet_margin_with_distance_loss,
         ("distance_function", "margin", "swap")),
        ("MultiLabelSoftMarginLoss", multilabel_soft_margin_loss, ("weight",)),
        ("MultiMarginLoss", multi_margin_loss, ("p", "margin", "weight")),
        ("MultiLabelMarginLoss", multilabel_margin_loss, ()),
    )
    return {name: type(name, (_Loss,), {"_fn": staticmethod(fn), "_keys": keys})
            for name, fn, keys in table}


for _name, _loss_cls in _make_losses().items():
    globals()[_name] = _loss_cls
    setattr(nn, _name, _loss_cls)


class PairwiseDistance(Module):
    """짝지어진 두 줄 사이의 거리. **`eps` 는 차에 더한다** — 함수 쪽에 적었다."""

    def __init__(self, p=2.0, eps=1e-6, keepdim=False):
        super().__init__()
        self.p, self.eps, self.keepdim = p, eps, keepdim

    def forward(self, x1, x2):
        return pairwise_distance(x1, x2, p=self.p, eps=self.eps,
                                 keepdim=self.keepdim)

    def __repr__(self):
        return "PairwiseDistance()"


class CosineSimilarity(Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, x1, x2):
        return cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)

    def __repr__(self):
        return "CosineSimilarity()"


nn.PairwiseDistance = PairwiseDistance
nn.CosineSimilarity = CosineSimilarity


# ---------------------------------------------------------------- 패딩 층
#
# **열다섯 개가 한 기계에서 나온다.** 셋(1·2·3 차원) × 다섯(reflect·replicate·zero·
# constant·circular)이고, 갈리는 것은 모드 이름과 짝의 개수뿐이다. 손으로 열다섯 벌을
# 적으면 열다섯 자리가 어긋날 수 있는데, 실제로 갈리는 것은 두 가지뿐이다.
#
# **`ConstantPad` 만 다르게 찍는다** — 나머지는 짝만 찍고 그쪽은 이름을 붙인다
# (`ConstantPad1d(padding=(2, 2), value=7.0)`). 진짜 torch 가 그렇고, 골든이 글자를
# 굳혔으므로 그 차이가 답의 일부다.

class _PadNd(Module):
    _mode = "constant"
    _dims = 1

    def __init__(self, padding, value=0.0):
        super().__init__()
        pairs = 2 * self._dims
        self.padding = ((padding,) * pairs if isinstance(padding, int)
                        else tuple(padding))
        self.value = value

    def forward(self, x):
        return pad(x, self.padding, mode=self._mode, value=self.value)

    def __repr__(self):
        return f"{type(self).__name__}({self.padding})"


class _ConstantPadNd(_PadNd):
    def __repr__(self):
        return (f"{type(self).__name__}(padding={self.padding}, "
                f"value={self.value})")


def _make_pads():
    """열다섯 개를 여기서 찍는다. 이름과 모드와 차수만 다르다."""
    made = {}
    for kind, mode, base in (("Reflection", "reflect", _PadNd),
                             ("Replication", "replicate", _PadNd),
                             ("Circular", "circular", _PadNd),
                             ("Zero", "constant", _PadNd),
                             ("Constant", "constant", _ConstantPadNd)):
        for dims in (1, 2, 3):
            name = f"{kind}Pad{dims}d"
            made[name] = type(name, (base,), {"_mode": mode, "_dims": dims})
    return made


for _name, _pad_cls in _make_pads().items():
    globals()[_name] = _pad_cls
    setattr(nn, _name, _pad_cls)


class Upsample(Module):
    """확대. 한 칸이 s×s 로 복제되므로 **역방향은 그 블록을 합하는 것**이다.

    **첫 자리는 `size` 다.** torch 가 그렇다 — `Upsample(2)` 는 배율 2 가 아니라
    "출력을 2×2 로" 다. 배율을 첫 자리에 두고 있었고, 그러면 `Upsample(2)` 가 같은
    코드에서 늘리는 것과 줄이는 것으로 갈린다. 모양이 그럴듯해서 값으로만 걸린다.

    **`mode='bilinear'` 를 거절하고 있었다.** 계산은 `interpolate` 에 이미 있었고
    `F.upsample_bilinear` 로는 돌았다 — 같은 계산에 이름이 둘인데 한쪽만 되던
    자리다. 교재가 쓰는 꼴은 층 쪽이다.
    """

    def __init__(self, size=None, scale_factor=None, mode="nearest",
                 align_corners=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor
        self.mode, self.align_corners = mode, align_corners

    def forward(self, x):
        if self.size is None and self.scale_factor is None:
            raise RuntimeError(_like_torch(
                "size 나 scale_factor 중 하나는 주어야 합니다.",
                "either size or scale_factor should be defined"))
        return interpolate(x, size=self.size, scale_factor=self.scale_factor,
                           mode=self.mode, align_corners=self.align_corners)

    def __repr__(self):
        return f"Upsample(scale_factor={self.scale_factor}, mode={self.mode!r})"


for _cls in (GELU, SiLU, LeakyReLU, ELU, Softmax, LogSoftmax, AvgPool2d,
             AdaptiveAvgPool2d, Unflatten, L1Loss, SmoothL1Loss, NLLLoss, BatchNorm1d,
             Conv1d, Conv3d, MaxPool1d, MaxPool3d, BatchNorm3d, Upsample):
    setattr(nn, _cls.__name__, _cls)
nn.RNN = RNN
nn.LSTM = LSTM
nn.GRU = GRU
nn.MultiheadAttention = MultiheadAttention
nn.TransformerEncoderLayer = TransformerEncoderLayer
nn.TransformerEncoder = TransformerEncoder
nn.TransformerDecoderLayer = TransformerDecoderLayer
nn.TransformerDecoder = TransformerDecoder
nn.Transformer = Transformer
nn.MaxPool2d = MaxPool2d
nn.Sequential = Sequential
nn.ModuleList = ModuleList
nn.ModuleDict = ModuleDict
nn.ParameterList = ParameterList
nn.ParameterDict = ParameterDict
for _cls in (CELU, GLU, Hardshrink, Hardsigmoid, Hardswish, Hardtanh, LogSigmoid,
             Mish, PReLU, ReLU6, SELU, Softmin, Softplus, Softshrink, Softsign,
             Tanhshrink, Threshold,
             ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, GroupNorm,
             InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, RMSNorm,
             AvgPool1d, AvgPool3d, AdaptiveAvgPool1d, AdaptiveAvgPool3d,
             AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d,
             LPPool1d, LPPool2d, LPPool3d,
             MaxUnpool1d, MaxUnpool2d, MaxUnpool3d,
             FractionalMaxPool2d, FractionalMaxPool3d, CTCLoss,
             AdaptiveLogSoftmaxWithLoss):
    setattr(nn, _cls.__name__, _cls)
nn.MSELoss = MSELoss
nn.BCELoss = BCELoss
nn.BCEWithLogitsLoss = BCEWithLogitsLoss
nn.CrossEntropyLoss = CrossEntropyLoss


def one_hot(t, num_classes=-1):
    idx = t.data.astype(int)
    n = int(idx.max()) + 1 if num_classes == -1 else num_classes
    return Tensor(_np.eye(n, dtype=_np.int64)[idx])


class _Functional(_Namespace):
    # 손실과 거리. **층과 같은 함수를 가리킨다** — 두 벌이면 어긋난다.
    huber_loss = staticmethod(huber_loss)
    kl_div = staticmethod(kl_div)
    poisson_nll_loss = staticmethod(poisson_nll_loss)
    gaussian_nll_loss = staticmethod(gaussian_nll_loss)
    margin_ranking_loss = staticmethod(margin_ranking_loss)
    cosine_embedding_loss = staticmethod(cosine_embedding_loss)
    hinge_embedding_loss = staticmethod(hinge_embedding_loss)
    soft_margin_loss = staticmethod(soft_margin_loss)
    triplet_margin_loss = staticmethod(triplet_margin_loss)
    triplet_margin_with_distance_loss = staticmethod(
        triplet_margin_with_distance_loss)
    multilabel_soft_margin_loss = staticmethod(multilabel_soft_margin_loss)
    multi_margin_loss = staticmethod(multi_margin_loss)
    multilabel_margin_loss = staticmethod(multilabel_margin_loss)
    pairwise_distance = staticmethod(pairwise_distance)
    pdist = staticmethod(pdist)
    # 자리 옮기기와 채널째 dropout.
    pixel_shuffle = staticmethod(pixel_shuffle)
    pixel_unshuffle = staticmethod(pixel_unshuffle)
    channel_shuffle = staticmethod(channel_shuffle)
    # `native_channel_shuffle` 은 안 낸다. ATen 의 밑단 진입점이고 위에 부르는 이름이
    # 따로 있다 — 빈자리 표에 그렇게 적어 두고 여기서 만들면 표가 거짓말을 한다.
    # 실제로 한 번 만들었다가 `test_gap.py` 가 그 모순을 잡았다.
    dropout1d = staticmethod(dropout1d)
    dropout2d = staticmethod(dropout2d)
    dropout3d = staticmethod(dropout3d)
    alpha_dropout = staticmethod(alpha_dropout)
    feature_alpha_dropout = staticmethod(feature_alpha_dropout)
    # 창 펴기와 나머지. **`F.unfold` 는 im2col 이다** — `Tensor.unfold` 와 다르다.
    unfold = staticmethod(unfold_im2col)
    fold = staticmethod(fold)
    bilinear = staticmethod(bilinear)
    local_response_norm = staticmethod(local_response_norm)
    rrelu = staticmethod(rrelu)
    softmax = staticmethod(softmax)
    log_softmax = staticmethod(log_softmax)
    relu = staticmethod(relu)
    leaky_relu = staticmethod(leaky_relu)
    elu = staticmethod(elu)
    silu = staticmethod(silu)
    gelu = staticmethod(gelu)
    sigmoid = staticmethod(sigmoid)
    tanh = staticmethod(tanh)
    celu = staticmethod(celu)
    hardshrink = staticmethod(hardshrink)
    hardsigmoid = staticmethod(hardsigmoid)
    hardswish = staticmethod(hardswish)
    hardtanh = staticmethod(hardtanh)
    logsigmoid = staticmethod(logsigmoid)
    mish = staticmethod(mish)
    relu6 = staticmethod(relu6)
    selu = staticmethod(selu)
    softplus = staticmethod(softplus)
    softshrink = staticmethod(softshrink)
    softsign = staticmethod(softsign)
    tanhshrink = staticmethod(tanhshrink)
    threshold = staticmethod(threshold_fn)
    softmin = staticmethod(softmin)
    glu = staticmethod(glu)
    prelu = staticmethod(prelu)
    group_norm = staticmethod(group_norm)
    instance_norm = staticmethod(instance_norm)
    rms_norm = staticmethod(rms_norm)
    scaled_dot_product_attention = staticmethod(scaled_dot_product_attention)
    avg_pool1d = staticmethod(avg_pool1d)
    avg_pool3d = staticmethod(avg_pool3d)
    adaptive_avg_pool1d = staticmethod(adaptive_avg_pool1d)
    adaptive_avg_pool3d = staticmethod(adaptive_avg_pool3d)
    adaptive_max_pool1d = staticmethod(adaptive_max_pool1d)
    adaptive_max_pool2d = staticmethod(adaptive_max_pool2d)
    adaptive_max_pool3d = staticmethod(adaptive_max_pool3d)
    lp_pool1d = staticmethod(lp_pool1d)
    lp_pool2d = staticmethod(lp_pool2d)
    lp_pool3d = staticmethod(lp_pool3d)
    # 이긴 자리를 함께 내는 판과, 그 자리로 되돌리는 짝.
    adaptive_max_pool1d_with_indices = staticmethod(adaptive_max_pool1d_with_indices)
    adaptive_max_pool2d_with_indices = staticmethod(adaptive_max_pool2d_with_indices)
    adaptive_max_pool3d_with_indices = staticmethod(adaptive_max_pool3d_with_indices)
    max_pool1d_with_indices = staticmethod(max_pool1d_with_indices)
    max_pool2d_with_indices = staticmethod(max_pool2d_with_indices)
    max_pool3d_with_indices = staticmethod(max_pool3d_with_indices)
    max_unpool1d = staticmethod(max_unpool1d)
    max_unpool2d = staticmethod(max_unpool2d)
    max_unpool3d = staticmethod(max_unpool3d)
    ctc_loss = staticmethod(ctc_loss)
    # 제자리 활성. 계산은 밑줄 없는 쪽이 하고 여기서는 제 버퍼에 되쓴다.
    celu_ = staticmethod(celu_)
    elu_ = staticmethod(elu_)
    hardtanh_ = staticmethod(hardtanh_)
    leaky_relu_ = staticmethod(leaky_relu_)
    relu_ = staticmethod(relu_)
    rrelu_ = staticmethod(rrelu_)
    selu_ = staticmethod(selu_)
    threshold_ = staticmethod(threshold_)
    multi_head_attention_forward = staticmethod(multi_head_attention_forward)
    affine_grid = staticmethod(affine_grid)
    grid_sample = staticmethod(grid_sample)
    batch_norm = staticmethod(batch_norm)
    embedding_bag = staticmethod(embedding_bag)
    gumbel_softmax = staticmethod(gumbel_softmax)
    upsample = staticmethod(upsample)
    upsample_bilinear = staticmethod(upsample_bilinear)
    upsample_nearest = staticmethod(upsample_nearest)
    fractional_max_pool2d = staticmethod(fractional_max_pool2d)
    fractional_max_pool3d = staticmethod(fractional_max_pool3d)
    fractional_max_pool2d_with_indices = staticmethod(fractional_max_pool2d_with_indices)
    fractional_max_pool3d_with_indices = staticmethod(fractional_max_pool3d_with_indices)
    conv_transpose1d = staticmethod(conv_transpose1d)
    conv_transpose2d = staticmethod(conv_transpose2d)
    conv_transpose3d = staticmethod(conv_transpose3d)
    one_hot = staticmethod(one_hot)
    dropout = staticmethod(dropout)
    avg_pool2d = staticmethod(avg_pool2d)
    layer_norm = staticmethod(layer_norm)
    embedding = staticmethod(embedding)
    nll_loss = staticmethod(nll_loss)
    l1_loss = staticmethod(l1_loss)
    smooth_l1_loss = staticmethod(smooth_l1_loss)
    pad = staticmethod(pad)
    normalize = staticmethod(normalize)
    cosine_similarity = staticmethod(cosine_similarity)
    # 1차원·3차원 계열. **자매에는 있고 여기에는 없던 것들이다.**
    conv1d = staticmethod(conv1d)
    conv2d = staticmethod(conv2d)
    conv3d = staticmethod(conv3d)
    max_pool1d = staticmethod(max_pool1d)
    max_pool2d = staticmethod(max_pool2d)
    max_pool3d = staticmethod(max_pool3d)
    adaptive_avg_pool2d = staticmethod(adaptive_avg_pool2d)
    interpolate = staticmethod(interpolate)

    @staticmethod
    def mse_loss(pred, target, reduction="mean"):
        return MSELoss(reduction)(pred, target)

    @staticmethod
    def binary_cross_entropy_with_logits(logits, target, reduction="mean"):
        return BCEWithLogitsLoss(reduction)(logits, target)

    @staticmethod
    def binary_cross_entropy(p, target, reduction="mean"):
        return BCELoss(reduction)(p, target)

    @staticmethod
    def linear(x, weight, bias=None):
        out = x @ weight.transpose(-2, -1)
        return out + bias if bias is not None else out

    conv2d = staticmethod(conv2d)
    max_pool2d = staticmethod(max_pool2d)

    @staticmethod
    def cross_entropy(logits, target, reduction="mean"):
        return CrossEntropyLoss(reduction)(logits, target)


nn.functional = _Functional()


