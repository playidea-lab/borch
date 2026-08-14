"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._tensor import (
    Tensor,
)
from ._base import (
    _DEFAULT_DTYPE, _math, _np, _unsupported,
)
from ._ops import (
    _Namespace, _gelu, _pool_all, _rng, adaptive_avg_pool2d, avg_pool2d, conv1d, conv2d,
    conv3d, cosine_similarity, dropout, elu, embedding, gelu, interpolate, l1_loss,
    layer_norm, leaky_relu, log_softmax, max_pool1d, max_pool2d, max_pool3d, nll_loss,
    no_grad, norm, normalize, pad, relu, sigmoid, silu, smooth_l1_loss, softmax, stack,
    tanh, zeros,
)

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
        inner = "\n".join(f"  ({n}): {m}" for n, m in self._modules.items())
        return f"{type(self).__name__}(\n{inner}\n)" if inner else f"{type(self).__name__}()"


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 진짜 torch 와 같은 초기화 (Kaiming uniform 계열): U(-1/√fan_in, 1/√fan_in)
        bound = 1.0 / _math.sqrt(in_features)
        self.weight = Parameter(_rng.uniform(-bound, bound, (out_features, in_features)).astype(_DEFAULT_DTYPE))
        self.bias = Parameter(_rng.uniform(-bound, bound, (out_features,)).astype(_DEFAULT_DTYPE)) if bias else None

    def forward(self, x):
        out = x @ self.weight.transpose(-2, -1)
        return out + self.bias if self.bias is not None else out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"


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
    def __init__(self, mods=()):
        super().__init__()
        self._layers = list(mods)
        for i, m in enumerate(self._layers):
            self._modules[str(i)] = m

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, i):
        return self._layers[i]

    def __len__(self):
        return len(self._layers)


class MSELoss(Module):
    def forward(self, pred, target):
        return ((pred - target) ** 2).mean()


class BCEWithLogitsLoss(Module):
    def forward(self, logits, target):
        # log(1+e^-|x|) + max(x,0) - x*t  — 큰 값에서도 안전한 형태
        x, t = logits, target
        return (relu(x) - x * t + (1 + (-(x.abs())).exp()).log()).mean()


class BCELoss(Module):
    def forward(self, p, t):
        eps = 1e-12
        return -(t * (p + eps).log() + (1 - t) * (1 - p + eps).log()).mean()


class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        n = logits.data.shape[0]
        sm = softmax(logits, dim=-1)
        idx = target.data.astype(int)
        picked = sm[_np.arange(n), idx]
        return -(picked + 1e-12).log().mean()


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
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return max_pool2d(x, self.kernel_size, self.stride)


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
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(_np.ones(shape, dtype=_DEFAULT_DTYPE))
        self.bias = Parameter(_np.zeros(shape, dtype=_DEFAULT_DTYPE))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        normed = centered / (var + self.eps) ** 0.5
        return normed * self.weight + self.bias


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
        # **랭크를 안 따진다.** 채널 축만 남기고 나머지를 전부 줄이므로 (N,C,H,W) 도
        # (N,C,D,H,W) 도 같은 코드로 돈다 — `BatchNorm3d` 가 이것을 그대로 물려받는다.
        rank = x.data.ndim
        shape = (1, -1) + (1,) * (rank - 2)
        reduced = tuple(i for i in range(rank) if i != 1)
        if self.training:
            # 평균·분산을 **그래프 안에서** 계산해야 한다. numpy 로 빼서 상수처럼 쓰면
            # x → mean → y 로 흐르는 길이 끊겨 기울기가 틀리고, weight 에는 아예 안 간다.
            # (BatchNorm 순방향만 대조하고 역방향은 안 봤던 탓에 오래 남아 있었다.)
            mean = x.mean(dim=0)
            for _ in range(rank - 2):
                mean = mean.mean(dim=1)                           # (C,)
            centered = x - mean.reshape(shape)
            var = (centered * centered).mean(dim=0)
            for _ in range(rank - 2):
                var = var.mean(dim=1)

            # 진짜 torch 는 두 곳에서 다른 분산을 쓴다 — 정규화는 편향(ddof=0),
            # running_var 갱신은 비편향(ddof=1). 둘 다 편향으로 두면 값이 2.6% 어긋난다.
            with no_grad():
                unbiased = x.data.var(axis=reduced, ddof=1)
                self.running_mean = ((1 - self.momentum) * self.running_mean
                                     + self.momentum * mean.data)
                self.running_var = ((1 - self.momentum) * self.running_var
                                    + self.momentum * unbiased)
                self.num_batches_tracked = self.num_batches_tracked + 1
            normed = centered / (var.reshape(shape) + self.eps) ** 0.5
        else:
            normed = ((x - Tensor(self.running_mean.reshape(shape)))
                      / Tensor(_np.sqrt(self.running_var + self.eps).reshape(shape)))
        return normed * self.weight.reshape(shape) + self.bias.reshape(shape)



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

    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=True):
        key = query if key is None else key
        value = query if value is None else value
        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        B, T, E = query.data.shape
        S = key.data.shape[1]
        w, b = self.in_proj_weight, self.in_proj_bias

        def project(t, index, length):
            piece = w[index * E:(index + 1) * E]
            out = t @ piece.transpose(0, 1)
            return out + b[index * E:(index + 1) * E] if b is not None else out

        q = _split_heads(project(query, 0, T), B, T, self.num_heads, self.head_dim)
        k = _split_heads(project(key, 1, S), B, S, self.num_heads, self.head_dim)
        v = _split_heads(project(value, 2, S), B, S, self.num_heads, self.head_dim)

        scores = (q @ k.transpose(-2, -1)) / _math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = _apply_mask(scores, attn_mask)
        weights = softmax(scores, dim=-1)

        merged = (weights @ v).transpose(1, 2).reshape(B, T, E)
        out = self.out_proj(merged)
        if not self.batch_first:
            out = out.transpose(0, 1)
        if not need_weights:
            return out, None
        return out, weights.mean(dim=1)          # torch 는 헤드 평균을 돌려준다

    def __repr__(self):
        return f"MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"


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


class GELU(_Activation):
    fn = staticmethod(gelu)


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


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return softmax(x, dim=self.dim)


class LogSoftmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return log_softmax(x, dim=self.dim)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return avg_pool2d(x, self.kernel_size, self.stride)


class AdaptiveAvgPool2d(Module):
    """출력 크기 1 만 지원한다 — 실무에서 쓰이는 것은 대개 그것이고, 나머지는 거절한다."""

    def __init__(self, output_size):
        super().__init__()
        if output_size not in (1, (1, 1)):
            _unsupported("AdaptiveAvgPool2d(출력 크기가 1 이 아닌 것)")
        self.output_size = output_size

    def forward(self, x):
        return _pool_all(x)


class Unflatten(Module):
    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim, self.unflattened_size = dim, tuple(unflattened_size)

    def forward(self, x):
        return x.reshape(x.data.shape[:self.dim] + self.unflattened_size)


class L1Loss(Module):
    def forward(self, pred, target):
        return l1_loss(pred, target)


class SmoothL1Loss(Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        return smooth_l1_loss(pred, target, self.beta)


class NLLLoss(Module):
    def forward(self, log_probs, target):
        return nll_loss(log_probs, target)


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
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return max_pool1d(x, self.kernel_size, self.stride)

    def __repr__(self):
        return f"MaxPool1d(kernel_size={self.kernel_size}, stride={self.stride})"


class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return max_pool3d(x, self.kernel_size, self.stride)

    def __repr__(self):
        return f"MaxPool3d(kernel_size={self.kernel_size}, stride={self.stride})"


class BatchNorm3d(BatchNorm2d):
    """`BatchNorm2d` 와 **같은 코드다** — 위에서 랭크를 안 따지게 고쳤으므로
    (N,C,D,H,W) 도 그대로 통한다. 자매도 같은 구조다."""


class Upsample(Module):
    """최근접 확대. 한 칸이 s×s 로 복제되므로 **역방향은 그 블록을 합하는 것**이다."""

    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        if mode != "nearest":
            _unsupported(f"Upsample(mode={mode!r})")
        self.scale_factor, self.mode = scale_factor, mode

    def forward(self, x):
        return interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

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
nn.MSELoss = MSELoss
nn.BCELoss = BCELoss
nn.BCEWithLogitsLoss = BCEWithLogitsLoss
nn.CrossEntropyLoss = CrossEntropyLoss


def one_hot(t, num_classes=-1):
    idx = t.data.astype(int)
    n = int(idx.max()) + 1 if num_classes == -1 else num_classes
    return Tensor(_np.eye(n, dtype=_np.int64)[idx])


class _Functional(_Namespace):
    softmax = staticmethod(softmax)
    log_softmax = staticmethod(log_softmax)
    relu = staticmethod(relu)
    leaky_relu = staticmethod(leaky_relu)
    elu = staticmethod(elu)
    silu = staticmethod(silu)
    gelu = staticmethod(gelu)
    sigmoid = staticmethod(sigmoid)
    tanh = staticmethod(tanh)
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
    def mse_loss(pred, target):
        return MSELoss()(pred, target)

    @staticmethod
    def binary_cross_entropy_with_logits(logits, target):
        return BCEWithLogitsLoss()(logits, target)

    @staticmethod
    def binary_cross_entropy(p, target):
        return BCELoss()(p, target)

    @staticmethod
    def linear(x, weight, bias=None):
        out = x @ weight.transpose(-2, -1)
        return out + bias if bias is not None else out

    conv2d = staticmethod(conv2d)
    max_pool2d = staticmethod(max_pool2d)

    @staticmethod
    def cross_entropy(logits, target):
        return CrossEntropyLoss()(logits, target)


nn.functional = _Functional()


