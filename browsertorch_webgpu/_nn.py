"""browsertorch_webgpu 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import numpy as _np

try:
    import js as _js
    from pyodide.ffi import create_proxy as _create_proxy
    from pyodide.ffi import to_js as _to_js
except ImportError as _exc:                                          # pragma: no cover
    raise ImportError(
        "browsertorch_webgpu 는 브라우저(Pyodide) 안에서만 돕니다. "
        "네이티브에서는 `browsertorch` 를 쓰세요 — 이쪽을 CPU 로 흉내 내면 "
        "GPU 로 돌렸다고 착각하게 됩니다."
    ) from _exc

_tf = getattr(_js, "tf", None)
if _tf is None:                                                      # pragma: no cover
    raise ImportError("TF.js 가 페이지에 없습니다. tf.min.js 를 먼저 실으세요.")

from ._tensor import (
    Tensor, _canonical, _wrap,
)
from ._functional import (
    _Functional, adaptive_avg_pool2d, avg_pool2d, batch_norm, binary_cross_entropy,
    binary_cross_entropy_with_logits, conv1d, conv2d, conv3d, cross_entropy, dropout,
    elu, embedding, gelu, interpolate, l1_loss, layer_norm, leaky_relu, log_softmax,
    max_pool1d, max_pool2d, max_pool3d, mse_loss, nll_loss, pad, silu, smooth_l1_loss,
    softmax,
)
from ._base import (
    _keep, _like_torch, _shape_of, _to_tf, _unsupported, bool_, float32,
)
from ._ops import (
    _rng, norm, relu, sigmoid, stack, tanh, tensor, zeros,
)

# ---------------------------------------------------------------- nn.Module
#
# 구조는 코어와 같게 둔다 — 이름 규약(`0.weight` …)이 같아야 체크포인트가 오가고,
# 같은 학습 코드가 임포트만 바꿔서 돈다.

class Parameter(Tensor):
    """학습 대상. 처음부터 requires_grad 다."""

    def __init__(self, data):
        handle = data._h if isinstance(data, Tensor) else _to_tf(_np.asarray(data))
        super().__init__(handle, requires_grad=True)


class Module:
    def __init__(self):
        self._modules = {}
        self._params = {}
        self._buffers = {}          # 학습은 안 하지만 저장·복원되는 값 (running_mean 등)
        self.training = True

    def register_buffer(self, name, value):
        """`state_dict` 에는 들어가고 학습 대상은 아닌 값.

        빠뜨리면 저장했다 불러왔을 때 **평가 모드가 초기값으로 돌아간다** — 학습은
        멀쩡해 보이고 추론만 틀린다. 코어가 ROADMAP 8번에서 겪은 그대로다.
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

    def state_dict(self):
        out = {n: Tensor(_tf.clone(p._h)) for n, p in self.named_parameters()}
        for name, buf in self.named_buffers():
            out[name] = Tensor(_tf.clone(buf._h)) if isinstance(buf, Tensor) else buf
        return out

    def load_state_dict(self, state, strict=True):
        own = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        missing = [k for k in list(own) + list(buffers) if k not in state]
        unexpected = [k for k in state if k not in own and k not in buffers]
        if strict and (missing or unexpected):
            raise RuntimeError(f"state_dict 가 안 맞습니다. 빠진 것: {missing}, 남는 것: {unexpected}")
        for name, value in state.items():
            if name in buffers:
                holder = self
                *path, leaf = name.split(".")
                for part in path:
                    holder = holder._modules[part]
                holder.register_buffer(
                    leaf, tensor(value) if isinstance(value, Tensor) else value)
                continue
            if name not in own:
                continue
            target = own[name]
            incoming = value._h if isinstance(value, Tensor) else _to_tf(_np.asarray(value))
            if _shape_of(incoming) != target.shape:
                raise RuntimeError(
                    f"{name} 의 모양이 다릅니다: {_shape_of(incoming)} vs {target.shape}")
            target._h.dispose()
            target._h = _tf.clone(incoming)
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, *a, **k):
        raise NotImplementedError("forward 를 구현하세요.")

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        # 코어와 같은 초기화 — U(-1/√fan_in, 1/√fan_in)
        # 모듈 수준 `_rng` 를 쓴다. 매번 `default_rng(0)` 을 새로 만들면 **층마다 같은
        # 가중치가 나온다** — 값을 명시적으로 넣는 골든에서는 안 드러나는 종류다.
        bound = 1.0 / _np.sqrt(in_features)
        self.weight = Parameter(
            _rng.uniform(-bound, bound, (out_features, in_features)).astype(_np.float32))
        self.bias = Parameter(
            _rng.uniform(-bound, bound, out_features).astype(_np.float32)) if bias else None

    def forward(self, x):
        out = x @ self.weight.transpose(0, 1)
        return out + self.bias if self.bias is not None else out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"


class _Activation(Module):
    fn = staticmethod(relu)

    def forward(self, x):
        return type(self).fn(x)


class ReLU(_Activation):
    pass


class Sigmoid(_Activation):
    fn = staticmethod(sigmoid)


class Tanh(_Activation):
    fn = staticmethod(tanh)


class GELU(_Activation):
    fn = staticmethod(gelu)


class Identity(Module):
    def forward(self, x):
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return dropout(x, self.p, self.training)


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


class SiLU(_Activation):
    fn = staticmethod(silu)


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


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        shape = ((normalized_shape,) if isinstance(normalized_shape, int)
                 else tuple(normalized_shape))
        self.eps = eps
        self.weight = Parameter(_np.ones(shape, dtype=_np.float32))
        self.bias = Parameter(_np.zeros(shape, dtype=_np.float32))

    def forward(self, x):
        return layer_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = Parameter(_np.ones(num_features, dtype=_np.float32))
        self.bias = Parameter(_np.zeros(num_features, dtype=_np.float32))
        self.register_buffer("running_mean", Tensor(_keep(_tf.zeros(_to_js([num_features])))))
        self.register_buffer("running_var", Tensor(_keep(_tf.ones(_to_js([num_features])))))
        self.register_buffer("num_batches_tracked", 0)

    def forward(self, x):
        x = _canonical(x)
        if self.training:
            mean = x.mean(dim=0)
            centered = x - mean
            var = (centered * centered).mean(dim=0)
            n = x.shape[0]
            keep = 1.0 - self.momentum
            self.running_mean = Tensor(_keep(_tf.add(
                _tf.mul(keep, self.running_mean._h), _tf.mul(self.momentum, mean._h))))
            self.running_var = Tensor(_keep(_tf.add(
                _tf.mul(keep, self.running_var._h),
                _tf.mul(self.momentum * n / (n - 1), var._h))))
            self.num_batches_tracked = self.num_batches_tracked + 1
            normed = centered / (var + self.eps) ** 0.5
        else:
            inv = Tensor(_tf.rsqrt(_tf.add(self.running_var._h, float(self.eps))))
            normed = (x - self.running_mean) * inv
        return normed * self.weight + self.bias


class Embedding(Module):
    """번호를 벡터로 바꾸는 학습 가능한 표."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Parameter(
            _rng.standard_normal((num_embeddings, embedding_dim)).astype(_np.float32))

    def forward(self, idx):
        return embedding(idx, self.weight)

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


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


class Unflatten(Module):
    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim, self.unflattened_size = dim, tuple(unflattened_size)

    def forward(self, x):
        return x.reshape(tuple(x.shape[:self.dim]) + self.unflattened_size)


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


class BCELoss(Module):
    def forward(self, p, target):
        return binary_cross_entropy(p, target)


class BCEWithLogitsLoss(Module):
    def forward(self, logits, target):
        return binary_cross_entropy_with_logits(logits, target)


class _RNNBase(Module):
    """RNN·LSTM·GRU 의 공통 부분 — 파라미터 만들기와 층·시간 루프.

    파라미터 이름을 torch 와 같게 둔다(`weight_ih_l0` …). 이름이 맞아야 `state_dict`
    키가 맞고 체크포인트가 양쪽을 오간다.

    시간 방향은 파이썬 반복문이다. 순환은 앞을 끝내야 뒤를 볼 수 있어서 병렬화가 안 되고,
    그 느림이 곧 트랜스포머가 나온 이유다. 다만 **입력 쪽 곱은 h 에 안 걸리므로**
    시간 전체를 한 번에 계산해 둔다 — 반복문 안에는 은닉 쪽 곱만 남는다.
    """

    gates = 1

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.num_layers, self.batch_first, self.has_bias = num_layers, batch_first, bias

        bound = 1.0 / _np.sqrt(hidden_size)
        g = self.gates
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            setattr(self, f"weight_ih_l{layer}", Parameter(
                _rng.uniform(-bound, bound, (g * hidden_size, in_size)).astype(_np.float32)))
            setattr(self, f"weight_hh_l{layer}", Parameter(
                _rng.uniform(-bound, bound, (g * hidden_size, hidden_size)).astype(_np.float32)))
            if bias:
                setattr(self, f"bias_ih_l{layer}", Parameter(
                    _rng.uniform(-bound, bound, g * hidden_size).astype(_np.float32)))
                setattr(self, f"bias_hh_l{layer}", Parameter(
                    _rng.uniform(-bound, bound, g * hidden_size).astype(_np.float32)))

    def _weights(self, layer):
        return (getattr(self, f"weight_ih_l{layer}"), getattr(self, f"weight_hh_l{layer}"),
                getattr(self, f"bias_ih_l{layer}", None),
                getattr(self, f"bias_hh_l{layer}", None))

    def _run(self, x, init):
        if self.batch_first:
            x = x.transpose(0, 1)                       # (N,T,I) → (T,N,I)
        steps_n = x.shape[0]

        layer_input, finals = x, []
        for layer in range(self.num_layers):
            w_ih, w_hh, b_ih, b_hh = self._weights(layer)
            pre = layer_input @ w_ih.transpose(0, 1)     # (T, N, gates*H) — h 와 무관
            if self.has_bias:
                pre = pre + b_ih
            state = init(layer)
            outs = []
            for t in range(steps_n):
                state, out = self._step(pre[t], state, w_hh, b_hh)
                outs.append(out)
            layer_input = stack(outs)
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
    """h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh)."""

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
        batch = x.shape[0 if self.batch_first else 1]
        if hx is None:
            hx = zeros(self.num_layers, batch, self.hidden_size)
        out, finals = self._run(x, lambda layer: hx[layer])
        return out, stack(finals)


class LSTM(_RNNBase):
    """게이트 넷으로 무엇을 잊고 무엇을 남길지 배운다.

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
        batch = x.shape[0 if self.batch_first else 1]
        if hx is None:
            hx = (zeros(self.num_layers, batch, self.hidden_size),
                  zeros(self.num_layers, batch, self.hidden_size))
        h0, c0 = hx
        out, finals = self._run(x, lambda layer: (h0[layer], c0[layer]))
        return out, (stack([h for h, _ in finals]), stack([c for _, c in finals]))


class GRU(_RNNBase):
    """게이트 셋. **`n` 게이트에서 `r` 은 편향까지 포함한 은닉 항에 곱한다** —
    편향을 밖에 두면 미세하게 어긋나고 눈에 안 띈다."""

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
        batch = x.shape[0 if self.batch_first else 1]
        if hx is None:
            hx = zeros(self.num_layers, batch, self.hidden_size)
        out, finals = self._run(x, lambda layer: hx[layer])
        return out, stack(finals)


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


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        bound = 1.0 / _np.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = Parameter(_rng.uniform(
            -bound, bound,
            (out_channels, in_channels, kernel_size, kernel_size)).astype(_np.float32))
        self.bias = Parameter(
            _rng.uniform(-bound, bound, out_channels).astype(_np.float32)) if bias else None

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        bound = 1.0 / _np.sqrt(in_channels * kernel_size)
        self.weight = Parameter(_rng.uniform(
            -bound, bound, (out_channels, in_channels, kernel_size)).astype(_np.float32))
        self.bias = Parameter(
            _rng.uniform(-bound, bound, out_channels).astype(_np.float32)) if bias else None

    def forward(self, x):
        return conv1d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv1d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return max_pool1d(x, self.kernel_size, self.stride)


class Upsample(Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor, self.mode = scale_factor, mode

    def forward(self, x):
        return interpolate(x, self.scale_factor, self.mode)


class Conv3d(Module):
    """역방향이 `tf.grad` 를 타서 2차원만큼 빠르지 않다 — 처음 부를 때 경고한다."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        k = kernel_size
        bound = 1.0 / _np.sqrt(in_channels * k * k * k)
        self.weight = Parameter(_rng.uniform(
            -bound, bound, (out_channels, in_channels, k, k, k)).astype(_np.float32))
        self.bias = Parameter(
            _rng.uniform(-bound, bound, out_channels).astype(_np.float32)) if bias else None

    def forward(self, x):
        return conv3d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv3d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")


class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return max_pool3d(x, self.kernel_size, self.stride)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return max_pool2d(x, self.kernel_size, self.stride)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        return avg_pool2d(x, self.kernel_size, self.stride)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if output_size not in (1, (1, 1)):
            _unsupported("AdaptiveAvgPool2d(출력 크기가 1 이 아닌 것)")
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avg_pool2d(x, self.output_size)


class Flatten(Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        return x.flatten(self.start_dim)


class BatchNorm2d(Module):
    """학습 중에는 이번 배치로, 평가 때는 모아둔 값으로.

    평균·분산을 **그래프 안에서** 계산해야 한다. 손잡이로 빼서 상수처럼 쓰면
    x → mean → y 로 흐르는 길이 끊겨 기울기가 틀리고 weight 에는 아예 안 간다 —
    코어가 그 상태로 오래 있었고, 순방향만 대조하고 있어서 안 드러났다.

    그리고 torch 는 두 곳에서 **다른 분산**을 쓴다. 정규화는 편향(ddof=0),
    running_var 갱신은 비편향(ddof=1). 둘 다 편향으로 두면 2.6% 어긋난다.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = Parameter(_np.ones(num_features, dtype=_np.float32))
        self.bias = Parameter(_np.zeros(num_features, dtype=_np.float32))
        # running 통계는 **GPU 에 둔다.** 스텝마다 읽어오면 층마다 동기화가 한 번씩
        # 걸리고, ResNet-18 은 그런 층이 20개다. 그리고 **버퍼로 등록한다** —
        # state_dict 에서 빠지면 저장·복원 뒤 추론만 조용히 틀린다.
        self.register_buffer("running_mean", Tensor(_keep(_tf.zeros(_to_js([num_features])))))
        self.register_buffer("running_var", Tensor(_keep(_tf.ones(_to_js([num_features])))))
        self.register_buffer("num_batches_tracked", 0)

    def forward(self, x):
        x = _wrap(x)          # `batch_norm` 이 레이아웃을 보고 축을 고른다 — 되돌리면 안 된다
        raw = _shape_of(x._h)
        rank = len(raw)
        caxis = rank - 1 if x._nhwc else 1
        bshape = [1] * rank
        bshape[caxis] = self.num_features

        if self.training:
            out, mu, var = batch_norm(x, self.weight, self.bias, self.eps)
            n = int(_np.prod([raw[i] for i in range(rank) if i != caxis]))
            flat = _to_js([self.num_features])
            keep = 1.0 - self.momentum
            self.running_mean = Tensor(_keep(_tf.add(
                _tf.mul(keep, self.running_mean._h),
                _tf.mul(self.momentum, _tf.reshape(mu, flat)))))
            # torch 는 running_var 에만 **비편향** 분산을 쓴다. 둘 다 편향으로 두면 2.6% 어긋난다.
            self.running_var = Tensor(_keep(_tf.add(
                _tf.mul(keep, self.running_var._h),
                _tf.mul(self.momentum * n / (n - 1), _tf.reshape(var, flat)))))
            self.num_batches_tracked = self.num_batches_tracked + 1
            return out

        mean_t = Tensor(_tf.reshape(self.running_mean._h, _to_js(bshape)))
        inv_t = Tensor(_tf.reshape(
            _tf.rsqrt(_tf.add(self.running_var._h, float(self.eps))), _to_js(bshape)))
        mean_t._nhwc = inv_t._nhwc = x._nhwc          # 이미 속 순서로 만들었다
        w = Tensor(_tf.reshape(self.weight._h, _to_js(bshape)))
        b = Tensor(_tf.reshape(self.bias._h, _to_js(bshape)))
        w._nhwc = b._nhwc = x._nhwc
        return (x - mean_t) * inv_t * w + b


def _apply_mask(scores, mask):
    """torch 의 마스크는 두 가지다.

      불리언 — True 인 자리를 가린다(-inf 로 채운다)
      실수   — 점수에 **더한다.** `generate_square_subsequent_mask` 가 주는 0/-inf 가 그것이다

    실수 마스크를 "0 이 아니면 가림" 으로 뭉뚱그리면 인과 마스크는 우연히 맞지만
    가중치를 조절하는 마스크에서 어긋난다.
    """
    m = _wrap(mask)
    if m._dtype is bool_:
        return scores.masked_fill(m, float("-inf"))
    return scores + m


def _split_heads(t, batch, length, heads, head_dim):
    return t.reshape(batch, length, heads, head_dim).transpose(1, 2)


class MultiheadAttention(Module):
    """torch 는 Q·K·V 의 가중치를 **하나로 묶어** `in_proj_weight` (3E, E) 에 담는다 —
    행렬곱을 세 번이 아니라 한 번 하려는 것이고, 그래서 체크포인트도 그 모양이다.
    나눠 들면 값은 같아도 `state_dict` 가 안 맞는다."""

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=False):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim({embed_dim}) 이 num_heads({num_heads}) 로 안 나뉩니다.")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        bound = _np.sqrt(1.0 / embed_dim)
        self.in_proj_weight = Parameter(
            _rng.uniform(-bound, bound, (3 * embed_dim, embed_dim)).astype(_np.float32))
        self.in_proj_bias = Parameter(
            _np.zeros(3 * embed_dim, dtype=_np.float32)) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key=None, value=None, attn_mask=None, need_weights=True):
        key = query if key is None else key
        value = query if value is None else value
        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        B, T, E = query.shape
        S = key.shape[1]
        w, b = self.in_proj_weight, self.in_proj_bias

        def project(t, index):
            piece = w[index * E:(index + 1) * E]
            out = t @ piece.transpose(0, 1)
            return out + b[index * E:(index + 1) * E] if b is not None else out

        q = _split_heads(project(query, 0), B, T, self.num_heads, self.head_dim)
        k = _split_heads(project(key, 1), B, S, self.num_heads, self.head_dim)
        v = _split_heads(project(value, 2), B, S, self.num_heads, self.head_dim)

        scores = (q @ k.transpose(-2, -1)) / float(_np.sqrt(self.head_dim))
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


class TransformerEncoderLayer(Module):
    """어텐션 + 피드포워드, 각각에 잔차와 정규화."""

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
            gelu if activation == "gelu" else activation)

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


class TransformerDecoderLayer(Module):
    """인코더 층과 다른 점은 가운데 하나다 — `multihead_attn` 이 자기 자신이 아니라
    인코더의 출력(memory)을 본다."""

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
            gelu if activation == "gelu" else activation)

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


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        import copy as _copy
        self.layers = ModuleList([_copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self._modules["layers"] = self.layers
        self.num_layers, self.norm = num_layers, norm

    def forward(self, src, mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x) if self.norm is not None else x


class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        import copy as _copy
        self.layers = ModuleList([_copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self._modules["layers"] = self.layers
        self.num_layers, self.norm = num_layers, norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x) if self.norm is not None else x


class Transformer(Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 batch_first=False, norm_first=False, layer_norm_eps=1e-5):
        super().__init__()
        common = dict(dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                      batch_first=batch_first, norm_first=norm_first,
                      layer_norm_eps=layer_norm_eps)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, **common), num_encoder_layers,
            LayerNorm(d_model, eps=layer_norm_eps))
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, **common), num_decoder_layers,
            LayerNorm(d_model, eps=layer_norm_eps))
        self.d_model, self.nhead, self.batch_first = d_model, nhead, batch_first

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, mask=src_mask)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        """윗삼각을 -inf 로 채운 **실수** 마스크. 더해서 쓴다."""
        m = _np.zeros((sz, sz), dtype=_np.float32)
        m[_np.triu_indices(sz, 1)] = -_np.inf
        return Tensor(_to_tf(m), dt=float32)


class BatchNorm3d(BatchNorm2d):
    """`BatchNorm2d` 와 **같은 코드다.**

    `batch_norm` 이 랭크를 안 따지고 채널 축만 남기므로 (N,C,D,H,W) 도 그대로 통한다 —
    3차원이라고 새로 쓸 것이 없었다. 처음에 이것을 conv3d·maxPool3d 와 한 덩이로 묶어
    거절했는데, 셋의 사정이 전혀 달랐다.
    """


class MSELoss(Module):
    def forward(self, pred, target):
        return mse_loss(pred, target)


class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        return cross_entropy(logits, target)


class _NN:
    functional = _Functional()
    Conv3d = Conv3d
    MaxPool3d = MaxPool3d
    BatchNorm3d = BatchNorm3d
    Module = Module
    Parameter = Parameter
    Linear = Linear
    ReLU = ReLU
    Sigmoid = Sigmoid
    Tanh = Tanh
    GELU = GELU
    Sequential = Sequential
    ModuleList = ModuleList
    Identity = Identity
    Dropout = Dropout
    LeakyReLU = LeakyReLU
    ELU = ELU
    SiLU = SiLU
    Softmax = Softmax
    LogSoftmax = LogSoftmax
    LayerNorm = LayerNorm
    BatchNorm1d = BatchNorm1d
    Embedding = Embedding
    Unflatten = Unflatten
    L1Loss = L1Loss
    SmoothL1Loss = SmoothL1Loss
    NLLLoss = NLLLoss
    BCELoss = BCELoss
    BCEWithLogitsLoss = BCEWithLogitsLoss
    RNN = RNN
    LSTM = LSTM
    GRU = GRU
    MultiheadAttention = MultiheadAttention
    TransformerEncoderLayer = TransformerEncoderLayer
    TransformerEncoder = TransformerEncoder
    TransformerDecoderLayer = TransformerDecoderLayer
    TransformerDecoder = TransformerDecoder
    Transformer = Transformer
    Conv1d = Conv1d
    Conv2d = Conv2d
    MaxPool1d = MaxPool1d
    MaxPool2d = MaxPool2d
    Upsample = Upsample
    AvgPool2d = AvgPool2d
    AdaptiveAvgPool2d = AdaptiveAvgPool2d
    Flatten = Flatten
    BatchNorm2d = BatchNorm2d
    MSELoss = MSELoss
    CrossEntropyLoss = CrossEntropyLoss


nn = _NN()


# ---------------------------------------------------------------- nn.utils.rnn

def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """길이가 제각각인 텐서들을 가장 긴 것에 맞춰 채워 하나로 쌓는다.

    **그래프를 잇는다.** 이미 있는 `pad` 와 `stack` 으로만 짜서 역방향이 저절로 따라온다.
    numpy 로 자리를 메워 맨 텐서로 돌려주면 값은 맞고 기울기가 조용히 사라지는데, 이
    저장소가 `var`·`std` 에서 정확히 그것을 한 번 겪었다(커밋 3ada1db).

    채운 자리가 진짜 값처럼 보이면 안 되므로 진짜 torch 도 이 함수를 마스크와 짝으로 쓴다.
    """
    tensors = [_canonical(_wrap(s)) for s in sequences]
    if not tensors:
        raise ValueError("빈 목록은 쌓을 수 없습니다.")
    rest = tuple(tensors[0].shape[1:])
    for t in tensors:
        if tuple(t.shape[1:]) != rest:
            raise RuntimeError(_like_torch(
                f"첫 차원 말고는 모양이 같아야 합니다 — {rest} 와 {tuple(t.shape[1:])} 가 다릅니다.",
                "pad_sequence expects trailing dimensions to match",
            ))
    longest = max(t.shape[0] for t in tensors)
    # `pad` 는 torch 규칙대로 **마지막 차원부터** 받는다. 첫 차원만 뒤로 늘리려면
    # 나머지 차원을 0 으로 채운 뒤 맨 끝에 그 한 쌍을 둔다.
    padded = []
    for t in tensors:
        gap = longest - t.shape[0]
        spec = [0, 0] * (len(rest)) + [0, gap]
        padded.append(pad(t, spec, padding_value) if gap else t)
    return stack(padded, 0 if batch_first else 1)


class _NnUtilsRnn:
    pad_sequence = staticmethod(pad_sequence)


class _NnUtils:
    rnn = _NnUtilsRnn()


nn.utils = _NnUtils()


