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

import numpy as _np

import js as _js
from pyodide.ffi import to_js as _to_js

from ._base import Tensor, _js_floats, _js_list, guarded, handle, settle, wrap
from ._ops import _arg, camel, positional
# **모듈째 든다.** `manual_seed` 가 `_ops._rng` 를 **새 생성기로 갈아끼우므로**,
# 이름으로 들여오면 그 갈아끼움이 여기까지 안 온다 — 씨앗을 심어도 이 층만 안 바뀐다.
# 같은 갈래로 한 번 당한 자리다.
from . import _ops as _ops_mod

_ts = _js.borch


class _Functional:
    """`nn.functional`. 첫 인자의 메서드로 넘긴다 — torch 의 규칙 그대로다."""

    def __getattr__(self, name):
        # 모듈 쪽에 손으로 쓴 것은 여기서도 같은 것을 쓴다 — `F.pad` 가 그 예다.
        from . import _ops
        if name == "embedding":
            return embedding
        # **이름이나 인자가 borch.ts 와 다른 것들.** 규칙으로 못 넘긴다:
        # `rms_norm` 은 torch 가 모양을 받고 저쪽은 축 개수를 받으며,
        # `conv_transpose2d` 는 저쪽에 차원별 이름이 없고 `convTransposeND` 하나다.
        if name in _HAND_WRITTEN:
            return _HAND_WRITTEN[name]
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


def _rms_norm(x, normalized_shape, weight=None, eps=None):
    """torch 는 **모양**을 받고 borch.ts 는 **축 개수**를 받는다."""
    dims = len(normalized_shape) if isinstance(normalized_shape, (list, tuple)) else 1
    out = wrap(handle(x).rmsNorm(dims) if eps is None
               else handle(x).rmsNorm(dims, eps))
    return out if weight is None else out * weight


def _conv_transpose(x, weight, bias=None, stride=1, padding=0):
    """차원별 이름이 저쪽엔 없다 — `convTransposeND` 하나가 전부를 한다."""
    return wrap(handle(x).convTransposeND(
        handle(weight), handle(bias) if bias is not None else None, stride, padding))


def _pool_fn(kind, adaptive):
    """풀링 하나. **차원별 이름이 저쪽엔 없다** — `poolND` 하나가 전부를 한다."""
    def call(x, size, stride=None, return_indices=False, **kw):
        h = handle(x)
        if return_indices:
            return _pool_with_indices(x, size, stride, adaptive)
        if adaptive:
            return wrap(h.adaptivePool(kind, size))
        return wrap(h.poolND(kind, size, stride if stride is not None else size))
    return call


def _pool_with_indices(x, size, stride=None, adaptive=False, **kw):
    """값과 **이긴 자리**를 함께 낸다. 자리는 평면 안의 평평한 번호다.

    저쪽은 `{values, indices}` 를 돌려주므로 이름으로 꺼내 튜플로 바꾼다 — torch 가
    튜플을 주고 교재 코드가 `out, idx = ...` 로 푼다.
    """
    h = handle(x)
    got = (h.adaptiveMaxPoolWithIndices(size) if adaptive
           else h.maxPoolWithIndices(size, stride if stride is not None else size))
    return wrap(got.values), wrap(got.indices)


def _unpool(x, indices, kernel_size, stride=None, padding=0, output_size=None):
    """자리표가 가리키는 칸으로 값을 되돌린다. 나머지는 0 이다."""
    return wrap(handle(x).maxUnpool(
        handle(indices), kernel_size, stride, padding,
        _js_list(list(output_size)) if output_size is not None else None))


def _fractional(spatial):
    """창 자리를 표본이 흔드는 최대 풀링.

    **표본을 파이썬에서 만든다.** 창 자리는 CPU 에서 정해야 셰이더에 구울 수 있고,
    borch.ts 쪽에서 GPU 난수를 읽어 오려면 기다려야 한다. numpy 가 이미 여기 있으므로
    이쪽에서 뽑아 넘긴다 — `manual_seed` 도 그 난수를 잡는다.
    """
    def call(x, kernel_size, output_size=None, output_ratio=None,
             return_indices=False, _random_samples=None, **kw):
        h = handle(x)
        shape = [int(n) for n in h.shape]
        if (output_size is None) == (output_ratio is None):
            raise ValueError(
                "fractional_max_pool 은 output_size 나 output_ratio 중 하나만 받습니다.")
        if output_size is not None:
            sizes = ([output_size] * spatial if isinstance(output_size, int)
                     else list(output_size))
        else:
            ratios = ([output_ratio] * spatial if isinstance(output_ratio, float)
                      else list(output_ratio))
            sizes = [int(shape[2 + k] * ratios[k]) for k in range(spatial)]

        n, c = shape[0], shape[1]
        if _random_samples is None:
            samples = _ops_mod._rng.random((n, c, spatial))
        else:
            samples = _np.asarray(_random_samples.numpy()).reshape(n, c, spatial)
        flat = [[float(v) for v in samples[i, j]] for i in range(n) for j in range(c)]
        got = h.fractionalMaxPool(
            kernel_size, _js_list(sizes),
            _to_js([_js_floats(row) for row in flat]))
        out, idx = wrap(got.values), wrap(got.indices)
        return (out, idx) if return_indices else out
    return call


def _fractional_indices(spatial):
    fn = _fractional(spatial)

    def call(x, kernel_size, output_size=None, output_ratio=None,
             _random_samples=None, **kw):
        return fn(x, kernel_size, output_size, output_ratio, True, _random_samples)
    return call


def _lp_pool(x, norm_type, kernel_size, stride=None, **kw):
    return wrap(handle(x).lpPool(norm_type, kernel_size, stride))


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
          scale=None):
    """borch.ts 쪽은 자유 함수라 텐서 메서드로 안 간다."""
    return wrap(_js.borch.nn.scaledDotProductAttention(
        handle(query), handle(key), handle(value),
        handle(attn_mask) if attn_mask is not None else None, bool(is_causal)))


# ── 손실과 거리 ─────────────────────────────────────────────────────────
#
# **자리 순서를 손으로 적는다.** 일반 길(`positional`)은 빈 자리를 `None` 으로 메우는데,
# torch 의 손실은 `reduction` 이 가운데 있고 `delta`·`margin` 이 뒤에 있어서 하나만
# 이름으로 주면 앞자리에 구멍이 생긴다. 그 구멍이 저쪽의 기본값을 지운다.

def _lay_out(order, args, kw):
    """이름으로 받은 것을 **자리로 편다.** 빈 자리는 비워서 넘긴다.

    **가운데가 비면 멈추면 안 된다.** 처음에 그렇게 적었더니
    `huber_loss(x, y, reduction="none")` 이 `delta` 자리에서 끊겨 `reduction` 을
    통째로 잃었다 — 예외가 아니라 **기본값으로 조용히 다른 답**이었고, 열세 케이스가
    한꺼번에 그 모양으로 갈렸다.

    빈 자리는 `None` 으로 둔다. Pyodide 가 그것을 `undefined` 로 넘기므로 저쪽의
    기본값이 그대로 산다. 뒤에 남는 것은 잘라낸다.
    """
    got = dict(zip(order, args))
    got.update(kw)
    laid = [_arg(got[key]) if key in got else None for key in order]
    while laid and laid[-1] is None:
        laid.pop()
    return laid


def _loss(js_name, order):
    """torch 의 인자를 **borch.ts 의 자리 순서**로 옮긴다."""
    def call(x, *args, **kw):
        return guarded(getattr(handle(x), js_name), *_lay_out(order, args, kw))

    call.__name__ = js_name
    return call


_LOSSES = {
    "huber_loss": ("huberLoss", ("target", "delta", "reduction")),
    "kl_div": ("klDiv", ("target", "reduction", "log_target")),
    "poisson_nll_loss": ("poissonNllLoss",
                         ("target", "log_input", "full", "eps", "reduction")),
    "gaussian_nll_loss": ("gaussianNllLoss",
                          ("target", "var", "full", "eps", "reduction")),
    "margin_ranking_loss": ("marginRankingLoss",
                            ("input2", "target", "margin", "reduction")),
    "cosine_embedding_loss": ("cosineEmbeddingLoss",
                              ("input2", "target", "margin", "reduction")),
    "hinge_embedding_loss": ("hingeEmbeddingLoss",
                             ("target", "margin", "reduction")),
    "soft_margin_loss": ("softMarginLoss", ("target", "reduction")),
    "triplet_margin_loss": ("tripletMarginLoss",
                            ("positive", "negative", "margin", "p", "eps",
                             "swap", "reduction")),
    "multilabel_soft_margin_loss": ("multilabelSoftMarginLoss",
                                    ("target", "reduction")),
    "multi_margin_loss": ("multiMarginLoss",
                          ("target", "p", "margin", "weight", "reduction")),
    "multilabel_margin_loss": ("multilabelMarginLoss", ("target", "reduction")),
    "pairwise_distance": ("pairwiseDistance", ("x2", "p", "eps", "keepdim")),
    "pdist": ("pdist", ("p",)),
    # 자리 옮기기. 이름만 갈린다.
    # 창 펴기와 나머지. **`F.unfold` 는 im2col 이다** — `Tensor.unfold` 와 다르다.
    "unfold": ("unfoldIm2col", ("kernel_size", "dilation", "padding", "stride")),
    "fold": ("fold", ("output_size", "kernel_size", "dilation", "padding",
                      "stride")),
    "local_response_norm": ("localResponseNorm", ("size", "alpha", "beta", "k")),
    "rrelu": ("rrelu", ("lower", "upper", "training")),
    "pixel_shuffle": ("pixelShuffle", ("upscale_factor",)),
    "pixel_unshuffle": ("pixelUnshuffle", ("downscale_factor",)),
    "channel_shuffle": ("channelShuffle", ("groups",)),
    # **채널째 떨구는 것들.** 저쪽은 이름 하나(`featureDropout`)이고 랭크는 안 따진다 —
    # 랭크 검사는 `dropout1d` 만 하므로 아래에서 따로 단다.
    "dropout2d": ("featureDropout", ("p", "training")),
    "dropout3d": ("featureDropout", ("p", "training")),
}


def _dropout1d(x, p=0.5, training=True, **kw):
    """**4 차원을 거절한다.** torch 가 그렇고, 이름에 공간 축의 수가 들어 있다."""
    rank = len(handle(x).shape)
    if rank not in (2, 3):
        raise RuntimeError(
            f"dropout1d: Expected 2D or 3D input, but received a {rank}D input. "
            "Note that dropout1d exists to provide channel-wise dropout on inputs "
            "with 1 spatial dimension, a channel dimension, and an optional batch "
            "dimension (i.e. 2D or 3D inputs).")
    return wrap(guarded(handle(x).featureDropout, float(p), bool(training)))


def _alpha_dropout(per_channel):
    def call(x, p=0.5, training=False, **kw):
        return wrap(guarded(handle(x).alphaDropout, float(p), bool(training),
                            per_channel))
    return call


def _triplet_with_distance(anchor, positive, negative, distance_function=None,
                           margin=1.0, swap=False, reduction="mean"):
    """거리 함수를 받는 삼중항. 층 쪽에 있는 것을 함수 이름으로도 낸다."""
    layer = _ts.nn.TripletMarginWithDistanceLoss.new(
        distance_function, float(margin), bool(swap), reduction)
    return wrap(guarded(layer.call, handle(anchor), handle(positive),
                        handle(negative)))


def _interpolate(x, size=None, scale_factor=2, mode="nearest",
                 align_corners=None, **kw):
    """**`mode` 를 받아만 놓고 안 쓰던 자리다.**

    일반 길은 `upsample` 로 넘기는데 그쪽은 최근접뿐이라, 겹선형을 달라고 해도
    최근접이 나왔다 — 예외가 아니라 조용히 다른 값이다. `F.pad` 가 같은 모양으로
    한 번 걸렸고, 그때와 마찬가지로 골든에 그 갈래를 묻는 케이스가 생기고서 드러났다.
    """
    h = handle(x)
    if mode == "nearest":
        return wrap(guarded(h.upsample, scale_factor))
    if mode != "bilinear":
        raise RuntimeError(f"interpolate(mode={mode!r}) — 최근접과 겹선형만 있습니다")
    if size is not None:
        oh, ow = (size, size) if isinstance(size, int) else tuple(size)
    else:
        s = scale_factor if isinstance(scale_factor, int) else scale_factor[0]
        oh, ow = int(h.shape[2] * s), int(h.shape[3] * s)
    return wrap(guarded(h.interpolateBilinear, oh, ow, bool(align_corners)))


def _bilinear(x1, x2, weight, bias=None):
    """가중치를 밖에서 받는 꼴. 층 쪽과 같은 텐서 메서드로 간다."""
    return wrap(guarded(handle(x1).bilinear, handle(x2), handle(weight),
                        handle(bias) if bias is not None else None))


_HAND_WRITTEN = {
    "interpolate": _interpolate,
    "bilinear": _bilinear,
    "dropout1d": _dropout1d,
    "alpha_dropout": _alpha_dropout(False),
    "feature_alpha_dropout": _alpha_dropout(True),
    "triplet_margin_with_distance_loss": _triplet_with_distance,
    "scaled_dot_product_attention": _sdpa,
    "avg_pool1d": _pool_fn("avg", False),
    "avg_pool3d": _pool_fn("avg", False),
    "adaptive_avg_pool1d": _pool_fn("avg", True),
    "adaptive_avg_pool3d": _pool_fn("avg", True),
    "adaptive_max_pool1d": _pool_fn("max", True),
    "adaptive_max_pool2d": _pool_fn("max", True),
    "adaptive_max_pool3d": _pool_fn("max", True),
    "lp_pool1d": _lp_pool,
    "lp_pool2d": _lp_pool,
    "lp_pool3d": _lp_pool,
    # 이긴 자리를 함께 내는 판. torch 는 같은 계산에 이름을 둘 준다 —
    # `return_indices=True` 와 `*_with_indices` 다.
    "max_pool1d_with_indices": _pool_with_indices,
    "max_pool2d_with_indices": _pool_with_indices,
    "max_pool3d_with_indices": _pool_with_indices,
    "adaptive_max_pool1d_with_indices": lambda x, s, **k: _pool_with_indices(
        x, s, adaptive=True),
    "adaptive_max_pool2d_with_indices": lambda x, s, **k: _pool_with_indices(
        x, s, adaptive=True),
    "adaptive_max_pool3d_with_indices": lambda x, s, **k: _pool_with_indices(
        x, s, adaptive=True),
    "max_unpool1d": _unpool,
    "max_unpool2d": _unpool,
    "max_unpool3d": _unpool,
    # `max_pool*d` 는 `return_indices` 를 받아야 해서 일반 길로 못 간다 — 일반 길은
    # 인자를 저쪽 메서드에 그대로 넘기고, 저쪽 `maxPool2d` 는 그 이름을 모른다.
    "max_pool1d": _pool_fn("max", False),
    "max_pool2d": _pool_fn("max", False),
    "max_pool3d": _pool_fn("max", False),
    "fractional_max_pool2d": _fractional(2),
    "fractional_max_pool3d": _fractional(3),
    "fractional_max_pool2d_with_indices": _fractional_indices(2),
    "fractional_max_pool3d_with_indices": _fractional_indices(3),
    "rms_norm": _rms_norm,
    "conv_transpose1d": _conv_transpose,
    "conv_transpose2d": _conv_transpose,
    "conv_transpose3d": _conv_transpose,
    **{name: _loss(js, order) for name, (js, order) in _LOSSES.items()},
}


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
            # **`_Holder` 를 빠뜨리면 컨테이너가 컨테이너 노릇을 못 한다.** 그 클래스들이
            # 있는 이유가 "맨 리스트는 안 보인다" 인데, 여기 안 적으면 컨테이너 자신이
            # 안 보이게 되어 같은 자리로 돌아온다.
            if isinstance(value, (Module, _Wrap, _Sequential, _Holder, Tensor)):
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

    def __repr__(self):
        """**찍는 것도 borch.ts 에 물어본다.**

        층이 자기를 어떻게 찍는지는 값과 같은 자격의 답이다 — 교재가 `print(model)`
        을 하고 골든이 그 글자를 굳혔다. 파이썬 쪽에서 다시 조립하면 두 벌이 되고,
        두 벌은 어긋난다. 저쪽에 `describe` 가 없으면 클래스 이름만 준다.
        """
        if self._m is None:
            return f"{type(self).__name__}()"
        fn = getattr(self._m, "describe", None)
        return str(fn()) if fn is not None else f"{type(self).__name__}()"

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
                    # **`no_grad` 안에서 옮긴다.** 파라미터는 기울기를 받는 잎이고,
                    # 잎을 제자리에서 고치는 것은 거절된다(torch 도 그렇다). borch.ts
                    # 쪽 `loadStateDict` 는 이미 감싸는데 이 갈래만 안 감싸고 있었다.
                    from ._ops import no_grad
                    with no_grad():
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
    """저쪽 층을 감싼다.

    **여기서 나오는 것은 전부 `Module` 한 클래스다.** 그래서
    `type(model.fc).__name__` 이 늘 `Module` 이고, torch 는 그 자리에서 `Linear` 라고
    말한다. 이름별 클래스를 찍어 주면 되지만 borch.ts 쪽에 차원별 이름이 없어서
    (`BatchNorm2d` 가 저쪽엔 `BatchNormND` 하나다) 반만 맞는 이름이 나온다 —
    반만 맞는 이름은 `Module` 보다 더 헷갈린다. 저쪽 이름을 먼저 갈라야 하는 일이다.
    """
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


# ── 컨테이너 ────────────────────────────────────────────────────────────────
#
# **여기가 조용히 틀리는 자리다.** 맨 리스트에 층이나 파라미터를 담아 속성으로 붙이면
# `Module._children()` 이 그것을 못 알아본다 — `parameters()` 가 안 내놓고, 옵티마이저가
# 못 보고, 그런데 **손실은 내려간다**(남은 파라미터가 대신 맞춘다). 예외도 경고도 없다.
#
# torch 도 똑같이 못 알아보고, 그래서 torch 에 이 네 클래스가 있다.

def Parameter(value, requires_grad=True):
    """학습되는 텐서. torch 는 `Tensor` 의 하위 클래스인데 여기서는 **잎 텐서**다.

    borch.ts 에 `Parameter` 라는 것이 따로 없다 — 기울기를 받는 잎이면 그것이
    파라미터다. 값을 CPU 로 한 번 돌려 새 잎으로 세우는데, 모델을 세울 때 한 번씩만
    도는 자리라 학습 루프의 비용이 아니다.
    """
    from ._base import tensor as _t

    arr = value.numpy() if isinstance(value, Tensor) else value
    return _t(arr, requires_grad=requires_grad)


class _Holder:
    """컨테이너 넷이 나눠 쓰는 살림. **이름 규칙이 여기 한 군데 있어야 한다.**

    `state_dict` 열쇠는 `layers.0.weight` 처럼 자리 이름을 앞에 붙여 만드는데, 그
    규칙이 컨테이너마다 따로 적히면 하나가 갈렸을 때 어디서 갈렸는지 못 찾는다.
    """

    def parameters(self):
        return [p for _, m in self._entries() for p in _params_of(m)]

    def state_dict(self):
        out = {}
        for name, m in self._entries():
            for k, v in _state_of(m).items():
                out[f"{name}.{k}"] = v
        return out

    def named_parameters(self):
        return list(self.state_dict().items())

    def load_state_dict(self, values, strict=True):
        own = dict(self._entries())
        groups = {}
        for key, v in values.items():
            head, _, rest = key.partition(".")
            groups.setdefault(head, {})[rest] = v
        for head, sub in groups.items():
            if head in own:
                own[head].load_state_dict(sub, strict)
            elif strict:
                raise RuntimeError(f"load_state_dict: 모르는 이름 '{head}'")
        return self

    def train(self, mode=True):
        for _, m in self._entries():
            if hasattr(m, "train"):
                m.train(mode)
        return self

    def eval(self):
        return self.train(False)


def _ordered(mapping):
    """torch 의 순서 규칙. **평범한 dict 는 열쇠를 정렬해서 넣는다.**

    `OrderedDict` 로 주면 넣은 순서를 지키고 그냥 `dict` 면 정렬한다. 안 맞추면
    `named_parameters` 의 순서가 갈리고 그것이 곧 `state_dict` 의 순서다 — 골든이
    실제로 이 자리를 잡았다(`{"w":…, "b":…}` 에 torch 는 `ws.b ws.w` 를 냈다).
    """
    import collections as _c

    items = dict(mapping or {})
    if isinstance(mapping, (_ModuleDict, _ParameterDict, _c.OrderedDict)):
        return list(items.items())
    return sorted(items.items(), key=lambda kv: str(kv[0]))


class _ModuleList(_Holder):
    """층 목록. 번호가 곧 이름이다 — `layers.0.weight`.

    `append` 가 없으면 층 수가 정해지지 않은 모델을 쓸 방법이 없다.
    """

    def __init__(self, mods=()):
        self.layers = list(mods)

    def _entries(self):
        return [(str(i), m) for i, m in enumerate(self.layers)]

    def append(self, module):
        self.layers.append(module)
        return self

    def extend(self, mods):
        self.layers.extend(mods)
        return self

    def insert(self, index, module):
        self.layers.insert(index, module)
        return self

    def __getitem__(self, i):
        return self.layers[i]

    def __setitem__(self, i, module):
        self.layers[i] = module

    def __iadd__(self, mods):
        return self.extend(mods)

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)


class _ModuleDict(_Holder):
    """이름 붙은 층 묶음. 준 이름이 그대로 `state_dict` 열쇠가 된다."""

    def __init__(self, mods=None):
        self.mods = dict(_ordered(mods))

    def _entries(self):
        return list(self.mods.items())

    def __getitem__(self, key):
        return self.mods[key]

    def __setitem__(self, key, module):
        self.mods[str(key)] = module

    def __contains__(self, key):
        return key in self.mods

    def __iter__(self):
        return iter(self.mods)

    def __len__(self):
        return len(self.mods)

    def keys(self):
        return self.mods.keys()

    def values(self):
        return self.mods.values()

    def items(self):
        return self.mods.items()

    def update(self, mods):
        self.mods.update(dict(_ordered(mods)))
        return self


class _ParamHolder(_Holder):
    """파라미터를 직접 담는 쪽. 잎이 텐서라 `_Holder` 의 순회를 못 쓴다."""

    def parameters(self):
        return [p for _, p in self._entries()]

    def state_dict(self):
        return dict(self._entries())

    def load_state_dict(self, values, strict=True):
        own = dict(self._entries())
        for key, v in values.items():
            if key in own:
                own[key]._h.copyFrom(handle(v))
            elif strict:
                raise RuntimeError(f"load_state_dict: 모르는 이름 '{key}'")
        return self

    def train(self, mode=True):
        return self


class _ParameterList(_ParamHolder):
    """`Parameter` 목록. **이것이 없으면 대신할 방법이 없다.**"""

    def __init__(self, params=()):
        self.params = list(params)

    def _entries(self):
        return [(str(i), p) for i, p in enumerate(self.params)]

    def append(self, param):
        self.params.append(param)
        return self

    def extend(self, params):
        self.params.extend(params)
        return self

    def __getitem__(self, i):
        return self.params[i]

    def __setitem__(self, i, param):
        self.params[i] = param

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)


class _ParameterDict(_ParamHolder):
    """이름 붙은 `Parameter` 묶음."""

    def __init__(self, params=None):
        self.params = dict(_ordered(params))

    def _entries(self):
        return list(self.params.items())

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, param):
        self.params[str(key)] = param

    def __contains__(self, key):
        return key in self.params

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def update(self, params):
        self.params.update(dict(_ordered(params)))
        return self


def ModuleList(mods=()):
    return _ModuleList(mods)


def ModuleDict(mods=None):
    return _ModuleDict(mods)


def ParameterList(params=()):
    return _ParameterList(params)


def ParameterDict(params=None):
    return _ParameterDict(params)


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


# ── 되풀이의 한 걸음 ────────────────────────────────────────────────────
#
# **`call` 이 아니라 `step` 으로 간다.** 저쪽 `Module.call(x)` 은 인자 하나짜리라
# 상태를 못 받는다. `LSTMCell` 은 상태가 짝이고 답도 짝이라 그 자리도 여기서 푼다.

class _Cell(Module):
    _pairs = False

    def __call__(self, x, hx=None):
        if not self._pairs:
            args = (handle(x),) if hx is None else (handle(x), handle(hx))
            return wrap(guarded(self._m.step, *args))
        if hx is None:
            got = settle(self._m.step(handle(x)))
        else:
            # **`_js_list` 는 정수 목록 전용이다** — 텐서를 넣으면 `int()` 에서 멈춘다.
            # 짝은 `Array.of` 로 만든다.
            got = settle(self._m.step(
                handle(x), _js.Array.of(handle(hx[0]), handle(hx[1]))))
        return (got[0], got[1])


def _cell(name, pairs=False):
    def make(input_size, hidden_size, bias=True, **kw):
        args = [input_size, hidden_size, bias]
        if name == "RNNCell":
            args.append(kw.get("nonlinearity", "tanh"))
        made = _Cell(getattr(_ts.nn, name).new(*args))
        object.__setattr__(made, "_pairs", pairs)
        return made

    make.__name__ = name
    return make


RNNCell = _cell("RNNCell")
GRUCell = _cell("GRUCell")
LSTMCell = _cell("LSTMCell", pairs=True)


# ── 나머지 층 ───────────────────────────────────────────────────────────
#
# **인자가 둘 이상인 층은 `_layer` 로 못 간다.** 감싼 쪽의 `__call__` 이 저쪽
# `call(x)` 로 넘기는데 `Bilinear` 는 둘을, `EmbeddingBag(offsets)` 은 목록을 받는다.
# 그 둘만 손으로 잇는다.

class _Bilinear(Module):
    def __call__(self, x1, x2):
        return wrap(guarded(self._m.call2, handle(x1), handle(x2)))


def Bilinear(in1, in2, out, bias=True):
    return _Bilinear(_ts.nn.Bilinear.new(in1, in2, out))


class _EmbeddingBag(Module):
    def __call__(self, idx, offsets=None):
        if offsets is None:
            return wrap(guarded(self._m.call, handle(idx)))
        starts = [int(v) for v in _to_list(offsets)]
        return wrap(guarded(self._m.callOffsets, handle(idx),
                            _js_list(starts)))


def _to_list(t):
    got = handle(t).toArray() if hasattr(handle(t), "toArray") else t
    return settle(got) if hasattr(got, "then") else list(got)


def EmbeddingBag(num, dim, mode="mean", **kw):
    return _EmbeddingBag(_ts.nn.EmbeddingBag.new(num, dim, mode))


def _misc_layer(name):
    def make(*args, **kw):
        # **파이썬 튜플을 그대로 넘기면 안 된다.** 저쪽에서 잠깐 빌린 프록시가 되어
        # 곧 버려지고, 실패는 나중에 `borrowed proxy was automatically destroyed` 로
        # 나온다 — `Fold((4,4), 2)` 가 그 자리였다.
        laid = [_js_list(list(a)) if isinstance(a, (list, tuple)) else a
                for a in args]
        if "scale_factor" in kw:
            laid.append(kw["scale_factor"])
        return _layer(name, *laid)

    make.__name__ = name
    return make


for _misc in ("Unfold", "Fold", "LocalResponseNorm", "Softmax2d", "RReLU",
              "UpsamplingNearest2d", "UpsamplingBilinear2d"):
    globals()[_misc] = _misc_layer(_misc)


# ── 자리 옮기기·채널째 dropout ──────────────────────────────────────────

for _shuffle in ("PixelShuffle", "PixelUnshuffle", "ChannelShuffle",
                 "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout",
                 "FeatureAlphaDropout"):
    globals()[_shuffle] = (lambda name: lambda *a: _layer(name, *a))(_shuffle)


# ── 게으른 층 열셋 ──────────────────────────────────────────────────────
#
# 전부 borch.ts 쪽에 있다. **굳는 자리도 저쪽이다** — 프로토타입을 갈아 끼우므로
# 파이썬 쪽 감싼 물건은 그대로 두고 속만 바뀐다. `__repr__` 이 `describe()` 를
# 물으므로 굳기 전후의 글자도 저절로 따라온다.

for _lazy in ("LazyLinear",
              *(f"Lazy{k}{d}d" for k in ("Conv", "ConvTranspose", "BatchNorm",
                                         "InstanceNorm") for d in (1, 2, 3))):
    globals()[_lazy] = (lambda name: lambda *a: _layer(name, *a))(_lazy)


# ── 손실 층 ─────────────────────────────────────────────────────────────
#
# 전부 borch.ts 쪽에 있다. 여기서는 **인자 순서만 옮긴다** — torch 는 `reduction` 을
# 가운데 두고 저쪽은 뒤에 두므로, 이름으로 받아 자리로 편다.

_LOSS_LAYERS = {
    "HuberLoss": ("delta", "reduction"),
    "KLDivLoss": ("reduction", "log_target"),
    "PoissonNLLLoss": ("log_input", "full", "eps", "reduction"),
    "GaussianNLLLoss": ("full", "eps", "reduction"),
    "MarginRankingLoss": ("margin", "reduction"),
    "CosineEmbeddingLoss": ("margin", "reduction"),
    "HingeEmbeddingLoss": ("margin", "reduction"),
    "SoftMarginLoss": ("reduction",),
    "TripletMarginLoss": ("margin", "p", "eps", "swap", "reduction"),
    "TripletMarginWithDistanceLoss": ("distance_function", "margin", "swap",
                                      "reduction"),
    "MultiLabelSoftMarginLoss": ("reduction",),
    "MultiMarginLoss": ("p", "margin", "weight", "reduction"),
    "MultiLabelMarginLoss": ("reduction",),
    "PairwiseDistance": ("p", "eps", "keepdim"),
    "CosineSimilarity": ("dim", "eps"),
}


def _loss_layer(name, order):
    def make(*args, **kw):
        return _layer(name, *_lay_out(order, args, kw))

    make.__name__ = name
    return make


for _loss_name, _loss_order in _LOSS_LAYERS.items():
    globals()[_loss_name] = _loss_layer(_loss_name, _loss_order)


# ── 패딩 층 열다섯 ──────────────────────────────────────────────────────
#
# 전부 borch.ts 쪽에 있다. 여기서는 이름을 잇고 **파이썬의 `int`/튜플을 JS 가 읽을
# 수 있는 것으로 바꾸는 일**만 한다 — 파이썬 튜플을 그대로 넘기면 저쪽에서
# `typeof padding === "number"` 도 아니고 배열도 아닌 프록시가 된다.

def _pad_arg(padding):
    return padding if isinstance(padding, int) else _js_list(list(padding))


def _pad_layer(name):
    def make(padding, value=0.0):
        args = (_pad_arg(padding),)
        if name.startswith("ConstantPad"):
            args += (float(value),)
        return _layer(name, *args)
    make.__name__ = name
    return make


for _dims in (1, 2, 3):
    for _kind in ("Reflection", "Replication", "Circular", "Zero", "Constant"):
        _pad_name = f"{_kind}Pad{_dims}d"
        globals()[_pad_name] = _pad_layer(_pad_name)


def _batchnorm(n, eps=1e-5, momentum=0.1):
    return _layer("BatchNormND", n, eps, momentum)


BatchNorm1d = BatchNorm2d = BatchNorm3d = _batchnorm


def ReLU():
    return _layer("ReLU")


def _max_pool_layer(js_name):
    """`return_indices` 를 켜면 답이 둘이 된다 — 값과 이긴 자리."""
    def make(k=2, stride=None, return_indices=False):
        if return_indices:
            return _Wrap(lambda x: _pool_with_indices(x, k, stride))
        return _Wrap(lambda x: wrap(getattr(handle(x), js_name)(k, stride)))
    return make


MaxPool1d = _max_pool_layer("maxPool1d")
MaxPool2d = _max_pool_layer("maxPool2d")
MaxPool3d = _max_pool_layer("maxPool3d")


def _spread(v, n):
    """수 하나면 축마다 같은 값으로, 목록이면 그대로. 코어의 같은 이름과 같은 규칙이다."""
    return (v,) * n if isinstance(v, int) else tuple(v)


class _MaxUnpool(_Wrap):
    """`MaxPool` 이 고른 자리로 값을 되돌린다.

    **`forward` 가 인자를 둘 받는다** — 값과 자리표. 다른 층과 모양이 달라 `Sequential`
    에 그냥 못 넣는데 torch 도 같다. 자리표는 값과 함께 흘러야 하고, 층 안에 숨기면
    같은 층을 두 번 쓸 때 남의 자리표를 쓰게 된다.
    """

    def __init__(self, dim, kernel_size, stride=None, padding=0):
        super().__init__(lambda x, indices, output_size=None: _unpool(
            x, indices, kernel_size, stride, padding, output_size))
        self.dim = dim
        # **축마다 펴서 든다** — torch 가 그렇게 들고 `repr` 에 그 튜플이 그대로 나온다.
        self.kernel_size = _spread(kernel_size, dim)
        self.stride = _spread(kernel_size if stride is None else stride, dim)
        self.padding = _spread(padding, dim)

    def __repr__(self):
        return (f"MaxUnpool{self.dim}d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")


def _unpool_layer(dim):
    def make(kernel_size, stride=None, padding=0):
        return _MaxUnpool(dim, kernel_size, stride, padding)
    return make


MaxUnpool1d = _unpool_layer(1)
MaxUnpool2d = _unpool_layer(2)
MaxUnpool3d = _unpool_layer(3)


class _FractionalMaxPool(_Wrap):
    """**`repr` 이 비어 있다** — torch 의 `extra_repr` 가 아무것도 안 낸다(재봤다)."""

    def __init__(self, dim, kernel_size, output_size=None, output_ratio=None,
                 return_indices=False, _random_samples=None):
        fn = _fractional(dim)
        super().__init__(lambda x: fn(x, kernel_size, output_size, output_ratio,
                                      return_indices, _random_samples))
        self.dim = dim

    def __repr__(self):
        return f"FractionalMaxPool{self.dim}d()"


def FractionalMaxPool2d(*args, **kw):
    return _FractionalMaxPool(2, *args, **kw)


def FractionalMaxPool3d(*args, **kw):
    return _FractionalMaxPool(3, *args, **kw)


def Flatten(start_dim=1, end_dim=-1):
    from ._ops import flatten
    return _Wrap(lambda x: flatten(x, start_dim, end_dim))


def Identity():
    return _Wrap(lambda x: x)


def _pool_layer(kind, adaptive):
    def make(size, stride=None, return_indices=False):
        n = size[0] if isinstance(size, (list, tuple)) else size
        fn = _pool_fn(kind, adaptive)
        return _Wrap(lambda x: fn(x, n, stride, return_indices=return_indices))
    return make


AdaptiveAvgPool2d = _pool_layer("avg", True)
AdaptiveAvgPool1d = _pool_layer("avg", True)
AdaptiveAvgPool3d = _pool_layer("avg", True)
AdaptiveMaxPool1d = _pool_layer("max", True)
AdaptiveMaxPool2d = _pool_layer("max", True)
AdaptiveMaxPool3d = _pool_layer("max", True)
AvgPool1d = _pool_layer("avg", False)
AvgPool3d = _pool_layer("avg", False)


def LPPool1d(norm_type, kernel_size, stride=None):
    return _Wrap(lambda x: _lp_pool(x, norm_type, kernel_size, stride))


LPPool2d = LPPool3d = LPPool1d


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


# ── 활성함수 층. 전부 borch.ts 의 메서드 하나를 감싼다. ─────────────────────
#
# 인자 없는 것은 `unary` 표를 그대로 부르고, 인자를 받는 것은 그 인자를 상수로
# 구운 커널로 간다 — 어느 쪽이든 파이썬은 이름만 옮긴다.

def _unary_layer(name):
    return lambda: _Wrap(lambda x, n=name: wrap(handle(x).unary(n)))


Hardsigmoid = _unary_layer("hardsigmoid")
Hardswish = _unary_layer("hardswish")
LogSigmoid = _unary_layer("logsigmoid")
Mish = _unary_layer("mish")
ReLU6 = _unary_layer("relu6")
SELU = _unary_layer("selu")
Softsign = _unary_layer("softsign")
Tanhshrink = _unary_layer("tanhshrink")


def CELU(alpha=1.0):
    return _Wrap(lambda x: wrap(handle(x).celu(alpha)))


def Hardshrink(lambd=0.5):
    return _Wrap(lambda x: wrap(handle(x).hardshrink(lambd)))


def Softshrink(lambd=0.5):
    return _Wrap(lambda x: wrap(handle(x).softshrink(lambd)))


def Hardtanh(min_val=-1.0, max_val=1.0):
    return _Wrap(lambda x: wrap(handle(x).hardtanh(min_val, max_val)))


def Softplus(beta=1.0, threshold=20.0):
    return _Wrap(lambda x: wrap(handle(x).softplus(beta, threshold)))


def Threshold(threshold, value):
    return _Wrap(lambda x: wrap(handle(x).threshold(threshold, value)))


def Softmin(dim=-1):
    return _Wrap(lambda x: wrap(handle(x).softmin(dim)))


def GLU(dim=-1):
    return _Wrap(lambda x: wrap(handle(x).glu(dim)))


class PReLU(Module):
    """음수 쪽 기울기를 **학습한다.** 이 부류에서 유일하게 파라미터가 있다.

    `_Wrap` 이 아니라 `Module` 인 이유가 그것이다 — `weight` 가 `named_parameters`
    에 잡혀야 하고, 그 이름이 `state_dict` 열쇠가 된다.
    """

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        import numpy as _np

        self.weight = Parameter(_np.full(num_parameters, init, dtype=_np.float32))

    def forward(self, x):
        return wrap(handle(x).prelu(handle(self.weight)))


class GroupNorm(Module):
    """채널을 그룹으로 묶어 정규화. 가중치가 붙으므로 `_Wrap` 이 아니라 `Module` 이다."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        import numpy as _np

        self.num_groups, self.eps = num_groups, eps
        if affine:
            self.weight = Parameter(_np.ones(num_channels, dtype=_np.float32))
            self.bias = Parameter(_np.zeros(num_channels, dtype=_np.float32))

    def forward(self, x):
        h = handle(x)
        out = h.groupNorm(self.num_groups, self.eps)
        if getattr(self, "weight", None) is None:
            return wrap(out)
        shape = [1, int(handle(self.weight).size)] + [1] * (len(h.shape) - 2)
        return (wrap(out) * self.weight.reshape(*shape)) + self.bias.reshape(*shape)


def _instance_norm_layer(eps=1e-5):
    return _Wrap(lambda x: wrap(handle(x).instanceNorm(eps)))


InstanceNorm1d = InstanceNorm2d = InstanceNorm3d = (
    lambda num_features=0, eps=1e-5, **kw: _instance_norm_layer(eps))


class RMSNorm(Module):
    """**평균을 안 뺀다.** 그것이 `LayerNorm` 과의 유일한 차이다."""

    def __init__(self, normalized_shape, eps=None, elementwise_affine=True):
        super().__init__()
        import numpy as _np

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.dims = len(normalized_shape)
        if elementwise_affine:
            self.weight = Parameter(_np.ones(normalized_shape, dtype=_np.float32))

    def forward(self, x):
        out = wrap(handle(x).rmsNorm(self.dims))
        return out if getattr(self, "weight", None) is None else out * self.weight


class _ConvTransposeND(Module):
    """전치 합성곱. **가중치가 `(입력, 출력, …)` 이다** — `Conv2d` 와 뒤집혀 있다."""

    nd = 2

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True):
        super().__init__()
        import numpy as _np

        self.stride, self.padding = stride, padding
        nd = type(self).nd
        shape = (in_channels, out_channels) + (kernel_size,) * nd
        bound = 1.0 / (out_channels * kernel_size ** nd) ** 0.5
        rng = _np.random.default_rng(0)
        self.weight = Parameter(rng.uniform(-bound, bound, shape).astype(_np.float32))
        if bias:
            self.bias = Parameter(
                rng.uniform(-bound, bound, out_channels).astype(_np.float32))

    def forward(self, x):
        b = getattr(self, "bias", None)
        return wrap(handle(x).convTransposeND(
            handle(self.weight), handle(b) if b is not None else None,
            self.stride, self.padding))


class ConvTranspose1d(_ConvTransposeND):
    nd = 1


class ConvTranspose2d(_ConvTransposeND):
    nd = 2


class ConvTranspose3d(_ConvTransposeND):
    nd = 3


class Dropout(Module):
    """학습 때만 떨군다. **`_Wrap` 이 아니라 `Module` 인 이유가 모드다** —
    `_Wrap` 은 `training` 을 안 들고, 그러면 `eval()` 이 내려가도 계속 떨군다."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True

    def train(self, mode=True):
        self.training = mode
        return self

    def forward(self, x):
        return wrap(handle(x).dropout(self.p, self.training))


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
