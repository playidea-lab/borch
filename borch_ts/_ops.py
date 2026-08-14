"""파이썬 이름을 borch.ts 메서드로 넘긴다 — **목록 없이.**

손으로 이름 192 개를 적으면 그중 하나가 다른 연산을 부르는 날이 오고, 그것은 값
대조로만 보인다. 표를 읽어 다는 방법도 있지만 그러려면 `index.ts` 가 커널 표를
내보내야 하고, 그건 안쪽 사정을 공개 표면에 올리는 것이다.

그래서 **모듈의 `__getattr__`** 을 쓴다(PEP 562). `L.exp(x)` 가 오면 그때
`x.exp()` 로 넘기고, borch.ts 에 없으면 `AttributeError` 로 멈춘다. 없는 것을
근사하지 않는다 — 골든이 그 자리를 실패로 세고, 그 수가 곧 이 결속의 진도다.

이름 규칙 하나만 다르다. 파이썬은 `masked_select`, JS 는 `maskedSelect` 다.
"""

import js as _js

from ._base import Tensor, _js_list, handle, wrap

_ts = _js.borch

# 두 언어에서 철자가 아예 다른 것들. 규칙으로 안 되는 것만 적는다.
_RENAME = {
    "adaptive_avg_pool2d": "adaptiveAvgPool",
    "adaptive_avg_pool1d": "adaptiveAvgPool",
    "absolute": "abs",
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctanh": "atanh",
    "clip": "clamp",
    "fix": "trunc",
    "swapdims": "transpose",
    "interpolate": "upsample",
    "as_tensor": "from_",
}

# **이름 붙은 인자를 자리로 바꾼다.**
#
# torch 코드는 `clamp(x, min=-0.5, max=0.5)` 처럼 이름으로 부르는 자리가 많은데,
# JS 에는 그런 것이 없다. 처음에 `**kw` 를 그냥 버렸더니 `clamp(x, undefined,
# undefined)` 가 셰이더로 내려가 WGSL 이 파싱에서 멈췄다 — 실패 72 건이 그것이었다.
#
# 그래서 자리 이름을 적어 둔다. borch.ts 쪽 **인자 순서**이고, torch 의 이름을 그
# 자리에 놓는다. 여기 없는 함수는 이름 붙은 인자를 안 받는다는 뜻이다.
_SIGNATURE = {
    "clamp": ("min", "max"),
    "clip": ("min", "max"),
    "sum": ("dim", "keepdim"),
    "mean": ("dim", "keepdim"),
    "prod": ("dim", "keepdim"),
    "amax": ("dim", "keepdim"),
    "amin": ("dim", "keepdim"),
    "var": ("dim", "keepdim"),
    "std": ("dim", "keepdim"),
    "logsumexp": ("dim", "keepdim"),
    "argmax": ("dim",),
    "argmin": ("dim",),
    "softmax": ("dim",),
    "log_softmax": ("dim",),
    "cumsum": ("dim",),
    "cumprod": ("dim",),
    "sort": ("dim", "descending"),
    "topk": ("k", "dim", "largest"),
    "squeeze": ("dim",),
    "unsqueeze": ("dim",),
    "flatten": ("start_dim", "end_dim"),
    "flip": ("dims",),
    "roll": ("shifts", "dims"),
    "norm": ("p", "dim", "keepdim"),
    "diff": ("n", "dim"),
    "median": ("dim", "keepdim"),
    "gather": ("dim", "index"),
    "index_select": ("dim", "index"),
    "narrow": ("dim", "start", "length"),
    "transpose": ("dim0", "dim1"),
    "swapdims": ("dim0", "dim1"),
    "movedim": ("source", "destination"),
    "repeat_interleave": ("repeats", "dim"),
    "cat": ("dim",),
    "stack": ("dim",),
    "split": ("size", "dim"),
    "chunk": ("chunks", "dim"),
    "unbind": ("dim",),
    "conv1d": ("weight", "bias", "stride", "padding"),
    "conv2d": ("weight", "bias", "stride", "padding"),
    "conv3d": ("weight", "bias", "stride", "padding"),
    "max_pool1d": ("kernel_size", "stride"),
    "max_pool2d": ("kernel_size", "stride"),
    "max_pool3d": ("kernel_size", "stride"),
    "avg_pool2d": ("kernel_size", "stride"),
    "adaptive_avg_pool2d": ("output_size",),
    "normalize": ("dim", "eps"),
    "cosine_similarity": ("other", "dim", "eps"),
    "layer_norm": ("dim", "eps"),
    "leaky_relu": ("negative_slope",),
    "one_hot": ("num_classes",),
    "smooth_l1_loss": ("target", "beta"),
    "interpolate": ("scale_factor",),
}


def camel(name):
    """`masked_select` → `maskedSelect`. 밑줄 뒤 첫 글자를 올린다."""
    if name in _RENAME:
        return _RENAME[name]
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def _arg(a):
    """텐서면 손잡이로, 목록이면 JS 배열로, 나머지는 그대로."""
    if isinstance(a, Tensor):
        return a._h
    if isinstance(a, (list, tuple)):
        return _js_list(a)
    return a


def positional(name, args, kw):
    """이름 붙은 인자를 **자리로 편다.**

    JS 에는 이름 붙은 인자가 없다. 버리면 `undefined` 가 셰이더까지 내려가고, WGSL 은
    그것을 파싱에서 거절한다 — 조용히 틀리지는 않지만 원인이 한참 멀리서 나온다.

    뒤에 붙는 `undefined` 는 잘라낸다. borch.ts 의 기본값(`stride = kernel`)이
    살아나야 하는데, `undefined` 를 명시로 넘기면 그 자리가 안 채워진다.
    """
    if not kw:
        out = list(args)
    else:
        order = _SIGNATURE.get(name)
        if order is None:
            raise TypeError(
                f"`{name}` 은 이름 붙은 인자를 안 받는다 (받은 것: {sorted(kw)})\n"
                f"  받아야 한다면 `_SIGNATURE` 에 자리 순서를 적어라.")
        out = list(args)
        for i, key in enumerate(order):
            if key in kw:
                while len(out) <= i:
                    out.append(None)
                out[i] = kw[key]
    while out and out[-1] is None:
        out.pop()
    return [_arg(a) for a in out]


def __getattr__(name):
    """모듈에 없는 이름은 **첫 인자의 메서드**로 넘긴다.

    `torch.exp(x)` 와 `x.exp()` 가 같은 것이라는 torch 의 규칙을 그대로 쓴다.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    js_name = camel(name)

    def call(x, *args, **kw):
        h = handle(x)
        fn = getattr(h, js_name, None)
        if fn is None:
            raise AttributeError(
                f"borch.ts 에 `{js_name}` 이 없다 (파이썬 이름 `{name}`)")
        out = fn(*positional(name, args, kw))
        return wrap(out) if _ts.isTensor(out) else out

    call.__name__ = name
    return call


# ── 첫 인자가 텐서가 아닌 것들. 여기만 손으로 적는다. ────────────────────

def arange(*args):
    """`arange(n)` · `arange(a, b)` · `arange(a, b, step)`."""
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        (start, stop), step = args, 1
    else:
        start, stop, step = args
    return wrap(_ts.Tensor.arange(start, stop, step))


def _shape_of(shape):
    """`zeros(2, 3)` 와 `zeros([2, 3])` 을 둘 다 받는다 — torch 가 그렇다."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        return _js_list(shape[0])
    return _js_list(shape)


def zeros(*shape, **kw):
    return wrap(_ts.Tensor.zeros(_shape_of(shape)))


def ones(*shape, **kw):
    return wrap(_ts.Tensor.ones(_shape_of(shape)))


def full(shape, value, **kw):
    return wrap(_ts.Tensor.full(_js_list(shape), float(value)))


def eye(n, m=None, **kw):
    return wrap(_ts.Tensor.eye(n, n if m is None else m))


def cat(parts, dim=0):
    return wrap(_ts.Tensor.cat(_js.Array.from_([p._h for p in parts]), dim))


def stack(parts, dim=0):
    return wrap(_ts.Tensor.stack(_js.Array.from_([p._h for p in parts]), dim))
