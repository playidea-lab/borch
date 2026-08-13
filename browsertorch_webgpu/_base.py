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


class BrowserTorchError(NotImplementedError):
    """축소판이 지원하지 않는 것. 근사하지 않고 여기서 멈춘다."""


def _like_torch(korean, torch_phrase):
    """오류 메시지의 규격 — 코어와 같다.

    한국어 설명만 두면 학습자가 검색해서 답을 못 찾고, 영문만 베끼면 이 교재가
    한국어인 이유가 사라진다. 둘 다 넣는다 — 설명은 읽고, 영문 문구는 검색한다.
    """
    return f"{korean}\n(torch: {torch_phrase})"


def _broadcast_error(a, b):
    """torch 가 내는 것과 같은 자리·같은 문구로 알린다."""
    bad = next((i for i in range(1, min(len(a), len(b)) + 1)
                if a[-i] != b[-i] and a[-i] != 1 and b[-i] != 1), 1)
    raise RuntimeError(_like_torch(
        f"모양 {tuple(a)} 과 {tuple(b)} 은 브로드캐스팅되지 않습니다 — "
        "뒤에서부터 맞춰볼 때 크기가 같거나 한쪽이 1이어야 합니다.",
        f"The size of tensor a ({a[-bad]}) must match the size of tensor b "
        f"({b[-bad]}) at non-singleton dimension {len(a) - bad}"))


def _unsupported(what):
    raise BrowserTorchError(
        f"{what} 은(는) 아직 browsertorch-webgpu 에 없습니다. "
        "코어 `browsertorch` 나 자기 컴퓨터의 진짜 PyTorch 를 쓰세요."
    )


# ---------------------------------------------------------------- 경계

def _shape_of(handle):
    return tuple(int(n) for n in handle.shape)


# ---------------------------------------------------------------- dtype
#
# **저장은 float32 한 가지, dtype 은 그 위의 라벨이다.**
#
# TF.js 에는 int64 도 float64 도 없다. 그리고 `cast(int32 → float32)` 가 WebGPU 에서
# dtype 라벨만 바꾸고 **비트를 안 바꾼다**(실측: 2 가 2.8e-45 로 읽힌다). 그래서 정수를
# int32 로 저장하면 정수+실수 승격을 아예 할 수 없다.
#
# 라벨로 두면 승격이 **캐스팅 없이 라벨 변경만으로** 끝나 그 버그를 통째로 피한다.
# 대신 정수는 float32 가 정확히 담는 2^24 까지다 — 넘으면 조용히 자르지 않고 던진다.
# 불리언만 TF.js 의 bool 로 든다(비교 결과가 그것으로 나온다).

_INT_EXACT = 2 ** 24


class dtype:
    def __init__(self, name, np_type, category, rank, storage):
        self.name = name
        self.np = np_type
        self.category = category        # bool(0) < 정수(1) < 실수(2)
        self.rank = rank
        self.storage = storage          # TF.js 가 실제로 드는 것

    def __repr__(self):
        return f"torch.{self.name}"

    def __eq__(self, other):
        return isinstance(other, dtype) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


float32 = dtype("float32", _np.float32, 2, 20, "float32")
int64 = dtype("int64", _np.int64, 1, 10, "float32")
long = int64
bool_ = dtype("bool", _np.bool_, 0, 0, "bool")

_BY_CATEGORY = {0: bool_, 1: int64, 2: float32}


def _dtype_of(arr):
    """numpy 배열의 dtype → 우리 dtype. torch 의 규칙을 따른다."""
    kind = _np.asarray(arr).dtype.kind
    if kind == "b":
        return bool_
    if kind in "iu":
        return int64
    if _np.asarray(arr).dtype == _np.float64:
        # 조용히 float32 로 떨어뜨리지 않는다. 코어는 float64 를 진짜로 지원한다.
        return float32
    return float32


def _reject_float64(dt):
    if dt is not None and getattr(dt, "name", None) == "float64":
        _unsupported("float64 (TF.js 에 배정도가 없습니다)")


def _to_tf(arr, dt=None):
    """numpy → tf.Tensor. 평평하게 펴서 올리고 모양을 따로 준다."""
    arr = _np.asarray(arr)
    dt = dt or _dtype_of(arr)
    if dt is bool_:
        flat = _np.ascontiguousarray(arr.astype(_np.bool_)).reshape(-1)
        buf = _js.Uint8Array.new(flat.size)
        buf.assign(flat.view(_np.uint8))
        return _tf.tensor(buf, _to_js(list(arr.shape)), "bool")
    if dt is int64 and arr.size and _np.abs(arr.astype(_np.float64)).max() > _INT_EXACT:
        raise RuntimeError(
            f"정수가 {_INT_EXACT} 를 넘습니다. 이 라이브러리는 정수를 float32 에 담으므로 "
            "그 위로는 정확하지 않습니다 — 조용히 자르지 않고 여기서 멈춥니다.")
    flat = _np.ascontiguousarray(arr, dtype=_np.float32).reshape(-1)
    buf = _js.Float32Array.new(flat.size)
    buf.assign(flat)
    return _tf.tensor(buf, _to_js(list(arr.shape)), "float32")


def _to_np(handle):
    """tf.Tensor → numpy.

    `dataSync()` 를 쓴다. WebGPU 에 동기 읽기 API 가 없는데도 TF.js 가 이것을
    받아준다는 것을 실측으로 확인했고(3.35ms 대 비동기 1.72ms), 그래서 이 라이브러리의
    파이썬 API 가 통째로 동기로 남을 수 있다.
    """
    shape = _shape_of(handle)
    size = int(_np.prod(shape)) if shape else 1

    # dtype 을 보고 받을 그릇을 고른다. 전부 float32 로 받으면 int32 텐서의 **비트를
    # 그대로 실수로 읽는다** — one_hot 의 1 이 1.4e-45 로 나왔던 것이 그것이다.
    # 조용히 틀리는 종류라, 값을 보기 전에는 안 드러난다.
    kind = str(handle.dtype)
    if kind == "int32":
        flat = _np.empty(size, dtype=_np.int32)
    elif kind == "bool":
        flat = _np.empty(size, dtype=_np.uint8)
    else:
        flat = _np.empty(size, dtype=_np.float32)
    handle.dataSync().assign_to(flat)
    out = flat.reshape(shape)
    return out.astype(bool) if kind == "bool" else out


# ---------------------------------------------------------------- 표현(repr)
#
# 학습자가 가장 많이 하는 일이 print(tensor) 다. 진짜와 다르게 찍히면 교재의 예시와
# 화면이 안 맞고, 그때마다 "내가 뭘 잘못했나" 를 의심하게 된다.
#
# **코어와 같은 알고리즘이다.** 두 벌로 쓰면 언젠가 갈리므로 규칙을 그대로 옮겼다 —
# torch/_tensor_str.py 의 규칙이고, 코어가 15/15 로 맞춰둔 것이다.

_PRINT_PRECISION = 4
_LINE_WIDTH = 80


def set_printoptions(precision=None, linewidth=None):
    global _PRINT_PRECISION, _LINE_WIDTH
    if precision is not None:
        _PRINT_PRECISION = precision
    if linewidth is not None:
        _LINE_WIDTH = linewidth


def _float_formatter(arr):
    """torch 의 규칙: 값이 전부 정수면 `1.`, 아니면 소수 네 자리, 범위가 넓으면 지수."""
    finite = arr[_np.isfinite(arr)]
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return lambda v: f"{v:.0f}."
    amax, amin = _np.abs(nonzero).max(), _np.abs(nonzero).min()
    integral = bool(_np.all(finite == _np.floor(finite)))

    if integral and amax < 1e8:
        return lambda v: f"{v:.0f}."
    if amax / amin > 1000 or amax > 1e8 or amin < 1e-4:
        return lambda v, p=_PRINT_PRECISION: f"{v:.{p}e}"
    return lambda v, p=_PRINT_PRECISION: f"{v:.{p}f}"


def _tensor_str(data):
    if data.size == 0:
        return "[]"
    if data.dtype.kind == "f":
        fmt = _float_formatter(data)
        # torch 는 원소를 같은 너비로 오른쪽 정렬한다 — 음수가 섞이면 양수 앞에 자리가 생긴다.
        width = max((len(fmt(v)) for v in data.reshape(-1)), default=0)
        padded = lambda v, f=fmt, w=width: f(v).rjust(w)             # noqa: E731
        body = _np.array2string(
            data, formatter={"float_kind": padded}, separator=", ",
            max_line_width=_LINE_WIDTH - 8, threshold=1000)
    else:
        body = _np.array2string(data, separator=", ",
                                max_line_width=_LINE_WIDTH - 8, threshold=1000)
    # numpy 는 이어지는 줄을 한 칸 들여쓴다. torch 는 "tensor(" 만큼(8칸) 들여쓴다.
    return body.replace("\n ", "\n" + " " * 8)


def _tensor_repr(t):
    parts = [_tensor_str(t.numpy())]
    if t._op:
        parts.append(f"grad_fn=<{t._op}>")
    elif t.requires_grad:
        parts.append("requires_grad=True")
    return f"tensor({', '.join(parts)})"


class Size(tuple):
    def __repr__(self):
        return f"torch.Size([{', '.join(str(x) for x in self)}])"


# 아래 일곱은 **`Tensor` 라는 이름을 안 쓴다.** `t._make(...)` 처럼 넘겨받은 것의
# 메서드를 부를 뿐이라 `Tensor` 보다 먼저 놓을 수 있고, 그래서 여기가 제자리다.
# 예전에는 파일 한참 아래에 있었는데 `Tensor` 의 메서드들이 그것을 부르고 있었다 —
# 파일 하나일 때는 안 보이던 층위 뒤집힘이다.

class _ValuesIndices:
    """`x.max(dim=0)` 이 돌려주는 (values, indices). 진짜 torch 와 같은 모양."""

    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def __iter__(self):
        yield self.values
        yield self.indices

    def __getitem__(self, i):
        return (self.values, self.indices)[i]


def _pick_last(t, idx32):
    """마지막 축에서 번호대로 뽑되 **그래프를 잇는다.**

    자리를 원-핫으로 만들어 곱하고 접으면 역전파가 저절로 따라온다. 값만 떼어
    돌려주면 뽑은 자리로 기울기가 안 가고, top-k 샘플링이나 정렬을 끼운 손실에서
    **학습이 조용히 멈춘다** — 코어가 ROADMAP 11번에서 겪은 그대로다.
    """
    shape = t.shape
    n = shape[-1]
    rows = int(_np.prod(shape[:-1])) if len(shape) > 1 else 1
    k = _shape_of(idx32)[-1]

    flat = t.reshape(rows, n)
    onehot = _tf.cast(_tf.oneHot(_tf.reshape(idx32, _to_js([rows * k])), n), "float32")
    onehot = _tf.reshape(onehot, _to_js([rows, k, n]))
    picked = _tf.sum(_tf.mul(onehot, _tf.reshape(flat._h, _to_js([rows, 1, n]))), 2)

    def back(g):
        return (_tf.sum(_tf.mul(onehot, _tf.reshape(g, _to_js([rows, k, 1]))), 1),)

    out = flat._make(picked, (flat,), back, "TopkBackward0")
    return out.reshape(tuple(shape[:-1]) + (k,)) if len(shape) > 1 else out.reshape(k)


def _last_axis_only(t, dim, what):
    """TF.js 의 `topk` 는 **마지막 축만** 본다. 다른 축을 받으면 조용히 다른 값이
    나오므로 여기서 멈춘다 — 없는 기능이 틀린 답보다 낫다."""
    if dim not in (-1, t.ndim - 1):
        _unsupported(f"{what}(마지막 축이 아닌 dim)")


# 랭크 5 부터는 `tf.pad` 를 못 믿는다 — 아래 `_pad_const` 참고.
_PAD_SAFE_RANK = 4


def _pad_const(handle, shape, pads, value=0.0):
    """상수로 두른다. `shape` 는 `handle` 의 현재 모양, `pads` 는 (축, 앞, 뒤) 목록이다.

    **랭크 5 이상에서는 `tf.pad` 를 쓰지 않는다.** 거기서 pad 는 모양을 맞게 돌려주고
    값을 깨뜨리며, 예외를 안 던진다 — 부르는 쪽은 아무것도 모른 채 틀린 답을 받는다.
    conv3d 를 굳히다 잡았다: 1×1×1 항등 커널을 씌운 결과의 합이 28 이어야 하는데
    0.238 이었다.

    한 번에 안 끝났다는 것을 적어둔다. 처음에는 conv3d 만 고쳤고, 그 다음 케이스를
    세워 물어보니 자르기의 역방향도 같은 함수를 불러 **narrow·unbind·split 셋이 랭크 5
    에서 조용히 틀린 기울기**를 내고 있었다. 거기서 또 멈췄는데, 랭크 6 을 물어보니
    이번에는 **`F.pad` 자신** — 사용자가 직접 부르는 문 — 이 랭크 5·6 양쪽에서 틀리고
    있었다. 세 번 다 "고쳤다"고 생각한 뒤에 나왔다. 그러니 호출 지점은 여기 하나로
    모으고, 랭크 판단도 여기서만 한다.

    랭크 6 자체는 멀쩡하다는 것도 그때 같이 확인했다 — 원소별·축 합·permute·reshape·
    기울기 전부 맞았다. 고장난 것은 랭크가 아니라 `pad` 다.

    랭크 4 이하는 `tf.pad` 그대로 둔다. 골든 427 건과 ResNet-18 의 매 스텝이 지나는
    길이고 거기서는 값이 맞는다 — 안 깨진 것을 바꾸면 바꾼 쪽이 새 위험이 된다.
    """
    if len(shape) <= _PAD_SAFE_RANK:
        pairs = [[0, 0] for _ in shape]
        for axis, before, after in pads:
            pairs[axis] = [before, after]
        return _tf.pad(handle, _to_js(pairs), float(value))

    cur = list(shape)
    for axis, before, after in pads:
        for width, at_front in ((before, True), (after, False)):
            if not width:
                continue
            block = list(cur)
            block[axis] = width
            zeros = (_tf.zeros(_to_js(block)) if value == 0.0
                     else _tf.fill(_to_js(block), float(value)))
            parts = [zeros, handle] if at_front else [handle, zeros]
            handle = _tf.concat(_to_js(parts), axis)
            cur[axis] += width
    return handle


def _slice_along(handle, axis, start, length):
    shape = _shape_of(handle)
    begin = [0] * len(shape)
    size = list(shape)
    begin[axis], size[axis] = start, length
    return _tf.slice(handle, _to_js(begin), _to_js(size))


def _slice_tensor(t, dim, start, length):
    """잘라내되 **그래프를 잇는다.** 역방향은 잘라낸 자리 밖을 0 으로 채우는 것이다."""
    shape = list(t.shape)
    # 메울 대상은 들어온 기울기, 즉 **잘라낸 뒤의** 모양이다.
    out_shape = list(shape)
    out_shape[dim] = length
    pads = [(dim, start, shape[dim] - start - length)]
    return t._make(_slice_along(t._h, dim, start, length), (t,),
                   lambda g: (_pad_const(g, out_shape, pads),), "SliceBackward0")


def _keep(handle):
    """스코프가 끝나도 살려둘 것. 파라미터와 옵티마이저 상태가 여기 해당한다.

    **`_data` 쪽에 있던 것을 여기로 올렸다.** `Tensor` 와 `nn` 이 둘 다 부르는데
    정의는 파일 맨 아래였다 — 파일 하나일 때는 안 보이던 층위 뒤집힘이다.
    """
    try:
        return _tf.keep(handle)
    except Exception:                                                # noqa: BLE001
        return handle          # 스코프 밖이면 keep 이 필요 없다


