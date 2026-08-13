"""browsertorch-webgpu — TF.js WebGPU 위에 얹은 browsertorch 모양의 층.

코어 `browsertorch` 를 **대체하지 않는다.** 코어는 numpy 위에서 MNIST 급까지 가고,
이쪽은 GPU 위에서 그 위를 간다. 왜 별도인지는 ROADMAP.md 의 ADR-001 에 있다.

## 브라우저 안에서만 돈다

`js.tf` 를 부른다. 네이티브 CPython 에서 임포트하면 바로 멈춘다 — 조용히 다른 것으로
폴백하면 "GPU 로 돌렸다"고 착각하게 되고, 그건 이 프로젝트가 가장 싫어하는 종류다.

## 왜 자체 autograd 인가

TF.js 의 `tf.grad` 를 쓰지 않는다. 재봤더니 conv 역방향 커널이 순방향의 1/26 이었고
(WEBGPU-DESIGN.md 1.5절), 역방향을 **순방향 conv 로 다시 써야** 목표에 닿는다.
그러려면 역방향을 우리가 들고 있어야 한다. 테이프 구조는 코어와 같게 둔다 —
같은 모양이면 코어에서 고친 것을 여기로 옮기기 쉽다.

## 한계 — 코어와 다른 점

- **dtype 은 `float32` · `int64` · `bool` 세 가지다.** 승격 규칙은 torch 와 같게
  맞췄지만(72/72), **`float64` 는 없다** — TF.js 에 배정도가 없어서 거절한다.
  그리고 정수는 float32 에 담으므로 **2^24 까지 정확**하고, 넘으면 조용히 자르지 않고
  던진다. 코어는 float64 까지 112/112 다
- **뷰가 저장소를 공유하지 않는다.** torch 는 `b = a.view(2,2); b[0,0] = 9` 로 `a` 가
  바뀌는데, TF.js 텐서는 불변이라 그럴 수 없다. 코어는 이것을 13/13 로 맞춘다.
  조용히 다르게 두지 않고, **뷰에 대입하면 거절한다**
- **`scope()` 를 노출한다.** 역전파 클로저가 든 중간 버퍼는 파이썬 GC 가 못 놓는다.
  코어와 다른 한 줄이고, 이유는 WEBGPU-DESIGN.md 7절에 있다

없는 것은 근사하지 않고 예외를 던진다. `topk(largest=False)` 처럼 TF.js 가 못 하는
인자도 조용히 무시하지 않고 거절한다 — 조용히 다른 값을 내는 것이 가장 나쁘다.

## 이 파일을 고칠 때의 규칙 하나

**`_wrap(x)._h` 를 인라인으로 쓰지 마라.** `x` 가 텐서가 아니면 `_wrap` 이 임시를
만드는데, `._h` 를 꺼내는 순간 그 임시의 참조가 0 이 되어 `__del__` 이 버퍼를 놓는다.
그 다음 줄이 이미 죽은 손잡이를 넘긴다.

    나쁨:  _tf.add(a, _wrap(b)._h)
    좋음:  bt = _wrap(b); _tf.add(a, bt._h)

이 함정에 **세 번** 걸렸다(`_cmp`, `__setitem__`, 그리고 디버깅용 프로브). 증상은
`TypeError: Cannot read properties of undefined (reading 'backend')` 다.
"""

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


# ---------------------------------------------------------------- Tensor

_grad_enabled = True


_NCHW_TO_NHWC = [0, 2, 3, 1]
_NHWC_TO_NCHW = [0, 3, 1, 2]


class Tensor:
    """torch 의 모양(NCHW)을 말하되, 4차원은 **속으로 NHWC 를 들 수 있다.**

    TF.js 의 conv·풀링은 NHWC 만 빠르다(NCHW 로 부르면 346 GFLOPS, NHWC 는 2,306 —
    실측). 그래서 conv 를 지날 때마다 전치하면 33.5MB 를 두 번 훑고, 그게 순방향
    시간의 88% 였다. 대신 **레이아웃을 들고 다니게** 해서 conv·BN·활성·잔차 덧셈이
    전부 NHWC 로 이어지게 한다. 전치는 들어올 때와 나갈 때 한 번씩이다.

    `shape` 는 언제나 torch 순서로 답한다 — 밖에서는 이 사정이 보이면 안 된다.
    """

    def __init__(self, handle, requires_grad=False, _parents=(), _backward=None, dt=None):
        self._h = handle
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = _parents
        self._backward = _backward
        self._op = None
        self._nhwc = False
        self._freed = False         # backward 한 번이면 그래프를 놓는다 (torch 와 같다)
        self._derived = False       # 다른 텐서에서 나왔는가 — 제자리 대입을 막는 근거다
        # 라벨이 없으면 저장이 말해주는 대로. bool 저장은 bool, 나머지는 실수다.
        self._dtype = dt or (bool_ if str(handle.dtype) == "bool" else float32)

        if self.requires_grad and _backward is None and self._dtype.category != 2:
            raise RuntimeError(
                "정수 텐서에는 기울기가 흐르지 않습니다. 미분은 실수에서만 정의됩니다 "
                "— `.float()` 로 바꾸세요.")

    # ---- 기본 정보

    @property
    def shape(self):
        raw = _shape_of(self._h)
        if self._nhwc:
            n, h, w, c = raw
            return Size((n, c, h, w))
        return Size(raw)

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def dim(self):
        return self.ndim

    def numel(self):
        shape = self.shape
        return int(_np.prod(shape)) if shape else 1

    def numpy(self):
        arr = _to_np(self._h)
        if arr.dtype != self._dtype.np:
            arr = arr.astype(self._dtype.np)      # 라벨이 말하는 dtype 으로 내보낸다
        return arr.transpose(0, 3, 1, 2) if self._nhwc else arr

    # ---- 형 변환

    def to(self, *args, **kwargs):
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, dtype):
                return self._cast_to(a)
            if isinstance(a, str) and a != "cpu":
                _unsupported(f"장치 '{a}'")
        return self

    def _cast_to(self, dt):
        if dt is self._dtype:
            return self
        handle = self._h
        if dt is bool_:
            handle = _tf.notEqual(handle, 0.0)
        elif self._dtype is bool_:
            handle = _tf.cast(handle, "float32")
        elif dt is int64:
            handle = _tf.floor(handle)            # torch 의 정수 변환은 0 쪽으로 자른다
        out = Tensor(handle if handle is not self._h else _tf.clone(handle), dt=dt)
        out._nhwc = self._nhwc
        return out

    def float(self):
        return self._cast_to(float32)

    def long(self):
        return self._cast_to(int64)

    int = long

    def bool(self):
        return self._cast_to(bool_)

    def tolist(self):
        return self.numpy().tolist()

    def item(self):
        if self.numel() != 1:
            raise RuntimeError(_like_torch(
                f"값이 {self.numel()}개인 텐서는 하나의 숫자로 바꿀 수 없습니다. "
                "`.tolist()` 나 인덱싱을 쓰세요.",
                f"a Tensor with {self.numel()} elements cannot be converted to Scalar"))
        return self.numpy().reshape(-1)[0].item()

    def detach(self):
        # **복제해야 한다.** 손잡이를 나눠 가지면 둘 중 하나가 사라질 때 다른 쪽의
        # 버퍼까지 놓아버린다. tf.clone 은 데이터를 공유하고 참조만 하나 더 든다.
        out = Tensor(_tf.clone(self._h), dt=self._dtype)
        out._nhwc = self._nhwc          # 레이아웃도 같이 물려준다 — 안 그러면 속이 밖으로 샌다
        out._derived = True             # torch 의 detach 는 저장소를 **공유한다** — 우리는 못 한다
        return out

    def dispose(self):
        """GPU 버퍼를 놓는다."""
        if self._h is not None:
            try:
                self._h.dispose()
            except Exception:                                        # noqa: BLE001
                pass          # 이미 놓인 것 — 두 번 놓는 것은 실패가 아니다
            self._h = None

    def __del__(self):
        """파이썬이 이 텐서를 놓을 때 GPU 버퍼도 같이 놓는다.

        TF.js 는 수동 해제라 안 걸어두면 **스텝마다 샌다**(실측: 학습 한 스텝에
        텐서 118개). CPython 은 참조 세기라 순환만 없으면 즉시 떨어지고, 이 그래프는
        자식이 부모를 가리킬 뿐 반대가 없어서 순환이 없다.
        """
        try:
            self.dispose()
        except Exception:                                            # noqa: BLE001
            pass          # 종료 중일 수 있다. 여기서 시끄러워봐야 얻을 것이 없다

    def __repr__(self):
        return _tensor_repr(self)

    # ---- 그래프

    def _make(self, handle, parents, backward, op=None, dt=None):
        needs = _grad_enabled and any(p.requires_grad for p in parents)
        out = Tensor(handle, requires_grad=needs,
                     _parents=parents if needs else (),
                     _backward=backward if needs else None,
                     dt=dt or self._dtype)
        out._op = op if needs else None
        # 원소별 연산은 레이아웃을 그대로 물려준다. 랭크가 4 에서 벗어나면 뜻이 없다.
        out._nhwc = self._nhwc and len(_shape_of(handle)) == 4
        # no_grad 안에서도 표시한다 — 그래프가 아니라 **어디서 나왔는가**의 문제다.
        out._derived = True
        return out

    def backward(self, gradient=None, retain_graph=False):
        if not self.requires_grad:
            raise RuntimeError(_like_torch(
                "requires_grad 가 아닌 텐서에는 backward() 를 부를 수 없습니다.",
                "element 0 of tensors does not require grad and does not have a grad_fn"))
        if self._freed:
            raise RuntimeError(_like_torch(
                "이미 backward() 를 부른 그래프입니다. 한 번 되짚으면 그래프를 놓습니다 — "
                "다시 계산하거나 `backward(retain_graph=True)` 를 쓰세요.",
                "Trying to backward through the graph a second time"))
        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError(_like_torch(
                    "값이 하나가 아닌 텐서에는 gradient 를 줘야 합니다. "
                    "보통은 손실을 스칼라로 만든 뒤 부릅니다.",
                    "grad can be implicitly created only for scalar outputs"))
            gradient = _tf.onesLike(self._h)
        elif isinstance(gradient, Tensor):
            gradient = gradient._h          # torch 처럼 텐서를 받는다

        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        visit(self)

        grads = {id(self): gradient}
        for t in reversed(order):
            g = grads.get(id(t))
            if g is None:
                continue
            if t._backward is None:                     # 잎 — 여기에 쌓는다
                if t.requires_grad:
                    # **스코프를 넘겨 살린다.** 안 그러면 `with scope():` 안에서 backward 한
                    # 뒤 밖에서 `p.grad` 를 읽으면 이미 놓인 버퍼다. torch 는 zero_grad
                    # 할 때까지 기울기를 들고 있고, 여기도 그래야 한다.
                    total = g if t.grad is None else _tf.add(t.grad._h, g)
                    t.grad = Tensor(_keep(total))
                continue
            for parent, pg in zip(t._parents, t._backward(g)):
                if pg is None:
                    continue
                # 부모의 **속 모양**으로 되돌린다. `parent.shape` 는 torch 순서라
                # 부모가 NHWC 를 들고 있으면 축이 어긋난다.
                pg = _unbroadcast(pg, _shape_of(parent._h))
                grads[id(parent)] = pg if id(parent) not in grads else _tf.add(grads[id(parent)], pg)

        if not retain_graph:
            for t in order:
                if t._backward is not None:
                    t._freed = True

    # ---- 산술

    def _binary(self, o, forward, back, op, force=None):
        """레이아웃을 맞추고 dtype 을 승격한 뒤 계산한다.

        승격은 **라벨만 바꾼다** — 수치 저장이 전부 float32 라서 캐스팅이 필요 없다.
        불리언만 저장이 달라서, 수치 연산에 들어갈 때 실수로 바꾼다.
        """
        target = _result_dtype(self, o)
        a, b = _align(self, _wrap(o))
        out_dt = force(target) if force else target
        ah, bh = _storage_for(a, out_dt), _storage_for(b, out_dt)
        try:
            handle = forward(ah, bh)
        except Exception:                                            # noqa: BLE001
            # TF.js 는 자기 말로 던진다. torch 를 쓰던 사람이 검색할 수 있는 문구로 바꾼다.
            _broadcast_error(a.shape, b.shape)
        return a._make(handle, (a, b), lambda g: back(g, ah, bh), op, dt=out_dt)

    def __add__(self, o):
        if _both_bool(self, o):
            return self._binary(o, _tf.logicalOr, lambda g, x, y: (g, g), "AddBackward0")
        return self._binary(o, _tf.add, lambda g, x, y: (g, g), "AddBackward0")

    __radd__ = __add__

    def __sub__(self, o):
        _no_bool_subtract(self, o)
        return self._binary(o, _tf.sub, lambda g, x, y: (g, _tf.neg(g)), "SubBackward0")

    def __rsub__(self, o):
        return _wrap(o).__sub__(self)

    def __mul__(self, o):
        if _both_bool(self, o):
            return self._binary(o, _tf.logicalAnd, lambda g, x, y: (g, g), "MulBackward0")
        return self._binary(o, _tf.mul,
                            lambda g, x, y: (_tf.mul(g, y), _tf.mul(g, x)), "MulBackward0")

    __rmul__ = __mul__

    def __truediv__(self, o):
        # torch 의 나눗셈은 정수끼리여도, 불리언끼리여도 **기본 실수형**을 낸다.
        return self._binary(
            o, _tf.div,
            lambda g, x, y: (_tf.div(g, y),
                             _tf.neg(_tf.div(_tf.mul(g, x), _tf.mul(y, y)))),
            "DivBackward0", force=lambda dt: float32)

    def __rtruediv__(self, o):
        return _wrap(o).__truediv__(self)

    def __neg__(self):
        return self._make(_tf.neg(self._h), (self,), lambda g: (_tf.neg(g),), "NegBackward0")

    def __pow__(self, p):
        if isinstance(p, Tensor):
            _unsupported("텐서 지수")
        return self._make(
            _tf.pow(self._h, float(p)), (self,),
            lambda g: (_tf.mul(g, _tf.mul(float(p), _tf.pow(self._h, float(p) - 1))),),
            "PowBackward0")

    def __matmul__(self, o):
        self, o = _canonical(self), _canonical(_wrap(o))
        sa, sb = self.shape, o.shape
        if len(sa) >= 2 and len(sb) >= 2 and sa[-1] != sb[-2]:
            left = "x".join(str(n) for n in sa[-2:])
            right = "x".join(str(n) for n in sb[-2:])
            raise RuntimeError(_like_torch(
                f"행렬곱의 모양이 안 맞습니다 ({left} @ {right}) — "
                f"앞의 열({sa[-1]})과 뒤의 행({sb[-2]})이 같아야 합니다.",
                f"mat1 and mat2 shapes cannot be multiplied ({left} and {right})"))
        return self._make(
            _tf.matMul(self._h, o._h), (self, o),
            lambda g: (_tf.matMul(g, o._h, False, True),
                       _tf.matMul(self._h, g, True, False)),
            "MmBackward0")

    def matmul(self, o):
        return self.__matmul__(o)

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        t = _canonical(self)          # 모양을 다시 짜는 것은 torch 순서에서만 뜻이 있다
        old = t.shape
        want = list(shape)
        if -1 not in want and int(_np.prod(want)) != t.numel():
            raise RuntimeError(_like_torch(
                f"모양 {want} 은 원소 {t.numel()}개짜리 텐서에 맞지 않습니다.",
                f"shape '{want}' is invalid for input of size {t.numel()}"))
        return t._make(_tf.reshape(t._h, _to_js(list(shape))), (t,),
                       lambda g: (_tf.reshape(g, _to_js(list(old))),), "ViewBackward0")

    def view(self, *shape):
        return self.reshape(*shape)

    def flatten(self, start_dim=0):
        return self.reshape(self.shape[:start_dim] + (-1,))

    def transpose(self, d0, d1):
        t = _canonical(self)
        perm = list(range(t.ndim))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        return t._make(_tf.transpose(t._h, _to_js(perm)), (t,),
                       lambda g: (_tf.transpose(g, _to_js(perm)),), "TransposeBackward0")

    @property
    def T(self):
        return self.transpose(-2, -1)

    # ---- 인덱싱

    def __getitem__(self, idx):
        """정수·슬라이스·정수목록·불리언 마스크. **그래프를 잇는다.**

        축마다 잘라내는 것을 겹쳐서 만든다 — 자르기의 역방향이 이미 0 채우기라
        역전파가 저절로 따라온다. 걸음이 1 이 아닌 슬라이스처럼 못 하는 것은
        근사하지 않고 거절한다.
        """
        t = _canonical(self)
        keys = idx if isinstance(idx, tuple) else (idx,)

        if len(keys) == 1:
            key = keys[0]
            if isinstance(key, Tensor) and key._dtype is bool_:
                return masked_select(t, key)
            if isinstance(key, (list, _np.ndarray)) or (
                    isinstance(key, Tensor) and key._dtype is int64):
                return index_select(t, 0, key)

        out, drop, axis = t, [], 0
        for key in keys:
            if isinstance(key, slice):
                start, stop, step = key.indices(out.shape[axis])
                if step != 1:
                    _unsupported("걸음(step)이 1 이 아닌 슬라이스")
                out = _slice_tensor(out, axis, start, max(0, stop - start))
                axis += 1
            elif isinstance(key, int):
                n = out.shape[axis]
                i = key + n if key < 0 else key
                if not 0 <= i < n:
                    raise IndexError(_like_torch(
                        f"인덱스 {key} 는 크기 {n} 인 축 {axis} 의 범위를 벗어납니다.",
                        f"index {key} is out of bounds for dimension {axis} with size {n}"))
                out = _slice_tensor(out, axis, i, 1)
                drop.append(axis)
                axis += 1
            else:
                _unsupported(f"인덱싱에 {type(key).__name__}")
        if drop:
            out = out.reshape(tuple(s for i, s in enumerate(out.shape) if i not in drop))
        return out

    def __setitem__(self, idx, value):
        """축 0 의 정수·슬라이스에만. 잘라서 새 값을 끼워 다시 붙인다.

        TF.js 텐서는 불변이라 제자리 수정이 없다 — 다시 만드는 수밖에 없다.
        """
        if self.requires_grad and _grad_enabled:
            raise RuntimeError(
                "기울기가 필요한 텐서에는 제자리 대입을 할 수 없습니다. "
                "`with torch.no_grad():` 안에서 하세요.")
        if self._derived:
            # torch 는 뷰가 **저장소를 공유**해서 뷰를 고치면 원본도 바뀐다.
            # TF.js 텐서는 불변이라 그렇게 할 수 없다. 조용히 다르게 도느니 멈춘다.
            _unsupported(
                "다른 텐서에서 나온 것(뷰·슬라이스·연산 결과)에 대입하는 것 — "
                "torch 는 저장소를 공유해 원본까지 바꾸는데 TF.js 텐서는 불변이라 "
                "여기서는 그럴 수 없습니다")
        t = _canonical(self)
        n = t.shape[0]
        if isinstance(idx, int):
            start, length = (idx + n if idx < 0 else idx), 1
        elif isinstance(idx, slice):
            a, b, step = idx.indices(n)
            if step != 1:
                _unsupported("걸음(step)이 1 이 아닌 슬라이스에 대입")
            start, length = a, max(0, b - a)
        else:
            _unsupported(f"{type(idx).__name__} 로 대입")

        # `_wrap(value)._h` 로 쓰면 안 된다 — 이름을 붙여 호출이 끝날 때까지 살려둔다.
        val = _wrap(value)
        target = list(t.shape)
        target[0] = length
        piece = _tf.broadcastTo(val._h, _to_js(target))
        parts = []
        if start:
            parts.append(_slice_along(t._h, 0, 0, start))
        parts.append(piece)
        if start + length < n:
            parts.append(_slice_along(t._h, 0, start + length, n - start - length))
        old = self._h
        self._h = _keep(_tf.concat(_to_js(parts), 0) if len(parts) > 1 else parts[0])
        self._nhwc = t._nhwc
        old.dispose()

    # ---- 제자리 갱신 — no_grad 안에서만 (진짜 torch 도 같은 규칙)

    def _inplace(self, fn, other):
        if self.requires_grad and _grad_enabled:
            raise RuntimeError(
                "기울기가 필요한 텐서를 제자리에서 바꿀 수 없습니다. "
                "`with torch.no_grad():` 안에서 하세요.")
        o = _wrap(other)
        old = self._h
        self._h = _keep(fn(self._h, _storage_for(o, self._dtype)))
        old.dispose()
        return self

    def __iadd__(self, o):
        return self._inplace(_tf.add, o)

    def __isub__(self, o):
        return self._inplace(_tf.sub, o)

    def __imul__(self, o):
        return self._inplace(_tf.mul, o)

    # ---- 비교 (기울기 없음)

    def _cmp(self, o, fn):
        # `fn(self._h, _wrap(o)._h)` 로 쓰면 안 된다. `._h` 를 꺼내는 순간 임시 텐서의
        # 참조가 0 이 되어 `__del__` 이 버퍼를 놓고, **그 뒤에** fn 이 불린다.
        # 이름을 붙여 호출이 끝날 때까지 살려둔다.
        a, b = _align(self, _wrap(o))
        out = Tensor(fn(a._h, b._h))
        out._nhwc = a._nhwc and len(_shape_of(out._h)) == 4
        return out

    def __gt__(self, o): return self._cmp(o, _tf.greater)
    def __ge__(self, o): return self._cmp(o, _tf.greaterEqual)
    def __lt__(self, o): return self._cmp(o, _tf.less)
    def __le__(self, o): return self._cmp(o, _tf.lessEqual)
    def __eq__(self, o): return self._cmp(o, _tf.equal)
    def __ne__(self, o): return self._cmp(o, _tf.notEqual)

    # `__eq__` 를 정의하면 파이썬이 해시를 지운다. 텐서는 값이 아니라 **자기 자신**으로
    # 식별되어야 한다 — 역전파의 방문 표시가 id 에 걸려 있다.
    def __hash__(self):
        return id(self)

    # ---- 축약

    def sum(self, dim=None, keepdim=False):
        # 축을 지정하지 않으면 레이아웃과 무관하다 — 그때는 되돌릴 이유가 없다.
        t = self if dim is None else _canonical(self)
        handle = _tf.sum(t._h) if dim is None else _tf.sum(t._h, dim, keepdim)

        def back(g):
            return (_tf.mul(_tf.onesLike(t._h),
                            _reshape_for_broadcast(g, t.shape, dim, keepdim)),)

        return t._make(handle, (t,), back,
                       "SumBackward0" if dim is None else "SumBackward1")

    def mean(self, dim=None, keepdim=False):
        t = self if dim is None else _canonical(self)
        n = t.numel() if dim is None else t.shape[dim]
        handle = _tf.mean(t._h) if dim is None else _tf.mean(t._h, dim, keepdim)

        def back(g):
            spread = _reshape_for_broadcast(g, t.shape, dim, keepdim)
            return (_tf.div(_tf.mul(_tf.onesLike(t._h), spread), float(n)),)

        return t._make(handle, (t,), back,
                       "MeanBackward0" if dim is None else "MeanBackward1")


# ---------------------------------------------------------------- 도우미

def result_type(a, b):
    """두 dtype 의 결과 타입 — torch 의 규칙.

    **범주**(bool < 정수 < 실수)로 먼저 가르고 **그 범주 안에서만** 올린다. 낮은 범주가
    높은 것을 끌어올리지 않는다. 이걸 numpy 에 맡기면 학습자가 틀린 규칙을 배운다.
    """
    cat = max(a.category, b.category)
    same = [d for d in (a, b) if d.category == cat]
    return max(same, key=lambda d: d.rank)


def _scalar_dtype(t_dtype, value):
    """파이썬 스칼라는 텐서보다 약하다.

    범주가 텐서보다 낮거나 같으면 텐서를 따르고, 높을 때만 그 범주의 기본형으로
    올라간다 — 정수 텐서 + 파이썬 float 가 float32 인 이유다.
    """
    cat = 0 if isinstance(value, bool) else (1 if isinstance(value, int) else 2)
    return t_dtype if cat <= t_dtype.category else _BY_CATEGORY[cat]


def _result_dtype(t, o):
    if isinstance(o, Tensor):
        return result_type(t._dtype, o._dtype)
    if isinstance(o, (bool, int, float)):
        return _scalar_dtype(t._dtype, o)
    return result_type(t._dtype, _dtype_of(o))


def _storage_for(t, target):
    """연산에 넣을 손잡이. 불리언 저장을 수치 연산에 쓰려면 실수로 바꾼다."""
    if target is bool_ or t._dtype is not bool_:
        return t._h
    return _tf.cast(t._h, "float32")


def _both_bool(t, o):
    return t._dtype is bool_ and isinstance(o, Tensor) and o._dtype is bool_


def _no_bool_subtract(t, o):
    """torch 는 불리언에 `-` 를 허용하지 않고 `~`·`^` 를 쓰라고 안내한다."""
    other = o._dtype if isinstance(o, Tensor) else (bool_ if isinstance(o, bool) else None)
    if t._dtype is bool_ or other is bool_:
        raise RuntimeError(
            "불리언 텐서에는 뺄셈(`-`)을 쓸 수 없습니다. `^` 나 `~` 를 쓰세요.\n"
            "(torch: Subtraction, the `-` operator, with a bool tensor is not supported. "
            "If you are trying to invert a mask use the `~` or `logical_not()` operator instead.)")


def _relayout(t, to_nhwc):
    """레이아웃을 바꾼다. **그래프 안에서** 하므로 역전파가 그냥 따라온다."""
    if t._nhwc == to_nhwc or len(_shape_of(t._h)) != 4:
        return t
    perm = _NCHW_TO_NHWC if to_nhwc else _NHWC_TO_NCHW
    inv = _NHWC_TO_NCHW if to_nhwc else _NCHW_TO_NHWC
    out = t._make(_tf.transpose(t._h, _to_js(perm)), (t,),
                  lambda g: (_tf.transpose(g, _to_js(inv)),), "LayoutBackward0")
    out._nhwc = to_nhwc
    return out


def _canonical(t):
    """torch 순서(NCHW)로 되돌린다. 레이아웃을 모르는 연산은 전부 이것을 먼저 부른다 —
    느릴 수는 있어도 **틀리지는 않는다.**"""
    return _relayout(_wrap(t), False)


def _align(a, b):
    """이항 연산의 두 짝을 같은 레이아웃으로.

    4차원끼리면 한쪽을 맞추고, 짝이 4차원이 아니면 **안전한 쪽(NCHW)으로 되돌린다** —
    1차원 편향 같은 것은 마지막 축에 붙는데, 그 축이 레이아웃마다 다르기 때문이다.
    스칼라는 어느 쪽이든 같아서 그냥 둔다.
    """
    ra, rb = len(_shape_of(a._h)), len(_shape_of(b._h))
    if ra == 4 and rb == 4:
        return a, (_relayout(b, a._nhwc) if a._nhwc != b._nhwc else b)
    if rb == 0 or ra == 0:
        return a, b
    return _relayout(a, False), _relayout(b, False)


def _wrap(x):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, bool):                       # bool 이 int 의 하위형이라 먼저 본다
        return Tensor(_to_tf(_np.asarray(x), bool_), dt=bool_)
    if isinstance(x, (int, float)):
        return Tensor(_tf.scalar(float(x)), dt=int64 if isinstance(x, int) else float32)
    arr = _np.asarray(x)
    dt = _dtype_of(arr)
    return Tensor(_to_tf(arr, dt), dt=dt)


def _reshape_for_broadcast(g, shape, dim, keepdim):
    """축약의 역방향 — 접힌 축을 되살려 원래 모양에 퍼질 수 있게 한다."""
    if dim is None or keepdim:
        return g
    return _tf.expandDims(g, dim)


def _unbroadcast(g, shape):
    """브로드캐스팅으로 늘어난 축을 되돌린다. 역전파의 필수 단계다."""
    gshape = _shape_of(g)
    while len(gshape) > len(shape):
        g = _tf.sum(g, 0)
        gshape = _shape_of(g)
    for i, n in enumerate(shape):
        if n == 1 and gshape[i] != 1:
            g = _tf.sum(g, i, True)
            gshape = _shape_of(g)
    return _tf.reshape(g, _to_js(list(shape)))


# ---------------------------------------------------------------- 만들기

def tensor(data, dtype=None, requires_grad=False):
    _reject_float64(dtype)
    if isinstance(data, Tensor):
        out = Tensor(_tf.clone(data._h), requires_grad,      # 손잡이를 나눠 갖지 않는다
                     dt=dtype or data._dtype)
        out._nhwc = data._nhwc
        return out
    arr = _np.asarray(data)
    dt = dtype or _dtype_of(arr)
    return Tensor(_to_tf(arr, dt), requires_grad, dt=dt)


def from_numpy(arr):
    arr = _np.asarray(arr)
    dt = _dtype_of(arr)
    return Tensor(_to_tf(arr, dt), dt=dt)


def zeros(*shape, dtype=None, requires_grad=False):
    _reject_float64(dtype)
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_tf.zeros(_to_js(list(shape))), requires_grad, dt=dtype or float32)


def ones(*shape, dtype=None, requires_grad=False):
    _reject_float64(dtype)
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_tf.ones(_to_js(list(shape))), requires_grad, dt=dtype or float32)


def arange(*args, dtype=None):
    _reject_float64(dtype)
    arr = _np.arange(*args)
    dt = dtype or _dtype_of(arr)
    return Tensor(_to_tf(arr, dt), dt=dt)


def randn(*shape, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_tf.randomNormal(_to_js(list(shape))), requires_grad)


# ---------------------------------------------------------------- 원소별
#
# TF.js 이름과 torch 이름이 갈리는 자리가 있어서 표로 둔다(matMul·notEqual 등).
# 미분이 정의되지 않는 것(sign·floor·ceil·round)은 기울기를 0 으로 둔다 — torch 도 그렇다.

def _unary(name, forward, derivative=None, keeps_dtype=False):
    """`keeps_dtype` 은 정수를 정수로 돌려주는 것들이다 — torch 에서 `abs`·`sign`·
    `floor`·`relu` 가 그렇고, `exp`·`log` 같은 것은 정수를 넣어도 실수를 준다."""
    def fn(t):
        t = _wrap(t)          # 원소별이라 레이아웃과 무관하다 — 되돌리면 안 된다
        dt = t._dtype if keeps_dtype else float32
        h = _storage_for(t, dt)
        out = forward(h)
        if derivative is None:
            return Tensor(out, dt=dt)
        return t._make(out, (t,), lambda g: (_tf.mul(g, derivative(h, out)),),
                       f"{name}Backward0", dt=dt)
    fn.__name__ = name
    return fn


_LN2 = float(_np.log(2.0))
_LN10 = float(_np.log(10.0))

exp = _unary("Exp", lambda x: _tf.exp(x), lambda x, o: o)
log = _unary("Log", lambda x: _tf.log(x), lambda x, o: _tf.div(1.0, x))
log2 = _unary("Log2", lambda x: _tf.div(_tf.log(x), _LN2),
              lambda x, o: _tf.div(1.0, _tf.mul(x, _LN2)))
log10 = _unary("Log10", lambda x: _tf.div(_tf.log(x), _LN10),
               lambda x, o: _tf.div(1.0, _tf.mul(x, _LN10)))
sqrt = _unary("Sqrt", lambda x: _tf.sqrt(x), lambda x, o: _tf.div(0.5, o))
rsqrt = _unary("Rsqrt", lambda x: _tf.rsqrt(x), lambda x, o: _tf.div(_tf.mul(-0.5, o), x))
square = _unary("Square", lambda x: _tf.square(x), lambda x, o: _tf.mul(2.0, x))
reciprocal = _unary("Reciprocal", lambda x: _tf.reciprocal(x), lambda x, o: _tf.neg(_tf.mul(o, o)))
abs = _unary("Abs", lambda x: _tf.abs(x), lambda x, o: _tf.sign(x), keeps_dtype=True)
sin = _unary("Sin", lambda x: _tf.sin(x), lambda x, o: _tf.cos(x))
cos = _unary("Cos", lambda x: _tf.cos(x), lambda x, o: _tf.neg(_tf.sin(x)))
tan = _unary("Tan", lambda x: _tf.tan(x), lambda x, o: _tf.add(1.0, _tf.mul(o, o)))
sinh = _unary("Sinh", lambda x: _tf.sinh(x), lambda x, o: _tf.cosh(x))
cosh = _unary("Cosh", lambda x: _tf.cosh(x), lambda x, o: _tf.sinh(x))
tanh = _unary("Tanh", lambda x: _tf.tanh(x), lambda x, o: _tf.sub(1.0, _tf.mul(o, o)))
erf = _unary("Erf", lambda x: _tf.erf(x),
             lambda x, o: _tf.mul(2.0 / float(_np.sqrt(_np.pi)), _tf.exp(_tf.neg(_tf.square(x)))))
relu = _unary("Relu", lambda x: _tf.relu(x), lambda x, o: _tf.step(x), keeps_dtype=True)
sigmoid = _unary("Sigmoid", lambda x: _tf.sigmoid(x),
                 lambda x, o: _tf.mul(o, _tf.sub(1.0, o)))
# 계단 모양 — 미분이 거의 모든 곳에서 0 이다.
sign = _unary("Sign", lambda x: _tf.sign(x), keeps_dtype=True)
floor = _unary("Floor", lambda x: _tf.floor(x), keeps_dtype=True)
ceil = _unary("Ceil", lambda x: _tf.ceil(x), keeps_dtype=True)
round = _unary("Round", lambda x: _tf.round(x), keeps_dtype=True)


def neg(t):
    return -_wrap(t)


def prod(t, dim=None):
    t = _canonical(t)
    out = _tf.prod(t._h) if dim is None else _tf.prod(t._h, dim)
    return t._make(out, (t,), lambda g: (_tf.div(_tf.mul(g, out), t._h),), "ProdBackward0")


def count_nonzero(t, dim=None):
    t = _canonical(t)
    nz = _tf.cast(_tf.notEqual(t._h, 0.0), "float32")
    return Tensor(_tf.sum(nz) if dim is None else _tf.sum(nz, dim))


def matmul(a, b):
    return _wrap(a) @ _wrap(b)


def mm(a, b):
    return _wrap(a) @ _wrap(b)


# ---------------------------------------------------------------- 비교·클램프

def maximum(a, b):
    a, b = _align(_wrap(a), _wrap(b))
    pick = _tf.cast(_tf.greaterEqual(a._h, b._h), "float32")
    return a._make(_tf.maximum(a._h, b._h), (a, b),
                   lambda g: (_tf.mul(g, pick), _tf.mul(g, _tf.sub(1.0, pick))),
                   "MaximumBackward0")


def minimum(a, b):
    a, b = _align(_wrap(a), _wrap(b))
    pick = _tf.cast(_tf.lessEqual(a._h, b._h), "float32")
    return a._make(_tf.minimum(a._h, b._h), (a, b),
                   lambda g: (_tf.mul(g, pick), _tf.mul(g, _tf.sub(1.0, pick))),
                   "MinimumBackward0")


def clamp(t, min=None, max=None):
    t = _canonical(t)
    lo = -1e30 if min is None else float(min)
    hi = 1e30 if max is None else float(max)
    inside = _tf.cast(_tf.logicalAnd(_tf.greaterEqual(t._h, lo), _tf.lessEqual(t._h, hi)), "float32")
    return t._make(_tf.clipByValue(t._h, lo, hi), (t,),
                   lambda g: (_tf.mul(g, inside),), "ClampBackward0")


# ---------------------------------------------------------------- 선형대수

def dot(a, b):
    return (_wrap(a) * _wrap(b)).sum()


def outer(a, b):
    a, b = _wrap(a), _wrap(b)
    return reshape(a, (-1, 1)) @ reshape(b, (1, -1))


def reshape(t, shape):
    t = _canonical(t)
    old = t.shape
    return t._make(_tf.reshape(t._h, _to_js(list(shape))), (t,),
                   lambda g: (_tf.reshape(g, _to_js(list(old))),), "ViewBackward0")


def diag(t):
    """torch 의 `diag` 는 **행렬에서 대각을 뽑는다.** TF.js 의 `diag` 는 반대로
    벡터에서 행렬을 만든다 — 이름이 같고 뜻이 반대라, 그대로 부르면 조용히 다른 값이 나온다."""
    t = _canonical(t)
    n = t.shape[0]
    eye = _tf.eye(n)
    return Tensor(_tf.sum(_tf.mul(t._h, eye), 1))


def trace(t):
    t = _canonical(t)
    n = t.shape[0]
    return Tensor(_tf.sum(_tf.mul(t._h, _tf.eye(n))))


def norm(t, p=2, dim=None):
    t = _canonical(t)
    if p == 1:
        return abs(t).sum(dim=dim)
    return (t * t).sum(dim=dim) ** 0.5


def cumsum(t, dim):
    t = _canonical(t)
    return t._make(_tf.cumsum(t._h, dim), (t,),
                   lambda g: (_tf.reverse(_tf.cumsum(_tf.reverse(g, dim), dim), dim),),
                   "CumsumBackward0")


def cumprod(t, dim):
    t = _canonical(t)
    return Tensor(_tf.cumprod(t._h, dim))


# ---------------------------------------------------------------- 뽑기·모양

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


def topk(t, k, dim=-1, largest=True):
    t = _canonical(t)
    _last_axis_only(t, dim, "topk")
    if not largest:
        _unsupported("topk(largest=False)")
    idx = _tf.topk(t._h, k).indices
    return _ValuesIndices(_pick_last(t, idx), Tensor(idx))


def sort(t, dim=-1, descending=False):
    """TF.js 에는 정렬이 없다. `topk` 로 전부 뽑으면 내림차순이므로, 오름차순은 뒤집는다."""
    t = _canonical(t)
    _last_axis_only(t, dim, "sort")
    idx = _tf.topk(t._h, t.shape[-1]).indices
    if not descending:
        idx = _tf.reverse(idx, -1)
    return _ValuesIndices(_pick_last(t, idx), Tensor(idx))


def unique(t, sorted=True, return_counts=False):
    """**TF.js 의 WebGPU 백엔드에 `Unique` 커널이 없다**(실측: Kernel not registered).

    그리고 결과의 크기가 값에 따라 정해지므로 GPU 에 두기도 어렵다. 읽어와서 numpy 로
    하고 다시 올린다 — 동기화 한 번을 무는 대신 값이 맞는다. 학습 경로에는 안 쓰인다.
    """
    values = _np.unique(_wrap(t).numpy())
    return Tensor(_to_tf(values))


def masked_select(t, mask):
    """골라낸 개수가 값에 따라 달라진다. TF.js 의 `booleanMask` 는 **비동기**라
    동기 API 를 지키려면 쓸 수 없다. unique 와 같은 이유로 읽어와서 처리한다."""
    t = _canonical(t)
    m = mask.numpy() if isinstance(mask, Tensor) else _np.asarray(mask)
    return Tensor(_to_tf(t.numpy()[m.astype(bool)]))


def median(t, dim=None):
    """torch 는 원소가 짝수일 때 **가운데 둘 중 작은 쪽**을 준다."""
    t = _canonical(t)
    if dim is None:
        n = t.numel()
        flat = _tf.reshape(t._h, _to_js([n]))
        asc = _tf.reverse(_tf.topk(flat, n).values, -1)
        picked = _tf.slice(asc, _to_js([(n - 1) // 2]), _to_js([1]))
        return Tensor(_tf.reshape(picked, _to_js([])))      # torch 는 0차원을 준다
    _last_axis_only(t, dim, "median")
    if t.ndim != 2:
        _unsupported("median(2차원이 아닌 것에 dim 을 준 경우)")
    rows, n = t.shape
    order = _tf.reverse(_tf.topk(t._h, n).indices, -1)          # 오름차순 자리 번호
    idx = _tf.slice(order, _to_js([0, (n - 1) // 2]), _to_js([rows, 1]))
    # **진짜 번호를 준다.** 예전에는 0 으로 채워 돌려줬는데, 값만 맞고 번호는 거짓이었다.
    #
    # 번호는 int32 그대로 둔다. `tf.cast(int32 → float32)` 는 WebGPU 에서 dtype 라벨만
    # 바꾸고 **비트를 안 바꾼다**(실측: 2 가 2.8e-45 로 읽힌다). torch 도 번호는 정수다.
    return _ValuesIndices(_pick_last(t, idx).reshape(rows),
                          Tensor(_tf.reshape(idx, _to_js([rows]))))


def flip(t, dims):
    t = _canonical(t)
    dims = [dims] if isinstance(dims, int) else list(dims)
    return t._make(_tf.reverse(t._h, _to_js(dims)), (t,),
                   lambda g: (_tf.reverse(g, _to_js(dims)),), "FlipBackward0")


def roll(t, shifts, dims=None):
    """TF.js 에 `roll` 이 없다. 잘라서 순서를 바꿔 붙인다."""
    t = _canonical(t)
    axis = 0 if dims is None else (dims if isinstance(dims, int) else dims[0])
    n = t.shape[axis]
    s = int(shifts) % n
    if s == 0:
        return Tensor(_tf.clone(t._h))
    head = _slice_along(t._h, axis, n - s, s)
    tail = _slice_along(t._h, axis, 0, n - s)
    return Tensor(_tf.concat(_to_js([head, tail]), axis))


def _slice_along(handle, axis, start, length):
    shape = _shape_of(handle)
    begin = [0] * len(shape)
    size = list(shape)
    begin[axis], size[axis] = start, length
    return _tf.slice(handle, _to_js(begin), _to_js(size))


def _slice_tensor(t, dim, start, length):
    """잘라내되 **그래프를 잇는다.** 역방향은 잘라낸 자리 밖을 0 으로 채우는 것이다."""
    shape = t.shape
    pairs = [[0, 0] for _ in shape]
    pairs[dim] = [start, shape[dim] - start - length]
    return t._make(_slice_along(t._h, dim, start, length), (t,),
                   lambda g: (_tf.pad(g, _to_js(pairs)),), "SliceBackward0")


def narrow(t, dim, start, length):
    return _slice_tensor(_canonical(t), dim, start, length)


def split(t, size, dim=0):
    t = _canonical(t)
    n = t.shape[dim]
    sizes = size if isinstance(size, (list, tuple)) else \
        [size] * (n // size) + ([n % size] if n % size else [])
    out, start = [], 0
    for sz in sizes:
        out.append(_slice_tensor(t, dim, start, sz))
        start += sz
    return tuple(out)


def chunk(t, chunks, dim=0):
    t = _canonical(t)
    n = t.shape[dim]
    return split(t, -(-n // chunks), dim)


def unbind(t, dim=0):
    t = _canonical(t)
    shape = t.shape
    rest = tuple(s for i, s in enumerate(shape) if i != dim)
    return tuple(_slice_tensor(t, dim, i, 1).reshape(rest) for i in range(shape[dim]))


def gather(t, dim, index):
    """torch 의 `gather` — 원소마다 자리를 고른다. TF.js 의 `gather` 는 축을 통째로
    뽑는 다른 연산이라 그대로 못 쓴다.

    자리를 원-핫으로 만들어 곱하고 접는다. 그러면 **역전파가 그냥 따라온다** —
    뽑기만 하고 그래프를 끊으면 뽑은 자리로 기울기가 안 가고, 분류 손실이 통째로
    미분 불가가 된다(실제로 그랬다).
    """
    t = _canonical(t)
    if t.ndim != 2 or dim != 1:
        _unsupported("gather(2차원 · dim=1 이 아닌 것)")
    rows, cols = t.shape
    idx32 = _to_int32(index)
    k = _shape_of(idx32)[1]
    onehot = _tf.cast(_tf.oneHot(_tf.reshape(idx32, _to_js([rows * k])), cols), "float32")
    onehot = _tf.reshape(onehot, _to_js([rows, k, cols]))

    picked = _tf.sum(_tf.mul(onehot, _tf.reshape(t._h, _to_js([rows, 1, cols]))), 2)

    def back(g):
        return (_tf.sum(_tf.mul(onehot, _tf.reshape(g, _to_js([rows, k, 1]))), 1),)

    return t._make(picked, (t,), back, "GatherBackward0")


def _to_int32(index):
    handle = index._h if isinstance(index, Tensor) else _to_tf(_np.asarray(index))
    return _tf.cast(handle, "int32")


def index_select(t, dim, index):
    """원-핫 행렬을 곱해서 뽑는다 — 그래야 역전파가 따라온다."""
    t = _canonical(t)
    if dim != 0:
        _unsupported("index_select(dim=0 이 아닌 것)")
    shape = t.shape
    n = shape[0]
    idx32 = _tf.reshape(_to_int32(index), _to_js([-1]))
    k = _shape_of(idx32)[0]
    onehot = _tf.cast(_tf.oneHot(idx32, n), "float32")           # (k, n)
    rest = int(_np.prod(shape[1:])) if len(shape) > 1 else 1
    flat = t.reshape(n, rest)
    picked = flat._make(_tf.matMul(onehot, flat._h), (flat,),
                        lambda g: (_tf.matMul(onehot, g, True, False),),
                        "IndexSelectBackward0")
    return picked.reshape((k,) + tuple(shape[1:]))


# ---------------------------------------------------------------- nn.functional

def softmax(t, dim=-1):
    t = _canonical(t)
    out = _tf.softmax(t._h, dim)

    def back(g):
        s = _tf.sum(_tf.mul(g, out), dim, True)
        return (_tf.mul(out, _tf.sub(g, s)),)

    return t._make(out, (t,), back, "SoftmaxBackward0")


def log_softmax(t, dim=-1):
    t = _canonical(t)
    out = _tf.logSoftmax(t._h, dim)
    soft = _tf.exp(out)

    def back(g):
        return (_tf.sub(g, _tf.mul(soft, _tf.sum(g, dim, True))),)

    return t._make(out, (t,), back, "LogSoftmaxBackward0")


def leaky_relu(t, negative_slope=0.01):
    t = _canonical(t)
    pick = _tf.cast(_tf.greater(t._h, 0.0), "float32")
    return t._make(
        _tf.leakyRelu(t._h, float(negative_slope)), (t,),
        lambda g: (_tf.mul(g, _tf.add(pick, _tf.mul(_tf.sub(1.0, pick), float(negative_slope)))),),
        "LeakyReluBackward0")


def elu(t, alpha=1.0):
    t = _canonical(t)
    out = _tf.elu(t._h)
    pick = _tf.cast(_tf.greater(t._h, 0.0), "float32")
    return t._make(
        out, (t,),
        lambda g: (_tf.mul(g, _tf.add(pick, _tf.mul(_tf.sub(1.0, pick),
                                                    _tf.add(out, float(alpha))))),),
        "EluBackward0")


def silu(t):
    """x·σ(x). Swish 라고도 한다."""
    t = _canonical(t)
    sig = _tf.sigmoid(t._h)
    return t._make(
        _tf.mul(t._h, sig), (t,),
        lambda g: (_tf.mul(g, _tf.mul(sig, _tf.add(1.0, _tf.mul(t._h, _tf.sub(1.0, sig))))),),
        "SiluBackward0")


_SQRT2 = float(_np.sqrt(2.0))
_SQRT2PI = float(_np.sqrt(2.0 * _np.pi))


def gelu(t):
    """torch 의 기본 gelu(정확형) — 0.5·x·(1 + erf(x/√2)). TF.js 에 erf 가 있다."""
    t = _canonical(t)
    ope = _tf.add(1.0, _tf.erf(_tf.div(t._h, _SQRT2)))

    def back(g):
        bell = _tf.div(_tf.exp(_tf.neg(_tf.div(_tf.square(t._h), 2.0))), _SQRT2PI)
        return (_tf.mul(g, _tf.add(_tf.mul(0.5, ope), _tf.mul(t._h, bell))),)

    return t._make(_tf.mul(0.5, _tf.mul(t._h, ope)), (t,), back, "GeluBackward0")


def one_hot(t, num_classes=-1):
    t = _canonical(t)
    depth = int(t.numpy().max()) + 1 if num_classes == -1 else int(num_classes)
    return Tensor(_tf.oneHot(_to_int32(t), depth))


def pad(x, padding, value=0.0):
    """마지막 차원부터 (앞, 뒤) 순으로 받는다 — torch 의 규칙이다."""
    x = _canonical(x)
    old = x.shape
    pairs = [[0, 0] for _ in range(x.ndim)]
    for i in range(0, len(padding), 2):
        pairs[-(i // 2 + 1)] = [int(padding[i]), int(padding[i + 1])]

    def back(g):
        return (_tf.slice(g, _to_js([p[0] for p in pairs]), _to_js(list(old))),)

    return x._make(_tf.pad(x._h, _to_js(pairs), float(value)), (x,), back, "PadBackward0")


def normalize(x, p=2, dim=1, eps=1e-12):
    x = _canonical(x)
    denom = norm(x, p=p, dim=dim)
    return x / maximum(unsqueeze(denom, dim), _wrap(eps))


def unsqueeze(t, dim):
    t = _canonical(t)
    old = t.shape
    return t._make(_tf.expandDims(t._h, dim), (t,),
                   lambda g: (_tf.reshape(g, _to_js(list(old))),), "UnsqueezeBackward0")


def cosine_similarity(a, b, dim=1, eps=1e-8):
    a, b = _wrap(a), _wrap(b)
    return (a * b).sum(dim=dim) / maximum(norm(a, dim=dim) * norm(b, dim=dim), _wrap(eps))


def l1_loss(pred, target):
    return abs(_wrap(pred) - _wrap(target)).mean()


def mse_loss(pred, target):
    diff = _wrap(pred) - _wrap(target)
    return (diff * diff).mean()


def smooth_l1_loss(pred, target, beta=1.0):
    """작은 오차는 제곱, 큰 오차는 절댓값. 이상치에 덜 흔들린다."""
    diff = _wrap(pred) - _wrap(target)
    small = _tf.cast(_tf.less(_tf.abs(diff._h), float(beta)), "float32")
    quad = diff * diff * (0.5 / float(beta))
    lin = abs(diff) - 0.5 * float(beta)
    return (Tensor(small) * quad + Tensor(_tf.sub(1.0, small)) * lin).mean()


def nll_loss(log_probs, target):
    log_probs = _wrap(log_probs)
    rows = log_probs.shape[0]
    idx = _tf.reshape(_to_int32(target), _to_js([rows, 1]))
    picked = gather(log_probs, 1, Tensor(_tf.cast(idx, "float32")))
    return -picked.mean()


def cross_entropy(logits, target):
    return nll_loss(log_softmax(logits, dim=-1), target)


# ---------------------------------------------------------------- 합성곱
#
# torch 는 NCHW, TF.js 는 NHWC 다. 축을 바꿔 넣고 되돌린다.
#
# **역방향을 직접 짠다.** TF.js 의 conv 역방향 커널은 순방향의 1/26 이고(S0 실측),
# 같은 계산을 **순방향 conv 로 다시 쓰면 33배 빠르다**. 그래서 `tf.grad` 에 안 맡긴다 —
# 이 라이브러리가 자체 테이프를 드는 이유가 이것이다.

def _to_nhwc(h):
    return _tf.transpose(h, _to_js([0, 2, 3, 1]))


def _to_nchw(h):
    return _tf.transpose(h, _to_js([0, 3, 1, 2]))


def _dilate(handle, stride, extra):
    """기울기 사이에 0 을 끼운다.

    스트라이드가 1보다 크면 순방향이 칸을 건너뛴 것이므로, 역방향에서는 그 자리를
    0 으로 되살려야 스트라이드 1 짜리 conv 로 계산할 수 있다. `extra` 는 입력 크기가
    스트라이드로 나누어떨어지지 않을 때 남는 칸이다.

    처음에는 6차원으로 펴서 `pad` 로 끼웠는데, **모양은 맞고 값이 전부 0 으로 나왔다.**
    던지지도 않았다 — 6차원 경로가 조용히 0 을 준다. 그래서 5차원을 넘지 않게
    `stack` 으로 다시 짰다.
    """
    n, oh, ow, f = _shape_of(handle)

    zeros_h = _tf.zerosLike(handle)
    stacked = _tf.stack(_to_js([handle] + [zeros_h] * (stride - 1)), 2)   # (n,oh,s,ow,f)
    x = _tf.reshape(stacked, _to_js([n, oh * stride, ow, f]))

    zeros_w = _tf.zerosLike(x)
    stacked = _tf.stack(_to_js([x] + [zeros_w] * (stride - 1)), 3)        # (n,oh*s,ow,s,f)
    x = _tf.reshape(stacked, _to_js([n, oh * stride, ow * stride, f]))

    keep_h = (oh - 1) * stride + 1 + extra
    keep_w = (ow - 1) * stride + 1 + extra
    return _tf.slice(x, _to_js([0, 0, 0, 0]), _to_js([n, keep_h, keep_w, f]))


def conv2d(x, weight, bias=None, stride=1, padding=0):
    x, weight = _wrap(x), _wrap(weight)
    N, C, H, W = x.shape
    F, C2, KH, KW = weight.shape
    if C != C2:
        raise RuntimeError(f"채널이 안 맞습니다: 입력 {C}, 필터 {C2}")
    if KH != KW:
        _unsupported("정사각형이 아닌 커널")

    # **들어올 때 한 번만 바꾼다.** 이미 NHWC 면 그냥 지나가고, 결과도 NHWC 로 나가서
    # 다음 conv·BN·활성까지 전치 없이 이어진다.
    xin = _relayout(x, True)
    xh = xin._h
    wh = _tf.transpose(weight._h, _to_js([2, 3, 1, 0]))          # (F,C,KH,KW) → (KH,KW,C,F)
    out = _tf.conv2d(xh, wh, stride, padding)
    bias_t = _wrap(bias) if bias is not None else None
    if bias_t is not None:
        # NHWC 는 마지막 축이 채널이라 1차원 편향이 그대로 붙는다. 모양을 만들 필요가 없다.
        out = _tf.add(out, bias_t._h)
    extra = (H + 2 * padding - KH) % stride

    def back(gd):                                                # gd 는 NHWC 로 온다
        if stride > 1:
            gd = _dilate(gd, stride, extra)

        # dx — 커널을 공간 반전하고 입출력 채널을 바꾸면 **순방향 conv 가 된다**
        wflip = _tf.transpose(_tf.reverse(wh, _to_js([0, 1])), _to_js([0, 1, 3, 2]))
        dx = _tf.conv2d(gd, wflip, 1, KH - 1 - padding)

        # dw — 배치와 채널을 맞바꾸고 기울기를 필터로 쓰면 역시 순방향 conv 가 된다
        xpad = _tf.pad(xh, _to_js([[0, 0], [padding, padding],
                                   [padding, padding], [0, 0]]))
        xt = _tf.transpose(xpad, _to_js([3, 1, 2, 0]))           # (C, H', W', N)
        gt = _tf.transpose(gd, _to_js([1, 2, 0, 3]))             # (OH, OW, N, F)
        dw = _tf.transpose(_tf.conv2d(xt, gt, 1, "valid"), _to_js([3, 0, 1, 2]))

        if bias is None:
            return (dx, dw)
        return (dx, dw, _tf.sum(gd, _to_js([0, 1, 2])))

    parents = (xin, weight) if bias_t is None else (xin, weight, bias_t)
    return xin._make(out, parents, back, "ConvolutionBackward0")


def max_pool2d(x, kernel_size, stride=None):
    """역방향은 TF.js 에 맡긴다.

    conv 와 달리 풀링은 전체 계산에서 차지하는 몫이 작고, 최댓값이 **어느 자리에**
    있었는지를 우리가 다시 만들면 동점일 때 torch 와 갈린다. 정확한 쪽을 고른다.
    """
    xin = _relayout(_wrap(x), True)
    stride = stride or kernel_size
    xh = xin._h
    ksize, strides = _to_js([kernel_size, kernel_size]), _to_js([stride, stride])
    out = _tf.maxPool(xh, ksize, strides, "valid")

    def back(g):
        # 파이썬 함수를 JS 로 넘길 때는 프록시를 직접 들고 있어야 한다. 그냥 넘기면
        # 호출이 끝나는 순간 파괴되는데, tf.grad 는 **나중에** 부른다.
        fn = _create_proxy(lambda t: _tf.maxPool(t, ksize, strides, "valid"))
        try:
            return (_tf.grad(fn)(xh, g),)
        finally:
            fn.destroy()

    return xin._make(out, (xin,), back, "MaxPool2DBackward0")


def batch_norm(x, weight, bias, eps=1e-5):
    """학습 모드 배치 정규화 — **역전파를 손으로 썼다.**

    우리 연산으로 조립하면 레이아웃마다 축이 달라져 다루기 까다롭고 커널도 는다.
    식은 알려진 그대로다.

        x̂ = (x-μ)/√(σ²+ε),   y = γ·x̂ + β
        dx = γ/(m√(σ²+ε)) · (m·dy − Σdy − x̂·Σ(dy·x̂))

    (μ, σ², 결과)를 돌려준다 — running 통계는 부르는 쪽이 갱신한다. torch 가 정규화에는
    편향 분산을, running_var 에는 비편향을 쓰는 것도 거기서 처리한다.
    """
    x, weight, bias = _wrap(x), _wrap(weight), _wrap(bias)
    raw = _shape_of(x._h)
    axes = _to_js([0, 1, 2] if x._nhwc else [0, 2, 3])
    m = float(raw[0] * (raw[1] * raw[2] if x._nhwc else raw[2] * raw[3]))
    bshape = _to_js([1, 1, 1, raw[3]] if x._nhwc else [1, raw[1], 1, 1])

    mu = _tf.mean(x._h, axes, True)
    centered = _tf.sub(x._h, mu)
    var = _tf.mean(_tf.square(centered), axes, True)
    inv = _tf.rsqrt(_tf.add(var, float(eps)))
    xhat = _tf.mul(centered, inv)
    gamma = _tf.reshape(weight._h, bshape)
    out = _tf.add(_tf.mul(xhat, gamma), _tf.reshape(bias._h, bshape))

    def back(g):
        dxhat = _tf.mul(g, gamma)
        s1 = _tf.sum(dxhat, axes, True)
        s2 = _tf.sum(_tf.mul(dxhat, xhat), axes, True)
        dx = _tf.mul(_tf.div(inv, m),
                     _tf.sub(_tf.sub(_tf.mul(m, dxhat), s1), _tf.mul(xhat, s2)))
        return (dx, _tf.sum(_tf.mul(g, xhat), axes), _tf.sum(g, axes))

    return x._make(out, (x, weight, bias), back, "NativeBatchNormBackward0"), mu, var


def adaptive_avg_pool2d(x, output_size=1):
    """출력 크기 1 만 지원한다 — ResNet 이 쓰는 것이 그것이고, 나머지는 거절한다.

    우리 연산으로 조립한다. 그래야 역전파가 그냥 따라온다.
    """
    if output_size not in (1, (1, 1)):
        _unsupported("adaptive_avg_pool2d(출력 크기가 1 이 아닌 것)")
    xin = _relayout(_wrap(x), True)
    _, h, w, _ = _shape_of(xin._h)
    out = _tf.mean(xin._h, _to_js([1, 2]), True)                 # (N,1,1,C)

    def back(g):
        return (_tf.div(_tf.mul(_tf.onesLike(xin._h), g), float(h * w)),)

    return xin._make(out, (xin,), back, "MeanBackward1")


def avg_pool2d(x, kernel_size, stride=None):
    xin = _relayout(_wrap(x), True)
    stride = stride or kernel_size
    ksize, strides = _to_js([kernel_size, kernel_size]), _to_js([stride, stride])
    out = _tf.avgPool(xin._h, ksize, strides, "valid")

    def back(g):
        fn = _create_proxy(lambda t: _tf.avgPool(t, ksize, strides, "valid"))
        try:
            return (_tf.grad(fn)(xin._h, g),)
        finally:
            fn.destroy()

    return xin._make(out, (xin,), back, "AvgPool2DBackward0")


class _Functional:
    softmax = staticmethod(softmax)
    log_softmax = staticmethod(log_softmax)
    relu = staticmethod(relu)
    sigmoid = staticmethod(sigmoid)
    tanh = staticmethod(tanh)
    leaky_relu = staticmethod(leaky_relu)
    elu = staticmethod(elu)
    silu = staticmethod(silu)
    gelu = staticmethod(gelu)
    one_hot = staticmethod(one_hot)
    pad = staticmethod(pad)
    normalize = staticmethod(normalize)
    cosine_similarity = staticmethod(cosine_similarity)
    l1_loss = staticmethod(l1_loss)
    mse_loss = staticmethod(mse_loss)
    smooth_l1_loss = staticmethod(smooth_l1_loss)
    nll_loss = staticmethod(nll_loss)
    cross_entropy = staticmethod(cross_entropy)
    avg_pool2d = staticmethod(avg_pool2d)
    conv2d = staticmethod(conv2d)
    max_pool2d = staticmethod(max_pool2d)
    adaptive_avg_pool2d = staticmethod(adaptive_avg_pool2d)


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
        bound = 1.0 / _np.sqrt(in_features)
        rng = _np.random.default_rng(0)
        self.weight = Parameter(
            rng.uniform(-bound, bound, (out_features, in_features)).astype(_np.float32))
        self.bias = Parameter(
            rng.uniform(-bound, bound, out_features).astype(_np.float32)) if bias else None

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
        rng = _np.random.default_rng(0)
        self.weight = Parameter(rng.uniform(
            -bound, bound,
            (out_channels, in_channels, kernel_size, kernel_size)).astype(_np.float32))
        self.bias = Parameter(
            rng.uniform(-bound, bound, out_channels).astype(_np.float32)) if bias else None

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")


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
        if self.training:
            out, mu, var = batch_norm(x, self.weight, self.bias, self.eps)
            raw = _shape_of(x._h)
            n = raw[0] * (raw[1] * raw[2] if x._nhwc else raw[2] * raw[3])
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

        bshape = [1, 1, 1, self.num_features] if x._nhwc else [1, self.num_features, 1, 1]
        mean_t = Tensor(_tf.reshape(self.running_mean._h, _to_js(bshape)))
        inv_t = Tensor(_tf.reshape(
            _tf.rsqrt(_tf.add(self.running_var._h, float(self.eps))), _to_js(bshape)))
        mean_t._nhwc = inv_t._nhwc = x._nhwc          # 이미 속 순서로 만들었다
        w = Tensor(_tf.reshape(self.weight._h, _to_js(bshape)))
        b = Tensor(_tf.reshape(self.bias._h, _to_js(bshape)))
        w._nhwc = b._nhwc = x._nhwc
        return (x - mean_t) * inv_t * w + b


class MSELoss(Module):
    def forward(self, pred, target):
        return mse_loss(pred, target)


class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        return cross_entropy(logits, target)


class _NN:
    functional = _Functional()
    Module = Module
    Parameter = Parameter
    Linear = Linear
    ReLU = ReLU
    Sigmoid = Sigmoid
    Tanh = Tanh
    GELU = GELU
    Sequential = Sequential
    Conv2d = Conv2d
    MaxPool2d = MaxPool2d
    AvgPool2d = AvgPool2d
    AdaptiveAvgPool2d = AdaptiveAvgPool2d
    Flatten = Flatten
    BatchNorm2d = BatchNorm2d
    MSELoss = MSELoss
    CrossEntropyLoss = CrossEntropyLoss


nn = _NN()


# ---------------------------------------------------------------- optim
#
# 갱신은 **GPU 에서** 한다. 파라미터를 읽어와 numpy 로 고치면 매 스텝 전량 왕복이
# 생기고, 그 순간 GPU 를 쓰는 의미가 사라진다(WEBGPU-DESIGN.md 8절 S3).

def _replace(state, key, handle):
    """옵티마이저 상태를 갈아끼우고 옛 버퍼를 놓는다.

    상태는 `Tensor` 가 아니라 손잡이라 파이썬 GC 가 안 봐준다 — 여기서 직접 놓지 않으면
    모멘텀·Adam 상태가 스텝마다 쌓인다.
    """
    old = state.get(key)
    state[key] = _keep(handle)
    if old is not None and old is not handle:
        try:
            old.dispose()
        except Exception:                                            # noqa: BLE001
            pass
    return handle


class Optimizer:
    def __init__(self, params, defaults):
        params = list(params)
        if params and isinstance(params[0], dict):
            self.param_groups = [dict(defaults, **g) for g in params]
            for g in self.param_groups:
                g["params"] = list(g["params"])
        else:
            self.param_groups = [dict(defaults, params=params)]
        self.state = {}
        self.defaults = defaults

    @property
    def params(self):
        return [p for g in self.param_groups for p in g["params"]]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def _state(self, p):
        return self.state.setdefault(id(p), {})

    def _assign(self, p, handle):
        """새 값으로 갈아끼우고 **옛 버퍼를 놓는다.** TF.js 는 수동 해제라
        안 놓으면 스텝마다 GPU 메모리가 는다."""
        old = p._h
        p._h = _keep(handle)          # 스코프를 나가도 파라미터는 살아야 한다
        old.dispose()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad._h
                if group["weight_decay"]:
                    g = _tf.add(g, _tf.mul(float(group["weight_decay"]), p._h))
                if group["momentum"]:
                    st = self._state(p)
                    buf = st.get("momentum_buffer")
                    # 첫 스텝에서 **복제해야 한다.** 그대로 두면 버퍼가 `p.grad` 의
                    # 손잡이를 물고, 다음 zero_grad 에서 grad 가 사라질 때 __del__ 이
                    # 그 버퍼까지 놓는다 — 두 번째 스텝이 죽은 손잡이를 읽는다.
                    g = (_tf.clone(g) if buf is None
                         else _tf.add(_tf.mul(float(group["momentum"]), buf), g))
                    _replace(st, "momentum_buffer", g)
                self._assign(p, _tf.sub(p._h, _tf.mul(float(group["lr"]), g)))


class Adam(Optimizer):
    decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _tf.zerosLike(p._h))
                st.setdefault("exp_avg_sq", _tf.zerosLike(p._h))
                st["step"] += 1
                g = p.grad._h
                if group["weight_decay"] and not self.decoupled:
                    g = _tf.add(g, _tf.mul(float(group["weight_decay"]), p._h))
                # 상태를 갈아끼울 때 **옛 버퍼를 놓는다.** 파라미터와 달리 이쪽은
                # 텐서가 아니라 손잡이라 파이썬 GC 가 안 봐준다.
                st["exp_avg"] = _replace(st, "exp_avg",
                                         _tf.add(_tf.mul(float(b1), st["exp_avg"]),
                                                 _tf.mul(1.0 - float(b1), g)))
                st["exp_avg_sq"] = _replace(st, "exp_avg_sq",
                                            _tf.add(_tf.mul(float(b2), st["exp_avg_sq"]),
                                                    _tf.mul(1.0 - float(b2), _tf.mul(g, g))))
                mh = _tf.div(st["exp_avg"], 1.0 - float(b1) ** st["step"])
                vh = _tf.div(st["exp_avg_sq"], 1.0 - float(b2) ** st["step"])
                new = _tf.sub(p._h, _tf.mul(float(group["lr"]),
                                            _tf.div(mh, _tf.add(_tf.sqrt(vh),
                                                                float(group["eps"])))))
                if group["weight_decay"] and self.decoupled:
                    new = _tf.sub(new, _tf.mul(float(group["lr"]) * float(group["weight_decay"]),
                                               p._h))
                self._assign(p, new)


class AdamW(Adam):
    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


class _Optim:
    Optimizer = Optimizer
    SGD = SGD
    Adam = Adam
    AdamW = AdamW


optim = _Optim()


class scope:
    """한 스텝 동안 만들어진 GPU 버퍼를 통째로 놓는다.

    파이썬 GC 는 `Tensor` 가 든 손잡이만 놓아준다. 그런데 **역전파 클로저가 붙들고 있는
    중간 버퍼**(gelu 의 `ope`, gather 의 `onehot` 같은 것)는 `Tensor` 가 아니라
    아무도 안 놓는다 — 실측으로 학습 스텝당 92.7개가 남았다.

    설계 문서 7절은 "backward() 시점에 묶으면 사용자 API 에 스코프를 노출하지 않아도
    된다"고 적었는데, **그 전제가 틀렸다.** 클로저가 든 것은 그래프를 훑어서 찾을 수
    없다. 그래서 노출한다 — 코어와 다른 한 줄이 생기지만, 새는 것보다 낫다.

        with torch.scope():
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

    파라미터와 옵티마이저 상태는 `tf.keep` 으로 살려두므로 스코프를 나가도 남는다.
    """

    def __enter__(self):
        _tf.engine().startScope()
        return self

    def __exit__(self, *exc):
        _tf.engine().endScope()
        return False


def _keep(handle):
    """스코프가 끝나도 살려둘 것. 파라미터와 옵티마이저 상태가 여기 해당한다."""
    try:
        return _tf.keep(handle)
    except Exception:                                                # noqa: BLE001
        return handle          # 스코프 밖이면 keep 이 필요 없다


class no_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev
        return False


# ---------------------------------------------------------------- utils.data
#
# 데이터는 **CPU(numpy)에 둔다.** CIFAR-10 을 통째로 GPU 에 올리면 614MB 이고,
# 배치 하나는 3MB 다. 매 배치 올리는 쪽이 싸고, GPU 메모리를 모델에 남긴다.

class TensorDataset:
    def __init__(self, *arrays):
        self.arrays = [_np.asarray(a) for a in arrays]

    def __len__(self):
        return len(self.arrays[0])


class DataLoader:
    """배치마다 GPU 로 올린다. 셔플은 CPU 에서 번호만 섞는다."""

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = _np.random.default_rng(seed)

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        order = self._rng.permutation(n) if self.shuffle else _np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            yield tuple(tensor(a[idx]) for a in self.dataset.arrays)


class _UtilsData:
    TensorDataset = TensorDataset
    DataLoader = DataLoader


class _Utils:
    data = _UtilsData()


utils = _Utils()


# ---------------------------------------------------------------- 내려받기·캐시

def _u8_to_np(view):
    out = _np.empty(int(view.length), dtype=_np.uint8)
    view.assign_to(out)
    return out


def _np_to_u8(arr):
    buf = _js.Uint8Array.new(arr.size)
    buf.assign(arr)
    return buf


async def _opfs_read(name):
    root = await _js.navigator.storage.getDirectory()
    handle = await root.getFileHandle(name)
    blob = await handle.getFile()
    return _u8_to_np(_js.Uint8Array.new(await blob.arrayBuffer()))


async def _opfs_write(name, arr):
    root = await _js.navigator.storage.getDirectory()
    opts = _to_js({"create": True}, dict_converter=_js.Object.fromEntries)
    handle = await root.getFileHandle(name, opts)
    writable = await handle.createWritable()
    await writable.write(_np_to_u8(arr))
    await writable.close()


async def fetch_cached(url, name=None):
    """받아서 OPFS 에 넣고, 다음부터는 넣어둔 것을 쓴다.

    **비동기다.** OPFS 에 동기 API 가 없다(워커 안에서만 있다). 다만 이것은 학습
    루프가 아니라 **준비 단계에서 한 번** 부르는 것이라, 스텝을 동기로 유지한다는
    약속은 그대로다.

    URL 은 부르는 쪽이 준다. 데이터셋 주소를 라이브러리에 박아두면 그것이 사라졌을 때
    라이브러리를 고쳐야 한다.
    """
    key = name or url.rsplit("/", 1)[-1]
    try:
        return await _opfs_read(key)
    except Exception:                                                # noqa: BLE001
        pass                       # 아직 없다 — 받아온다
    response = await _js.fetch(url)
    if not response.ok:
        raise RuntimeError(f"내려받기 실패 {response.status}: {url}")
    data = _u8_to_np(_js.Uint8Array.new(await response.arrayBuffer()))
    await _opfs_write(key, data)
    return data


async def cache_put(name, data):
    """받아온 바이트를 캐시에 직접 넣는다.

    **CIFAR-10 원본(`cs.toronto.edu`)은 CORS 헤더를 주지 않는다**(실측: 브라우저가
    차단한다). 그래서 `fetch_cached` 로는 못 받는다. 사용자가 파일을 골라 넣거나
    CORS 를 주는 미러에서 받은 바이트를 여기로 넣으면 그다음은 같다.
    """
    await _opfs_write(name, _np.asarray(data, dtype=_np.uint8))


async def cache_get(name):
    """캐시에 있는 바이트. 없으면 None."""
    try:
        return await _opfs_read(name)
    except Exception:                                                # noqa: BLE001
        return None


_CIFAR_RECORD = 1 + 3 * 32 * 32          # 라벨 1바이트 + 픽셀 3072바이트


def decode_cifar10(raw):
    """CIFAR-10 의 바이너리 한 덩이를 (x, y) 로 푼다.

    한 장이 3073 바이트다 — 라벨 1 바이트에 R·G·B 가 각각 1024 바이트씩 이어 붙는다.
    그 순서가 곧 (3, 32, 32) 이라 torch 의 NCHW 와 같다.
    """
    arr = _np.asarray(raw, dtype=_np.uint8)
    if arr.size % _CIFAR_RECORD:
        raise ValueError(
            f"CIFAR-10 바이너리가 아닙니다 — {arr.size} 바이트는 {_CIFAR_RECORD} 의 배수가 아닙니다")
    rows = arr.reshape(-1, _CIFAR_RECORD)
    y = rows[:, 0].astype(_np.int64)
    x = rows[:, 1:].reshape(-1, 3, 32, 32).astype(_np.float32) / 255.0
    return x, y


def backend():
    """지금 붙어 있는 TF.js 백엔드. 'webgpu' 가 아니면 GPU 로 돌고 있지 않다."""
    return str(_tf.getBackend())
