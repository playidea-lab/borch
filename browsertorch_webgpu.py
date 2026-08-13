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

## 아직 없는 것

S2 범위다. 원소별·축약·행렬곱과 그 역전파까지. conv·풀링·BatchNorm 은 S3,
옵티마이저의 GPU 화도 S3 다. 없는 것은 근사하지 않고 예외를 던진다.
"""

import numpy as _np

try:
    import js as _js
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


def _unsupported(what):
    raise BrowserTorchError(
        f"{what} 은(는) 아직 browsertorch-webgpu 에 없습니다. "
        "코어 `browsertorch` 나 자기 컴퓨터의 진짜 PyTorch 를 쓰세요."
    )


# ---------------------------------------------------------------- 경계

def _shape_of(handle):
    return tuple(int(n) for n in handle.shape)


def _to_tf(arr):
    """numpy → tf.Tensor. 평평하게 펴서 올리고 모양을 따로 준다."""
    arr = _np.ascontiguousarray(arr, dtype=_np.float32)
    buf = _js.Float32Array.new(arr.size)
    buf.assign(arr.reshape(-1))
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


# ---------------------------------------------------------------- Tensor

_grad_enabled = True


class Tensor:
    def __init__(self, handle, requires_grad=False, _parents=(), _backward=None):
        self._h = handle
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = _parents
        self._backward = _backward
        self._op = None

    # ---- 기본 정보

    @property
    def shape(self):
        return _shape_of(self._h)

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
        return _to_np(self._h)

    def tolist(self):
        return self.numpy().tolist()

    def item(self):
        if self.numel() != 1:
            raise RuntimeError(
                f"값이 {self.numel()}개인 텐서는 하나의 숫자로 바꿀 수 없습니다.\n"
                "(torch: a Tensor with more than one element cannot be converted to Scalar)")
        return self.numpy().reshape(-1)[0].item()

    def detach(self):
        # **복제해야 한다.** 손잡이를 나눠 가지면 둘 중 하나가 사라질 때 다른 쪽의
        # 버퍼까지 놓아버린다. tf.clone 은 데이터를 공유하고 참조만 하나 더 든다.
        return Tensor(_tf.clone(self._h))

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
        return f"tensor({self.numpy()!r})"

    # ---- 그래프

    def _make(self, handle, parents, backward, op=None):
        needs = _grad_enabled and any(p.requires_grad for p in parents)
        out = Tensor(handle, requires_grad=needs,
                     _parents=parents if needs else (),
                     _backward=backward if needs else None)
        out._op = op if needs else None
        return out

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeError(
                "requires_grad 가 아닌 텐서에는 backward() 를 부를 수 없습니다.\n"
                "(torch: element 0 of tensors does not require grad and does not have a grad_fn)")
        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError(
                    "값이 하나가 아닌 텐서에는 gradient 를 줘야 합니다.\n"
                    "(torch: grad can be implicitly created only for scalar outputs)")
            gradient = _tf.onesLike(self._h)

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
                    t.grad = Tensor(g) if t.grad is None else Tensor(_tf.add(t.grad._h, g))
                continue
            for parent, pg in zip(t._parents, t._backward(g)):
                if pg is None:
                    continue
                pg = _unbroadcast(pg, parent.shape)
                grads[id(parent)] = pg if id(parent) not in grads else _tf.add(grads[id(parent)], pg)

    # ---- 산술

    def __add__(self, o):
        o = _wrap(o)
        return self._make(_tf.add(self._h, o._h), (self, o),
                          lambda g: (g, g), "AddBackward0")

    __radd__ = __add__

    def __sub__(self, o):
        o = _wrap(o)
        return self._make(_tf.sub(self._h, o._h), (self, o),
                          lambda g: (g, _tf.neg(g)), "SubBackward0")

    def __rsub__(self, o):
        return _wrap(o).__sub__(self)

    def __mul__(self, o):
        o = _wrap(o)
        return self._make(_tf.mul(self._h, o._h), (self, o),
                          lambda g: (_tf.mul(g, o._h), _tf.mul(g, self._h)), "MulBackward0")

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _wrap(o)
        return self._make(
            _tf.div(self._h, o._h), (self, o),
            lambda g: (_tf.div(g, o._h),
                       _tf.neg(_tf.div(_tf.mul(g, self._h), _tf.mul(o._h, o._h)))),
            "DivBackward0")

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
        o = _wrap(o)
        return self._make(
            _tf.matMul(self._h, o._h), (self, o),
            lambda g: (_tf.matMul(g, o._h, False, True),
                       _tf.matMul(self._h, g, True, False)),
            "MmBackward0")

    def matmul(self, o):
        return self.__matmul__(o)

    def transpose(self, d0, d1):
        perm = list(range(self.ndim))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        return self._make(_tf.transpose(self._h, _to_js(perm)), (self,),
                          lambda g: (_tf.transpose(g, _to_js(perm)),), "TransposeBackward0")

    @property
    def T(self):
        return self.transpose(-2, -1)

    # ---- 비교 (기울기 없음)

    def _cmp(self, o, fn):
        # `fn(self._h, _wrap(o)._h)` 로 쓰면 안 된다. `._h` 를 꺼내는 순간 임시 텐서의
        # 참조가 0 이 되어 `__del__` 이 버퍼를 놓고, **그 뒤에** fn 이 불린다.
        # 이름을 붙여 호출이 끝날 때까지 살려둔다.
        other = _wrap(o)
        return Tensor(fn(self._h, other._h))

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
        shape = self.shape
        handle = (_tf.sum(self._h) if dim is None
                  else _tf.sum(self._h, dim, keepdim))

        def back(g):
            return (_tf.mul(_tf.onesLike(self._h), _reshape_for_broadcast(g, shape, dim, keepdim)),)

        return self._make(handle, (self,), back,
                          "SumBackward0" if dim is None else "SumBackward1")

    def mean(self, dim=None, keepdim=False):
        shape = self.shape
        n = self.numel() if dim is None else shape[dim]
        handle = (_tf.mean(self._h) if dim is None
                  else _tf.mean(self._h, dim, keepdim))

        def back(g):
            spread = _reshape_for_broadcast(g, shape, dim, keepdim)
            return (_tf.div(_tf.mul(_tf.onesLike(self._h), spread), float(n)),)

        return self._make(handle, (self,), back,
                          "MeanBackward0" if dim is None else "MeanBackward1")


# ---------------------------------------------------------------- 도우미

def _wrap(x):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (int, float)):
        return Tensor(_tf.scalar(float(x)))
    return Tensor(_to_tf(_np.asarray(x)))


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
    if isinstance(data, Tensor):
        return Tensor(_tf.clone(data._h), requires_grad)     # 손잡이를 나눠 갖지 않는다
    return Tensor(_to_tf(_np.asarray(data)), requires_grad)


def from_numpy(arr):
    return Tensor(_to_tf(arr))


def zeros(*shape, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_tf.zeros(_to_js(list(shape))), requires_grad)


def ones(*shape, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_tf.ones(_to_js(list(shape))), requires_grad)


def randn(*shape, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_tf.randomNormal(_to_js(list(shape))), requires_grad)


# ---------------------------------------------------------------- 원소별
#
# TF.js 이름과 torch 이름이 갈리는 자리가 있어서 표로 둔다(matMul·notEqual 등).
# 미분이 정의되지 않는 것(sign·floor·ceil·round)은 기울기를 0 으로 둔다 — torch 도 그렇다.

def _unary(name, forward, derivative=None):
    def fn(t):
        t = _wrap(t)
        out = forward(t._h)
        if derivative is None:
            return Tensor(out)
        return t._make(out, (t,), lambda g: (_tf.mul(g, derivative(t._h, out)),),
                       f"{name}Backward0")
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
abs = _unary("Abs", lambda x: _tf.abs(x), lambda x, o: _tf.sign(x))
sin = _unary("Sin", lambda x: _tf.sin(x), lambda x, o: _tf.cos(x))
cos = _unary("Cos", lambda x: _tf.cos(x), lambda x, o: _tf.neg(_tf.sin(x)))
tan = _unary("Tan", lambda x: _tf.tan(x), lambda x, o: _tf.add(1.0, _tf.mul(o, o)))
sinh = _unary("Sinh", lambda x: _tf.sinh(x), lambda x, o: _tf.cosh(x))
cosh = _unary("Cosh", lambda x: _tf.cosh(x), lambda x, o: _tf.sinh(x))
tanh = _unary("Tanh", lambda x: _tf.tanh(x), lambda x, o: _tf.sub(1.0, _tf.mul(o, o)))
erf = _unary("Erf", lambda x: _tf.erf(x),
             lambda x, o: _tf.mul(2.0 / float(_np.sqrt(_np.pi)), _tf.exp(_tf.neg(_tf.square(x)))))
relu = _unary("Relu", lambda x: _tf.relu(x), lambda x, o: _tf.step(x))
sigmoid = _unary("Sigmoid", lambda x: _tf.sigmoid(x),
                 lambda x, o: _tf.mul(o, _tf.sub(1.0, o)))
# 계단 모양 — 미분이 거의 모든 곳에서 0 이다.
sign = _unary("Sign", lambda x: _tf.sign(x))
floor = _unary("Floor", lambda x: _tf.floor(x))
ceil = _unary("Ceil", lambda x: _tf.ceil(x))
round = _unary("Round", lambda x: _tf.round(x))


def neg(t):
    return -_wrap(t)


def prod(t, dim=None):
    t = _wrap(t)
    out = _tf.prod(t._h) if dim is None else _tf.prod(t._h, dim)
    return Tensor(out)


def count_nonzero(t, dim=None):
    t = _wrap(t)
    nz = _tf.cast(_tf.notEqual(t._h, 0.0), "float32")
    return Tensor(_tf.sum(nz) if dim is None else _tf.sum(nz, dim))


def matmul(a, b):
    return _wrap(a) @ _wrap(b)


def mm(a, b):
    return _wrap(a) @ _wrap(b)


# ---------------------------------------------------------------- 비교·클램프

def maximum(a, b):
    a, b = _wrap(a), _wrap(b)
    pick = _tf.cast(_tf.greaterEqual(a._h, b._h), "float32")
    return a._make(_tf.maximum(a._h, b._h), (a, b),
                   lambda g: (_tf.mul(g, pick), _tf.mul(g, _tf.sub(1.0, pick))),
                   "MaximumBackward0")


def minimum(a, b):
    a, b = _wrap(a), _wrap(b)
    pick = _tf.cast(_tf.lessEqual(a._h, b._h), "float32")
    return a._make(_tf.minimum(a._h, b._h), (a, b),
                   lambda g: (_tf.mul(g, pick), _tf.mul(g, _tf.sub(1.0, pick))),
                   "MinimumBackward0")


def clamp(t, min=None, max=None):
    t = _wrap(t)
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
    t = _wrap(t)
    old = t.shape
    return t._make(_tf.reshape(t._h, _to_js(list(shape))), (t,),
                   lambda g: (_tf.reshape(g, _to_js(list(old))),), "ViewBackward0")


def diag(t):
    """torch 의 `diag` 는 **행렬에서 대각을 뽑는다.** TF.js 의 `diag` 는 반대로
    벡터에서 행렬을 만든다 — 이름이 같고 뜻이 반대라, 그대로 부르면 조용히 다른 값이 나온다."""
    t = _wrap(t)
    n = t.shape[0]
    eye = _tf.eye(n)
    return Tensor(_tf.sum(_tf.mul(t._h, eye), 1))


def trace(t):
    t = _wrap(t)
    n = t.shape[0]
    return Tensor(_tf.sum(_tf.mul(t._h, _tf.eye(n))))


def norm(t, p=2, dim=None):
    t = _wrap(t)
    if p == 1:
        return abs(t).sum(dim=dim)
    return (t * t).sum(dim=dim) ** 0.5


def cumsum(t, dim):
    t = _wrap(t)
    return t._make(_tf.cumsum(t._h, dim), (t,),
                   lambda g: (_tf.reverse(_tf.cumsum(_tf.reverse(g, dim), dim), dim),),
                   "CumsumBackward0")


def cumprod(t, dim):
    return Tensor(_tf.cumprod(_wrap(t)._h, dim))


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


def topk(t, k, dim=-1, largest=True):
    t = _wrap(t)
    out = _tf.topk(t._h, k)
    return _ValuesIndices(Tensor(out.values), Tensor(_tf.cast(out.indices, "float32")))


def sort(t, dim=-1, descending=False):
    """TF.js 에는 정렬이 없다. `topk` 로 전부 뽑으면 내림차순이므로, 오름차순은 뒤집는다."""
    t = _wrap(t)
    n = t.shape[dim]
    out = _tf.topk(t._h, n)
    values, idx = out.values, out.indices
    if not descending:
        values, idx = _tf.reverse(values, -1), _tf.reverse(idx, -1)
    return _ValuesIndices(Tensor(values), Tensor(_tf.cast(idx, "float32")))


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
    t = _wrap(t)
    m = mask.numpy() if isinstance(mask, Tensor) else _np.asarray(mask)
    return Tensor(_to_tf(t.numpy()[m.astype(bool)]))


def median(t, dim=None):
    """torch 는 원소가 짝수일 때 **가운데 둘 중 작은 쪽**을 준다."""
    t = _wrap(t)
    if dim is None:
        n = t.numel()
        flat = _tf.reshape(t._h, _to_js([n]))
        asc = _tf.reverse(_tf.topk(flat, n).values, -1)
        picked = _tf.slice(asc, _to_js([(n - 1) // 2]), _to_js([1]))
        return Tensor(_tf.reshape(picked, _to_js([])))      # torch 는 0차원을 준다
    n = t.shape[dim]
    asc = _tf.reverse(_tf.topk(t._h, n).values, -1)
    idx = (n - 1) // 2
    picked = _tf.squeeze(_tf.slice(asc, _to_js([0, idx]), _to_js([t.shape[0], 1])), _to_js([1]))
    return _ValuesIndices(Tensor(picked), Tensor(_tf.zerosLike(picked)))


def flip(t, dims):
    t = _wrap(t)
    dims = [dims] if isinstance(dims, int) else list(dims)
    return t._make(_tf.reverse(t._h, _to_js(dims)), (t,),
                   lambda g: (_tf.reverse(g, _to_js(dims)),), "FlipBackward0")


def roll(t, shifts, dims=None):
    """TF.js 에 `roll` 이 없다. 잘라서 순서를 바꿔 붙인다."""
    t = _wrap(t)
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


def narrow(t, dim, start, length):
    t = _wrap(t)
    return t._make(_slice_along(t._h, dim, start, length), (t,),
                   lambda g: _unsupported("narrow 의 역전파"), "SliceBackward0")


def split(t, size, dim=0):
    t = _wrap(t)
    n = t.shape[dim]
    sizes = size if isinstance(size, (list, tuple)) else \
        [size] * (n // size) + ([n % size] if n % size else [])
    out, start = [], 0
    for sz in sizes:
        out.append(Tensor(_slice_along(t._h, dim, start, sz)))
        start += sz
    return tuple(out)


def chunk(t, chunks, dim=0):
    t = _wrap(t)
    n = t.shape[dim]
    return split(t, -(-n // chunks), dim)


def unbind(t, dim=0):
    t = _wrap(t)
    return tuple(Tensor(h) for h in _tf.unstack(t._h, dim))


def gather(t, dim, index):
    """torch 의 `gather` — 원소마다 자리를 고른다. TF.js 의 `gather` 는 축을 통째로
    뽑는 다른 연산이라 그대로 못 쓴다.

    자리를 원-핫으로 만들어 곱하고 접는다. 그러면 **역전파가 그냥 따라온다** —
    뽑기만 하고 그래프를 끊으면 뽑은 자리로 기울기가 안 가고, 분류 손실이 통째로
    미분 불가가 된다(실제로 그랬다).
    """
    t = _wrap(t)
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
    t = _wrap(t)
    return Tensor(_tf.gather(t._h, _tf.reshape(_to_int32(index), _to_js([-1])), dim))


# ---------------------------------------------------------------- nn.functional

def softmax(t, dim=-1):
    t = _wrap(t)
    out = _tf.softmax(t._h, dim)

    def back(g):
        s = _tf.sum(_tf.mul(g, out), dim, True)
        return (_tf.mul(out, _tf.sub(g, s)),)

    return t._make(out, (t,), back, "SoftmaxBackward0")


def log_softmax(t, dim=-1):
    t = _wrap(t)
    out = _tf.logSoftmax(t._h, dim)
    soft = _tf.exp(out)

    def back(g):
        return (_tf.sub(g, _tf.mul(soft, _tf.sum(g, dim, True))),)

    return t._make(out, (t,), back, "LogSoftmaxBackward0")


def leaky_relu(t, negative_slope=0.01):
    t = _wrap(t)
    pick = _tf.cast(_tf.greater(t._h, 0.0), "float32")
    return t._make(
        _tf.leakyRelu(t._h, float(negative_slope)), (t,),
        lambda g: (_tf.mul(g, _tf.add(pick, _tf.mul(_tf.sub(1.0, pick), float(negative_slope)))),),
        "LeakyReluBackward0")


def elu(t, alpha=1.0):
    t = _wrap(t)
    out = _tf.elu(t._h)
    pick = _tf.cast(_tf.greater(t._h, 0.0), "float32")
    return t._make(
        out, (t,),
        lambda g: (_tf.mul(g, _tf.add(pick, _tf.mul(_tf.sub(1.0, pick),
                                                    _tf.add(out, float(alpha))))),),
        "EluBackward0")


def silu(t):
    """x·σ(x). Swish 라고도 한다."""
    t = _wrap(t)
    sig = _tf.sigmoid(t._h)
    return t._make(
        _tf.mul(t._h, sig), (t,),
        lambda g: (_tf.mul(g, _tf.mul(sig, _tf.add(1.0, _tf.mul(t._h, _tf.sub(1.0, sig))))),),
        "SiluBackward0")


_SQRT2 = float(_np.sqrt(2.0))
_SQRT2PI = float(_np.sqrt(2.0 * _np.pi))


def gelu(t):
    """torch 의 기본 gelu(정확형) — 0.5·x·(1 + erf(x/√2)). TF.js 에 erf 가 있다."""
    t = _wrap(t)
    ope = _tf.add(1.0, _tf.erf(_tf.div(t._h, _SQRT2)))

    def back(g):
        bell = _tf.div(_tf.exp(_tf.neg(_tf.div(_tf.square(t._h), 2.0))), _SQRT2PI)
        return (_tf.mul(g, _tf.add(_tf.mul(0.5, ope), _tf.mul(t._h, bell))),)

    return t._make(_tf.mul(0.5, _tf.mul(t._h, ope)), (t,), back, "GeluBackward0")


def one_hot(t, num_classes=-1):
    t = _wrap(t)
    depth = int(t.numpy().max()) + 1 if num_classes == -1 else int(num_classes)
    return Tensor(_tf.oneHot(_to_int32(t), depth))


def pad(x, padding, value=0.0):
    """마지막 차원부터 (앞, 뒤) 순으로 받는다 — torch 의 규칙이다."""
    x = _wrap(x)
    pairs = [[0, 0] for _ in range(x.ndim)]
    for i in range(0, len(padding), 2):
        pairs[-(i // 2 + 1)] = [int(padding[i]), int(padding[i + 1])]
    return Tensor(_tf.pad(x._h, _to_js(pairs), float(value)))


def normalize(x, p=2, dim=1, eps=1e-12):
    x = _wrap(x)
    denom = norm(x, p=p, dim=dim)
    return x / maximum(unsqueeze(denom, dim), _wrap(eps))


def unsqueeze(t, dim):
    t = _wrap(t)
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


def avg_pool2d(x, kernel_size, stride=None):
    """torch 는 NCHW, TF.js 는 NHWC 다. 축을 바꿔 넣고 되돌린다."""
    x = _wrap(x)
    stride = stride or kernel_size
    nhwc = _tf.transpose(x._h, _to_js([0, 2, 3, 1]))
    pooled = _tf.avgPool(nhwc, _to_js([kernel_size, kernel_size]),
                         _to_js([stride, stride]), "valid")
    return Tensor(_tf.transpose(pooled, _to_js([0, 3, 1, 2])))


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
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_params", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

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
        return {n: Tensor(_tf.clone(p._h)) for n, p in self.named_parameters()}

    def load_state_dict(self, state, strict=True):
        own = dict(self.named_parameters())
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        if strict and (missing or unexpected):
            raise RuntimeError(f"state_dict 가 안 맞습니다. 빠진 것: {missing}, 남는 것: {unexpected}")
        for name, value in state.items():
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
                    g = g if buf is None else _tf.add(_tf.mul(float(group["momentum"]), buf), g)
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


def backend():
    """지금 붙어 있는 TF.js 백엔드. 'webgpu' 가 아니면 GPU 로 돌고 있지 않다."""
    return str(_tf.getBackend())
