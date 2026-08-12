"""nanotorch — numpy 위에 얹은 PyTorch 모양의 얇은 층.

설치 없이 브라우저(Pyodide)에서 PyTorch **문법**을 연습하기 위한 것이다.
torch 는 wasm 으로 포팅되지 않는다 — 수백 MB의 네이티브 코드에, 손튜닝된 AVX·NEON
커널은 wasm SIMD 로 옮겨지지 않고, OpenMP 스레드는 Pyodide 가 싣지 않는 헤더를 요구한다.
그런데 **문법을 익히는 데는 그중 아무것도 필요하지 않다.** numpy 면 된다.

## 설계 원칙 — 틀린 답보다 없는 기능이 낫다

축소판이 진짜와 조금이라도 다르게 동작하면 학생은 거짓을 배운다. 그래서
**없는 것은 근사하지 않고 예외를 던진다.** 조용히 다른 값을 내느니 시끄럽게 멈춘다.

지원 범위 밖을 만나면 `NanoTorchError` 가 나고, 메시지가 "자기 컴퓨터에서 하라"고 말한다.

## 어떻게 보장하는가

두 겹이다.

1. `nanotorch-check` — 같은 **랩 테스트**를 진짜 torch 와 축소판 양쪽에서 돌린다.
2. `nanotorch-diff` — 랩과 무관하게 **같은 연산의 숫자를 직접 비교**한다
   (`tests/test_nanotorch_diff.py`). 1번만으로는 랩이 지나가는 길만 봐서
   축소판의 73% 였고, 그 사각지대에 역전파가 들어 있었다.

지금은 두 검사가 86% 를 덮는다. 남은 곳은 `__repr__` 처럼 값이 걸리지 않는 자리다.

`nanotorch-diff` 가 실제로 잡은 것: BatchNorm 은 정규화에 편향 분산을,
running_var 갱신에는 비편향 분산을 쓴다. 둘 다 편향으로 두면 2.6% 어긋난다.
"""

import math as _math

import numpy as _np

__all__ = ["Tensor", "tensor", "nn", "optim", "no_grad"]

_DEFAULT_DTYPE = _np.float32


class NanoTorchError(NotImplementedError):
    """축소판이 지원하지 않는 것. 근사하지 않고 여기서 멈춘다."""


def _like_torch(korean: str, torch_phrase: str) -> str:
    """오류 메시지의 규격.

    한국어 설명만 두면 학습자가 검색해서 답을 못 찾고, 영문만 베끼면 이 교재가
    한국어인 이유가 사라진다. 둘 다 넣는다 — 설명은 읽고, 영문 문구는 검색한다.
    """
    return f"{korean}\n(torch: {torch_phrase})"


def _unsupported(what: str):
    raise NanoTorchError(
        f"{what} 은(는) 브라우저 축소판에 없습니다.\n"
        "자기 컴퓨터에서 `uv add torch` 로 진짜 PyTorch 를 쓰세요 — "
        "축소판은 문법 연습용이고, 없는 것을 흉내 내면 틀린 것을 배우게 됩니다."
    )


# ---------------------------------------------------------------- dtype

class dtype:
    def __init__(self, name, np_type):
        self.name = name
        self.np = np_type

    def __repr__(self):
        return f"torch.{self.name}"

    def __eq__(self, other):
        return isinstance(other, dtype) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


float32 = dtype("float32", _np.float32)
float64 = dtype("float64", _np.float64)
int64 = dtype("int64", _np.int64)
long = int64
bool_ = dtype("bool", _np.bool_)

_NP_TO_DTYPE = {_np.dtype("float32"): float32, _np.dtype("float64"): float64,
                _np.dtype("int64"): int64, _np.dtype("bool"): bool_}


def _resolve(data, dt):
    """진짜 torch 의 규칙을 따른다 — 정수만 있으면 int64, 하나라도 실수면 float32."""
    if dt is not None:
        return dt.np
    arr = _np.asarray(data)
    if arr.dtype.kind == "b":
        return _np.bool_
    if arr.dtype.kind in "iu":
        return _np.int64
    return _np.float32



# ---------------------------------------------------------------- 표현(repr)
#
# 학습자가 가장 많이 하는 일이 print(tensor) 다. 진짜와 다르게 찍히면 교재의 예시와
# 화면이 안 맞고, 그때마다 "내가 뭘 잘못했나" 를 의심하게 된다.
# torch/_tensor_str.py 의 규칙을 따른다.

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
        return "[]" if data.ndim else "[]"
    if data.dtype.kind == "f":
        fmt = _float_formatter(data)
        # torch 는 원소를 같은 너비로 오른쪽 정렬한다 — 음수가 섞이면 양수 앞에 자리가 생긴다.
        width = max((len(fmt(v)) for v in data.reshape(-1)), default=0)
        padded = lambda v, f=fmt, w=width: f(v).rjust(w)
        body = _np.array2string(
            data, formatter={"float_kind": padded}, separator=", ",
            max_line_width=_LINE_WIDTH - 8, threshold=1000)
    else:
        body = _np.array2string(data, separator=", ",
                                max_line_width=_LINE_WIDTH - 8, threshold=1000)
    # numpy 는 이어지는 줄을 한 칸 들여쓴다. torch 는 "tensor(" 만큼(8칸) 들여쓴다.
    return body.replace("\n ", "\n" + " " * 8)


def _tensor_repr(t):
    parts = [_tensor_str(t.data)]
    if t.data.dtype not in (_np.dtype("float32"), _np.dtype("int64"), _np.dtype("bool")):
        parts.append(f"dtype={t.dtype}")
    if t._op:
        parts.append(f"grad_fn=<{t._op}>")
    elif t.requires_grad:
        parts.append("requires_grad=True")
    return f"tensor({', '.join(parts)})"


# ---------------------------------------------------------------- Size

class Size(tuple):
    def __repr__(self):
        return f"torch.Size([{', '.join(str(x) for x in self)}])"


# ---------------------------------------------------------------- Tensor

def _unbroadcast(grad, shape):
    """브로드캐스팅으로 늘어난 축을 되돌린다. 역전파의 필수 단계다."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


# torch 의 dtype 승격은 numpy 와 다르다 — **범주**로 먼저 가르고, 그 범주 안에서만 올린다.
#
#   범주:  bool(0) < 정수(1) < 실수(2)
#   규칙:  참여한 것 중 가장 높은 범주를 고르고, 그 범주에 속한 것들 중 큰 것을 쓴다.
#          낮은 범주는 높은 범주를 **끌어올리지 않는다.**
#
# 그래서 float32 + int64 가 torch 에서는 float32 다 (numpy 는 float64 로 올린다).
# 여기를 numpy 에 맡기면 학습자는 틀린 규칙을 배운다.

_CATEGORY = {"b": 0, "i": 1, "u": 1, "f": 2}
_RANK = {_np.dtype("bool"): 0, _np.dtype("int64"): 10,
         _np.dtype("float32"): 20, _np.dtype("float64"): 21}
_DEFAULT_BY_CATEGORY = {0: _np.dtype("bool"), 1: _np.dtype("int64"), 2: _np.dtype("float32")}


def _category(dt):
    return _CATEGORY.get(_np.dtype(dt).kind, 2)


def result_type(a, b):
    """두 텐서 dtype 의 결과 타입. torch.result_type 과 같은 규칙."""
    da, db = _np.dtype(a), _np.dtype(b)
    cat = max(_category(da), _category(db))
    same = [d for d in (da, db) if _category(d) == cat]
    return max(same, key=lambda d: _RANK.get(d, 0))


def _scalar_category(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return 1
    return 2


def _no_bool_subtract(dtype, other):
    """torch 는 불리언에 `-` 를 허용하지 않는다. `^` 나 `~` 를 쓰라고 안내한다."""
    other_dtype = other.data.dtype if isinstance(other, Tensor) else _np.asarray(other).dtype
    if _np.dtype(dtype).kind == "b" or other_dtype.kind == "b":
        raise RuntimeError(_like_torch(
            "불리언 텐서에는 뺄셈(`-`)을 쓸 수 없습니다. "
            "`^`(배타적 논리합)나 `~`(부정)를 쓰세요.",
            "Subtraction, the `-` operator, with a bool tensor is not supported. "
            "If you are trying to invert a mask use the `~` or `logical_not()` operator instead."))


def _promote(data, scalar):
    """파이썬 스칼라와 섞을 때의 dtype.

    스칼라는 텐서보다 약하다 — 범주가 텐서보다 낮거나 같으면 텐서의 dtype 을 따르고,
    높을 때만 그 범주의 기본형으로 올라간다. int 텐서 + 파이썬 float 가
    float64 가 아니라 **float32** 인 이유다.
    """
    tcat = _category(data.dtype)
    scat = _scalar_category(scalar)
    return data.dtype if scat <= tcat else _DEFAULT_BY_CATEGORY[scat]


_grad_enabled = True


class _DataDescriptor:
    """`t.data` 는 읽을 때 numpy 를, 쓸 때는 **텐서만** 받는다.

    torch 가 `p.data = ndarray` 를 거부하기 때문이다. 여기서 받아주면 브라우저에서 돌던
    코드가 자기 컴퓨터에서 깨진다 — 관대한 것도 갈리는 것이다.
    """

    def __get__(self, obj, owner=None):
        return obj._array if obj is not None else self

    def __set__(self, obj, value):
        if isinstance(value, Tensor):
            obj._array = value._array
            return
        if isinstance(value, _np.ndarray):
            raise TypeError(_like_torch(
                "`.data` 에는 텐서만 넣을 수 있습니다. `torch.tensor(...)` 로 감싸세요.",
                "Variable data has to be a tensor, but got numpy.ndarray"))
        raise TypeError(_like_torch(
            f"`.data` 에는 텐서만 넣을 수 있습니다 ({type(value).__name__} 을 받았습니다).",
            f"Variable data has to be a tensor, but got {type(value).__name__}"))


class Tensor:
    data = _DataDescriptor()

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self._array = data if isinstance(data, _np.ndarray) else _np.asarray(data)
        # no_grad 는 **연산의 결과**가 그래프를 안 갖게 할 뿐, 직접 만든 잎의 requires_grad 를
        # 끄지는 않는다. torch 도 그렇다 — 여기서 끄면 no_grad 블록 안에서 만든 파라미터가
        # 학습 대상에서 조용히 빠진다.
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = _parents
        self._backward = _backward
        self._freed = False        # backward 한 번이면 그래프를 놓는다 (torch 와 같다)
        self._op = None            # grad_fn 표시용 — 어느 연산에서 나왔는가

        if self.requires_grad and self.data.dtype.kind not in "fc":
            raise RuntimeError(
                "정수 텐서에는 기울기가 흐르지 않습니다. 미분은 실수에서만 정의됩니다 "
                "— `.float()` 로 바꾸세요."
            )

    # ---- 기본 정보

    @property
    def shape(self):
        return Size(self.data.shape)

    @property
    def dtype(self):
        return _NP_TO_DTYPE.get(self.data.dtype, float32)

    @property
    def ndim(self):
        return self.data.ndim

    def size(self, dim=None):
        return self.shape if dim is None else self.data.shape[dim]

    def dim(self):
        return self.data.ndim

    def numel(self):
        return int(self.data.size)

    def item(self):
        if self.data.size != 1:
            raise RuntimeError(_like_torch(
                f"값이 {self.data.size}개인 텐서는 하나의 숫자로 바꿀 수 없습니다. "
                "`.tolist()` 나 인덱싱을 쓰세요.",
                f"a Tensor with {self.data.size} elements cannot be converted to Scalar"))
        return self.data.reshape(-1)[0].item()

    def tolist(self):
        return self.data.tolist()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return _tensor_repr(self)

    __str__ = __repr__

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]

    def __bool__(self):
        return bool(self.data)

    def __float__(self):
        return float(self.item())

    def __hash__(self):
        return id(self)

    # ---- 그래프

    def _make(self, data, parents, backward, op=None):
        needs = _grad_enabled and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=False, _parents=parents if needs else (),
                     _backward=backward if needs else None)
        out.requires_grad = needs
        out._op = op if needs else None
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
            if self.data.size != 1:
                raise RuntimeError(_like_torch(
                    "값이 하나가 아닌 텐서에는 gradient 를 줘야 합니다. "
                    "보통은 손실을 스칼라로 만든 뒤 부릅니다.",
                    "grad can be implicitly created only for scalar outputs"))
            gradient = _np.ones_like(self.data)

        # 위상 정렬 — 뒤에서 앞으로 한 번씩만 지나간다
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        visit(self)

        grads = {id(self): _np.asarray(gradient, dtype=self.data.dtype)}
        for t in reversed(order):
            g = grads.get(id(t))
            if g is None:
                continue
            if t._backward is None:                 # 잎 — 여기에 쌓는다
                if t.requires_grad:
                    t.grad = Tensor(g) if t.grad is None else Tensor(t.grad.data + g)
                continue
            for parent, pg in zip(t._parents, t._backward(g)):
                if pg is None:
                    continue
                # 잎의 .grad 는 위의 분기에서만 채운다. 여기서도 채우면 두 번 쌓인다.
                pg = _unbroadcast(_np.asarray(pg), parent.data.shape)
                grads[id(parent)] = pg if id(parent) not in grads else grads[id(parent)] + pg

        if not retain_graph:
            for t in order:
                if t._backward is not None:
                    t._freed = True

    def detach(self):
        return Tensor(self.data)

    def clone(self):
        return self._make(self.data.copy(), (self,), lambda g: (g,))

    def numpy(self):
        return self.data

    # ---- 형 변환

    def float(self):
        return Tensor(self.data.astype(_np.float32), self.requires_grad)

    def long(self):
        return Tensor(self.data.astype(_np.int64))

    def int(self):
        return Tensor(self.data.astype(_np.int64))

    def bool(self):
        return Tensor(self.data.astype(_np.bool_))

    def double(self):
        return Tensor(self.data.astype(_np.float64), self.requires_grad)

    def type(self, dt):
        return Tensor(self.data.astype(dt.np if isinstance(dt, dtype) else dt), self.requires_grad)

    def to(self, *args, **kwargs):
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, str) and a != "cpu":
                _unsupported(f"장치 '{a}'")
        return self

    def cpu(self):
        return self

    # ---- 산술

    def _binary(self, other, forward, back_self, back_other, op=None):
        if isinstance(other, Tensor):
            target = result_type(self.data.dtype, other.data.dtype)
            o = other if other.data.dtype == target else Tensor(other.data.astype(target))
            mine = self.data if self.data.dtype == target else self.data.astype(target)
        else:
            # 파이썬 스칼라를 텐서 dtype 으로 끌어온 뒤 계산한다. numpy 에 맡기면
            # int64 + float32 가 float64 로 올라가는데 torch 는 float32 를 준다.
            target = _promote(self.data, other)
            o = Tensor(_np.asarray(other, dtype=target))
            mine = self.data.astype(target) if self.data.dtype != target else self.data
        try:
            out = forward(mine, o.data)
        except ValueError:
            a, b = mine.shape, o.data.shape
            bad = next((i for i in range(1, min(len(a), len(b)) + 1)
                        if a[-i] != b[-i] and a[-i] != 1 and b[-i] != 1), 1)
            raise RuntimeError(_like_torch(
                f"모양 {tuple(a)} 과 {tuple(b)} 은 브로드캐스팅되지 않습니다 — "
                "뒤에서부터 맞춰볼 때 크기가 같거나 한쪽이 1이어야 합니다.",
                f"The size of tensor a ({a[-bad]}) must match the size of tensor b "
                f"({b[-bad]}) at non-singleton dimension {len(a) - bad}")) from None
        return self._make(out, (self, o), lambda g: (back_self(g, mine, o.data),
                                                     back_other(g, mine, o.data)), op)

    def __add__(self, o):
        return self._binary(o, _np.add, lambda g, a, b: g, lambda g, a, b: g, "AddBackward0")

    __radd__ = __add__

    def __sub__(self, o):
        _no_bool_subtract(self.data.dtype, o)
        return self._binary(o, _np.subtract, lambda g, a, b: g, lambda g, a, b: -g, "SubBackward0")

    def __rsub__(self, o):
        _no_bool_subtract(self.data.dtype, o)
        return Tensor(_np.asarray(o, dtype=self.data.dtype)).__sub__(self)

    def __mul__(self, o):
        return self._binary(o, _np.multiply, lambda g, a, b: g * b, lambda g, a, b: g * a,
                            "MulBackward0")

    __rmul__ = __mul__

    def __truediv__(self, o):
        # torch 의 나눗셈은 정수·불리언끼리여도 기본 실수형(float32)을 낸다.
        # numpy 에 맡기면 int64/int64 가 float64 가 된다.
        def div(a, b):
            out = _np.divide(a, b)
            return out.astype(_DEFAULT_DTYPE) if a.dtype.kind not in "fc" else out
        return self._binary(o, div, lambda g, a, b: g / b,
                            lambda g, a, b: -g * a / (b * b), "DivBackward0")

    def __rtruediv__(self, o):
        return Tensor(_np.asarray(o, dtype=self.data.dtype)).__truediv__(self)

    def __pow__(self, p):
        if isinstance(p, Tensor):
            _unsupported("텐서 지수")
        return self._make(self.data ** p, (self,), lambda g: (g * p * self.data ** (p - 1),),
                          "PowBackward0")

    def __neg__(self):
        return self._make(-self.data, (self,), lambda g: (-g,), "NegBackward0")

    def __mod__(self, o):
        return Tensor(_np.mod(self.data, o.data if isinstance(o, Tensor) else o))

    def __floordiv__(self, o):
        return Tensor(_np.floor_divide(self.data, o.data if isinstance(o, Tensor) else o))

    def __matmul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        if self.data.ndim >= 2 and o.data.ndim >= 2 and self.data.shape[-1] != o.data.shape[-2]:
            a = "x".join(str(n) for n in self.data.shape[-2:])
            b = "x".join(str(n) for n in o.data.shape[-2:])
            raise RuntimeError(_like_torch(
                f"행렬곱의 모양이 안 맞습니다 ({a} @ {b}) — "
                f"앞의 열({self.data.shape[-1]})과 뒤의 행({o.data.shape[-2]})이 같아야 합니다.",
                f"mat1 and mat2 shapes cannot be multiplied ({a} and {b})"))
        return self._make(
            self.data @ o.data, (self, o),
            lambda g: (g @ _np.swapaxes(o.data, -1, -2), _np.swapaxes(self.data, -1, -2) @ g),
            "MmBackward0" if self.data.ndim == 2 else "BmmBackward0",
        )

    def matmul(self, o):
        return self.__matmul__(o)

    # 제자리 갱신 — no_grad 안에서만 허용한다 (진짜 torch 도 같은 규칙)
    def _inplace(self, fn, other):
        if self.requires_grad and _grad_enabled:
            raise RuntimeError(
                "기울기가 필요한 텐서를 제자리에서 바꿀 수 없습니다. "
                "`with torch.no_grad():` 안에서 하세요."
            )
        o = other.data if isinstance(other, Tensor) else other
        self._array = fn(self._array, o).astype(self._array.dtype)
        return self

    def __iadd__(self, o):
        return self._inplace(_np.add, o)

    def __isub__(self, o):
        return self._inplace(_np.subtract, o)

    def __imul__(self, o):
        return self._inplace(_np.multiply, o)

    # ---- 비교 (기울기 없음)

    def _cmp(self, o, fn):
        return Tensor(fn(self.data, o.data if isinstance(o, Tensor) else o))

    def __gt__(self, o): return self._cmp(o, _np.greater)
    def __ge__(self, o): return self._cmp(o, _np.greater_equal)
    def __lt__(self, o): return self._cmp(o, _np.less)
    def __le__(self, o): return self._cmp(o, _np.less_equal)
    def __eq__(self, o): return self._cmp(o, _np.equal)
    def __ne__(self, o): return self._cmp(o, _np.not_equal)

    def __and__(self, o): return self._cmp(o, _np.logical_and)
    def __or__(self, o): return self._cmp(o, _np.logical_or)

    def all(self):
        return Tensor(_np.all(self.data))

    def any(self):
        return Tensor(_np.any(self.data))

    # ---- 모양

    def reshape(self, *shape):
        # numpy 의 reshape 는 가능하면 뷰를 준다 — 그대로 들고 있으면 저장소가 공유되고,
        # 그것이 torch 의 동작이다. `b = a.view(2,2); b[0,0]=9` 가 a 를 바꾼다.
        # 복사해 두면 편하지만, 실무에서 사고가 나는 바로 그 지점을 안 가르치게 된다.
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        old = self.data.shape
        try:
            out = self.data.reshape(shape)
        except ValueError:
            want = list(shape)
            raise RuntimeError(_like_torch(
                f"모양 {want} 은 원소 {self.data.size}개짜리 텐서에 맞지 않습니다.",
                f"shape '{want}' is invalid for input of size {self.data.size}")) from None
        return self._make(out, (self,), lambda g: (g.reshape(old),), "ViewBackward0")

    def view(self, *shape):
        """`reshape` 과 달리 **저장소를 그대로 쓸 수 있을 때만** 된다.

        transpose 한 텐서처럼 메모리 순서가 어긋난 것에는 torch 가 거부하고
        `reshape` 을 쓰라고 안내한다. 둘의 차이를 여기서 배우는 게 맞다.
        """
        if not self.data.flags["C_CONTIGUOUS"]:
            raise RuntimeError(_like_torch(
                "메모리 순서가 어긋난 텐서에는 view() 를 쓸 수 없습니다 — "
                "`.contiguous().view(...)` 또는 `.reshape(...)` 을 쓰세요.",
                "view size is not compatible with input tensor's size and stride "
                "(at least one dimension spans across two contiguous subspaces). "
                "Use .reshape(...) instead."))
        return self.reshape(*shape)

    def unsqueeze(self, dim):
        old = self.data.shape
        return self._make(_np.expand_dims(self.data, dim), (self,), lambda g: (g.reshape(old),),
                          "UnsqueezeBackward0")

    def squeeze(self, dim=None):
        old = self.data.shape
        out = _np.squeeze(self.data) if dim is None else _np.squeeze(self.data, axis=dim)
        return self._make(out, (self,), lambda g: (g.reshape(old),))

    def transpose(self, d0, d1):
        return self._make(_np.swapaxes(self.data, d0, d1), (self,),
                          lambda g: (_np.swapaxes(g, d0, d1),), "TransposeBackward0")

    @property
    def T(self):
        return self.transpose(-2, -1)

    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        inv = _np.argsort(dims)
        return self._make(_np.transpose(self.data, dims), (self,),
                          lambda g: (_np.transpose(g, inv),))

    def contiguous(self):
        return self

    def flatten(self, start_dim=0):
        shape = self.data.shape[:start_dim] + (-1,)
        return self.reshape(shape)

    def __getitem__(self, idx):
        key = tuple(i.data if isinstance(i, Tensor) else i for i in idx) \
            if isinstance(idx, tuple) else (idx.data if isinstance(idx, Tensor) else idx)
        old = self.data.shape

        def back(g):
            z = _np.zeros(old, dtype=_np.asarray(g).dtype)
            _np.add.at(z, key, g)
            return (z,)

        return self._make(self.data[key], (self,), back)

    def __setitem__(self, idx, value):
        if self.requires_grad and _grad_enabled:
            raise RuntimeError("기울기가 필요한 텐서에는 제자리 대입을 할 수 없습니다.")
        key = tuple(i.data if isinstance(i, Tensor) else i for i in idx) \
            if isinstance(idx, tuple) else (idx.data if isinstance(idx, Tensor) else idx)
        self.data[key] = value.data if isinstance(value, Tensor) else value

    # ---- 축약

    def _reduce(self, fn, dim, keepdim, grad_fn, op=None):
        axis = dim
        out = fn(self.data, axis=axis, keepdims=keepdim)
        return self._make(out, (self,), lambda g: (grad_fn(g, axis, keepdim),), op)

    def sum(self, dim=None, keepdim=False):
        shape = self.data.shape

        def back(g, axis, kd):
            g = _np.asarray(g)
            if axis is not None and not kd:
                g = _np.expand_dims(g, axis)
            return _np.broadcast_to(g, shape).copy()

        return self._reduce(_np.sum, dim, keepdim, back,
                            "SumBackward0" if dim is None else "SumBackward1")

    def mean(self, dim=None, keepdim=False):
        shape = self.data.shape
        n = self.data.size if dim is None else shape[dim]

        def back(g, axis, kd):
            g = _np.asarray(g)
            if axis is not None and not kd:
                g = _np.expand_dims(g, axis)
            return _np.broadcast_to(g, shape).copy() / n

        return self._reduce(_np.mean, dim, keepdim, back,
                            "MeanBackward0" if dim is None else "MeanBackward1")

    def _argreduce(self, np_fn, np_arg, dim, keepdim):
        if dim is None:
            return Tensor(np_fn(self.data))
        idx = np_arg(self.data, axis=dim)
        values = _np.take_along_axis(self.data, _np.expand_dims(idx, dim), axis=dim)
        if not keepdim:
            values = _np.squeeze(values, axis=dim)
        shape, d = self.data.shape, dim

        def back(g):
            z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
            gg = _np.asarray(g)
            if not keepdim:
                gg = _np.expand_dims(gg, d)
            _np.put_along_axis(z, _np.expand_dims(idx, d), gg, axis=d)
            return (z,)

        out = self._make(values, (self,), back)
        return _MinMax(out, Tensor(idx))

    def max(self, dim=None, keepdim=False):
        return self._argreduce(_np.max, _np.argmax, dim, keepdim)

    def min(self, dim=None, keepdim=False):
        return self._argreduce(_np.min, _np.argmin, dim, keepdim)

    def argmax(self, dim=None):
        return Tensor(_np.argmax(self.data, axis=dim))

    def argmin(self, dim=None):
        return Tensor(_np.argmin(self.data, axis=dim))

    def std(self, dim=None, unbiased=True, keepdim=False):
        ddof = 1 if unbiased else 0
        return Tensor(_np.std(self.data, axis=dim, ddof=ddof, keepdims=keepdim))

    def var(self, dim=None, unbiased=True, keepdim=False):
        ddof = 1 if unbiased else 0
        return Tensor(_np.var(self.data, axis=dim, ddof=ddof, keepdims=keepdim))

    def abs(self):
        return self._make(_np.abs(self.data), (self,), lambda g: (g * _np.sign(self.data),))

    def exp(self):
        out = _np.exp(self.data)
        return self._make(out, (self,), lambda g: (g * out,))

    def log(self):
        return self._make(_np.log(self.data), (self,), lambda g: (g / self.data,))

    def sqrt(self):
        out = _np.sqrt(self.data)
        return self._make(out, (self,), lambda g: (g * 0.5 / out,))

    def masked_fill(self, mask, value):
        m = mask.data.astype(bool) if isinstance(mask, Tensor) else _np.asarray(mask, dtype=bool)
        out = _np.where(m, _np.asarray(value, dtype=self.data.dtype), self.data)
        return self._make(out, (self,), lambda g: (_np.where(m, 0, g),))

    def bincount(self):
        return Tensor(_np.bincount(self.data.astype(_np.int64)))


class _MinMax:
    """`x.max(dim=0)` 이 돌려주는 (values, indices). 진짜 torch 와 같은 모양."""

    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def __iter__(self):
        yield self.values
        yield self.indices

    def __getitem__(self, i):
        return (self.values, self.indices)[i]


# ---------------------------------------------------------------- 만들기

def tensor(data, dtype=None, requires_grad=False):
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(_np.asarray(data, dtype=_resolve(data, dtype)), requires_grad)


def as_tensor(data, dtype=None):
    if isinstance(data, Tensor) and dtype is None:
        return data
    if isinstance(data, Tensor):
        return Tensor(data.data.astype(dtype.np))
    return tensor(data, dtype)


def from_numpy(arr):
    return Tensor(arr)


def zeros(*shape, dtype=None, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_np.zeros(shape, dtype=(dtype.np if dtype else _DEFAULT_DTYPE)), requires_grad)


def ones(*shape, dtype=None, requires_grad=False, device=None):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_np.ones(shape, dtype=(dtype.np if dtype else _DEFAULT_DTYPE)), requires_grad)


def full(shape, value, dtype=None):
    return Tensor(_np.full(shape, value, dtype=(dtype.np if dtype else _DEFAULT_DTYPE)))


def zeros_like(t, dtype=None):
    return Tensor(_np.zeros_like(t.data if isinstance(t, Tensor) else t))


def ones_like(t, dtype=None):
    return Tensor(_np.ones_like(t.data if isinstance(t, Tensor) else t))


def full_like(t, value):
    return Tensor(_np.full_like(t.data, value))


def arange(*args, dtype=None):
    return Tensor(_np.arange(*args, dtype=(dtype.np if dtype else None)))


def linspace(start, end, steps):
    return Tensor(_np.linspace(start, end, steps, dtype=_DEFAULT_DTYPE))


def eye(n):
    return Tensor(_np.eye(n, dtype=_DEFAULT_DTYPE))


_rng = _np.random.default_rng(0)


class Generator:
    """씨앗을 담아 다니는 그릇. `random_split(generator=...)` 이 이것을 받는다 —
    나누기를 고정하지 않으면 모델을 바꿔 좋아진 건지 나누기가 운이 좋았던 건지 알 수 없다."""

    def __init__(self):
        self.seed = 0

    def manual_seed(self, seed):
        self.seed = seed
        return self

    def rng(self):
        return _np.random.default_rng(self.seed)


def manual_seed(seed):
    global _rng
    _rng = _np.random.default_rng(seed)
    return seed


def randn(*shape, requires_grad=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_rng.standard_normal(shape).astype(_DEFAULT_DTYPE), requires_grad)


def rand(*shape):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return Tensor(_rng.random(shape).astype(_DEFAULT_DTYPE))


def randint(low, high, shape):
    return Tensor(_rng.integers(low, high, shape).astype(_np.int64))


def randperm(n):
    return Tensor(_rng.permutation(n).astype(_np.int64))


def multinomial(probs, num_samples, replacement=True):
    p = probs.data / probs.data.sum(axis=-1, keepdims=True)
    if p.ndim == 1:
        return Tensor(_rng.choice(len(p), size=num_samples, p=p).astype(_np.int64))
    out = [_rng.choice(p.shape[-1], size=num_samples, p=row) for row in p]
    return Tensor(_np.asarray(out, dtype=_np.int64))


# ---------------------------------------------------------------- 함수

def _wrap(t):
    return t if isinstance(t, Tensor) else Tensor(_np.asarray(t))


def stack(items, dim=0):
    items = [_wrap(t) for t in items]
    out = _np.stack([t.data for t in items], axis=dim)
    if not items:
        return Tensor(out)
    return items[0]._make(
        out, tuple(items),
        lambda g: tuple(_np.take(_np.asarray(g), i, axis=dim) for i in range(len(items))),
        "StackBackward0")


def cat(items, dim=0):
    items = [_wrap(t) for t in items]
    out = _np.concatenate([t.data for t in items], axis=dim)
    sizes = [t.data.shape[dim] for t in items]

    def back(g):
        g = _np.asarray(g)
        cuts = _np.cumsum(sizes)[:-1]
        return tuple(_np.split(g, cuts, axis=dim))

    return items[0]._make(out, tuple(items), back, "CatBackward0")


def where(cond, a, b):
    c = cond.data if isinstance(cond, Tensor) else cond
    ta, tb = _wrap(a), _wrap(b)
    out = _np.where(c, ta.data, tb.data)
    return ta._make(out, (ta, tb), lambda g: (_np.where(c, g, 0), _np.where(c, 0, g)))


def sigmoid(t):
    out = 1.0 / (1.0 + _np.exp(-_np.clip(t.data, -60, 60)))
    return t._make(out, (t,), lambda g: (g * out * (1 - out),), "SigmoidBackward0")


def relu(t):
    return t._make(_np.maximum(t.data, 0), (t,), lambda g: (g * (t.data > 0),), "ReluBackward0")


def tanh(t):
    out = _np.tanh(t.data)
    return t._make(out, (t,), lambda g: (g * (1 - out * out),), "TanhBackward0")


def exp(t): return t.exp()
def log(t): return t.log()
def sqrt(t): return t.sqrt()
def abs(t): return t.abs()


def softmax(t, dim=-1):
    shifted = t.data - t.data.max(axis=dim, keepdims=True)
    e = _np.exp(shifted)
    out = e / e.sum(axis=dim, keepdims=True)

    def back(g):
        s = (g * out).sum(axis=dim, keepdims=True)
        return ((out * (g - s)),)

    return t._make(out, (t,), back, "SoftmaxBackward0")



def _pad2d(x, padding):
    if padding == 0:
        return x
    return _np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))


def _im2col(xd, KH, KW, stride):
    """(N,C,H,W) 를 (N*OH*OW, C*KH*KW) 로 편다. GEMM 한 번으로 합성곱을 끝내기 위한 것."""
    N, C, H, W = xd.shape
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (KH, KW), axis=(2, 3))
    win = win[:, :, ::stride, ::stride, :, :]          # (N, C, OH, OW, KH, KW)
    cols = win.transpose(0, 2, 3, 1, 4, 5)             # (N, OH, OW, C, KH, KW)
    return _np.ascontiguousarray(cols).reshape(N * OH * OW, C * KH * KW), OH, OW


def _col2im(gcols, shape, KH, KW, stride, OH, OW):
    """im2col 의 역. 출력 자리(OH×OW)가 아니라 **필터 자리(KH×KW)** 를 돈다 —
    28×28 이미지에서 784번 대신 9번이면 끝난다."""
    N, C, H, W = shape
    gx = _np.zeros(shape, dtype=gcols.dtype)
    g = gcols.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)   # (N,C,KH,KW,OH,OW)
    for i in range(KH):
        for j in range(KW):
            gx[:, :, i:i + OH * stride:stride, j:j + OW * stride:stride] += g[:, :, i, j]
    return gx


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """작은 입력용 합성곱. 26장에서 손으로 짠 이중 반복문과 같은 계산이다.

    im2col 로 펴서 행렬곱 한 번으로 끝낸다 — numpy 가 BLAS 를 부르므로,
    창을 돌며 einsum 하는 것보다 (실측) 20배 이상 빠르다.
    빠르다고 해도 실제 학습은 진짜 torch 로 한다.
    """
    xd = _pad2d(x.data, padding)
    wd = weight.data
    N, C, H, W = xd.shape
    F, C2, KH, KW = wd.shape
    if C != C2:
        raise RuntimeError(f"채널이 안 맞습니다: 입력 {C}, 필터 {C2}")
    if H < KH or W < KW:
        raise RuntimeError("필터가 입력보다 큽니다.")

    cols, OH, OW = _im2col(xd, KH, KW, stride)
    w2 = wd.reshape(F, -1)
    out = (cols @ w2.T).reshape(N, OH, OW, F).transpose(0, 3, 1, 2)

    def back(g):
        g = _np.asarray(g)
        g2 = g.transpose(0, 2, 3, 1).reshape(-1, F)
        gw = (g2.T @ cols).reshape(wd.shape)
        gx = _col2im(g2 @ w2, xd.shape, KH, KW, stride, OH, OW)
        if padding:
            gx = gx[:, :, padding:-padding, padding:-padding]
        return (gx, gw) if bias is None else (gx, gw, g.sum(axis=(0, 2, 3)))

    parents = (x, weight) if bias is None else (x, weight, bias)
    return x._make(out if bias is None else out + bias.data.reshape(1, -1, 1, 1), parents, back)


def max_pool2d(x, kernel_size, stride=None):
    stride = stride or kernel_size
    xd = x.data
    N, C, H, W = xd.shape
    OH = (H - kernel_size) // stride + 1
    OW = (W - kernel_size) // stride + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (kernel_size, kernel_size), axis=(2, 3))
    win = win[:, :, ::stride, ::stride, :, :].reshape(N, C, OH, OW, -1)
    idx = win.argmax(axis=-1)
    out = _np.take_along_axis(win, idx[..., None], axis=-1).squeeze(-1)

    def back(g):
        # 최댓값이 있던 자리로만 기울기를 보낸다. 자리를 평평한 번호로 바꿔 한 번에 흩뿌린다 —
        # N·C·OH·OW 를 파이썬으로 도는 것보다 훨씬 빠르고, 결과는 같다.
        g = _np.asarray(g)
        di, dj = _np.divmod(idx, kernel_size)
        n_i, c_i, oh_i, ow_i = _np.ogrid[:N, :C, :OH, :OW]
        h = oh_i * stride + di
        w = ow_i * stride + dj
        flat = ((n_i * C + c_i) * H + h) * W + w
        gx = _np.zeros(xd.size, dtype=g.dtype)
        _np.add.at(gx, flat.reshape(-1), g.reshape(-1))
        return (gx.reshape(xd.shape),)

    return x._make(out, (x,), back)


def sin(t): return t._make(_np.sin(t.data), (t,), lambda g: (g * _np.cos(t.data),), "SinBackward0")
def cos(t): return t._make(_np.cos(t.data), (t,), lambda g: (-g * _np.sin(t.data),), "CosBackward0")


def clamp(t, min=None, max=None):
    out = _np.clip(t.data, min, max)
    inside = _np.ones_like(t.data, dtype=bool)
    if min is not None:
        inside &= t.data >= min
    if max is not None:
        inside &= t.data <= max
    return t._make(out, (t,), lambda g: (g * inside,), "ClampBackward0")



# ---------------------------------------------------------------- 원소별 함수
#
# 대부분 numpy 한 줄에 미분 한 줄이다. 미분이 없는 것(floor·sign 등)은 기울기를 0 으로 둔다 —
# torch 도 그렇게 한다. 계단 함수의 미분은 거의 모든 곳에서 0 이기 때문이다.

def _unary(name, forward, derivative=None, op=None):
    def fn(t):
        t = _wrap(t)
        out = forward(t.data)
        if derivative is None:
            return Tensor(out)
        return t._make(out, (t,), lambda g: (g * derivative(t.data, out),), op or f"{name}Backward0")
    fn.__name__ = name
    return fn


log2 = _unary("Log2", _np.log2, lambda x, o: 1.0 / (x * _np.log(2)))
log10 = _unary("Log10", _np.log10, lambda x, o: 1.0 / (x * _np.log(10)))
rsqrt = _unary("Rsqrt", lambda x: 1.0 / _np.sqrt(x), lambda x, o: -0.5 * o / x)
square = _unary("Square", _np.square, lambda x, o: 2 * x)
reciprocal = _unary("Reciprocal", _np.reciprocal, lambda x, o: -o * o)
tan = _unary("Tan", _np.tan, lambda x, o: 1 + o * o)
sinh = _unary("Sinh", _np.sinh, lambda x, o: _np.cosh(x))
cosh = _unary("Cosh", _np.cosh, lambda x, o: _np.sinh(x))
erf = _unary("Erf", lambda x: _np.vectorize(_math.erf)(x).astype(x.dtype),
             lambda x, o: 2 / _np.sqrt(_np.pi) * _np.exp(-x * x))
# 계단 모양 — 미분이 거의 모든 곳에서 0 이다. torch 도 0 을 준다.
sign = _unary("Sign", _np.sign)
floor = _unary("Floor", _np.floor)
ceil = _unary("Ceil", _np.ceil)
round = _unary("Round", lambda x: _np.round(x))


def neg(t): return -_wrap(t)
def pow(t, exponent): return _wrap(t) ** exponent


# ---------------------------------------------------------------- 비교

def _compare(name, fn):
    def cmp(a, b):
        a = _wrap(a)
        bd = b.data if isinstance(b, Tensor) else b
        return Tensor(fn(a.data, bd))
    cmp.__name__ = name
    return cmp


eq = _compare("eq", _np.equal)
ne = _compare("ne", _np.not_equal)
lt = _compare("lt", _np.less)
le = _compare("le", _np.less_equal)
gt = _compare("gt", _np.greater)
ge = _compare("ge", _np.greater_equal)
logical_and = _compare("logical_and", _np.logical_and)
logical_or = _compare("logical_or", _np.logical_or)
isnan = _unary("IsNan", _np.isnan)
isinf = _unary("IsInf", _np.isinf)


def logical_not(t): return Tensor(_np.logical_not(_wrap(t).data))


def maximum(a, b):
    a, b = _wrap(a), _wrap(b)
    pick = a.data >= b.data
    return a._make(_np.maximum(a.data, b.data), (a, b),
                   lambda g: (g * pick, g * ~pick), "MaximumBackward0")


def minimum(a, b):
    a, b = _wrap(a), _wrap(b)
    pick = a.data <= b.data
    return a._make(_np.minimum(a.data, b.data), (a, b),
                   lambda g: (g * pick, g * ~pick), "MinimumBackward0")


# ---------------------------------------------------------------- 모양·선택

def split(t, size, dim=0):
    t = _wrap(t)
    n = t.data.shape[dim]
    sizes = size if isinstance(size, (list, tuple)) else \
        [size] * (n // size) + ([n % size] if n % size else [])
    cuts, out, start = [], [], 0
    for sz in sizes[:-1]:
        start += sz
        cuts.append(start)
    return tuple(t[_slice_at(dim, s, e)] for s, e in zip([0] + cuts, cuts + [n]))


def chunk(t, chunks, dim=0):
    t = _wrap(t)
    n = t.data.shape[dim]
    size = -(-n // chunks)
    return split(t, size, dim)


def _slice_at(dim, start, end):
    return tuple(slice(None) for _ in range(dim)) + (slice(start, end),)


def unbind(t, dim=0):
    t = _wrap(t)
    return tuple(t[_slice_at(dim, i, i + 1)].squeeze(dim) for i in range(t.data.shape[dim]))


def narrow(t, dim, start, length):
    return _wrap(t)[_slice_at(dim, start, start + length)]


def flip(t, dims):
    t = _wrap(t)
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    return t._make(_np.flip(t.data, dims).copy(), (t,),
                   lambda g: (_np.flip(_np.asarray(g), dims).copy(),), "FlipBackward0")


def roll(t, shifts, dims=None):
    t = _wrap(t)
    return t._make(_np.roll(t.data, shifts, dims), (t,),
                   lambda g: (_np.roll(_np.asarray(g), _negate(shifts), dims),), "RollBackward0")


def _negate(shifts):
    return -shifts if isinstance(shifts, int) else tuple(-s for s in shifts)


def index_select(t, dim, index):
    t = _wrap(t)
    idx = index.data.astype(int) if isinstance(index, Tensor) else _np.asarray(index, dtype=int)
    return t[_index_at(dim, idx)]


def _index_at(dim, idx):
    return tuple(slice(None) for _ in range(dim)) + (idx,)


def masked_select(t, mask):
    t = _wrap(t)
    m = mask.data.astype(bool) if isinstance(mask, Tensor) else _np.asarray(mask, dtype=bool)
    return t[m]


def gather(t, dim, index):
    """index 가 가리키는 자리를 뽑는다. 분류에서 정답 클래스의 확률을 꺼낼 때 쓴다."""
    t = _wrap(t)
    idx = index.data.astype(int) if isinstance(index, Tensor) else _np.asarray(index, dtype=int)
    out = _np.take_along_axis(t.data, idx, axis=dim)
    shape = t.data.shape

    def back(g):
        z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
        _np.put_along_axis(z, idx, _np.asarray(g), axis=dim)
        return (z,)

    return t._make(out, (t,), back, "GatherBackward0")


def repeat_interleave(t, repeats, dim=None):
    t = _wrap(t)
    return Tensor(_np.repeat(t.data, repeats, axis=dim))


def tile(t, reps):
    return Tensor(_np.tile(_wrap(t).data, reps))


def movedim(t, source, destination):
    return Tensor(_np.moveaxis(_wrap(t).data, source, destination))


# ---------------------------------------------------------------- 축약(추가)

def prod(t, dim=None):
    t = _wrap(t)
    out = _np.prod(t.data, axis=dim)
    return t._make(out, (t,), lambda g: (_np.asarray(g) * out / t.data,), "ProdBackward0")


def median(t, dim=None):
    """torch 는 원소가 짝수일 때 **가운데 둘 중 작은 쪽**을 준다. numpy 는 평균을 낸다 —
    그대로 쓰면 조용히 다른 값이 나온다."""
    t = _wrap(t)
    if dim is None:
        flat = _np.sort(t.data.reshape(-1))
        return Tensor(flat[(flat.size - 1) // 2])
    values = _np.sort(t.data, axis=dim)
    idx = (values.shape[dim] - 1) // 2
    picked = _np.take(values, idx, axis=dim)
    order = _np.argsort(t.data, axis=dim)
    return _MinMax(Tensor(picked), Tensor(_np.take(order, idx, axis=dim)))


def norm(t, p=2, dim=None):
    t = _wrap(t)
    if p == 1:
        return t.abs().sum(dim=dim)
    return (t * t).sum(dim=dim) ** 0.5


def cumsum(t, dim):
    t = _wrap(t)
    return t._make(_np.cumsum(t.data, axis=dim), (t,),
                   lambda g: (_np.flip(_np.cumsum(_np.flip(_np.asarray(g), dim), axis=dim), dim),),
                   "CumsumBackward0")


def cumprod(t, dim):
    return Tensor(_np.cumprod(_wrap(t).data, axis=dim))


def count_nonzero(t, dim=None):
    return Tensor(_np.count_nonzero(_wrap(t).data, axis=dim))


def _pick(t, idx, dim, op):
    """뽑은 값에 **기울기 길을 남긴다.** 뽑기만 하고 끊으면 학습이 조용히 멈춘다 —
    top-k 샘플링이나 정렬을 끼운 손실에서 그 일이 난다."""
    values = _np.take_along_axis(t.data, idx, axis=dim)
    shape = t.data.shape

    def back(g):
        z = _np.zeros(shape, dtype=_np.asarray(g).dtype)
        _np.put_along_axis(z, idx, _np.asarray(g), axis=dim)
        return (z,)

    return t._make(values, (t,), back, op)


def topk(t, k, dim=-1, largest=True):
    """상위 k개의 (값, 번호). 32장의 top-k 샘플링이 이것이다."""
    t = _wrap(t)
    order = _np.argsort(t.data, axis=dim)
    if largest:
        order = _np.flip(order, axis=dim)
    idx = _np.take(order, _np.arange(k), axis=dim)
    return _MinMax(_pick(t, idx, dim, "TopkBackward0"), Tensor(idx))


def sort(t, dim=-1, descending=False):
    t = _wrap(t)
    idx = _np.argsort(t.data, axis=dim)
    if descending:
        idx = _np.flip(idx, axis=dim)
    return _MinMax(_pick(t, idx, dim, "SortBackward0"), Tensor(idx))


def argsort(t, dim=-1, descending=False):
    return sort(t, dim, descending).indices


def unique(t, sorted=True, return_counts=False):
    values, counts = _np.unique(_wrap(t).data, return_counts=True)
    return (Tensor(values), Tensor(counts)) if return_counts else Tensor(values)


# ---------------------------------------------------------------- 선형대수

def mm(a, b): return _wrap(a) @ _wrap(b)
def bmm(a, b): return _wrap(a) @ _wrap(b)


def dot(a, b): return (_wrap(a) * _wrap(b)).sum()


def outer(a, b):
    a, b = _wrap(a), _wrap(b)
    return a.reshape(-1, 1) @ b.reshape(1, -1)


def diag(t):
    return Tensor(_np.diag(_wrap(t).data))


def trace(t):
    return Tensor(_np.trace(_wrap(t).data))


def einsum(equation, *operands):
    ops = [_wrap(o) for o in operands]
    return Tensor(_np.einsum(equation, *[o.data for o in ops]))


def empty(*shape, dtype=None):
    return zeros(*shape, dtype=dtype)




def leaky_relu(t, negative_slope=0.01):
    t = _wrap(t)
    pick = t.data > 0
    return t._make(_np.where(pick, t.data, negative_slope * t.data), (t,),
                   lambda g: (g * _np.where(pick, 1.0, negative_slope),), "LeakyReluBackward0")


def elu(t, alpha=1.0):
    t = _wrap(t)
    pick = t.data > 0
    out = _np.where(pick, t.data, alpha * (_np.exp(_np.minimum(t.data, 0)) - 1))
    return t._make(out, (t,), lambda g: (g * _np.where(pick, 1.0, out + alpha),),
                   "EluBackward0")


def silu(t):
    """x·σ(x). Swish 라고도 한다."""
    t = _wrap(t)
    sig = 1.0 / (1.0 + _np.exp(-_np.clip(t.data, -60, 60)))
    return t._make(t.data * sig, (t,),
                   lambda g: (g * (sig * (1 + t.data * (1 - sig))),), "SiluBackward0")


def gelu(t):
    return _gelu(_wrap(t))


def log_softmax(t, dim=-1):
    t = _wrap(t)
    shifted = t.data - t.data.max(axis=dim, keepdims=True)
    out = shifted - _np.log(_np.exp(shifted).sum(axis=dim, keepdims=True))
    soft = _np.exp(out)

    def back(g):
        g = _np.asarray(g)
        return (g - soft * g.sum(axis=dim, keepdims=True),)

    return t._make(out, (t,), back, "LogSoftmaxBackward0")


def dropout(t, p=0.5, training=True):
    if not training or p == 0:
        return _wrap(t)
    t = _wrap(t)
    mask = (_rng.random(t.data.shape) > p).astype(t.data.dtype) / (1 - p)
    return t * Tensor(mask)


def avg_pool2d(x, kernel_size, stride=None):
    stride = stride or kernel_size
    xd = x.data
    N, C, H, W = xd.shape
    OH = (H - kernel_size) // stride + 1
    OW = (W - kernel_size) // stride + 1
    win = _np.lib.stride_tricks.sliding_window_view(xd, (kernel_size, kernel_size), axis=(2, 3))
    win = win[:, :, ::stride, ::stride, :, :]
    out = win.mean(axis=(4, 5))
    area = kernel_size * kernel_size

    def back(g):
        g = _np.asarray(g) / area
        gx = _np.zeros_like(xd)
        for i in range(kernel_size):
            for j in range(kernel_size):
                gx[:, :, i:i + OH * stride:stride, j:j + OW * stride:stride] += g
        return (gx,)

    return x._make(out, (x,), back, "AvgPool2DBackward0")


def _pool_all(x):
    """AdaptiveAvgPool2d(1) 만 지원한다 — 흔한 것은 그것뿐이고, 나머지는 거절한다."""
    return x.mean(dim=2).mean(dim=2).reshape(x.data.shape[0], x.data.shape[1], 1, 1)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    out = centered / (var + eps) ** 0.5
    if weight is not None:
        out = out * weight
    return out + bias if bias is not None else out


def embedding(idx, weight):
    ids = idx.data.astype(int)
    dim = weight.data.shape[1]
    out = weight.data[ids]

    def back(g):
        gw = _np.zeros_like(weight.data)
        _np.add.at(gw, ids.reshape(-1), _np.asarray(g).reshape(-1, dim))
        return (gw,)

    return weight._make(out, (weight,), back, "EmbeddingBackward0")


def nll_loss(log_probs, target):
    n = log_probs.data.shape[0]
    picked = log_probs[_np.arange(n), target.data.astype(int)]
    return -picked.mean()


def l1_loss(pred, target):
    return (pred - target).abs().mean()


def smooth_l1_loss(pred, target, beta=1.0):
    """작은 오차는 제곱, 큰 오차는 절댓값. 이상치에 덜 흔들린다."""
    diff = pred - target
    small = _np.abs(diff.data) < beta
    return (where(Tensor(small), 0.5 * diff * diff / beta, diff.abs() - 0.5 * beta)).mean()


def pad(x, padding, value=0.0):
    """마지막 차원부터 (앞, 뒤) 순으로 받는다 — torch 의 규칙이다."""
    x = _wrap(x)
    pairs = [(0, 0)] * x.data.ndim
    for i in range(0, len(padding), 2):
        pairs[-(i // 2 + 1)] = (padding[i], padding[i + 1])
    out = _np.pad(x.data, pairs, constant_values=value)
    cuts = tuple(slice(a, s - b if b else None) for (a, b), s in zip(pairs, out.shape))
    return x._make(out, (x,), lambda g: (_np.asarray(g)[cuts],), "PadBackward0")


def normalize(x, p=2, dim=1, eps=1e-12):
    x = _wrap(x)
    denom = norm(x, p=p, dim=dim)
    return x / maximum(denom.unsqueeze(dim), Tensor(_np.array(eps, dtype=_DEFAULT_DTYPE)))


def cosine_similarity(a, b, dim=1, eps=1e-8):
    a, b = _wrap(a), _wrap(b)
    return (a * b).sum(dim=dim) / maximum(
        norm(a, dim=dim) * norm(b, dim=dim), Tensor(_np.array(eps, dtype=_DEFAULT_DTYPE)))



def tril(t, diagonal=0):
    return Tensor(_np.tril(t.data, k=diagonal))


def triu(t, diagonal=0):
    return Tensor(_np.triu(t.data, k=diagonal))


def allclose(a, b, rtol=1e-5, atol=1e-8):
    return bool(_np.allclose(_wrap(a).data, _wrap(b).data, rtol=rtol, atol=atol))


def equal(a, b):
    return bool(_np.array_equal(_wrap(a).data, _wrap(b).data))


def isfinite(t):
    return Tensor(_np.isfinite(_wrap(t).data))


def bincount(t):
    return Tensor(_np.bincount(_wrap(t).data.astype(_np.int64)))


def _to_plain(obj):
    """텐서를 numpy 로 바꿔 저장 가능한 형태로. 중첩 dict/list 도 따라간다."""
    if isinstance(obj, Tensor):
        return {"__tensor__": obj.data}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_plain(v) for v in obj)
    return obj


def _from_plain(obj):
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            return Tensor(obj["__tensor__"])
        return {k: _from_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_from_plain(v) for v in obj)
    return obj


def save(obj, path):
    """진짜 torch 와 달리 pickle 한 겹만 쓴다. 브라우저에도 가상 파일시스템이 있어 경로가 통한다."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(_to_plain(obj), f)


def load(path, **kwargs):
    import pickle
    with open(path, "rb") as f:
        return _from_plain(pickle.load(f))


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

    def __call__(self, fn):
        def wrapper(*a, **k):
            with no_grad():
                return fn(*a, **k)
        return wrapper


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed_all(seed):
        return None

    @staticmethod
    def synchronize():
        _unsupported("torch.cuda.synchronize")


cuda = _Cuda()


# ================================================================ nn

class _NN:
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
        shape = (1, -1, 1, 1)
        if self.training:
            # 평균·분산을 **그래프 안에서** 계산해야 한다. numpy 로 빼서 상수처럼 쓰면
            # x → mean → y 로 흐르는 길이 끊겨 기울기가 틀리고, weight 에는 아예 안 간다.
            # (BatchNorm 순방향만 대조하고 역방향은 안 봤던 탓에 오래 남아 있었다.)
            mean = x.mean(dim=0).mean(dim=1).mean(dim=1)          # (C,)
            centered = x - mean.reshape(shape)
            var = (centered * centered).mean(dim=0).mean(dim=1).mean(dim=1)

            # 진짜 torch 는 두 곳에서 다른 분산을 쓴다 — 정규화는 편향(ddof=0),
            # running_var 갱신은 비편향(ddof=1). 둘 다 편향으로 두면 값이 2.6% 어긋난다.
            with no_grad():
                unbiased = x.data.var(axis=(0, 2, 3), ddof=1)
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


def _gelu(t):
    """torch 의 기본 gelu(정확형)와 같은 식."""
    from math import erf, sqrt
    vec = _np.vectorize(lambda v: 0.5 * v * (1 + erf(v / sqrt(2))))
    out = vec(t.data).astype(t.data.dtype)
    grad = _np.vectorize(
        lambda v: 0.5 * (1 + erf(v / sqrt(2))) + v * _np.exp(-v * v / 2) / sqrt(2 * _np.pi))
    return t._make(out, (t,), lambda g: (g * grad(t.data),), "GeluBackward0")


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


for _cls in (GELU, SiLU, LeakyReLU, ELU, Softmax, LogSoftmax, AvgPool2d,
             AdaptiveAvgPool2d, Unflatten, L1Loss, SmoothL1Loss, NLLLoss, BatchNorm1d):
    setattr(nn, _cls.__name__, _cls)
for _name in ("Conv1d", "Conv3d", "BatchNorm3d", "Upsample", "MaxPool1d", "MaxPool3d"):
    setattr(nn, _name, _nn_unsupported(_name))
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


class _Functional:
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


# ================================================================ optim

class Optimizer:
    """torch 의 옵티마이저는 `param_groups` 라는 목록을 들고 있다.

    학습률을 읽고 쓰는 표준 경로가 `opt.param_groups[0]["lr"]` 이고, 스케줄러도 그것을
    고친다. `opt.lr` 로 두면 짧지만 **남의 코드가 안 돌고, 스케줄러를 직접 못 쓴다.**
    """

    def __init__(self, params, defaults):
        params = list(params)
        if params and isinstance(params[0], dict):
            self.param_groups = [dict(defaults, **g, ) for g in params]
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

    def state_dict(self):
        """torch 와 같은 모양 — 6장이 가르치는 "이어서 학습하기"가 이것에 걸려 있다.

        Adam 은 파라미터마다 보폭을 기억한다. 그 기억을 버리고 이어 학습하면
        손실이 한 번 튀었다가 다시 내려간다 — 오류는 안 나고 곡선만 이상해진다.
        """
        order = {id(p): i for i, p in enumerate(self.params)}
        return {
            "state": {order[k]: {n: (Tensor(v.copy()) if isinstance(v, _np.ndarray) else v)
                                 for n, v in st.items()}
                      for k, st in self.state.items() if k in order},
            "param_groups": [{k: v for k, v in g.items() if k != "params"}
                             | {"params": [order[id(p)] for p in g["params"]]}
                             for g in self.param_groups],
        }

    def load_state_dict(self, state):
        params = self.params
        self.state = {}
        for index, st in state.get("state", {}).items():
            self.state[id(params[int(index)])] = {
                n: (v.data.copy() if isinstance(v, Tensor) else v) for n, v in st.items()}
        for group, saved in zip(self.param_groups, state.get("param_groups", [])):
            for key, value in saved.items():
                if key != "params":
                    group[key] = value
        return self

    def step(self):
        raise NotImplementedError

    def __repr__(self):
        lr = self.param_groups[0].get("lr")
        return f"{type(self).__name__} (lr: {lr})"


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                if group["momentum"]:
                    st = self._state(p)
                    buf = st.get("momentum_buffer")
                    buf = g if buf is None else group["momentum"] * buf + g
                    st["momentum_buffer"] = buf
                    g = buf
                p._array = p.data - group["lr"] * g


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    decoupled = False          # AdamW 는 True — 감쇠를 기울기가 아니라 가중치에 직접 건다

    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("step", 0)
                st.setdefault("exp_avg", _np.zeros_like(p.data))
                st.setdefault("exp_avg_sq", _np.zeros_like(p.data))
                st["step"] += 1
                g = p.grad.data
                if group["weight_decay"] and not self.decoupled:
                    g = g + group["weight_decay"] * p.data
                st["exp_avg"] = b1 * st["exp_avg"] + (1 - b1) * g
                st["exp_avg_sq"] = b2 * st["exp_avg_sq"] + (1 - b2) * (g * g)
                mh = st["exp_avg"] / (1 - b1 ** st["step"])
                vh = st["exp_avg_sq"] / (1 - b2 ** st["step"])
                new = p.data - group["lr"] * mh / (_np.sqrt(vh) + group["eps"])
                if group["weight_decay"] and self.decoupled:
                    new = new - group["lr"] * group["weight_decay"] * p.data
                p._array = new


class AdamW(Adam):
    """Adam 과 같은데 가중치 감쇠를 **기울기가 아니라 가중치에 직접** 건다."""

    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st.setdefault("square_avg", _np.zeros_like(p.data))
                g = p.grad.data
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p.data
                st["square_avg"] = (group["alpha"] * st["square_avg"]
                                    + (1 - group["alpha"]) * g * g)
                p._array = p.data - group["lr"] * g / (_np.sqrt(st["square_avg"]) + group["eps"])




class _Scheduler:
    """스케줄러는 `optimizer.param_groups` 의 lr 을 고친다. 에폭마다 한 번 부른다."""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class StepLR(_Scheduler):
    """step_size 에폭마다 gamma 를 곱한다. 4장의 "멀리서는 성큼, 가까이서는 조심"."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.gamma ** (self.last_epoch // self.step_size)
                for base in self.base_lrs]


class MultiStepLR(_Scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones, self.gamma = sorted(milestones), gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        passed = sum(1 for m in self.milestones if m <= self.last_epoch)
        return [base * self.gamma ** passed for base in self.base_lrs]


class ExponentialLR(_Scheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.gamma ** self.last_epoch for base in self.base_lrs]


class CosineAnnealingLR(_Scheduler):
    """T_max 에폭에 걸쳐 코사인 곡선으로 내린다. 끝에서 부드럽게 멎는다."""

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base - self.eta_min)
                * (1 + _math.cos(_math.pi * self.last_epoch / self.T_max)) / 2
                for base in self.base_lrs]


class LambdaLR(_Scheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * self.lr_lambda(self.last_epoch) for base in self.base_lrs]


class ReduceLROnPlateau:
    """**좋아지지 않을 때** 내린다. 다른 것들과 달리 `step(metric)` 으로 값을 받는다 —
    6장의 조기 종료와 같은 발상이고, 멈추는 대신 보폭을 줄인다."""

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10,
                 threshold=1e-4, min_lr=0.0):
        self.optimizer = optimizer
        self.mode, self.factor, self.patience = mode, factor, patience
        self.threshold, self.min_lr = threshold, min_lr
        self.best = None
        self.num_bad_epochs = 0

    def _better(self, value):
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best * (1 - self.threshold)
        return value > self.best * (1 + self.threshold)

    def step(self, metric):
        metric = float(metric.item() if isinstance(metric, Tensor) else metric)
        if self._better(metric):
            self.best, self.num_bad_epochs = metric, 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                for group in self.optimizer.param_groups:
                    group["lr"] = max(group["lr"] * self.factor, self.min_lr)
                self.num_bad_epochs = 0

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class _LRScheduler:
    StepLR = StepLR
    MultiStepLR = MultiStepLR
    ExponentialLR = ExponentialLR
    CosineAnnealingLR = CosineAnnealingLR
    LambdaLR = LambdaLR
    ReduceLROnPlateau = ReduceLROnPlateau


class _Optim:
    SGD = SGD
    Adam = Adam
    AdamW = AdamW
    RMSprop = RMSprop
    Optimizer = Optimizer
    lr_scheduler = _LRScheduler


optim = _Optim()


# ================================================================ utils.data

class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)


class SequentialSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class WeightedRandomSampler:
    """드문 것을 더 자주 뽑는다. 1000명 중 10명이 환자인 데이터에서 배치에 환자가
    한 명도 없는 일을 막는다 — 5장이 가르치는 그것이다."""

    def __init__(self, weights, num_samples, replacement=True, generator=None):
        self.weights = _np.asarray(
            [float(w) for w in (weights.tolist() if isinstance(weights, Tensor) else weights)])
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rng = self.generator.rng() if self.generator is not None else _rng
        p = self.weights / self.weights.sum()
        return iter(rng.choice(len(p), size=self.num_samples,
                               replace=self.replacement, p=p).tolist())

    def __len__(self):
        return self.num_samples


class RandomSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(_rng.permutation(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 num_workers=0, drop_last=False, collate_fn=None):
        if sampler is not None and shuffle:
            raise ValueError("sampler 와 shuffle 은 같이 쓸 수 없습니다.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.sampler = sampler or (RandomSampler(dataset) if shuffle else SequentialSampler(dataset))

    def __iter__(self):
        batch = []
        for i in self.sampler:
            batch.append(self.dataset[i])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._collate(batch)

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _collate(self, batch):
        if self.collate_fn:
            return self.collate_fn(batch)
        cols = list(zip(*batch))
        return tuple(stack([_wrap(x) for x in col]) for col in cols)


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        for d in self.datasets:
            if i < len(d):
                return d[i]
            i -= len(d)
        raise IndexError(i)


def random_split(dataset, lengths, generator=None):
    rng = generator.rng() if generator is not None else _rng
    idx = rng.permutation(len(dataset)).tolist()
    out, start = [], 0
    for n in lengths:
        out.append(Subset(dataset, idx[start:start + n]))
        start += n
    return out


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


class _UtilsData:
    Dataset = Dataset
    TensorDataset = TensorDataset
    ConcatDataset = ConcatDataset
    Subset = Subset
    DataLoader = DataLoader
    RandomSampler = RandomSampler
    SequentialSampler = SequentialSampler
    WeightedRandomSampler = WeightedRandomSampler
    random_split = staticmethod(random_split)


class _Utils:
    data = _UtilsData()


utils = _Utils()
