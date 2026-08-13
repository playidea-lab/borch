"""browsertorch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._base import (
    Size, _DEFAULT_DTYPE, _NP_TO_DTYPE, _like_torch, _np, _tensor_repr, _unsupported,
    dtype, float32,
)

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


class _GradMode:
    """`no_grad` 의 스위치를 **객체 하나에 담는다.** 그냥 모듈 전역으로 두면 안 된다.

    파일을 쪼개면 `no_grad` 와 `_make` 가 다른 모듈에 놓이는데, 전역 이름은 모듈마다
    따로 생긴다. `no_grad` 가 자기 모듈의 이름만 False 로 바꾸고 `_make` 는 옛 값을
    계속 읽는다 — **예외도 경고도 없이 `no_grad` 가 안 먹는다.** 쪼개다 실제로 걸렸고,
    골든이 아니라 `test_diff` 가 잡았다.

    객체 하나를 들여오면 어느 모듈에서 보든 같은 것을 본다.
    """

    enabled = True


_grad_mode = _GradMode()


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
        needs = _grad_mode.enabled and any(p.requires_grad for p in parents)
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

    def _cast(self, target):
        """실수끼리의 형 변환은 **그래프를 잇는다.** torch 가 그렇다.

        전에는 `Tensor(..., self.requires_grad)` 였는데, 그러면 결과가 `requires_grad=True`
        라고 말하면서 부모가 없다. `backward()` 는 예외 없이 잘 돌고 원래 텐서의 `.grad`
        만 `None` 으로 남는다 — **예외도 경고도 없다.** 이 저장소가 가장 싫어하는 모양이고,
        `x.float()` 는 튜토리얼 코드에 흔해서 조용히 학습이 안 되는 자리가 된다.

        정수·불리언으로 가는 변환은 여기 안 온다. 거기서는 torch 도 기울기를 끊는다.
        """
        out = self.data.astype(target)
        return self._make(out, (self,), lambda g: (g.astype(self.data.dtype),), "ToCopyBackward0")

    def float(self):
        return self._cast(_np.float32)

    def long(self):
        return Tensor(self.data.astype(_np.int64))

    def int(self):
        return Tensor(self.data.astype(_np.int64))

    def bool(self):
        return Tensor(self.data.astype(_np.bool_))

    def double(self):
        return self._cast(_np.float64)

    def type(self, dt):
        target = dt.np if isinstance(dt, dtype) else dt
        if _np.dtype(target).kind != "f":
            return Tensor(self.data.astype(target))
        return self._cast(target)

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
        """나머지. **기울기는 나누어지는 쪽으로 그대로 흐른다** — `a % b` 는 `a` 에 대해
        기울기 1 이다(계단이 뛰는 자리를 빼면). 나누는 쪽으로는 `-floor(a/b)` 다."""
        od = o.data if isinstance(o, Tensor) else o
        parents = (self, o) if isinstance(o, Tensor) else (self,)

        def back(g):
            g = _np.asarray(g)
            if isinstance(o, Tensor):
                return (g, -g * _np.floor_divide(self.data, od))
            return (g,)

        return self._make(_np.mod(self.data, od), parents, back, "RemainderBackward0")

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
        if self.requires_grad and _grad_mode.enabled:
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
        if self.requires_grad and _grad_mode.enabled:
            raise RuntimeError("기울기가 필요한 텐서에는 제자리 대입을 할 수 없습니다.")
        key = tuple(i.data if isinstance(i, Tensor) else i for i in idx) \
            if isinstance(idx, tuple) else (idx.data if isinstance(idx, Tensor) else idx)
        self.data[key] = value.data if isinstance(value, Tensor) else value

    # ---- 제자리 연산
    #
    # **버퍼를 진짜로 고친다.** 코어의 뷰는 numpy 배열을 공유하므로(실측:
    # `np.shares_memory(a.data, a.view(2,2).data)` 가 참) 뷰를 고치면 원본도 바뀐다 —
    # torch 와 같다. 자매는 TF.js 텐서가 불변이라 그 전파를 못 하고, 거기서 둘이 갈린다.
    #
    # torch 가 거절하는 자리도 따라 거절한다. 잎에 기울기가 켜져 있으면 못 고치고,
    # 역전파에 필요한 값을 덮는 것도 못 한다 — 후자는 우리가 안 잡지만, 잎 쪽은 잡는다.

    def _inplace(self, fn, what):
        if self.requires_grad and _grad_mode.enabled:
            raise RuntimeError(_like_torch(
                f"기울기가 필요한 잎 텐서에는 `{what}` 을(를) 쓸 수 없습니다. "
                "`with torch.no_grad():` 안에서 하거나 제자리가 아닌 연산을 쓰세요.",
                "a leaf Variable that requires grad is being used in an in-place operation"))
        out = fn()
        self.data[...] = out.data if isinstance(out, Tensor) else out
        return self

    def add_(self, other, alpha=1):
        return self._inplace(lambda: self + (other * alpha if alpha != 1 else other), "add_")

    def sub_(self, other, alpha=1):
        return self._inplace(lambda: self - (other * alpha if alpha != 1 else other), "sub_")

    def mul_(self, other):
        return self._inplace(lambda: self * other, "mul_")

    def div_(self, other):
        return self._inplace(lambda: self / other, "div_")

    def pow_(self, exponent):
        return self._inplace(lambda: self ** exponent, "pow_")

    def neg_(self):
        return self._inplace(lambda: -self, "neg_")

    def zero_(self):
        return self._inplace(lambda: _np.zeros_like(self.data), "zero_")

    def fill_(self, value):
        return self._inplace(lambda: _np.full_like(self.data, value), "fill_")

    def copy_(self, other):
        return self._inplace(
            lambda: (other.data if isinstance(other, Tensor) else _np.asarray(other)),
            "copy_")

    def clamp_(self, min=None, max=None):
        return self._inplace(lambda: _np.clip(self.data, min, max), "clamp_")

    clip_ = clamp_

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

    def var(self, dim=None, unbiased=True, keepdim=False):
        """**그래프 안에서** 계산한다.

        전에는 `np.var` 로 값만 떼어 돌려줬다. 값은 맞지만 기울기가 안 흐른다 —
        분산을 손실에 끼우면 학습이 조용히 멈춘다. ROADMAP 11번이 topk·sort 에서
        잡은 것과 같은 종류인데, 여기는 검사가 없어서 남아 있었다.
        """
        n = self.data.size if dim is None else self.data.shape[dim]
        mean = self.mean(dim=dim, keepdim=True) if dim is not None else self.mean()
        centered = self - mean
        total = (centered * centered).sum(dim=dim, keepdim=keepdim)
        return total / float(n - 1 if unbiased else n)

    def std(self, dim=None, unbiased=True, keepdim=False):
        return self.var(dim=dim, unbiased=unbiased, keepdim=keepdim) ** 0.5

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
        # `intp` 다 — wasm32 에서 int64 를 주면 거절한다. `_ops.repeat_interleave` 참고.
        return Tensor(_np.bincount(self.data.astype(_np.intp)))


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


