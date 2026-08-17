"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._base import (
    Size, _DEFAULT_DTYPE, _NP_TO_DTYPE, _like_torch, _needs_float,
    _no_complex128, _np, _refuses_bool, _tensor_repr, _unsupported, dtype,
    float32,
)

# ---------------------------------------------------------------- Tensor

def _conj(x):
    """**정칙 함수의 역방향에 붙는 켤레.**

    복소수 기울기 규약이 `z.grad = ∂L/∂re + i·∂L/∂im` 이라(실측), 정칙 함수 `f` 의
    역방향이 `conj(f'(z))·g` 다. 실수에서는 켤레가 항등이라 **실수만 넣어 보면 이
    자리가 있는지 없는지 알 수 없다** — 그래서 실수 코드에 그냥 넣어 두어도 안전하고,
    빼먹으면 복소수에서만 부호가 뒤집힌다.
    """
    return _np.conj(x) if _np.asarray(x).dtype.kind == "c" else x


def _keep(out, source, dim, keepdim):
    """접힌 축을 크기 1 로 되살린다. `numpy` 의 `keepdims` 가 없는 함수들이 쓴다.

    **`argmax`·`argmin` 이 그런 자리다** — numpy 가 그 인자를 안 받아서, 축을 되살릴
    곳이 여기밖에 없다. 안 되살리면 축 하나가 사라진 채로 브로드캐스팅이 **맞아
    버리고**, 값만 틀린 채 끝까지 간다.
    """
    if not keepdim or dim is None:
        return out
    shape = list(_np.shape(source.data))
    shape[dim if dim >= 0 else dim + len(shape)] = 1
    return _np.reshape(out, shape)


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
#   범주:  bool(0) < 정수(1) < 실수(2) < 복소수(3)
#   규칙:  참여한 것 중 가장 높은 범주를 고르고, 그 범주에 속한 것들 중 큰 것을 쓴다.
#          낮은 범주는 높은 범주를 **끌어올리지 않는다.**
#
# 그래서 float32 + int64 가 torch 에서는 float32 다 (numpy 는 float64 로 올린다).
# 여기를 numpy 에 맡기면 학습자는 틀린 규칙을 배운다.
#
# **복소수는 한 칸 더 위다**(실측: `complex64 + int64` 가 complex64). 그런데 실수 쪽
# 정밀도가 복소수 쪽으로 **건너온다** — `complex64 + float64` 가 **complex128** 이다
# (실측). 범주 규칙만으로는 그 자리가 안 나오므로 따로 적는다.

_CATEGORY = {"b": 0, "i": 1, "u": 1, "f": 2, "c": 3}
_RANK = {_np.dtype("bool"): 0, _np.dtype("int64"): 10,
         _np.dtype("float32"): 20, _np.dtype("float64"): 21,
         _np.dtype("complex64"): 30, _np.dtype("complex128"): 31}
_DEFAULT_BY_CATEGORY = {0: _np.dtype("bool"), 1: _np.dtype("int64"),
                        2: _np.dtype("float32"), 3: _np.dtype("complex64")}
# 실수의 정밀도가 복소수로 건너오는 표. 배정도 실수 하나가 배정도 복소수를 만든다.
_WIDENS_COMPLEX = {_np.dtype("float64"): _np.dtype("complex128")}


def _category(dt):
    return _CATEGORY.get(_np.dtype(dt).kind, 2)


def result_type(a, b):
    """두 텐서 dtype 의 결과 타입. torch.result_type 과 같은 규칙."""
    da, db = _np.dtype(a), _np.dtype(b)
    cat = max(_category(da), _category(db))
    same = [d for d in (da, db) if _category(d) == cat]
    out = max(same, key=lambda d: _RANK.get(d, 0))
    if cat == 3:
        # **실수 쪽 정밀도가 건너온다.** `complex64 + float64` 가 complex128 이다.
        for d in (da, db):
            wide = _WIDENS_COMPLEX.get(d)
            if wide is not None and _RANK[wide] > _RANK[out]:
                out = wide
    return out


def _scalar_category(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return 1
    if isinstance(value, complex):
        return 3
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
        # **배정도가 여기서 막힌다.** `float64` 가 없으니 `complex128` 도 없다.
        #
        # 이 한 줄이 목문이다 — 승격이 그것을 만드는 유일한 길이 `complex64 + float64`
        # 이고, 그 결과도 여기를 지나 텐서가 된다. 자리마다 막으면 새 연산이 생길
        # 때마다 빠뜨린다.
        if self._array.dtype == _np.complex128:
            _no_complex128("이 연산")
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
            # **손실은 실수여야 한다.** torch 가 그 자리에서 멈춘다(실측).
            #
            # 그리고 이 한 줄이 복소수 기울기 규약 전체를 떠받친다 — 손실이 늘 실수라야
            # `z.grad = ∂L/∂re + i·∂L/∂im` 이 잘 정의된다. 복소 손실을 받아 주면
            # Wirtinger 의 나머지 절반을 정해야 하고, 그것은 안 정한 자리다.
            if self.data.dtype.kind == "c":
                raise RuntimeError(_like_torch(
                    "복소수 손실에는 backward() 를 부를 수 없습니다 — "
                    "`.real`·`.abs()` 로 실수를 만든 뒤 부르세요.",
                    "grad can be implicitly created only for real scalar outputs "
                    "but got torch.complex64"))
            gradient = _np.ones_like(self.data)

        seed = _np.asarray(gradient, dtype=self.data.dtype)
        if seed.shape != self.data.shape:
            # **모양을 여기서 본다.** 안 보면 numpy 가 나중에 브로드캐스팅을 시도하고,
            # 맞으면 조용히 틀린 기울기가 나오고 안 맞으면 `ValueError` 가 원인에서
            # 먼 자리에서 뜬다. torch 는 여기서 `RuntimeError` 로 멈춘다 — 실측했다.
            raise RuntimeError(_like_torch(
                f"gradient 의 모양 {tuple(seed.shape)} 이 값의 모양 "
                f"{tuple(self.data.shape)} 과 다릅니다.",
                f"Mismatch in shape: grad_output[0] has a shape of "
                f"torch.Size({list(seed.shape)}) and output[0] has a shape of "
                f"torch.Size({list(self.data.shape)})."))

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

        grads = {id(self): seed}
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
        """장치와 **형** 둘 다 받는다. torch 의 `to` 가 그 둘을 한 이름에 담는다.

        **오래 형을 조용히 버리고 있었다.** 장치 문자열만 보고 나머지는 무시한 채
        `self` 를 돌려줬으므로 `x.to(torch.float32)` 가 아무 일도 안 했다 — 예외도
        경고도 없이 원래 형 그대로다. 그 꼴이 교재 코드에 흔하고(`x.to(device)` 와
        나란히 쓰인다), 정수 텐서에서는 그 뒤 나눗셈이 **정수 나눗셈으로 조용히**
        갈린다. 축약에 `dtype=` 를 붙이다가 드러났다 — 그쪽이 이 함수를 부르는데
        형이 안 바뀌어서.
        """
        target = None
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, str):
                if a != "cpu":
                    _unsupported(f"장치 '{a}'")
                continue
            if isinstance(a, Tensor):
                target = a.data.dtype
            elif a is not None and not isinstance(a, bool):
                target = a
        return self if target is None else self.type(target)

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
        """곱셈. **복소수에서는 국소 도함수에 켤레가 붙는다.**

        규약이 `z.grad = ∂L/∂re + i·∂L/∂im` 이라, 정칙 함수 `f` 의 역방향은
        `conj(f'(z))·g` 다. `d(ab)/da = b` 이므로 `conj(b)·g` — 실수에서는 켤레가
        항등이라 같은 식이 되고, **복소수에서만 갈린다.** 실수 입력으로는 이 자리가
        절대 안 보인다.
        """
        return self._binary(o, _np.multiply,
                            lambda g, a, b: g * _conj(b),
                            lambda g, a, b: g * _conj(a), "MulBackward0")

    __rmul__ = __mul__

    def __truediv__(self, o):
        # torch 의 나눗셈은 정수·불리언끼리여도 기본 실수형(float32)을 낸다.
        # numpy 에 맡기면 int64/int64 가 float64 가 된다.
        def div(a, b):
            out = _np.divide(a, b)
            return out.astype(_DEFAULT_DTYPE) if a.dtype.kind not in "fc" else out
        # 곱셈과 같은 자리 — `d(a/b)/da = 1/b`, `d(a/b)/db = −a/b²` 에 켤레가 붙는다.
        return self._binary(o, div, lambda g, a, b: g / _conj(b),
                            lambda g, a, b: -g * _conj(a) / _conj(b * b),
                            "DivBackward0")

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
        def back(g):
            """**1차원은 자리를 하나 빌려 쓴다.**

            numpy 는 앞이 1차원이면 `(1, n)` 으로, 뒤가 1차원이면 `(n, 1)` 로 늘려
            곱하고 **그 빌린 축을 결과에서 지운다.** 역방향에서는 그 축을 도로 끼워야
            전치가 성립한다 — 안 끼우면 `swapaxes(v, -1, -2)` 가 1차원에서 그대로
            멈춘다(`axis -2 is out of bounds`). 2차원끼리로만 재면 안 드러나는 자리라
            `mv` 를 넣으면서 처음 밟았다.
            """
            g = _np.asarray(g)
            a, b = self.data, o.data
            aa = a.reshape((1,) + a.shape) if a.ndim == 1 else a
            bb = b.reshape(b.shape + (1,)) if b.ndim == 1 else b
            lead = _np.broadcast_shapes(aa.shape[:-2], bb.shape[:-2])
            gg = g.reshape(lead + (aa.shape[-2], bb.shape[-1]))
            da = gg @ _np.swapaxes(bb, -1, -2)
            db = _np.swapaxes(aa, -1, -2) @ gg
            # 배치가 한쪽으로만 퍼진 자리는 크기가 안 맞는다 — 그때는 그대로 둔다.
            return (da.reshape(a.shape) if da.size == a.size else da,
                    db.reshape(b.shape) if db.size == b.size else db)

        return self._make(
            self.data @ o.data, (self, o), back,
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

    def all(self, dim=None, keepdim=False):
        """전부 참인가. **축과 `keepdim` 을 받는다** — 안 받으면 조용히 틀린다.

        축을 안 받으면 `x.all(dim=1)` 이 전체 축약으로 떨어져 스칼라가 나오는데,
        그 뒤 브로드캐스팅이 **맞아 버려서** 값만 틀린 채 끝까지 간다. `keepdim` 도
        같은 갈래다 — 축 하나가 사라진 모양이 우연히 브로드캐스팅으로 들어맞는다.
        """
        return Tensor(_np.all(self.data, axis=dim, keepdims=bool(keepdim)))

    def any(self, dim=None, keepdim=False):
        return Tensor(_np.any(self.data, axis=dim, keepdims=bool(keepdim)))

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
        got = out.data if isinstance(out, Tensor) else _np.asarray(out)
        # **모양이 바뀌는 제자리 연산이 있다.** `transpose_`·`squeeze_`·`unsqueeze_` 는
        # 값이 아니라 보는 틀을 고친다. 값만 되쓰면 3×2 를 2×3 자리에 넣으라는 말이
        # 되어 터지고, 정사각으로만 물으면 **모양이 안 바뀐 채 통과한다** — 실제로
        # 2×2 케이스가 이것을 못 봤다. 그 자리에서는 배열을 갈아 끼운다. numpy 의
        # 전치·차원 넣기는 뷰라서 버퍼는 그대로 공유하고 틀만 바뀐다.
        if got.shape != self.data.shape:
            self._array = got                  # `.data` 는 일부러 ndarray 를 거절한다
            return self
        self.data[...] = got
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

    def _cast_first(self, dtype):
        """`dtype=` 를 받은 축약이 맨 앞에서 부른다. 안 주면 자기 자신이다.

        **규칙 한 줄이다: 넣기 전에 바꾼다.** 접고 나서 바꾸는 것이 아니다 —
        실측이 그것을 못 박는다:

            torch.tensor([1.7, -2.3, 0.9]).sum(dtype=torch.int64)  →  -1

        먼저 접으면 `0.3` 이고 그것을 정수로 깎아도 `0` 이다. 먼저 깎으면
        `[1, -2, 0]` 이라 합이 `-1` 이다. 답이 갈리는 자리이고, 그래서 이 한 줄이
        `dtype=` 의 정의다. 정수 입력에 `mean(dtype=float32)` 이 도는 것도 같은
        이유다 — 거절하던 것은 **결과 형을 못 정해서**였고, 그것을 받았으니 돈다.
        """
        return self if dtype is None else self.to(dtype)

    def _reduce(self, fn, dim, keepdim, grad_fn, op=None):
        axis = dim
        out = fn(self.data, axis=axis, keepdims=keepdim)
        return self._make(out, (self,), lambda g: (grad_fn(g, axis, keepdim),), op)

    def sum(self, dim=None, keepdim=False, dtype=None):
        if dtype is not None:
            # **결과 형도 못 박는다.** 캐스팅만 하면 누적 규칙이 다시 올린다 —
            # `sum(dtype=bool)` 이 int64 로 나온다(torch 는 `True` 다, 실측).
            return self._cast_first(dtype).sum(dim, keepdim).to(dtype)
        shape = self.data.shape

        def back(g, axis, kd):
            g = _np.asarray(g)
            if axis is not None and not kd:
                g = _np.expand_dims(g, axis)
            return _np.broadcast_to(g, shape).copy()

        return self._reduce(_np.sum, dim, keepdim, back,
                            "SumBackward0" if dim is None else "SumBackward1")

    def mean(self, dim=None, keepdim=False, dtype=None):
        if dtype is not None:
            # **정수로 내리라는 것은 거절한다**(실측). `dtype=` 이 입력 쪽 거절은
            # 풀어 주지만(정수 입력 + `dtype=float32` 는 돈다) 결과가 정수인 평균은
            # 여전히 답이 없다 — 푸는 것은 **결과 형을 못 정하던 것**뿐이다.
            _needs_float(
                _np.empty(0, dtype=_np.dtype(getattr(dtype, "np", dtype))),
                "평균의 결과 형은 실수여야 합니다.",
                "mean(): could not infer output dtype. Input dtype must be either "
                "a floating point or complex dtype")
            return self._cast_first(dtype).mean(dim, keepdim).to(dtype)
        _needs_float(
            self.data,
            "평균은 실수에만 있습니다 — 정수·참거짓 칸에는 나눗셈의 답이 안 들어갑니다. "
            "`.float()` 을 먼저 부르세요.",
            "mean(): could not infer output dtype. Input dtype must be either "
            "a floating point or complex dtype")
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
        # **번호도 축을 지켜야 한다.** 값만 살리면 `x.gather(1, m.indices)` 가 랭크
        # 어긋남으로 멈추거나 — 더 나쁘게 — 브로드캐스팅으로 통과한다. torch 는 둘
        # 다 `(2, 1)` 을 준다(실측).
        return _MinMax(out, Tensor(_np.expand_dims(idx, d) if keepdim else idx))

    def _elementwise_extreme(self, other, pick, name):
        """**동점이면 반씩 나눈다.** torch 가 그렇다 — `maximum(2, 2)` 의 기울기는
        양쪽 다 0.5 다. `_ops.maximum` 과 같은 규칙이고, 여기서 다시 적는 이유는
        `_tensor.py` 가 `_ops` 를 못 들여오기 때문이다(순환)."""
        other = other if isinstance(other, Tensor) else Tensor(_np.asarray(other))
        tie = self.data == other.data
        left = _np.where(tie, 0.5, (self.data > other.data).astype(self.data.dtype))
        if name == "MinimumBackward0":
            left = 1.0 - left
        return self._make(pick(self.data, other.data), (self, other),
                          lambda g: (g * left, g * (1.0 - left)), name)

    # **한 이름에 셋이 들어 있다.** torch 의 `max` 는 인자에 따라 다른 것을 낸다:
    # 인자가 없으면 전부의 최댓값 하나, 축이면 `(값, 번호)` 쌍, **텐서면 칸마다의
    # 최댓값**이다. 마지막 갈래가 없어서 `torch.max(a, b)` 가 축인 줄 알고
    # `'Tensor' object cannot be interpreted as an integer` 로 멈췄다.

    def max(self, dim=None, keepdim=False):
        if isinstance(dim, Tensor):
            return self._elementwise_extreme(dim, _np.maximum, "MaximumBackward0")
        return self._argreduce(_np.max, _np.argmax, dim, keepdim)

    def min(self, dim=None, keepdim=False):
        if isinstance(dim, Tensor):
            return self._elementwise_extreme(dim, _np.minimum, "MinimumBackward0")
        return self._argreduce(_np.min, _np.argmin, dim, keepdim)

    def argmax(self, dim=None, keepdim=False):
        _refuses_bool(self.data, "argmax 는 참거짓을 받지 않습니다.",
                      "argmax(): does not support bool input")
        return Tensor(_keep(_np.argmax(self.data, axis=dim), self, dim, keepdim))

    def argmin(self, dim=None, keepdim=False):
        _refuses_bool(self.data, "argmin 은 참거짓을 받지 않습니다.",
                      "argmin(): does not support bool input")
        return Tensor(_keep(_np.argmin(self.data, axis=dim), self, dim, keepdim))

    def var(self, dim=None, unbiased=True, keepdim=False):
        """**그래프 안에서** 계산한다.

        전에는 `np.var` 로 값만 떼어 돌려줬다. 값은 맞지만 기울기가 안 흐른다 —
        분산을 손실에 끼우면 학습이 조용히 멈춘다. ROADMAP 11번이 topk·sort 에서
        잡은 것과 같은 종류인데, 여기는 검사가 없어서 남아 있었다.
        """
        _needs_float(
            self.data,
            "분산·표준편차는 실수에만 있습니다. `.float()` 을 먼저 부르세요.",
            "std and var only support floating point and complex dtypes")
        n = self.data.size if dim is None else self.data.shape[dim]
        mean = self.mean(dim=dim, keepdim=True) if dim is not None else self.mean()
        centered = self - mean
        total = (centered * centered).sum(dim=dim, keepdim=keepdim)
        return total / float(n - 1 if unbiased else n)

    def std(self, dim=None, unbiased=True, keepdim=False):
        return self.var(dim=dim, unbiased=unbiased, keepdim=keepdim) ** 0.5

    def abs(self):
        """크기. **복소수에서는 결과가 실수이고 기울기가 `z/|z|` 다.**

        `sign` 을 그대로 쓰면 안 된다 — numpy 의 복소 `sign` 은 torch 의 것과 다르고,
        애초에 torch 는 복소수에 `sign` 을 거절한다(실측). 여기서 필요한 것은
        `∂|z|/∂re = re/|z|`, `∂|z|/∂im = im/|z|` 를 묶은 `z/|z|` 다.
        """
        if self.data.dtype.kind == "c":
            mag = _np.abs(self.data)
            safe = _np.where(mag == 0, 1.0, mag)
            return self._make(
                mag.astype(_DEFAULT_DTYPE), (self,),
                lambda g: ((_np.asarray(g) * self.data / safe).astype(self.data.dtype),))
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
        # **참·거짓은 거절한다** — torch 가 `"bincount_cpu" not implemented for
        # 'Bool'` 로 멈춘다(실측). `_ops.bincount` 에도 같은 가드가 있는데 메서드가
        # 그 문을 안 지나고 numpy 를 직접 부르고 있었다 — 두 벌은 이렇게 갈린다.
        _refuses_bool(self.data, "bincount 는 참거짓을 받지 않습니다.",
                      '"bincount_cpu" not implemented for \'Bool\'',
                      kind=NotImplementedError)
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


