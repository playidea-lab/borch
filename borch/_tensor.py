"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

from ._base import (
    Size, _DEFAULT_DTYPE, _NP_TO_DTYPE, _TYPE_NAMES, _like_torch, _needs_float,
    _no_complex128, _np, _refuses_bool, _tensor_repr, _unsupported,
    device as _device, dtype, float32,
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
        """**int32 가 없다 — 그래서 거절한다.**

        오래 int64 를 내주고 있었다. 값은 그럴듯한데 `x.int().dtype == torch.int32`
        를 보는 코드가 자기 컴퓨터에서 갈리고, 그때 원인은 이 줄이 아니라 훨씬
        뒤에서 드러난다. torch 의 `.int()` 는 int32 다(실측) — 우리에게 그 칸이
        없으면 다른 칸을 대신 주는 것이 아니라 멈추는 편이 낫다.
        """
        _unsupported("`.int()`(int32)")

    def bool(self):
        return Tensor(self.data.astype(_np.bool_))

    def double(self):
        return self._cast(_np.float64)

    def type_as(self, other):
        """`other` 의 형으로 맞춘다. **없는 기능이 아니라 안 적힌 기능이었다** —
        `type()` 은 있는데 이쪽이 없어서 `AttributeError` 로 멈췄다."""
        return self.type(other.dtype if isinstance(other, Tensor)
                         else _np.asarray(other).dtype)

    def cfloat(self):
        """complex64. 이 칸은 있다 — `cdouble`·`chalf` 와 달리."""
        return Tensor(self.data.astype(_np.complex64))


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
            # **`device` 물건도 받는다.** `x.to(device)` 가 튜토리얼의 꼴이고,
            # 그 `device` 는 문자열이 아니라 `torch.device(...)` 다. 문자열만 보면
            # 그 줄이 조용히 아무것도 안 한다 — 형 인자로 읽혀 `target` 이 되면
            # 오히려 `numpy` 가 "형이 아니다" 로 멈춘다.
            if isinstance(a, _device):
                if a.type != "cpu":
                    _unsupported(f"장치 '{a}'")
                continue
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
        """**`self` 를 그냥 돌려주고 있었다.** 이미 연속이면 그것이 맞지만 아니면
        틀리다 — 전치한 뒤 불러도 여전히 비연속이었고, 그래서 `is_contiguous()` 가
        torch 와 반대 답을 낸다. 기울기는 항등이다.
        """
        if self.data.flags["C_CONTIGUOUS"]:
            return self
        return self._make(_np.ascontiguousarray(self.data), (self,),
                          lambda g: (_np.asarray(g),), "CloneBackward0")

    # ── 묻는 것 넷 ────────────────────────────────────────────────────────
    #
    # 넷 다 **torch 에서 실제로 거짓이 나온다** — 그것을 먼저 재고 넣었다. 늘 참인
    # 술어는 우리 구현을 굳히는 것뿐이라 케이스로 물어도 묻는 게 아니다.
    #
    # `is_contiguous` 는 우리에게도 뜻이 있다. numpy 가 전치·permute·성긴 슬라이스를
    # **뷰**로 주므로 그 자리에서 거짓이 되고, torch 와 같은 답이다. 브라우저 쪽은
    # 뷰를 안 만들어서 늘 참인데, 그것은 이 술어의 이야기가 아니라 **뷰의 이야기**다.

    def is_floating_point(self):
        return bool(self.data.dtype.kind == "f")

    # ── 조밀 텐서에도 답이 있는 여섯 ──────────────────────────────────────
    #
    # 이름만 보면 **희소·장치·양자화라 없는 게 맞다**고 세게 된다. 실제로 재보니
    # torch 가 조밀 텐서에서 여섯을 그냥 해낸다 — 희소·양자화 기계가 필요한 것이
    # 아니라 "이 텐서는 조밀하다"·"CPU 다" 라는 답이 있는 것이다. 이름으로 세면
    # 없는 결함을 굳히게 되고, 나중에 누가 구현하면 **초록이던 케이스가 빨개진다.**
    #
    # `to_dense`·`dequantize` 는 **기울기를 나른다**(실측) — 항등이라 그대로 지난다.

    def dense_dim(self):
        """조밀 텐서는 축이 전부 조밀하다."""
        return self.data.ndim

    def sparse_dim(self):
        """조밀 텐서에 희소 축은 없다."""
        return 0

    def to_dense(self):
        """이미 조밀하다 — torch 도 같은 객체를 돌려준다(실측)."""
        return self

    def dequantize(self):
        """실수에서는 항등이다. 양자화 dtype 이 필요한 자리가 아니다."""
        return self

    def storage_offset(self):
        """우리 배열은 언제나 자기 버퍼의 처음부터다 — 저장을 나눠 갖지 않는다."""
        return 0

    def get_device(self):
        """CPU 텐서는 -1 이다(실측). 장치 번호가 없다는 뜻이지 오류가 아니다."""
        return -1

    def is_signed(self):
        """참거짓과 부호 없는 정수만 거짓이다 — 실수·정수·복소수는 참."""
        return bool(self.data.dtype.kind in "fci")

    def is_contiguous(self):
        return bool(self.data.flags["C_CONTIGUOUS"])

    def is_nonzero(self):
        """**원소가 하나여야 한다.** 여럿이면 torch 가 "모호하다" 로 멈춘다 —
        `if tensor:` 가 조용히 첫 원소를 보는 일을 막는 자리다."""
        if self.data.size != 1:
            raise RuntimeError(_like_torch(
                f"값이 {self.data.size}개인 텐서의 참거짓은 모호합니다.",
                "Boolean value of Tensor with "
                f"{'no values' if self.data.size == 0 else 'more than one value'}"
                " is ambiguous"))
        return bool(self.data.reshape(-1)[0] != 0)

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
            # **축이 없으면 번호도 없고, 그래서 규칙이 반대가 된다.** 번호를 주는
            # `max(dim=0)` 은 고른 한 자리에 기울기를 전부 주지만, 번호가 없는
            # `max()` 는 동점에 **고르게 나눈다** — `amax()` 와 같은 규칙이다
            # (실측: [3,5,5,1,5] → [0, ⅓, ⅓, 0, ⅓]).
            #
            # 예전에는 여기서 `Tensor(...)` 를 맨손으로 만들어 **그래프가 조용히
            # 끊겼다.** 값 검사는 전부 통과했다 — 값은 맞았기 때문이다. 드러난 것은
            # `backward()` 를 불렀을 때이고, 그때 나오는 말은 "requires_grad 가 아닌
            # 텐서" 라 사용자를 가리킨다. 연산이 없다고는 아무도 안 말해 준다.
            value = _np.asarray(np_fn(self.data))
            hit = (self.data == value).astype(self.data.dtype)
            share = hit / hit.sum()
            return self._make(value, (self,),
                              lambda g: (_np.asarray(g) * share,))
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




# ── 제자리 판을 짝에서 만든다 ────────────────────────────────────────────────
#
# torch 의 제자리 연산은 `x.add_(1)` 처럼 이름 끝에 밑줄이 붙고, **학습 루프에서
# 관용구다** — `p.data.add_(-lr * g)` 를 안 쓰는 교재가 드물다. 그런데 마흔한 개가
# 빠져 있었다. 짝(`x.add`)은 전부 있고 `_inplace` 관용구도 있었으니, 없던 것은
# **잇는 줄** 하나씩이었다.
#
# 손으로 마흔한 벌을 적지 않는다. 서로 다른 것이 이름뿐이면 그 마흔한 벌은 언젠가
# 한 자리만 다르게 고쳐지고, 그 한 자리는 아무도 안 본다. 표를 두고 붙인다.
#
# **`i0_`·`clamp_min_`·`clamp_max_` 는 셋 다 `_ops` 에 이미 있었다** — 모듈 함수로만
# 있고 메서드로는 없었던 것이다. `borch.i0_(x)` 는 되고 `x.i0_()` 는 안 됐는데,
# 교재가 쓰는 쪽은 뒤쪽이다.
#
# **둘은 이 표에 못 넣는다.** 이름 끝에 밑줄이 붙었다고 짝과 같은 연산이 아니다 —
# 마흔한 개를 전부 torch 와 대조해서 확인했고 거기서 갈렸다.
#
#   `bernoulli_` 는 **짝과 다른 연산이다.** `x.bernoulli()` 는 `x` 를 확률로 읽는데
#   `x.bernoulli_(p=0.5)` 는 `x` 를 무시하고 `p` 로 채운다(실측: `[0,1,0,1]` 을 넣어도
#   결과가 매번 다르다). 짝에서 만들었으면 확률이 0·1 인 자리는 확정이라 값이 맞고,
#   **가운데 확률에서만 조용히 틀렸을** 것이다.
#
#   `float_power_` 는 **torch 가 거절한다.** `float_power` 의 결과가 언제나 float64 라
#   float32 자리에 되쓸 수 없어서다. 우리에게는 float64 가 아예 없으므로 이 연산은
#   제자리로는 영영 안 된다.
_INPLACE_FROM_PAIR = (
    "bitwise_and_", "bitwise_left_shift_", "bitwise_not_",
    "bitwise_or_", "bitwise_right_shift_", "bitwise_xor_", "clamp_max_",
    "clamp_min_", "digamma_", "divide_", "erfinv_",
    "floor_divide_", "fmod_", "gcd_", "greater_", "greater_equal_", "i0_",
    # **셋이 빠져 있었다.** 마흔한 개를 재놓고 목록을 손으로 옮겨 적으면서
    # `index_reduce_`·`scatter_reduce_` 를 흘렸고, `resize_as_` 는 모듈에만 있던
    # 이름을 정리하다 같이 빠졌다 — **잰 것과 적은 것이 갈리는** 자리다.
    "index_reduce_", "scatter_reduce_",
    "lcm_", "lerp_", "less_", "less_equal_", "lgamma_", "logical_and_",
    "logical_not_", "logical_or_", "logical_xor_", "multiply_", "mvlgamma_",
    "nan_to_num_", "nextafter_", "not_equal_", "put_", "remainder_", "renorm_",
    "subtract_", "t_", "true_divide_",
)


def _bind_inplace(name):
    pair = name[:-1]

    def method(self, *args, **kw):
        return self._inplace(lambda: getattr(self, pair)(*args, **kw), name)

    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = f"`{pair}` 를 제자리에서. 값을 되쓰고 자기를 돌려준다."
    return method


for _name in _INPLACE_FROM_PAIR:
    setattr(Tensor, _name, _bind_inplace(_name))
del _name


def _float_power_(self, exponent):
    """**언제나 거절한다.** torch 도 float32 자리에서는 거절한다 — `float_power` 의
    결과가 float64 이고 그것을 float32 에 되쓸 수 없어서다. 우리에게는 float64 가
    없으므로 어떤 dtype 에서도 안 된다. 값을 내주면 그 코드가 진짜 torch 에서 깨진다."""
    del exponent
    raise RuntimeError(_like_torch(
        f"`float_power_` 는 {self.dtype} 자리에 쓸 수 없습니다 — 결과가 배정도라 "
        "되쓸 곳이 없습니다. `x.float_power(k)` 로 새 텐서를 받으세요.",
        f"the base given to float_power_ has dtype {str(self.dtype).split('.')[-1].capitalize()} "
        "but the operation's result requires dtype Double"))


Tensor.float_power_ = _float_power_


# ── 없는 형은 이름째 거절한다 ────────────────────────────────────────────────
#
# 예전에는 `AttributeError: 'Tensor' object has no attribute 'half'` 가 났다.
# **그 문구는 오타와 구별이 안 된다** — 배우는 사람은 자기가 이름을 잘못 쳤다고
# 읽지, 그 형이 여기 없다고는 안 읽는다.
#
# `half`·`bfloat16` 은 튜토리얼의 혼합정밀도 절에서 **실제로 치는 줄**이라, 그
# 자리에서 나는 말이 셋 다 같아야 한다. 값이 아니라 **거절 문구**를 맞추는 자리이고,
# 그런 자리는 서로 대조해도 안 걸린다 — `i0` 이 물렸던 갈래와 같다.
_ABSENT_DTYPES = {
    "half": "float16", "bfloat16": "bfloat16", "chalf": "complex32",
    "cdouble": "complex128", "byte": "uint8", "char": "int8", "short": "int16",
}


def _bind_absent_dtype(name, shown):
    def method(self):
        del self
        _unsupported(f"`.{name}()`({shown})")

    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = f"{shown} 은 이 축소판에 없다 — 다른 칸을 대신 주지 않는다."
    return method


for _dname, _shown in _ABSENT_DTYPES.items():
    setattr(Tensor, _dname, _bind_absent_dtype(_dname, _shown))
del _dname, _shown


# ── 모듈에만 있던 이름을 메서드로도 낸다 ──────────────────────────────────────
#
# torch 는 거의 모든 연산을 **둘 다** 준다 — `torch.igamma(x, y)` 와 `x.igamma(y)`.
# 우리는 모듈 쪽만 있고 메서드가 없는 자리가 열셋이었다. `borch.arctan2(x, y)` 는
# 되고 `x.arctan2(y)` 는 `AttributeError` 였는데, **교재가 쓰는 쪽은 메서드다.**
#
# 결속의 `_base.py` 에 같은 이야기가 이미 적혀 있었다("`borch.t(x)` 는 되고
# `x.t()` 는 안 되는 한쪽 고리만 남았다") — 그쪽은 그때 메꿨고 코어는 안 메꿨다.
#
# **`lstsq`·`solve` 는 여기 없다.** torch 가 1.9 에서 폐기하고 **지금은 거절한다** —
# 이름은 남아 있는데 부르면 멈춘다. 처음에 "torch 에 있는 이름" 으로 세어 붙였다가,
# 인자 차례를 재보려다 torch 쪽이 거절하는 것을 봤다. 우리가 답을 내주면 그 코드가
# 진짜 torch 에서 깨진다 — **관대한 것도 갈리는 것이다.**
_METHOD_FROM_MODULE = (
    "arctan2", "igamma", "igammac", "geqrf",
    # **짝이 없는 밑줄 이름.** `resize_as_` 는 모듈에만 있고 `resize_as` 라는 짝이
    # 아예 없어서, 파생표에 넣으면 없는 이름을 찾다가 `AttributeError` 로 멈춘다 —
    # 표에 이름을 적는 것과 그 표가 그 이름을 만들 수 있는 것은 다른 일이다.
    "resize_as_",
)


def _deprecated_by_torch(name, instead):
    def method(self, *args, **kw):
        del self, args, kw
        raise RuntimeError(_like_torch(
            f"`{name}` 은 torch 1.9 에서 없어졌습니다 — `{instead}` 을(를) 쓰세요.",
            f"This function was deprecated since version 1.9 and is now removed. "
            f"Please use the `torch.linalg.{instead}` function instead."))

    method.__name__ = name
    return method


def _polygamma(self, n):
    """**인자가 뒤집혀 있다.** 모듈은 `polygamma(n, x)` 이고 메서드는
    `x.polygamma(n)` 이다 — torch 가 그렇게 둔다(실측).

    표로 그냥 붙였다가 `TypeError` 로 걸렸다. 안 걸렸으면 차수와 입력이 뒤바뀐 채
    값이 나왔을 자리다 — `lu_solve` 가 같은 모양이었다.
    """
    from . import _ops
    return _ops.polygamma(n, self)


def _bind_from_module(name):
    def method(self, *args, **kw):
        from . import _ops
        return getattr(_ops, name)(self, *args, **kw)

    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = f"`borch.{name}` 을 메서드로. torch 는 둘 다 준다."
    return method


for _mname in _METHOD_FROM_MODULE:
    setattr(Tensor, _mname, _bind_from_module(_mname))
del _mname


def _is_same_size(self, other):
    """모양이 같은가. **값이 아니라 모양만** 본다."""
    return tuple(self.data.shape) == tuple(_np.asarray(other.data).shape)


def _fill_diagonal_(self, value, wrap=False):
    """대각을 채운다. `wrap` 은 세로로 긴 행렬에서 대각을 **감아 이어 간다**."""
    if self.requires_grad and _grad_mode.enabled:
        raise RuntimeError(_like_torch(
            "기울기가 필요한 잎 텐서에는 `fill_diagonal_` 을 쓸 수 없습니다.",
            "a leaf Variable that requires grad is being used in an in-place operation"))
    _np.fill_diagonal(self._array, value, wrap=wrap)
    return self


def _requires_grad_(self, requires_grad=True):
    """**교재의 관용구다** — `x.requires_grad_()` 로 잎을 켠다. 자기를 돌려준다."""
    if requires_grad and self.data.dtype.kind not in "fc":
        raise RuntimeError(
            "정수 텐서에는 기울기가 흐르지 않습니다. 미분은 실수에서만 정의됩니다 "
            "— `.float()` 로 바꾸세요.")
    self.requires_grad = bool(requires_grad)
    return self


def _share_memory_(self):
    """프로세스 사이 공유는 여기 없다. torch 도 CPU 에서는 자기를 돌려준다."""
    return self


Tensor.polygamma = _polygamma
Tensor.lstsq = _deprecated_by_torch("lstsq", "lstsq")
Tensor.solve = _deprecated_by_torch("solve", "solve")
Tensor.is_same_size = _is_same_size
Tensor.is_distributed = lambda self: False
Tensor.is_inference = lambda self: False
Tensor.fill_diagonal_ = _fill_diagonal_
Tensor.requires_grad_ = _requires_grad_
Tensor.share_memory_ = _share_memory_

# 짝이 생겼으니 밑줄 판도 같은 표에서 나온다.
for _iname in ("arctan2_", "igamma_", "igammac_", "polygamma_"):
    setattr(Tensor, _iname, _bind_inplace(_iname))
del _iname


# ── torch 가 **속성**으로 주는 술어들 ────────────────────────────────────────
#
# 대부분 "이 텐서가 어디 있는가·어떤 저장인가" 를 묻고 우리 답은 하나로 정해져
# 있다. **그래도 이름이 있어야 한다** — 없으면 `if x.is_cuda:` 가 `AttributeError`
# 로 멈추는데, torch 에서는 그냥 거짓으로 지나가는 줄이다.
#
# **`is_leaf` 만 진짜 계산이다.** 잎은 연산에서 나오지 않은 텐서이고, 그것이
# 거짓이면 `.grad` 가 안 쌓인다 — 값이 하나로 정해진 나머지와 성격이 다르다.
_ALWAYS_FALSE = (
    "is_cuda", "is_ipu", "is_maia", "is_meta", "is_mkldnn", "is_mps", "is_mtia",
    "is_nested", "is_quantized", "is_sparse", "is_sparse_csr", "is_vulkan",
    "is_xla", "is_xpu",
)

for _pname in _ALWAYS_FALSE:
    setattr(Tensor, _pname, property(lambda self: False, doc="이 축소판에는 없다."))
del _pname

# CPU 에 산다. **결속은 여기서 갈린다** — 값이 GPU 버퍼에 있으므로 거짓이다.
Tensor.is_cpu = property(lambda self: True)
Tensor.is_leaf = property(
    lambda self: not self._parents,
    doc="연산에서 안 나온 텐서. **거짓이면 `.grad` 가 안 쌓인다.**")
Tensor.retains_grad = property(
    lambda self: False,
    doc="`retain_grad()` 가 없으므로 언제나 거짓이다 — torch 도 잎에서는 거짓이다.")


def _is_pinned(self):
    """**메서드다** — 위의 것들과 달리 괄호가 있다(실측). 고정 메모리는 없다."""
    del self
    return False


def _is_coalesced(self):
    """희소 전용이라 **조밀 텐서에서는 멈춘다** — torch 도 그렇다(실측)."""
    raise RuntimeError(_like_torch(
        "조밀 텐서에는 coalesce 상태가 없습니다.",
        "is_coalesced expected sparse coordinate tensor layout but got Strided"))


# **`is_neg`·`is_pinned` 만 메서드다** — 괄호가 있다(실측). 속성으로 두면
# `x.is_neg` 가 참거짓이 아니라 묶인 메서드를 돌려주는데, torch 쪽은
# 묶인 메서드다 — 이번엔 **우리가 속성으로 만든 것이 갈림**이었다.
Tensor.is_neg = lambda self: False
Tensor.is_pinned = _is_pinned
Tensor.is_coalesced = _is_coalesced


# ── 짝이 없는 제자리 판 여덟 ────────────────────────────────────────────────
#
# 파생표로는 못 만든다 — 짝이 아예 없다. 다섯은 torch 가 해내고 셋은 희소 전용이라
# **torch 도 조밀 텐서에서 멈춘다.** 이름만 보고 "제자리니까 다 만든다" 로 묶으면
# 뒤의 셋에서 우리가 더 관대해진다.

def _apply_(self, fn):
    """칸마다 파이썬 함수를 건다. **torch 도 CPU 에서만 된다** — 느린 길이고,
    그래서 값이 아니라 편의를 주는 이름이다."""
    _refuse_leaf_inplace(self, "apply_")
    flat = self.data.reshape(-1)
    self.data[...] = _np.array([fn(v.item()) for v in flat],
                               dtype=self.data.dtype).reshape(self.data.shape)
    return self


def _map_(self, other, fn):
    _refuse_leaf_inplace(self, "map_")
    a, b = self.data.reshape(-1), _np.broadcast_to(other.data, self.data.shape).reshape(-1)
    self.data[...] = _np.array([fn(x.item(), y.item()) for x, y in zip(a, b)],
                               dtype=self.data.dtype).reshape(self.data.shape)
    return self


def _map2_(self, other, third, fn):
    _refuse_leaf_inplace(self, "map2_")
    a = self.data.reshape(-1)
    b = _np.broadcast_to(other.data, self.data.shape).reshape(-1)
    c = _np.broadcast_to(third.data, self.data.shape).reshape(-1)
    self.data[...] = _np.array(
        [fn(x.item(), y.item(), z.item()) for x, y, z in zip(a, b, c)],
        dtype=self.data.dtype).reshape(self.data.shape)
    return self


def _resize_(self, *sizes):
    """**키우면 새 칸의 값이 정해지지 않는다** — torch 는 쓰레기값을 준다(실측:
    우연히 0 이 나오기도 한다). 우리는 0 으로 채운다. 값을 굳힐 수 없는 자리라
    골든은 **줄이는 쪽과 모양만** 묻는다."""
    _refuse_leaf_inplace(self, "resize_")
    shape = tuple(sizes[0]) if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)) \
        else tuple(int(s) for s in sizes)
    want = int(_np.prod(shape)) if shape else 1
    flat = self.data.reshape(-1)
    if want <= flat.size:
        self._array = flat[:want].reshape(shape).copy()
    else:
        grown = _np.zeros(want, dtype=self.data.dtype)
        grown[:flat.size] = flat
        self._array = grown.reshape(shape)
    return self


def _set_(self, source=None):
    """**저장을 통째로 갈아 끼운다.** 인자가 없으면 빈 텐서가 된다."""
    _refuse_leaf_inplace(self, "set_")
    self._array = (_np.empty(0, dtype=self.data.dtype) if source is None
                   else _np.asarray(source.data))
    return self


def _refuse_leaf_inplace(self, name):
    if self.requires_grad and _grad_mode.enabled:
        raise RuntimeError(_like_torch(
            f"기울기가 필요한 잎 텐서에는 `{name}` 을(를) 쓸 수 없습니다.",
            "a leaf Variable that requires grad is being used in an in-place operation"))


def _sparse_only(name):
    def method(self, *args, **kw):
        del self, args, kw
        raise NotImplementedError(_like_torch(
            f"`{name}` 은 희소 텐서 전용입니다 — 조밀 텐서에는 쓸 수 없습니다.",
            f"Could not run 'aten::{name}' with arguments from the 'CPU' backend"))

    method.__name__ = name
    return method


Tensor.apply_ = _apply_
Tensor.map_ = _map_
Tensor.map2_ = _map2_
Tensor.resize_ = _resize_
Tensor.set_ = _set_
for _sname in ("resize_as_sparse_", "sparse_resize_", "sparse_resize_and_clear_"):
    setattr(Tensor, _sname, _sparse_only(_sname))
del _sname
