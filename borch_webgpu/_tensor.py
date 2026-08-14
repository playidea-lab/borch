"""borch_webgpu 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import numpy as _np

try:
    import js as _js
    from pyodide.ffi import create_proxy as _create_proxy
    from pyodide.ffi import to_js as _to_js
except ImportError as _exc:                                          # pragma: no cover
    raise ImportError(
        "borch_webgpu 는 브라우저(Pyodide) 안에서만 돕니다. "
        "네이티브에서는 `borch` 를 쓰세요 — 이쪽을 CPU 로 흉내 내면 "
        "GPU 로 돌렸다고 착각하게 됩니다."
    ) from _exc

_tf = getattr(_js, "tf", None)
if _tf is None:                                                      # pragma: no cover
    raise ImportError("TF.js 가 페이지에 없습니다. tf.min.js 를 먼저 실으세요.")

from ._base import (
    Size, _BY_CATEGORY, _ValuesIndices, _broadcast_error, _dtype_of, _keep,
    _last_axis_only, _like_torch, _pick_last, _shape_of, _slice_along, _slice_tensor,
    _tensor_repr, _to_np, _to_tf, _unsupported, bool_, dtype, float32, int64, long,
)

# ---------------------------------------------------------------- Tensor

class _GradMode:
    """`no_grad` 의 스위치를 **객체 하나에 담는다.** 모듈 전역으로 두면 안 된다.

    파일을 쪼개면 `no_grad` 와 `_make` 가 다른 모듈에 놓이는데, 전역 이름은 모듈마다
    따로 생긴다. `no_grad` 가 자기 모듈의 이름만 False 로 바꾸고 `_make` 는 옛 값을
    계속 읽는다 — **예외도 경고도 없이 `no_grad` 가 안 먹는다.** 코어를 쪼개다 실제로
    걸렸고(골든이 아니라 `test_diff` 가 잡았다), 여기도 같은 모양이라 미리 고친다.
    """

    enabled = True


_grad_mode = _GradMode()


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
        needs = _grad_mode.enabled and any(p.requires_grad for p in parents)
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

    # ---- 제자리 연산
    #
    # **된다. 다만 코어와 한 가지가 다르다.**
    #
    # `x.add_(1)` 처럼 자기 자신을 고치는 것은 손잡이를 새것으로 갈아 끼우면 끝이고,
    # 저장소 공유가 필요 없다. 튜토리얼이 쓰는 것은 거의 이쪽이다.
    #
    # 못 하는 것은 **뷰를 통한 전파**다. torch 는 `b = a.view(2,2); b.add_(10)` 으로
    # `a` 까지 바뀌는데, TF.js 텐서는 불변이라 그럴 수 없다. 코어는 numpy 배열을
    # 공유하므로 그것까지 torch 와 같다(실측). 여기서는 조용히 다르게 굴지 않고
    # **파생 텐서의 제자리 수정을 거절한다** — `_derived` 가 그 판단의 근거다.

    def _write_back(self, fn, what):
        # 이름이 `_inplace` 가 아닌 이유: **이미 그 이름이 있었다.** 아래쪽에
        # `+=` 계열이 쓰는 `_inplace(fn, other)` 가 따로 있고, 같은 이름으로 두면
        # 그것을 덮어써서 `x += 1` 이 이름 문자열을 값으로 받는다. 실제로 그렇게
        # 터졌고(`could not convert string to float: 'add_'`), 시끄럽게 터진 것이
        # 다행이다 — 서명이 우연히 맞았으면 조용히 틀렸다.
        if self.requires_grad and _grad_mode.enabled:
            raise RuntimeError(_like_torch(
                f"기울기가 필요한 잎 텐서에는 `{what}` 을(를) 쓸 수 없습니다. "
                "`with torch.no_grad():` 안에서 하거나 제자리가 아닌 연산을 쓰세요.",
                "a leaf Variable that requires grad is being used in an in-place operation"))
        if self._derived:
            _unsupported(
                f"다른 텐서에서 나온 것(뷰·슬라이스·연산 결과)에 `{what}` 을(를) 쓰는 것 — "
                "torch 는 저장소를 공유해 원본까지 바꾸는데 TF.js 텐서는 불변이라 "
                "여기서는 그럴 수 없습니다. 코어 `borch` 는 이것을 지원합니다")
        out = fn()
        old = self._h
        self._h = out._h if isinstance(out, Tensor) else out
        if isinstance(out, Tensor):
            out._h = None          # 손잡이를 넘겨받았다 — 둘이 나눠 들면 한쪽이 놓는다
            self._dtype = out._dtype
            self._nhwc = out._nhwc
        if old is not None and old is not self._h:
            try:
                old.dispose()
            except Exception:                                        # noqa: BLE001
                pass
        return self

    def add_(self, other, alpha=1):
        return self._write_back(lambda: self + (other * alpha if alpha != 1 else other), "add_")

    def sub_(self, other, alpha=1):
        return self._write_back(lambda: self - (other * alpha if alpha != 1 else other), "sub_")

    def mul_(self, other):
        return self._write_back(lambda: self * other, "mul_")

    def div_(self, other):
        return self._write_back(lambda: self / other, "div_")

    def pow_(self, exponent):
        return self._write_back(lambda: self ** exponent, "pow_")

    def neg_(self):
        return self._write_back(lambda: -self, "neg_")

    def zero_(self):
        return self._write_back(
            lambda: Tensor(_tf.zerosLike(self._h), dt=self._dtype), "zero_")

    def fill_(self, value):
        return self._write_back(
            lambda: Tensor(_tf.fill(_to_js(list(self.shape)), float(value))), "fill_")

    def copy_(self, other):
        return self._write_back(
            lambda: Tensor(_tf.clone(_wrap(other)._h), dt=self._dtype), "copy_")

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

    def __mod__(self, o):
        """나머지. 코어에는 있는데 여기에는 아예 없었다 — 골든이 표면을 넓히면서 드러났다.

        기울기는 나누어지는 쪽으로 **그대로** 흐른다(계단이 뛰는 자리를 빼면).
        나누는 쪽으로는 `-floor(a/b)` 다.
        """
        other = _wrap(o)
        parents = (self, other) if isinstance(o, Tensor) else (self,)

        def back(g):
            if isinstance(o, Tensor):
                return (g, _tf.neg(_tf.mul(g, _tf.floor(_tf.div(self._h, other._h)))))
            return (g,)

        return self._make(_tf.mod(self._h, other._h), parents, back, "RemainderBackward0")

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

    # ---- 모양·축약 (메서드로만 있는 것들)

    @property
    def data(self):
        """torch 처럼 **텐서**를 준다(코어는 numpy 를 준다).

        torch 의 `.data` 는 저장소를 공유하지만 우리는 못 하므로 사본이다.
        쓰는 쪽(`p.data = ...`)은 손잡이를 갈아끼우니 파라미터 초기화에는 그대로 통한다.
        """
        return self.detach()

    @data.setter
    def data(self, value):
        if not isinstance(value, Tensor):
            raise TypeError(_like_torch(
                f"`.data` 에는 텐서만 넣을 수 있습니다 ({type(value).__name__} 을 받았습니다).",
                f"Variable data has to be a tensor, but got {type(value).__name__}"))
        old = self._h
        self._h = _keep(_tf.clone(value._h))
        self._nhwc, self._dtype = value._nhwc, value._dtype
        old.dispose()

    def clone(self):
        return self._make(_tf.clone(self._h), (self,), lambda g: (g,), "CloneBackward0")

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def double(self):
        _unsupported("float64 (TF.js 에 배정도가 없습니다)")

    def type(self, dt):
        return self._cast_to(dt)

    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims
        t = _canonical(self)
        inv = [int(i) for i in _np.argsort(dims)]
        return t._make(_tf.transpose(t._h, _to_js(list(dims))), (t,),
                       lambda g: (_tf.transpose(g, _to_js(inv)),), "PermuteBackward0")

    def squeeze(self, dim=None):
        t = _canonical(self)
        old = t.shape
        out = (_tf.squeeze(t._h) if dim is None else _tf.squeeze(t._h, _to_js([dim])))
        return t._make(out, (t,), lambda g: (_tf.reshape(g, _to_js(list(old))),),
                       "SqueezeBackward0")

    def _argreduce(self, big, dim, keepdim):
        """dim 이 없으면 값 하나, 있으면 (값, 번호). torch 와 같은 모양이다."""
        t = _canonical(self)
        pick = _tf.max if big else _tf.min
        if dim is None:
            return Tensor(pick(t._h), dt=t._dtype)
        _last_axis_only(t, dim, "max/min")
        arg = (_tf.argMax if big else _tf.argMin)(t._h, -1)
        idx = _tf.reshape(arg, _to_js(list(t.shape[:-1]) + [1]))
        values = _pick_last(t, idx)
        if not keepdim:
            values = values.reshape(tuple(t.shape[:-1]))
        # **번호의 모양을 명시한다.** 1차원을 접으면 torch 는 스칼라를 주는데 여기서
        # 나오는 것은 (1,) 이었다. 값 쪽은 위에서 이미 못 박고 있었고 번호만 빠져 있었다.
        arg = _tf.reshape(arg, _to_js(list(t.shape[:-1])))
        return _ValuesIndices(values, Tensor(arg))

    def max(self, dim=None, keepdim=False):
        return self._argreduce(True, dim, keepdim)

    def min(self, dim=None, keepdim=False):
        return self._argreduce(False, dim, keepdim)

    def _argpick(self, pick, dim):
        t = _canonical(self)
        h = t._h if dim is not None else _tf.reshape(t._h, _to_js([-1]))
        out = pick(h, -1 if dim is None else dim)
        # dim 이 없으면 스칼라, 있으면 그 축만 빠진 모양이다 — torch 와 같게 못 박는다.
        keep = [] if dim is None else [n for i, n in enumerate(t.shape) if i != dim]
        return Tensor(_tf.reshape(out, _to_js(keep)), dt=int64)

    def argmax(self, dim=None):
        return self._argpick(_tf.argMax, dim)

    def argmin(self, dim=None):
        return self._argpick(_tf.argMin, dim)

    def var(self, dim=None, unbiased=True, keepdim=False):
        t = _canonical(self)
        n = t.numel() if dim is None else t.shape[dim]
        mean = t.mean(dim=dim, keepdim=True) if dim is not None else t.mean()
        centered = t - mean
        out = (centered * centered).sum(dim=dim, keepdim=keepdim)
        return out / float(n - 1 if unbiased else n)

    def std(self, dim=None, unbiased=True, keepdim=False):
        return self.var(dim=dim, unbiased=unbiased, keepdim=keepdim) ** 0.5

    def all(self):
        return Tensor(_tf.all(_storage_bool(self)), dt=bool_)

    def any(self):
        return Tensor(_tf.any(_storage_bool(self)), dt=bool_)

    def bincount(self):
        # `_ops` 는 이 모듈을 들여오므로 위에서 들여오면 순환이다. 부를 때 들여온다 —
        # 메서드가 모듈 함수에 넘기는 자리가 셋 있고, 셋 다 이렇게 막는다.
        from ._ops import bincount
        return bincount(self)

    # ---- 인덱싱

    def __getitem__(self, idx):
        """정수·슬라이스·정수목록·불리언 마스크. **그래프를 잇는다.**

        축마다 잘라내는 것을 겹쳐서 만든다 — 자르기의 역방향이 이미 0 채우기라
        역전파가 저절로 따라온다. 걸음이 1 이 아닌 슬라이스처럼 못 하는 것은
        근사하지 않고 거절한다.
        """
        from ._ops import index_select, masked_select      # 순환을 피한다 — 위 참고

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
        if self.requires_grad and _grad_mode.enabled:
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
        if self.requires_grad and _grad_mode.enabled:
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


def _storage_bool(t):
    """불리언 연산에 넣을 손잡이. 수치 저장이면 0 이 아닌 것을 참으로 본다."""
    return t._h if t._dtype is bool_ else _tf.notEqual(t._h, 0.0)


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


