"""JS 쪽 텐서를 파이썬에서 잡는 손잡이, 그리고 값을 **동기로** 읽는 길.

이 파일이 이 결속의 전부라고 해도 된다. 나머지는 이름을 옮겨 적는 일이다.
"""

import js as _js
import numpy as _np
from pyodide.ffi import run_sync as _run_sync, to_js as _to_js

_ts = _js.borch


def _js_list(seq):
    """파이썬 목록 → JS 배열. 프록시를 그냥 넘기면 JS 쪽이 배열로 안 본다."""
    return _to_js(list(int(n) for n in seq))


def _read(handle):
    """GPU 에서 값을 가져온다 — **`await` 없이.**

    WebGPU 에 동기 읽기가 없다. borch.ts 의 `toArray()` 는 `mapAsync` 위에 서 있어서
    약속(Promise)을 돌려준다. `run_sync` 가 JSPI(WebAssembly 의 Promise 통합)로 그
    자리를 메운다 — 파이썬 스택을 중단했다가 값이 오면 재개한다.

    **비동기 진입점 아래에서만 된다.** 페이지가 `runPythonAsync` 로 들어와야 스택에
    중단할 자리가 있고, `runPython` 으로 들어오면 `RuntimeError: No suspender` 로
    멈춘다. 라이브러리의 한계가 아니라 그 스택의 사정이다. 실측해서 안 것이고
    (`tests/browser/sync_probe.py`), 그것 하나로 이 결속이 성립한다 — 안 되면
    `await loss.item()` 이 되고 그러면 이 프로젝트의 주장이 깨진다.
    """
    return _np.asarray(_run_sync(handle.toArray()), dtype=_np.float32)


def int64_name():
    """색인 텐서의 형. 이름을 한 곳에서만 적는다."""
    return _DType("int64")


def _core_repr(shim):
    """코어의 `_tensor_repr` 을 빌려온다. 브라우저에서는 `/work` 아래에 있다."""
    global _REPR
    if _REPR is None:
        from borch._base import _tensor_repr as fn
        _REPR = fn
    return _REPR(shim)


_REPR = None


class _Shim:
    """`_tensor_repr` 이 보는 것만 흉내 낸다 — `.data` · `.dtype` · `._op` · `.requires_grad`."""

    __slots__ = ("data", "dtype", "_op", "requires_grad")

    def __init__(self, t):
        self.data = t.numpy()
        self.dtype = t.dtype
        self._op = t._h.gradName or None
        self.requires_grad = bool(t._h.requiresGrad)


class _DType(str):
    """형 이름. 값은 borch.ts 의 이름이고 **보이는 것은 torch 의 이름**이다.

    골든이 `str(x.dtype)` 를 답으로 굳혔고 그 답은 `torch.float32` 다. 그런데 우리
    내부에서는 `"float32"` 로 다녀야 borch.ts 에 그대로 넘길 수 있다 — 문자열을
    물려받아 두 이름을 한 물건에 담는다.
    """

    __slots__ = ()

    def __repr__(self):
        return f"torch.{self}" if self != "bool" else "torch.bool"

    def __str__(self):
        return f"torch.{str.__str__(self)}"

    @property
    def plain(self):
        return str.__str__(self)


class _Size(tuple):
    """모양. `torch.Size([2, 2])` 로 보여야 한다 — 골든이 그 문자열을 굳혔다."""

    __slots__ = ()

    def __repr__(self):
        return f"torch.Size([{', '.join(str(n) for n in self)}])"

    __str__ = __repr__


class Tensor:
    """borch.ts 텐서 하나를 감싼다.

    **값을 파이썬에 안 들고 있는다.** 손잡이만 들고 필요할 때 GPU 에서 읽는다 —
    양쪽에 두면 어느 쪽이 진짜인지 갈리는 날이 온다.
    """

    __slots__ = ("_h",)

    def __init__(self, handle):
        self._h = handle

    # ── 하네스가 요구하는 둘 ──────────────────────────────────────────────
    #
    # `tests/cases.py` 의 `to_numpy` 가 `t.detach().numpy()` 만 부른다. 그 둘만
    # 맞추면 골든 하네스를 한 줄도 안 고치고 이 구현을 대조할 수 있다.

    def detach(self):
        return Tensor(self._h.detach())

    def numpy(self):
        flat = _read(self._h)
        shape = self.shape
        out = flat.reshape(shape) if shape else flat.reshape(())
        # dtype 은 borch.ts 에서 float32 저장 위의 **이름표**다. 되돌릴 때 그 이름을
        # 따라간다 — 안 그러면 int64 케이스가 실수로 나오고, 값은 맞아 보인다.
        kind = str(self._h.dtype)
        if kind == "int64":
            return out.astype(_np.int64)
        if kind == "bool":
            return out.astype(bool)
        return out

    # ── 파이썬다움 ────────────────────────────────────────────────────────

    @property
    def shape(self):
        return _Size(int(n) for n in self._h.shape)

    @property
    def dtype(self):
        """**`torch.float32` 로 보여야 한다.** borch.ts 는 `"float32"` 라고 말한다.

        골든의 dtype 케이스는 값이 아니라 **형 이름 문자열**을 답으로 굳혔다. 이름이
        다르면 승격 규칙이 다 맞아도 전부 실패한다 — 실제로 그렇게 나왔다.
        """
        return _DType(str(self._h.dtype))

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return len(self.shape)

    def numel(self):
        return int(self._h.size)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def item(self):
        """**원소가 하나여야 한다.** torch 가 그 자리에서 던지고 골든이 그것을 굳혔다."""
        if self._h.size != 1:
            raise RuntimeError(
                f"a Tensor with {self._h.size} elements cannot be converted to Scalar")
        return float(_read(self._h)[0])

    def backward(self, *args):
        return guarded(self._h.backward, *[handle(a) for a in args])

    # dtype 을 바꾸는 이름들. borch.ts 는 `to("float32")` 하나로 받는다.
    def to(self, dtype):
        name = dtype.plain if isinstance(dtype, _DType) else str(dtype)
        return guarded(self._h.to, name.replace("torch.", ""))

    def float(self):
        return self.to("float32")

    def double(self):
        """**없다.** WebGPU 의 셰이더에 배정도가 없다.

        자매(`borch_webgpu`)가 TF.js 때문에 거절하는 것과 같은 자리다 — 이유가 다를
        뿐 결론이 같고, 골든이 그 거절을 답으로 굳혔다. 조용히 float32 로 돌려주면
        "배정도로 계산했다" 고 믿는 코드가 생긴다.
        """
        raise RuntimeError(
            "Only Tensors of floating point dtype float32 are supported — "
            "float64 는 WebGPU 셰이더에 없다")

    def long(self):
        return self.to("int64")

    def int(self):
        return self.to("int64")

    def bool(self):
        return self.to("bool")

    def type(self, dtype=None):
        return self.dtype if dtype is None else self.to(dtype)

    def tolist(self):
        return self.numpy().tolist()

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __repr__(self):
        """**코어의 규칙을 빌린다.** 여기서 다시 쓰면 두 번째가 다른 날이 온다.

        `borch/_base.py` 의 `_tensor_repr` 이 torch 의 출력 규칙(정렬·자릿수·줄바꿈·
        여덟 칸 들여쓰기)을 이미 담고 있고, 골든이 그 문자열을 답으로 굳혔다. 값과
        몇 가지 표시만 넘겨주면 그 함수가 답을 만든다.
        """
        return _core_repr(_Shim(self))

    __str__ = __repr__

    # ── 나머지는 전부 넘긴다 ──────────────────────────────────────────────

    def __getattr__(self, name):
        """이 클래스에 없는 이름은 **borch.ts 텐서 쪽**으로 넘긴다.

        `x.exp()` · `x.masked_select(m)` 처럼 케이스가 메서드로 부르는 자리가 많다.
        손으로 옮겨 적으면 그중 하나가 다른 연산을 부르는 날이 오므로 안 적고 넘긴다.
        없으면 `AttributeError` 로 멈춘다 — 근사하지 않는다.

        **메서드인지 속성인지를 JS 쪽에 물어서 가른다.** 전부 메서드로 감쌌더니
        `x.T` 나 `x.grad` 가 함수를 돌려주었고, 그것이 실패 96 건이었다 — 원인은
        `'function' object has no attribute 'detach'` 로 나왔다. 값을 돌려줄 자리에
        함수를 돌려주면 그 다음 줄에서야 터지고, 그러면 원인이 한 칸 밀린다.
        """
        from ._ops import _BINARY_ONLY, camel, positional

        # 모듈 쪽에 손으로 쓴 것들은 메서드로도 같은 것을 써야 한다 — 인자 순서가
        # 뒤집혔거나(`split`) 한쪽만 올 수 있는(`clamp`) 자리들이다.
        if name in ("clamp", "clip", "split", "chunk", "aminmax", "flip",
                    "pow", "squeeze", "repeat_interleave"):
            from . import _ops
            fn = getattr(_ops, name)
            return lambda *a, **k: fn(self, *a, **k)

        js_name = camel(name)
        if name in _BINARY_ONLY:
            # borch.ts 는 단항만 표에서 메서드로 만든다. 이항은 `binary(이름, 상대)` 다.
            return lambda other, *_: guarded(self._h.binary, js_name, handle(other))
        got = getattr(self._h, js_name, None)
        if got is None:
            raise AttributeError(
                f"borch.ts 텐서에 `{js_name}` 이 없다 (파이썬 이름 `{name}`)")
        if not callable(got):
            return settle(got)

        def call(*args, **kw):
            return guarded(got, *positional(name, args, kw))

        call.__name__ = name
        return call

    # 연산자. `x + y` 는 `x.add(y)` 이고, 상대가 수여도 받는다.
    def _op(js_name):                                        # noqa: N805
        def go(self, other):
            return guarded(self._h.binary, js_name, handle(other))
        return go

    def _rop(js_name):                                       # noqa: N805
        def go(self, other):
            return guarded(handle(other).binary, js_name, self._h)
        return go

    __add__, __radd__ = _op("add"), _rop("add")
    __sub__, __rsub__ = _op("sub"), _rop("sub")
    __mul__, __rmul__ = _op("mul"), _rop("mul")
    __truediv__, __rtruediv__ = _op("div"), _rop("div")
    __pow__ = _op("pow")
    __eq__, __ne__ = _op("eq"), _op("ne")
    __lt__, __le__ = _op("lt"), _op("le")
    __gt__, __ge__ = _op("gt"), _op("ge")

    def __mod__(self, other):
        return guarded(self._h.remainder, float(other))

    def __matmul__(self, other):
        return guarded(self._h.mm, handle(other))

    def __neg__(self):
        return wrap(self._h.neg())

    def __getitem__(self, key):
        """`x[0]` · `x[1:3]` · `x[:, 1]`. torch 코드가 가장 자주 하는 일이다."""
        keys = key if isinstance(key, tuple) else (key,)
        out, axis = self, 0
        for k in keys:
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                stop = out.shape[axis] if k.stop is None else k.stop
                out = wrap(out._h.narrow(axis, start, stop - start))
                axis += 1
            elif isinstance(k, (Tensor, list, tuple)):
                # `x[[2, 0]]` — 번호 목록으로 고르는 자리. torch 코드가 흔히 쓴다.
                idx = k if isinstance(k, Tensor) else tensor(list(k), int64_name())
                out = wrap(out._h.indexSelect(axis, idx._h))
                axis += 1
            else:
                n = out.shape[axis]
                out = wrap(out._h.select(axis, k + n if k < 0 else k))
        return out

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __bool__(self):
        return bool(_read(self._h)[0])

    def __float__(self):
        return self.item()

    def __hash__(self):
        return id(self)

    del _op, _rop


class RuntimeError_(RuntimeError):
    """이름은 파이썬의 것이고 문구는 torch 의 것이다."""


class IndexError_(IndexError):
    pass


def translate(exc):
    """JS 쪽 예외를 **torch 가 내는 종류**로 옮긴다.

    골든이 예외의 **종류 이름까지** 답으로 굳혔다(`RuntimeError|문구=True`). 그대로
    두면 파이썬 쪽에 `JsException` 이 올라오고, `except RuntimeError` 로 잡던 코드가
    안 잡힌다 — 예외의 종류도 API 다.

    문구는 안 바꾼다. borch.ts 가 이미 torch 의 원문을 담고 있고, 그것이 검색을
    통하게 하려고 그렇게 쓴 것이다.
    """
    text = str(exc)
    # **앞머리만 벗긴다.** `replace` 로 첫 번째 `Error: ` 를 지웠더니 `RuntimeError:
    # shape …` 이 `Runtimeshape …` 이 되어 문구가 망가졌다 — 검색이 통하라고 원문을
    # 담아 둔 것을 우리가 부순 셈이다.
    for head in ("RuntimeError: ", "IndexError: ", "Error: "):
        if text.startswith(head):
            text = text[len(head):]
            break
    kind = IndexError if ("index" in text.lower() or "색인" in text) else RuntimeError
    return kind(text)


class _Pair:
    """`values` 와 `indices` 를 함께 주는 것들 — `sort`·`topk`·`median`·`max(dim)`.

    borch.ts 는 평범한 JS 객체를 준다. 그대로 흘리면 `.values` 가 JS 프록시라
    `.numpy()` 가 없고, 실패는 `AttributeError: numpy` 로 한 칸 밀려서 나온다.
    """

    __slots__ = ("values", "indices")

    def __init__(self, obj):
        self.values = wrap(obj.values)
        self.indices = wrap(obj.indices)

    def __iter__(self):
        yield self.values
        yield self.indices

    def __getitem__(self, i):
        return (self.values, self.indices)[i]

    def __getattr__(self, name):
        """**값 쪽으로 넘긴다.** torch 의 `median()` 은 dim 없이 부르면 값 하나를
        주는데 borch.ts 는 늘 쌍을 준다 — 그 자리에서 `.numpy()` 나 `.shape` 를
        물으면 값을 묻는 것이다."""
        return getattr(self.values, name)


def settle(out):
    """돌려받은 것을 파이썬이 쓸 모양으로 만든다.

    **약속이면 여기서 기다린다.** borch.ts 에서 몇 개는 비동기다 — `unique`,
    `bincount`, `masked_select` 처럼 **결과의 크기가 값에 달린** 것들이라 GPU 에서
    한 번 읽어야 모양이 정해진다. 그것을 그대로 흘리면 파이썬 쪽에 `PyodideFuture` 가
    돌아다니고, 실패는 그 다음 줄에서 `'PyodideFuture' object has no attribute
    'detach'` 로 나온다 — 원인에서 한 칸 밀린 자리다.

    `run_sync` 가 여기서도 그 자리를 메운다. 값을 읽는 것과 같은 장치다.
    """
    from pyodide.ffi import JsException

    try:
        if hasattr(out, "then"):
            out = _run_sync(out)
    except JsException as exc:
        raise translate(exc) from None
    if _js.borch.isTensor(out):
        return wrap(out)
    # `{values, indices}` 쌍과 텐서 배열은 그대로 흘리면 프록시가 파이썬에 남는다.
    if hasattr(out, "values") and hasattr(out, "indices"):
        return _Pair(out)
    if _js.Array.isArray(out):
        return [wrap(x) if _js.borch.isTensor(x) else x for x in out]
    # **이름 붙은 자리를 여럿 주는 것들** — `slogdet` 의 `{sign, logabs}`,
    # `qr` 의 `{q, r}`, `svd` 의 `{u, s, vt}`. 그대로 흘리면 파이썬에서 첨자도
    # 속성 접근도 안 되는 프록시가 남는다.
    if hasattr(out, "constructor") and str(getattr(out, "constructor", "")) and \
            not callable(out) and hasattr(out, "toString"):
        keys = [str(k) for k in _js.Object.keys(out)]
        if keys and all(not k.isdigit() for k in keys):
            return _Fields({k: getattr(out, k) for k in keys})
    return out


class _Fields:
    """이름 붙은 자리를 여럿 주는 답. 첨자로도 이름으로도 닿는다 — torch 가 그렇다."""

    __slots__ = ("_d", "_order")

    def __init__(self, d):
        self._order = list(d)
        object.__setattr__(self, "_d", {
            k: (wrap(v) if _js.borch.isTensor(v) else v) for k, v in d.items()})

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, i):
        return self._d[self._order[i]] if isinstance(i, int) else self._d[i]

    def __iter__(self):
        for k in self._order:
            yield self._d[k]


def guarded(fn, *args):
    """부르고, JS 예외가 오면 torch 종류로 바꿔 던진다."""
    from pyodide.ffi import JsException

    try:
        return settle(fn(*args))
    except JsException as exc:
        raise translate(exc) from None


def wrap(x):
    """JS 텐서든 파이썬 수든 우리 `Tensor` 로."""
    if isinstance(x, Tensor):
        return x
    # **파이썬 수의 형이 승격 규칙에 들어간다.** `int64 + 2` 는 int64 이고
    # `int64 + 2.0` 은 float32 다. 전부 float32 스칼라로 만들었더니 승격이 다 float32 로
    # 무너졌다 — 값은 맞는데 형 이름만 갈리는 자리라 값 대조로는 안 보인다.
    if isinstance(x, bool):
        return Tensor(_ts.Tensor.from_(
            _js.Float32Array.new(_to_js([1.0 if x else 0.0])),
            _js_list([]), False, "bool"))
    if isinstance(x, int):
        return Tensor(_ts.Tensor.from_(
            _js.Float32Array.new(_to_js([float(x)])), _js_list([]), False, "int64"))
    if isinstance(x, float):
        return Tensor(_ts.Tensor.full(_js_list([]), x))
    return Tensor(x)


def handle(x):
    """상대가 우리 텐서면 손잡이를, 수면 스칼라 텐서를 만들어 그 손잡이를."""
    return wrap(x)._h


def tensor(data, dtype=None, requires_grad=False):
    """`torch.tensor` 자리. numpy 배열·중첩 리스트·수를 받는다."""
    arr = _np.asarray(data)
    if dtype is not None:
        # `torch.float32` 로 보이는 물건이 와도 borch.ts 에는 `float32` 로 넘긴다.
        name = dtype.plain if isinstance(dtype, _DType) else str(dtype)
    elif arr.dtype == bool:
        name = "bool"
    elif arr.dtype.kind in "iu":
        name = "int64"
    else:
        name = "float32"
    flat = _js.Float32Array.new(_to_js(arr.ravel().astype(_np.float32)))
    return Tensor(_ts.Tensor.from_(flat, _js_list(arr.shape), requires_grad, name))
