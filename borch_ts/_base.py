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
        return tuple(int(n) for n in self._h.shape)

    @property
    def dtype(self):
        return str(self._h.dtype)

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
        return float(_read(self._h)[0])

    def tolist(self):
        return self.numpy().tolist()

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __repr__(self):
        return f"tensor({self.numpy()!r})"

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
        from ._ops import camel, positional

        js_name = camel(name)
        got = getattr(self._h, js_name, None)
        if got is None:
            raise AttributeError(
                f"borch.ts 텐서에 `{js_name}` 이 없다 (파이썬 이름 `{name}`)")
        if not callable(got):
            return wrap(got) if _js.borch.isTensor(got) else got

        def call(*args, **kw):
            out = got(*positional(name, args, kw))
            return wrap(out) if _js.borch.isTensor(out) else out

        call.__name__ = name
        return call

    # 연산자. `x + y` 는 `x.add(y)` 이고, 상대가 수여도 받는다.
    def _op(js_name):                                        # noqa: N805
        def go(self, other):
            return wrap(self._h.binary(js_name, handle(other)))
        return go

    def _rop(js_name):                                       # noqa: N805
        def go(self, other):
            return wrap(handle(other).binary(js_name, self._h))
        return go

    __add__, __radd__ = _op("add"), _rop("add")
    __sub__, __rsub__ = _op("sub"), _rop("sub")
    __mul__, __rmul__ = _op("mul"), _rop("mul")
    __truediv__, __rtruediv__ = _op("div"), _rop("div")
    __pow__ = _op("pow")
    __eq__, __ne__ = _op("eq"), _op("ne")
    __lt__, __le__ = _op("lt"), _op("le")
    __gt__, __ge__ = _op("gt"), _op("ge")

    def __neg__(self):
        return wrap(self._h.neg())

    del _op, _rop


def wrap(x):
    """JS 텐서든 파이썬 수든 우리 `Tensor` 로."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, bool):
        return Tensor(_ts.Tensor.full(_js_list([]), 1.0 if x else 0.0))
    if isinstance(x, (int, float)):
        return Tensor(_ts.Tensor.full(_js_list([]), float(x)))
    return Tensor(x)


def handle(x):
    """상대가 우리 텐서면 손잡이를, 수면 스칼라 텐서를 만들어 그 손잡이를."""
    return wrap(x)._h


def tensor(data, dtype=None, requires_grad=False):
    """`torch.tensor` 자리. numpy 배열·중첩 리스트·수를 받는다."""
    arr = _np.asarray(data)
    if dtype is not None:
        name = str(dtype)
    elif arr.dtype == bool:
        name = "bool"
    elif arr.dtype.kind in "iu":
        name = "int64"
    else:
        name = "float32"
    flat = _js.Float32Array.new(_to_js(arr.ravel().astype(_np.float32)))
    return Tensor(_ts.Tensor.from_(flat, _js_list(arr.shape), requires_grad, name))
