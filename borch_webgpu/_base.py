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


def _js_options(**kw):
    """파이썬 딕셔너리 → **JS 평범한 객체**.

    `to_js` 는 딕셔너리를 기본으로 `Map` 으로 옮기는데, borch.ts 의 옵션 인자는
    `options.requiresGrad` 처럼 속성으로 읽으므로 `Map` 이 오면 **전부 `undefined` 다.**
    예외는 안 나고 기본값이 조용히 쓰인다 — 위치 인자가 옵션 객체로 바뀔 때 이 결속이
    정확히 그렇게 끊겼고, 브라우저 골든의 `edge::grad::*` 열다섯 건이 "기울기를 안
    켰다" 로 그것을 잡았다. TS 러너는 이 경로를 안 지난다.
    """
    return _to_js(kw, dict_converter=_js.Object.fromEntries)


def camel_name(name):
    """`requires_grad` → `requiresGrad`. `_ops.camel` 과 같은 규칙인데 여기서 쓰려면
    그쪽을 들여와야 하고 그것이 순환이라, 이 한 줄만 따로 둔다."""
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def _js_floats(seq):
    """`_js_list` 의 실수판. **정수로 깎으면 안 되는 자리**가 있다 — 분수 풀링의
    표본은 0..1 사이라 `int()` 를 지나면 전부 0 이 되고, 그 0 은 예외가 아니라
    **답이 있는 창 자리**라 조용히 다른 층이 된다."""
    return _to_js(list(float(v) for v in seq))


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

    ## `to_py()` 를 거치는 이유 — **음의 0**

    예전에는 `_np.asarray(js_array, dtype=float32)` 였는데, 그 길은 **`-0.0` 을
    `0.0` 으로 만든다**(실측). JS 쪽에서는 `Object.is(a[0], -0)` 이 참인 채로 왔고
    numpy 로 옮긴 뒤에만 부호가 없었다 — 원소를 하나씩 옮기는 길이라 그렇다.
    `to_py()` 는 **memoryview** 를 주므로 `frombuffer` 가 바이트를 그대로 읽는다.

    값 대조로는 절대 안 걸린다 — `-0.0 == 0.0` 이다. **글자로만** 걸린다:
    `tensor([1.-0.j])` 가 `tensor([1.+0.j])` 로 찍혔다. 복소수 repr 을 굳히다가
    나왔고, 실수 텐서에도 내내 있던 자리다.
    """
    raw = _run_sync(handle.toArray())
    # **사본을 뜬다.** `frombuffer` 는 WASM 힙을 가리키는 읽기 전용 뷰라, 그대로
    # 들고 있으면 그 자리가 다음 읽기에 덮인다.
    return _np.frombuffer(raw.to_py(), dtype=_np.float32).copy()


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

    def __setattr__(self, name, value):
        """**속성은 저쪽 텐서에 쓴다.** `p.grad = g` 가 torch 코드의 예사로운 줄이다.

        `__slots__` 이 `_h` 하나뿐이라 다른 이름은 그냥 `AttributeError` 였다.
        읽기(`__getattr__`)는 저쪽으로 넘기면서 쓰기는 안 넘겼던 자리다 — 옵티마이저를
        손으로 먹이는 코드가 그 첫 줄에서 멈춘다.
        """
        if name == "_h":
            object.__setattr__(self, name, value)
            return
        setattr(self._h, camel_name(name),
                handle(value) if isinstance(value, Tensor) else value)

    # ── 하네스가 요구하는 둘 ──────────────────────────────────────────────
    #
    # `tests/cases.py` 의 `to_numpy` 가 `t.detach().numpy()` 만 부른다. 그 둘만
    # 맞추면 골든 하네스를 한 줄도 안 고치고 이 구현을 대조할 수 있다.

    def detach(self):
        return Tensor(self._h.detach())

    def numpy(self):
        flat = _read(self._h)
        shape = self.shape
        kind = str(self._h.dtype)
        # **복소수만 칸 수와 버퍼 길이가 다르다.** borch.ts 의 저장이 인터리브라
        # `[re, im, re, im, …]` 로 2n 개가 온다 — 그대로 `reshape(shape)` 하면
        # 칸이 두 배라 거기서 멈춘다. 다른 형은 이름표일 뿐이라 이 자리가 없다.
        if kind == "complex64":
            pair = flat.reshape(-1, 2)
            # **자리에 써 넣는다 — `re + 1j*im` 이 아니다.** 그 식은 **음의 0** 을
            # 잃는다: `1j * (-0.0)` 의 허수부가 `-0.0` 인데 실수부의 `+0.0` 과
            # 더해지면서 `+0.0` 이 된다. `tensor([1.-0.j])` 가 `tensor([1.+0.j])` 로
            # 찍혔고, 값 대조로는 안 걸린다(둘은 `==` 로 같다) — **글자로만** 걸린다.
            out = _np.empty(pair.shape[0], dtype=_np.complex64)
            out.real, out.imag = pair[:, 0], pair[:, 1]
            return out.reshape(shape) if shape else out.reshape(())
        out = flat.reshape(shape) if shape else flat.reshape(())
        # dtype 은 borch.ts 에서 float32 저장 위의 **이름표**다. 되돌릴 때 그 이름을
        # 따라간다 — 안 그러면 int64 케이스가 실수로 나오고, 값은 맞아 보인다.
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
        flat = _read(self._h)
        # **복소수는 파이썬 `complex` 로 낸다** — torch 도 그렇다. borch.ts 쪽
        # `item()` 은 JS 에 복소수 값이 없어서 거절하는데, 파이썬에는 있다.
        if str(self._h.dtype) == "complex64":
            return complex(float(flat[0]), float(flat[1]))
        return float(flat[0])

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
        """**int32 가 없다 — 그래서 거절한다.** 오래 int64 를 내주고 있었다.

        값은 그럴듯한데 `x.int().dtype == torch.int32` 를 보는 코드가 진짜 torch 에서
        갈리고, 그때 원인은 이 줄이 아니라 훨씬 뒤에서 드러난다. 코어도 같이 고쳤다 —
        **거절 문구가 셋에서 같아야** 배우는 사람이 "구현마다 다른 것" 으로 안 읽는다.
        """
        _absent_dtype("int", "int32")

    def type_as(self, other):
        """`other` 의 형으로 맞춘다. 코어에도 없던 이름이라 양쪽에 같이 넣었다."""
        return self.to(other.dtype if isinstance(other, Tensor) else other)

    # ── 묻는 것 넷 ────────────────────────────────────────────────────────
    #
    # 셋은 형과 값에서 바로 나온다. 넷째(`is_contiguous`)만 성격이 다르다 —
    # **여기는 뷰가 없어서 늘 참이다.** 그것은 이 술어의 이야기가 아니라 뷰의
    # 이야기이고, 뷰 전파를 거절하는 자리와 같은 뿌리다. 코어는 numpy 가 전치를
    # 뷰로 주므로 거기서 거짓이 되고, 그 갈림을 골든에 따로 굳혀 뒀다.

    def is_floating_point(self):
        return self.dtype.plain in ("float32", "float64")

    # ── 조밀 텐서에도 답이 있는 다섯 ──────────────────────────────────────
    #
    # 이름만 보면 희소·장치라 없는 게 맞다고 세게 되는데, torch 는 조밀 텐서에서
    # 그냥 해낸다 — "이 텐서는 조밀하다"·"CPU 다" 라는 답이 있는 것이다. 코어에도
    # 같이 넣었다.

    def dense_dim(self):
        return self.ndim

    # ── 모듈에만 있던 이름을 메서드로도 낸다 ──────────────────────────────
    #
    # torch 는 거의 모든 연산을 둘 다 준다. 코어에 열셋이 모듈에만 있었고 여기도
    # 같았다 — 교재가 쓰는 쪽은 메서드다.

    def igamma(self, other):
        from . import _ops
        return _ops.igamma(self, other)

    def igammac(self, other):
        from . import _ops
        return _ops.igammac(self, other)

    def polygamma(self, n):
        """**인자가 뒤집혀 있다** — 모듈은 `polygamma(n, x)`, 메서드는
        `x.polygamma(n)` 이다(torch 가 그렇게 둔다). 표로 그냥 붙이면 차수와 입력이
        뒤바뀐 채 값이 나온다."""
        from . import _ops
        return _ops.polygamma(n, self)

    def polygamma_(self, n):
        return self._write_back(self.polygamma(n))

    def is_same_size(self, other):
        return tuple(self.shape) == tuple(other.shape)

    def is_inference(self):
        return False

    def is_distributed(self):
        return False

    def share_memory_(self):
        """프로세스 사이 공유는 없다. torch 도 CPU 에서는 자기를 돌려준다."""
        return self

    def requires_grad_(self, requires_grad=True):
        """**밑줄을 떼면 참거짓 속성이 나온다.** 일반 길로 두면 `requires_grad` 를
        찾아 부르려다 `'bool' object is not callable` 로 멈춘다 — 제자리 이름을
        짝에서 만드는 규칙이 **속성과 부딪히는** 자리다.
        """
        self.requires_grad = bool(requires_grad)
        return self

    # ── 분포에서 뽑아 제자리에 채우는 일곱 ────────────────────────────────
    #
    # **값은 못 굳힌다**(난수기가 셋 다 다르다 — `randn` 에서 이미 받아들인 자리다).
    # 그래서 맞출 것은 모양·형과 **거절**이다. torch 의 규칙이 분포마다 다르고
    # 예외 종류까지 다르다 — 코어에 그 표가 있으니 여기서는 그것을 빌려 쓴다.
    #
    # 값을 파이썬에서 만들어 되쓴다. 셰이더로 뽑으면 씨앗 규칙이 두 벌이 되고,
    # 그 두 벌은 언젠가 갈린다.
    def _draw_(self, name, *args, **kw):
        from borch._tensor import Tensor as _Core

        core = _Core(self.numpy().copy())
        getattr(core, name)(*args, **kw)
        from ._base import tensor as _t
        return self._write_back(_t(core.data))

    def normal_(self, mean=0.0, std=1.0, generator=None):
        del generator
        return self._draw_("normal_", mean, std)

    def uniform_(self, from_=0.0, to=1.0, generator=None):
        del generator
        return self._draw_("uniform_", from_, to)

    def exponential_(self, lambd=1.0, generator=None):
        del generator
        return self._draw_("exponential_", lambd)

    def cauchy_(self, median=0.0, sigma=1.0, generator=None):
        del generator
        return self._draw_("cauchy_", median, sigma)

    def log_normal_(self, mean=1.0, std=2.0, generator=None):
        del generator
        return self._draw_("log_normal_", mean, std)

    def geometric_(self, p, generator=None):
        """**이산이라 정수 텐서에서도 돈다** — 연속 다섯과 갈리는 하나다."""
        del generator
        return self._draw_("geometric_", p)

    def random_(self, from_=0, to=None, generator=None):
        del generator
        return self._draw_("random_", from_, to)

    def fill_diagonal_(self, value, wrap=False):
        from ._base import tensor as _t
        got = self.numpy().copy()
        _np.fill_diagonal(got, value, wrap=wrap)
        return self._write_back(_t(got))

    def sparse_dim(self):
        return 0

    def to_dense(self):
        return self

    def storage_offset(self):
        return 0

    def get_device(self):
        """장치 번호가 없다는 뜻으로 -1 이다 — 오류가 아니다(실측)."""
        return -1

    def is_signed(self):
        return self.dtype.plain not in ("bool", "uint8")

    def is_nonzero(self):
        if self.numel() != 1:
            raise RuntimeError(
                f"값이 {self.numel()}개인 텐서의 참거짓은 모호합니다. "
                "(torch: Boolean value of Tensor with "
                f"{'no values' if self.numel() == 0 else 'more than one value'}"
                " is ambiguous)")
        return bool(self.item() != 0)

    def is_contiguous(self):
        """**늘 참이다** — GPU 버퍼를 뷰로 나눠 갖지 않으므로 비연속이 될 자리가
        없다. 코어는 numpy 뷰 때문에 전치 뒤 거짓이 된다."""
        return True

    def contiguous(self):
        """이미 연속이라 자기를 돌려준다. 코어에서는 비연속이면 옮겨 담는다."""
        return self

    def cfloat(self):
        """complex64. **이름표 갈이가 아니다** — borch.ts 는 복소수를 `[re, im]`
        엇갈이로 저장하므로 칸 수가 두 배다. 허수부 0 을 붙여 진짜로 만든다."""
        from . import _ops
        return _ops.complex(self, _ops.zeros_like(self))

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
        from ._ops import (
            _BINARY_ONLY, _EXTREME, camel, positional, refuse_if_nullary,
        )

        # 모듈 쪽에 손으로 쓴 것들은 메서드로도 같은 것을 써야 한다 — 인자 순서가
        # 뒤집혔거나(`split`) 한쪽만 올 수 있는(`clamp`) 자리들이다.
        #
        # **제자리 판도 여기로 온다.** `transpose_` 를 borch.ts 로 그냥 넘기면 저쪽
        # `transpose_()` 는 축을 안 받아서 `transpose_(0, 1)` 이 멈춘다 — 축을 푸는
        # 일이 이 파일에 있으므로 밑줄 있는 쪽도 같은 자리를 지나야 한다.
        inplace = name.endswith("_") and not name.endswith("__")
        bare = name[:-1] if inplace else name
        if bare in ("clamp", "clip", "split", "chunk", "aminmax", "flip",
                    "pow", "squeeze", "repeat_interleave", "flatten",
                    "sum", "norm", "transpose", "swapdims", "remainder",
                    # `max`·`min` 은 인자에 따라 셋으로 갈린다 — 저쪽은 가운데
                    # 하나뿐이고 기본 축이 0 이라, 그냥 넘기면 `x.max()` 가 전체
                    # 최댓값 대신 축 0 을 줄인 쌍을 낸다.
                    "max", "min",
                    # 색인으로 쓰는 쪽 — 이름과 인자가 borch.ts 와 다르다.
                    "scatter", "scatter_add", "index_add", "index_copy",
                    "index_fill", "take", "take_along_dim",
                    # **모듈에만 있고 borch.ts 에는 없는 이름들.** 여기 있는 것은 전부
                    # 이미 있는 계산의 조합이라 borch.ts 쪽에 이름을 안 늘렸는데,
                    # 그러다 보니 `borch.t(x)` 는 되고 `x.t()` 는 안 되는 한쪽 고리만
                    # 남았다. torch 는 둘 다 주고 교재 코드는 메서드 꼴을 더 쓴다.
                    #
                    # 이 목록은 스스로 검사된다 — 이름마다 골든 케이스가 있고, 골든은
                    # **진짜 torch 를 돌려** 굳혔으므로 torch 에 없는 이름은 애초에
                    # 굳히는 자리에서 멈춘다.
                    "multiply", "true_divide", "floor_divide", "lerp",
                    "greater", "less_equal", "isclose", "nan_to_num", "fmax",
                    "inner", "adjoint", "moveaxis", "t", "corrcoef", "cov",
                    "vdot", "kron", "broadcast_to",
                    # 여섯이 빠져 있었다. **짝은 위에 있는데 별칭만 없는 꼴**이라
                    # `x.multiply_(3)` 은 되고 `x.divide_(2)` 는 안 됐다 — 두 이름이
                    # 나란히 서 있는데 한쪽만 도는 자리가 제일 안 보인다. 코어에
                    # 제자리 판 마흔한 개를 채우면서 셋을 나란히 재다가 걸렸다.
                    "divide", "subtract", "greater_equal", "less", "not_equal",
                    "logical_xor",
                    # 참거짓이면 논리 부정으로 갈라야 한다 — 그 갈림이 `_ops` 에 있다.
                    "bitwise_not",
                    # 모양·색인 중 **모듈 쪽에 손으로 쓴 것들.** 텐서 목록을 받거나
                    # (`index_put`) 묶음으로 답하는(`tensor_split`) 자리라, 이름을
                    # 그냥 넘기면 JS 쪽에서 정수 변환이나 목록 꼴에서 걸린다.
                    "index_put", "index_put_", "tensor_split",
                    "split_with_sizes", "unique_consecutive",
                    # 희소 전용이라 거절만 한다 — borch.ts 에는 이름이 없다.
                    "sspaddmm",
                    # **`linalg` 쪽과 이름이 겹치는 둘.** 그냥 넘기면 `luSolve` 가
                    # 잡혀서 인자 순서가 뒤집힌 채 다른 답이 나온다.
                    "lu", "lu_solve",
                    # 통계 중 모듈 쪽에 손으로 쓴 것들 — 난수와 거절, 조립, 그리고
                    # 경계를 목록으로 주는 `histogramdd`.
                    "bernoulli", "float_power", "stft", "istft", "hash_tensor", "trapz",
                    "histogramdd",
                    # 복소수의 이웃 — 항등이라 borch.ts 에 이름이 없다.
                    "real", "conj", "conj_physical", "conj_physical_",
                    "resolve_conj", "resolve_neg", "imag", "angle",
                    "is_complex", "is_conj", "is_neg"):
            from . import _ops
            # **`max`·`min` 은 모듈 전역에 없다.** 그 이름을 `_ops` 에 두면 그 파일
            # 안에서 파이썬 내장을 가리고, `max(a, b)` 로 크기를 재던 자리가 텐서
            # 함수를 부른다 — 증상이 GPU 버퍼 할당 실패라 원인에서 아주 멀다.
            # **밑줄 이름이 따로 있으면 그것을 먼저 쓴다.** 보통은 짝을 부르고 값을
            # 되쓰면 되는데, 밑줄이 붙었다고 **같은 연산이라는 보장이 없다** —
            # `bernoulli_(p)` 는 자기 값을 확률로 읽는 `bernoulli()` 와 달리 자기 값을
            # 무시하고 `p` 로 채운다. 짝으로 흘려보내면 인자 수부터 안 맞는다.
            #
            # **`__dict__` 로 본다.** `getattr` 은 이 모듈의 `__getattr__` 을 깨워서
            # 없는 이름을 자기 자신으로 되돌려 준다 — 무한 재귀가 되고, 증상이
            # `maximum recursion depth exceeded` 라 원인에서 멀다.
            exact = _ops.__dict__.get(name) if inplace else None
            if exact is not None:
                # **제자리 이름이므로 되쓴다.** 값만 돌려주면 `x.bernoulli_(0)` 이
                # `x` 를 안 바꾸고, 그것은 제자리 연산이 아니다.
                self._refuse_inplace_on_leaf(name)
                return lambda *a, **k: self._write_back(exact(self, *a, **k))
            fn = _EXTREME[bare] if bare in _EXTREME else getattr(_ops, bare)
            if not inplace:
                return lambda *a, **k: fn(self, *a, **k)
            self._refuse_inplace_on_leaf(name)
            return lambda *a, **k: self._write_back(fn(self, *a, **k))

        # **기울기가 켜진 잎은 제자리로 못 고친다.** torch 가 그 자리에서 던지고
        # 골든이 그것을 굳혔다 — 흘려보내면 역전파가 이미 지난 값을 보게 된다.
        # **`no_grad` 안에서는 된다.** torch 도 그렇다 — 기울기를 안 만드는 동안에는
        # 잎을 고쳐도 역전파가 볼 것이 없다. 그 조건을 빼먹었더니 옵티마이저가
        # 파라미터를 갱신하는 정상 경로까지 막혔다.
        # **`detach_` 는 예외다.** 값을 안 건드리고 그래프만 끊으므로 역전파가 이미
        # 지난 값을 볼 일이 없다 — torch 도 잎에서 허용한다.
        if inplace and bare != "detach":
            self._refuse_inplace_on_leaf(name)

        js_name = camel(name)

        # **borch.ts 에 제자리 판이 없는 이름들.** 저쪽은 단항 표에서만 `abs_` 꼴을
        # 자동으로 만들고, 이항(`eq_`)은 거기 없다. 그 자리에서는 **계산을 밑줄 없는
        # 쪽이 하고** 결과를 이 버퍼로 옮긴다 — 코어의 `_inplace` 와 같은 자리, 같은
        # 이유다. 식을 두 벌로 두면 언젠가 갈리고, 값이 그럴듯해서 안 보인다.
        if inplace and getattr(self._h, js_name, None) is None:
            return lambda *a, **k: self._write_back(
                getattr(self, bare)(*a, **k))

        if name in _BINARY_ONLY:
            # borch.ts 는 단항만 표에서 메서드로 만든다. 이항은 `binary(이름, 상대)` 다.
            return lambda other, *_: guarded(self._h.binary, js_name, handle(other))
        got = getattr(self._h, js_name, None)
        if got is None:
            # **첫 마디를 torch 와 같게 둔다.** 예전에는 `borch.ts 텐서에 X 이
            # 없다` 로만 말했는데, 같은 이름을 코어에 물으면 파이썬의 표준 문구가
            # 났다 — 배우는 사람은 그 둘을 보고 **구현마다 다른 것**으로 읽는다.
            # 진짜 torch 도 `'Tensor' object has no attribute 'x'` 라고 말한다.
            #
            # 뒤에 붙는 힌트는 우리를 위한 것이라 남긴다. 앞 마디로 무는 검사는
            # 그대로 통과하고, 결속을 고칠 때는 `js_name` 이 필요하다.
            raise AttributeError(
                f"'Tensor' object has no attribute '{name}'"
                f" — borch.ts 에 `{js_name}` 이 없다")
        if not callable(got):
            return settle(got)

        def call(*args, **kw):
            laid = positional(name, args, kw)
            refuse_if_nullary(js_name, got, len(laid))
            out = guarded(got, *laid)
            # **제자리 연산은 자기 자신을 돌려줘야 한다.** borch.ts 의 `abs_` 는 같은
            # 손잡이를 돌려주지만 `guarded` 가 그것을 **새 파이썬 텐서로 감싸므로**,
            # `x.absolute_() is x` 가 거짓이 된다. 그러면 `x.mul_(2).add_(1)` 처럼
            # 이어 부르는 코드가 원본이 아닌 사본을 고치기 시작한다.
            return self if inplace else out

        call.__name__ = name
        return call

    def _refuse_inplace_on_leaf(self, _name):
        """**기울기가 켜진 잎은 제자리로 못 고친다.** torch 가 그 자리에서 던지고
        골든이 그것을 굳혔다 — 흘려보내면 역전파가 이미 지난 값을 보게 된다.

        **`no_grad` 안에서는 된다.** torch 도 그렇다 — 기울기를 안 만드는 동안에는
        잎을 고쳐도 역전파가 볼 것이 없다. 그 조건을 빼먹었더니 옵티마이저가
        파라미터를 갱신하는 정상 경로까지 막혔다.
        """
        if (_ts.gradMode.enabled and bool(self._h.requiresGrad)
                and not self._h.parents.length):
            raise RuntimeError(
                "a leaf Variable that requires grad is being used in an "
                "in-place operation.")

    def _write_back(self, out):
        """계산한 값을 이 버퍼로 옮기고 **같은 텐서**를 돌려준다.

        borch.ts 의 `copyFrom` 은 모양이 달라지면 그것도 따라간다 — `transpose_` 는
        칸 수는 그대로 두고 보는 틀을 바꾼다.
        """
        self._h.copyFrom(handle(out))
        return self

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
    def __pow__(self, other):
        """**정수 지수는 곱셈으로 푼다.** WGSL 의 `pow` 는 `exp2(y·log2(x))` 라 밑이
        음수면 답이 없고, 짝수 지수의 순방향만 우연히 맞고 기울기가 nan 이 된다 —
        `borch.ts` 의 `powScalar` 가 그 자리를 위해 있다."""
        from ._ops import pow as _pow
        return _pow(self, other)
    __eq__, __ne__ = _op("eq"), _op("ne")
    __lt__, __le__ = _op("lt"), _op("le")
    __gt__, __ge__ = _op("gt"), _op("ge")

    def __mod__(self, other):
        return guarded(self._h.remainder, float(other))

    def __matmul__(self, other):
        return guarded(self._h.mm, handle(other))

    def _inplace(js_name):                                   # noqa: N805
        """`x += 1` 도 제자리 연산이다 — 잎에 기울기가 켜져 있으면 torch 가 거절한다."""
        def go(self, other):
            return getattr(self, f"{js_name}_")(other)
        return go

    __iadd__ = _inplace("add")
    __isub__ = _inplace("sub")
    __imul__ = _inplace("mul")
    __itruediv__ = _inplace("div")

    del _inplace

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
                at = k + n if k < 0 else k
                if not 0 <= at < n:
                    # torch 는 여기서 `IndexError` 다 — 종류도 API 이므로 맞춘다.
                    raise IndexError(
                        f"index {k} is out of bounds for dimension {axis} "
                        f"with size {n}")
                out = wrap(out._h.select(axis, at))
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


class LinAlgError(RuntimeError):
    """torch 의 `linalg.LinAlgError`.

    **이름이 하는 일이 있다.** 특이행렬을 만날 수 있는 코드는 `except
    linalg.LinAlgError` 로 감싸는 것이 보통인데, 그냥 `RuntimeError` 로 올리면 그
    감싸기를 지나쳐 프로그램이 죽는다. borch.ts 쪽에 같은 이름의 클래스가 있고
    `translate` 가 둘을 잇는다 — 예외의 종류도 API 다.
    """


def translate(exc):
    """JS 쪽 예외를 **torch 가 내는 종류**로 옮긴다.

    골든이 예외의 **종류 이름까지** 답으로 굳혔다(`RuntimeError|문구=True`). 그대로
    두면 파이썬 쪽에 `JsException` 이 올라오고, `except RuntimeError` 로 잡던 코드가
    안 잡힌다 — 예외의 종류도 API 다.

    문구는 안 바꾼다. borch.ts 가 이미 torch 의 원문을 담고 있고, 그것이 검색을
    통하게 하려고 그렇게 쓴 것이다.
    """
    text = str(exc)
    # 아래의 어림짐작(문구에 index 가 있으면 IndexError)에 맡기면 안 되는 것들.
    # **거절도 종류가 있다** — "아직 없다"(NotImplementedError)와 "부른 쪽이
    # 틀렸다"(RuntimeError)는 다른 말이고, 골든이 그 이름을 답으로 굳혔다.
    for head, cls in (("LinAlgError: ", LinAlgError),
                      ("NotImplementedError: ", NotImplementedError)):
        if text.startswith(head):
            return cls(text[len(head):])
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
            return _Fields({k: getattr(out, k) for k in keys},
                           _TORCH_FIELDS.get(tuple(keys)))
    return out


# borch.ts 의 자리 이름 → **torch 의 이름.**
#
# torch 는 이것들을 이름으로도 물을 수 있게 준다 — `slogdet(A).logabsdet`,
# `qr(A).Q`, `eigh(A).eigenvalues`. 자리로만 맞춰 두면 값이 맞는데도 교재 코드가
# 속성 접근에서 멈춘다. `lstsq` 가 `.solution` 으로 그 자리를 이미 겪었다.
#
# **낱개 이름이 아니라 자리 묶음으로 건다.** `values` 는 `eigh` 에서 고윳값이지만
# `sort`·`topk` 에서는 그냥 값이라, 이름 하나만 보고 바꾸면 엉뚱한 것까지 바뀐다.
_TORCH_FIELDS = {
    ("sign", "logabs"): {"logabs": "logabsdet"},
    ("q", "r"): {"q": "Q", "r": "R"},
    ("u", "s", "vt"): {"u": "U", "s": "S", "vt": "Vh"},
    ("values", "vectors"): {"values": "eigenvalues", "vectors": "eigenvectors"},
}


class _Fields:
    """이름 붙은 자리를 여럿 주는 답. 첨자로도 이름으로도 닿는다 — torch 가 그렇다."""

    __slots__ = ("_d", "_order")

    def __init__(self, d, alias=None):
        # **자리 순서는 JS 쪽 이름으로 둔다.** 별명을 순서에 넣으면 `[0]`·`[1]` 이
        # 밀린다 — 이름을 얹으려다 자리를 어긋내는 것이 된다.
        self._order = list(d)
        vals = {k: (wrap(v) if _js.borch.isTensor(v) else v) for k, v in d.items()}
        for js_name, torch_name in (alias or {}).items():
            if js_name in vals:
                vals[torch_name] = vals[js_name]
        object.__setattr__(self, "_d", vals)

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
            _js_list([]), _js_options(dtype="bool")))
    if isinstance(x, int):
        return Tensor(_ts.Tensor.from_(
            _js.Float32Array.new(_to_js([float(x)])),
            _js_list([]), _js_options(dtype="int64")))
    if isinstance(x, float):
        return Tensor(_ts.Tensor.full(_js_list([]), x))
    return Tensor(x)


def handle(x):
    """상대가 우리 텐서면 손잡이를, 수면 스칼라 텐서를 만들어 그 손잡이를."""
    return wrap(x)._h


def tensor(data, dtype=None, requires_grad=False):
    """`torch.tensor` 자리. numpy 배열·중첩 리스트·수를 받는다."""
    from pyodide.ffi import JsException

    arr = _np.asarray(data)
    if dtype is not None:
        # `torch.float32` 로 보이는 물건이 와도 borch.ts 에는 `float32` 로 넘긴다.
        name = dtype.plain if isinstance(dtype, _DType) else str(dtype)
    elif arr.dtype.kind == "c":
        name = "complex64"
    elif arr.dtype == bool:
        name = "bool"
    elif arr.dtype.kind in "iu":
        name = "int64"
    else:
        name = "float32"
    # **복소수는 다른 문으로 들어간다.** borch.ts 의 `Tensor.from` 은 형만
    # `complex64` 라고 붙는 것을 거절한다 — 저장이 칸당 f32 두 개라 이름표만 갈면
    # 뒤쪽 절반이 남의 메모리가 되기 때문이다. 실수부와 허수부로 갈라서 엮는다.
    if name == "complex64":
        if requires_grad:
            # **잎이 안 되기 때문에 거절한다.** 엮어서 만들면 `ComplexBackward0` 이
            # 붙은 **중간 마디**가 되고, torch 의 `requires_grad=True` 텐서는 잎이다.
            # 그 차이는 `.grad` 가 안 쌓이는 것으로만 드러난다 — 값은 다 맞는 채로.
            raise RuntimeError(
                "complex64 텐서에 requires_grad=True 를 여기서는 못 준다 — "
                "실수 잎 둘을 만들어 `complex(re, im)` 으로 엮어라.")
        parts = _np.asarray(arr, dtype=_np.complex64)
        pair = [_np.ascontiguousarray(half.ravel(), dtype=_np.float32)
                for half in (parts.real, parts.imag)]
        made = [_ts.Tensor.from_(_js.Float32Array.new(_to_js(half)),
                                 _js_list(parts.shape), _js_options())
                for half in pair]
        return Tensor(_ts.Tensor.complex(made[0], made[1]))
    flat = _js.Float32Array.new(_to_js(arr.ravel().astype(_np.float32)))
    try:
        return Tensor(_ts.Tensor.from_(
            flat, _js_list(arr.shape),
            _js_options(requiresGrad=bool(requires_grad), dtype=name)))
    except JsException as exc:
        # 정수·참거짓에 기울기를 켜는 것은 torch 도 거절한다. 종류를 옮겨 준다 —
        # `except RuntimeError` 로 잡던 코드가 안 잡히면 안 된다.
        raise translate(exc) from None


# ── 없는 형은 이름째 거절한다 — **코어와 같은 문구로** ────────────────────────
#
# 그냥 두면 `AttributeError: borch.ts 텐서에 'half' 이 없다` 가 났다. 코어는
# `BrowserTorchError: '.half()'(float16) 은(는) 브라우저 축소판에 없습니다` 였고,
# 배우는 사람은 그 둘을 보고 **구현마다 다른 것** 으로 읽는다. 값이 아니라 문구를
# 맞추는 자리이고, 그런 자리는 서로 대조해도 안 걸린다 — 아무도 안 물었기 때문이다.
_ABSENT_DTYPES = {
    "half": "float16", "bfloat16": "bfloat16", "chalf": "complex32",
    "cdouble": "complex128", "byte": "uint8", "char": "int8", "short": "int16",
}


def _absent_dtype(name, shown):
    # 예외 **종류**도 같아야 한다 — 코어의 `BrowserTorchError` 를 빌려온다.
    # 브라우저에서는 `borch` 도 `/work` 아래 있으므로 늦게 들여오면 된다
    # (`_core_repr` 이 같은 방식이다).
    from borch._base import BrowserTorchError
    raise BrowserTorchError(
        f"`.{name}()`({shown}) 은(는) 브라우저 축소판에 없습니다.\n"
        "자기 컴퓨터에서 `uv add torch` 로 진짜 PyTorch 를 쓰세요 — "
        "축소판은 문법 연습용이고, 없는 것을 흉내 내면 틀린 것을 배우게 됩니다.")


def _bind_absent(name, shown):
    def method(self):
        del self
        _absent_dtype(name, shown)

    method.__name__ = name
    return method


for _dname, _shown in _ABSENT_DTYPES.items():
    setattr(Tensor, _dname, _bind_absent(_dname, _shown))
del _dname, _shown
