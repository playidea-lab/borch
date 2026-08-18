"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np

__all__ = ["Tensor", "tensor", "nn", "optim", "no_grad"]

_DEFAULT_DTYPE = _np.float32


class BrowserTorchError(NotImplementedError):
    """축소판이 지원하지 않는 것. 근사하지 않고 여기서 멈춘다."""


def _like_torch(korean: str, torch_phrase: str) -> str:
    """오류 메시지의 규격.

    한국어 설명만 두면 학습자가 검색해서 답을 못 찾고, 영문만 베끼면 이 교재가
    한국어인 이유가 사라진다. 둘 다 넣는다 — 설명은 읽고, 영문 문구는 검색한다.
    """
    return f"{korean}\n(torch: {torch_phrase})"


def _unsupported(what: str):
    raise BrowserTorchError(
        f"{what} 은(는) 브라우저 축소판에 없습니다.\n"
        "자기 컴퓨터에서 `uv add torch` 로 진짜 PyTorch 를 쓰세요 — "
        "축소판은 문법 연습용이고, 없는 것을 흉내 내면 틀린 것을 배우게 됩니다."
    )


# torch 가 오류 문구에서 쓰는 형 이름. **복소수가 빠져 있었다** —
# `out=` 의 형 거절이 그 자리에서 `KeyError` 로 터지며 알려 주었다.
_TYPE_NAMES = {"b": "Bool", "i": "Long", "u": "Long", "f": "Float",
               "c": "ComplexFloat"}


def _needs_float(data, korean: str, torch_phrase: str):
    """**torch 가 멈추는 자리에서 우리도 멈춘다.**

    평균·분산·노름은 나눗셈과 제곱근이라 정수 칸에 답이 안 들어간다. numpy 는 조용히
    float64 로 올려 값을 내주는데, 그 값을 받은 사람은 torch 에서 같은 줄이 `RuntimeError`
    로 멈추는 것을 나중에야 안다 — 이 저장소의 첫 줄이 거절하는 종류다.

    빠져나가는 것이 numpy 쪽이라 자리마다 따로 막아야 한다. 한 곳에 모아 두면 어느
    함수가 이 규칙 아래 있는지가 목록으로 보인다.
    """
    if data.dtype.kind not in "fc":
        raise RuntimeError(_like_torch(korean, torch_phrase))


def _refuses_bool(data, korean: str, torch_phrase: str, kind=RuntimeError):
    """참·거짓만 거절하는 자리. `argmax`·`median` 이 그렇다(실측)."""
    if data.dtype.kind == "b":
        raise kind(_like_torch(korean, torch_phrase))


def _refuses_nonfloat_kernel(data, name: str, kernel: str):
    """torch 의 **커널 구멍**을 그대로 흉내 내는 자리.

    규칙이 아니다 — `logsumexp` 는 정수를 받는데 `logcumsumexp` 는 안 받는다(실측).
    torch 가 CPU 커널을 그 형으로 안 만들어 둔 것뿐이고, 그래서 `RuntimeError` 가
    아니라 `NotImplementedError` 로 나온다.

    **그래도 흉내 낸다.** 여기서 값을 내주면 그 코드가 진짜 torch 에서 깨지고,
    관대한 것도 갈리는 것이다. 커널 이름을 문구에 넣는 것은 torch 가 그렇게 하기
    때문이고, 검색이 통하는 쪽이 그 문구다.
    """
    if data.dtype.kind not in "fc":
        raise NotImplementedError(_like_torch(
            f"{name} 은(는) 실수에만 있습니다. `.float()` 을 먼저 부르세요.",
            f'"{kernel}" not implemented for '
            f"'{_TYPE_NAMES.get(data.dtype.kind, data.dtype.name)}'"))


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
# **복소수는 float32 둘이다.** 하드웨어 타입이 아니라 배치 규약이고(실측: 원소당
# 8 바이트, `view_as_real` 이 마지막 축에 `(re, im)`), 그래서 GPU 쪽에서도 표현된다.
complex64 = dtype("complex64", _np.complex64)
cfloat = complex64
# **`complex128` 은 영원히 없다.** WGSL 에 `f64` 가 없어서 `float64` 가 없고, 그러면
# 배정도 복소수도 없다. 이름만 두는 이유는 **승격이 그것을 만들기 때문**이다 —
# `complex64 + float64` 가 torch 에서 `complex128` 이라(실측), 그 자리에서 멈추려면
# 무엇을 만들려다 멈췄는지 말할 수 있어야 한다.
complex128 = dtype("complex128", _np.complex128)
cdouble = complex128

# ── torch 가 최상위에 두는 수 상수 다섯 ────────────────────────────────────
#
# **커버리지 표가 구조적으로 못 보던 자리다.** `tests/torch_gap.py` 는 `callable`
# 인 이름만 세는데 이 다섯은 부를 수 있는 것이 아니라 **값**이다. 그래서 분모에도
# 분자에도 안 들어갔고, "torch 79% · 검토 대상 0" 이라는 수가 이것들이 없는 채로
# 나왔다. 세는 잣대가 못 보는 자리는 아무리 세도 안 보인다.
#
# 다 교재가 실제로 쓰는 이름이다 — `torch.clamp(x, min=-torch.inf)`,
# `x[:, torch.newaxis]`, `torch.pi`. numpy 의 같은 이름을 그대로 가리키므로
# 값이 갈릴 자리도 없다.
e = _math.e
pi = _math.pi
inf = _math.inf
nan = _math.nan
# torch 도 이것이 그냥 `None` 이다 — `x[:, None]` 과 같은 뜻이라는 표시다.
newaxis = None


class _AbsentDtype(dtype):
    """torch 와 **이름은 같은데 이 축소판에 칸이 없는** 형.

    이름을 아예 안 두면 `dtype=torch.int` 가 `AttributeError` 로 멈추는데, 그 문구는
    **오타와 구별이 안 된다.** 이름은 두고 쓰려 할 때 무엇이 없는지 말한다 —
    `complex128` 이 같은 이유로 이름만 있다.

    부모의 `__init__` 을 안 부른다. 부모는 `self.np` 를 값으로 심는데 여기서는 그
    자리가 **읽을 때 멈추는 문**이어야 한다.
    """

    def __init__(self, name, instead):
        self.name = name
        self._instead = instead

    @property
    def np(self):
        _unsupported(f"`torch.{self.name}` (→ `{self._instead}` 로 맞추세요)")


# **`torch.int` 는 int32 다**(실측 — `torch.long` 이 int64다). 정수 칸을 int64 하나로
# 모았으므로 int32 는 없다. 그래도 이름은 둔다: 교재가 `dtype=torch.int` 를 쓰고,
# 그때 "없다" 와 "오타다" 는 다른 말이어야 한다.
int32 = _AbsentDtype("int32", "int64")
# 같은 까닭으로 이름만 두는 나머지. **반정밀은 WGSL 에 없고**(f16 확장은 기기마다
# 다르다) 좁은 정수는 int64 하나로 모았다. 이름을 안 두면 `dtype=torch.half` 가
# `'function' object has no attribute 'np'` 로 멈추는데, 그건 오타와 같은 문구다.
float16 = _AbsentDtype("float16", "float32")
bfloat16 = _AbsentDtype("bfloat16", "float32")
int16 = _AbsentDtype("int16", "int64")
complex32 = _AbsentDtype("complex32", "complex64")
half = float16
short = int16
chalf = complex32

_NP_TO_DTYPE = {_np.dtype("float32"): float32, _np.dtype("float64"): float64,
                _np.dtype("int64"): int64, _np.dtype("bool"): bool_,
                _np.dtype("complex64"): complex64,
                _np.dtype("complex128"): complex128}


def _resolve(data, dt):
    """진짜 torch 의 규칙을 따른다 — 정수만 있으면 int64, 하나라도 실수면 float32.

    **파이썬 `complex` 가 섞이면 complex64 다**(실측: `torch.tensor([1+1j])` 가
    `complex64`). numpy 에 맡기면 `complex128` 이 되고, 그것은 우리에게 없다.
    """
    if dt is not None:
        return dt.np
    arr = _np.asarray(data)
    if arr.dtype.kind == "c":
        return _np.complex64
    if arr.dtype.kind == "b":
        return _np.bool_
    if arr.dtype.kind in "iu":
        return _np.int64
    return _np.float32


def _no_complex128(what="이 연산"):
    """**배정도 복소수는 만들 수 없다.** `float64` 가 없는 것과 같은 자리다."""
    raise BrowserTorchError(
        f"{what} 이(가) complex128 을 만들려 합니다 — 브라우저 축소판에는 "
        "`float64` 가 없고(WGSL 에 `f64` 가 없습니다) 그래서 배정도 복소수도 "
        "없습니다. `complex64` 로 맞추세요.")



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


def _nonfinite_str(v):
    """`nan`·`inf`·`-inf`. **점을 안 붙인다** — torch 도 그렇다(실측)."""
    return "nan" if _np.isnan(v) else ("inf" if v > 0 else "-inf")


def _integral_str(v):
    """정수 판. 유한하지 않은 값은 점 없이 그대로 간다.

    **여기가 한동안 `nan.` 을 찍고 있었다.** `f"{v:.0f}."` 가 `nan` 에도 점을 붙여서,
    `tensor([nan, 1.])` 이 `tensor([nan., 1.])` 로 나왔다. 소수 판은 `f"{nan:.4f}"`
    가 이미 `nan` 이라 이 자리만 갈렸고, 그래서 nan 이 낀 **정수** 텐서를 찍을 때만
    드러났다 — 복소수를 붙이며 실수부에 nan 을 넣어 보다가 잡혔다.
    """
    return _nonfinite_str(v) if not _np.isfinite(v) else f"{v:.0f}."


def _float_formatter(arr):
    """torch 의 규칙: 값이 전부 정수면 `1.`, 아니면 소수 네 자리, 범위가 넓으면 지수."""
    finite = arr[_np.isfinite(arr)]
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return _integral_str
    amax, amin = _np.abs(nonzero).max(), _np.abs(nonzero).min()
    integral = bool(_np.all(finite == _np.floor(finite)))

    if integral and amax < 1e8:
        return _integral_str
    if amax / amin > 1000 or amax > 1e8 or amin < 1e-4:
        return lambda v, p=_PRINT_PRECISION: f"{v:.{p}e}"
    return lambda v, p=_PRINT_PRECISION: f"{v:.{p}f}"


def _field_width(arr, fmt):
    """오른쪽 정렬 폭.

    **유한한 값만 센다** — torch 가 그렇다(실측). `nan` 을 폭에 넣으면 정수 판에서
    폭이 3 이 되어 `1.` 이 ` 1.` 로 밀리는데, torch 는 `tensor([nan, 1.])` 이다.
    유한하지 않은 값은 폭보다 길면 그냥 삐져나온다.
    """
    return max((len(fmt(v)) for v in _np.asarray(arr).reshape(-1)
                if _np.isfinite(v)), default=0)


def _tensor_str(data):
    if data.size == 0:
        return "[]" if data.ndim else "[]"
    if data.dtype.kind == "f":
        fmt = _float_formatter(data)
        # torch 는 원소를 같은 너비로 오른쪽 정렬한다 — 음수가 섞이면 양수 앞에 자리가 생긴다.
        width = _field_width(data, fmt)
        padded = lambda v, f=fmt, w=width: f(v).rjust(w)
        body = _np.array2string(
            data, formatter={"float_kind": padded}, separator=", ",
            max_line_width=_LINE_WIDTH - 8, threshold=1000)
    elif data.dtype.kind == "c":
        # **실수부와 허수부를 따로 잰다**(실측). `[1+2j, -0.5-1j]` 에서 실수부는 소수
        # 네 자리를 요구하고 허수부는 정수라, torch 가 `1.0000+2.j` 를 찍는다 — 한
        # 형식으로 재면 `1.0000+2.0000j` 가 되어 글자가 갈린다.
        #
        # **자리맞춤은 실수부에만 건다**(실측). 허수부는 안 밀고 부호는 값의 부호를
        # 그대로 쓴다 — 그래서 `1.-0.j` 처럼 **음의 0** 도 부호가 산다.
        re_fmt = _float_formatter(data.real)
        im_fmt = _float_formatter(data.imag)
        width = _field_width(data.real, re_fmt)

        def one(v, rf=re_fmt, mf=im_fmt, w=width):
            im = mf(v.imag)
            return f"{rf(v.real).rjust(w)}{im if im.startswith('-') else '+' + im}j"

        # **줄바꿈 자리도 명세다.** torch 는 한 줄에 들어갈 개수를 글자 수가 아니라
        # **폭**으로 센다 — `floor((linewidth − 7) / (실수폭 + 허수폭 + 3))`. numpy 는
        # 실제 글자 길이로 끊으므로, 같은 자리에서 끊기게 예산을 되계산해 넘긴다.
        # 실수 경로의 `_LINE_WIDTH - 8` 을 그대로 쓰면 12 개짜리가 6+6 이 아니라
        # 5+5+2 로 접혀서, 값이 전부 맞는데 글자가 갈린다.
        im_width = _field_width(data.imag, im_fmt)
        per_line = max(1, (_LINE_WIDTH - 7) // (width + im_width + 3))
        budget = per_line * (width + im_width + 4)
        body = _np.array2string(
            data, formatter={"complex_kind": one}, separator=", ",
            max_line_width=budget, threshold=1000)
    else:
        body = _np.array2string(data, separator=", ",
                                max_line_width=_LINE_WIDTH - 8, threshold=1000)
    # numpy 는 이어지는 줄을 한 칸 들여쓴다. torch 는 "tensor(" 만큼(8칸) 들여쓴다.
    return body.replace("\n ", "\n" + " " * 8)


def _tensor_repr(t):
    parts = [_tensor_str(t.data)]
    dt = t.data.dtype
    plain = dt in (_np.dtype("float32"), _np.dtype("int64"), _np.dtype("bool"))
    # **complex64 는 값이 있으면 형을 안 찍는다**(실측). 끝의 `j` 가 이미 복소수라고
    # 말하고 있어서 torch 도 생략한다. **빈 텐서에는 그 단서가 없어서 찍는다** —
    # `tensor([], dtype=torch.complex64)`. 규칙이 형이 아니라 **단서의 유무**에 걸려
    # 있는 자리라, 형 목록에만 넣어 두면 빈 것에서 갈린다.
    if dt == _np.dtype("complex64") and t.data.size > 0:
        plain = True
    if not plain:
        parts.append(f"dtype={t.dtype}")
    if t._op:
        parts.append(f"grad_fn=<{t._op}>")
    elif t.requires_grad:
        parts.append("requires_grad=True")
    return f"tensor({', '.join(parts)})"


# ---------------------------------------------------------------- Size

class device:                                                   # noqa: N801
    """`torch.device` — **장치를 가리키는 이름표.**

    이 이름이 오래 없었고, 그것이 이 목록에서 제일 큰 구멍이었다:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    **튜토리얼 절반의 첫 줄**이고, 이름이 없으면 거기서 `AttributeError` 로 멈춘다.
    `cuda.is_available()` 이 거짓이라 실제로 만들어지는 것은 `cpu` 인데도 그렇다.

    **만드는 것과 쓰는 것을 가른다.** `torch.device("cuda")` 는 **하드웨어가 없어도
    만들어진다**(실측 — torch 도 그렇다). 거기서 멈추면 위의 삼항식이 통째로 못
    돌고, 그러면 학습자는 자기 코드가 틀렸다고 읽는다. 멈추는 자리는 그 장치로
    **텐서를 옮길 때**이고, 그때 나오는 문구가 원인을 가리킨다.
    """

    __slots__ = ("type", "index")

    def __init__(self, kind, index=None):
        if isinstance(kind, device):
            self.type, self.index = kind.type, kind.index
            return
        text = str(kind)
        if ":" in text:
            text, _, tail = text.partition(":")
            index = int(tail)
        self.type = text
        self.index = None if index is None else int(index)

    def __repr__(self):
        tail = "" if self.index is None else f", index={self.index}"
        return f"device(type='{self.type}'{tail})"

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __eq__(self, other):
        # **문자열과는 안 같다**(실측: `torch.device("cpu") == "cpu"` 가 거짓).
        # 관대하게 참을 주면 `if d == "cpu":` 가 여기서는 도는데 진짜 torch 에서는
        # 안 돈다 — 관대한 것도 갈리는 것이고, 이쪽은 **조건문의 방향**을 바꾼다.
        return (isinstance(other, device) and self.type == other.type
                and self.index == other.index)

    def __hash__(self):
        return hash((self.type, self.index))


class Size(tuple):
    def __repr__(self):
        return f"torch.Size([{', '.join(str(x) for x in self)}])"


