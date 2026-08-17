"""borch 를 쪼갠 조각. 공개 이름은 __init__ 이 모은다."""

import math as _math

import numpy as _np


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


_TYPE_NAMES = {"b": "Bool", "i": "Long", "u": "Long", "f": "Float"}


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


