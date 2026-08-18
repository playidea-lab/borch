"""**`out=` 을 받는 이름 표가 낡지도 짧지도 않은가.**

`borch/__init__.py` 의 `_TAKES_OUT` 에 적힌 이름만 `out=` 을 받는다. 코어는 torch 에
기대지 않으므로 표는 손으로 적혀 있고, 그 표를 torch 에 대 보는 것이 여기다.

## 왜 docstring 으로 못 만드나

torch 의 C 함수는 서명을 들여다볼 수 없어서 처음에는 docstring 의 `out=None` 으로
골랐다. 그 목록은 **넓다** — `rand_like`·`zeros_like`·`ones_like`·`median`·
`nanmedian`·`where`·`std_mean`·`var_mean`·`hamming_window` 는 거기 적혀 있는데 실제
오버로드는 `out=` 을 안 받는다. aten 스키마도 마찬가지로 `rand_like` 에 out 변종이
있다고 말한다 — 파이썬 층에서 막힐 뿐이다.

그래서 **실제로 불러 본다.** 인자를 몇 가지 꼴로 만들어 보고, 통하는 꼴을 찾으면
같은 인자에 `out=` 을 붙여 다시 부른다. `TypeError` 면 안 받는 것이고, 다른 오류는
**받은 뒤** 난 것이므로 받는 것으로 센다.

## 인자를 못 만든 이름

몇은 어떤 꼴로도 안 통한다(`from_file`·`hspmm`·`sparse_compressed_tensor` 처럼
우리에게 없거나 특별한 것들). 그런 이름은 **판정을 안 한다** — 모르는 것을 아는 척
하면 표가 거짓말을 시작한다. 우리에게 없는 이름이면 어차피 표에 못 들어간다.
"""

import inspect
import warnings

import torch

import borch

_V = torch.tensor([1.0, 2.0, 3.0])
_I = torch.tensor([0, 1, 2])
_M = torch.eye(3)
_B = torch.ones(2, 3, 3)
_P = torch.ones(2, 3)

PATTERNS = [
    (_V,), (_V, _V), (_M,), (_M, _M), (_V, 1), (_V, 0), (_I,), (_I, _I),
    ([_V, _V],), (_V, _V, _V), (3,), (0, 3), (_M, 0), (_M, _M, _M), (_B, _B),
    (_M, _V), (_V, 2), (_M, _I), (0.0, 1.0, 3), (_P, _P), (_B,), (_M, 1),
    (2, 3), (_M, 0, True),
]
# 꼴로 안 잡히는 것들은 손으로 준다. 여기 없고 꼴로도 안 되면 **판정을 안 한다.**
HAND = {
    "addbmm": (_M, _B, _B), "addmv": (_V, _M, _V), "baddbmm": (_B, _B, _B),
    "gather": (_M, 0, torch.zeros(3, 3, dtype=torch.int64)),
    "masked_select": (_V, torch.tensor([True, False, True])),
    "polygamma": (1, _V), "randint": (0, 5, (3,)), "renorm": (_M, 2, 0, 1.0),
    "narrow_copy": (_V, 0, 0, 2),
    "lu_solve": (_M, _M, torch.tensor([1, 2, 3], dtype=torch.int32)),
    "ormqr": (_M, _V, _M),
}


def _classify(name):
    """('단일'|'여럿'|'안 받음'|None). None 은 **판정 못 함**이다."""
    fn = getattr(torch, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for args in ([HAND[name]] if name in HAND else PATTERNS):
            try:
                got = fn(*args)
            except Exception:                               # noqa: BLE001
                continue
            many = (isinstance(got, tuple) and got
                    and all(isinstance(x, torch.Tensor) for x in got))
            if not isinstance(got, torch.Tensor) and not many:
                continue
            dst = (tuple(torch.empty_like(x) for x in got) if many
                   else torch.empty_like(got))
            try:
                fn(*args, out=dst)
            except TypeError:
                return "안 받음"
            except Exception:                               # noqa: BLE001
                pass
            return "여럿" if many else "단일"
    return None


def _candidates():
    for name in sorted(dir(torch)):
        if name.startswith("_"):
            continue
        fn = getattr(torch, name)
        if not callable(fn) or inspect.isclass(fn):
            continue
        if "out=None" in (fn.__doc__ or ""):
            yield name


def test_the_out_table_is_not_stale():
    """표에 적힌 이름은 torch 에서 실제로 `out=` 을 받아야 한다."""
    wrong = []
    for name in sorted(borch._TAKES_OUT | borch._TAKES_OUT_TUPLE):
        want = "여럿" if name in borch._TAKES_OUT_TUPLE else "단일"
        got = _classify(name)
        if got is not None and got != want:
            wrong.append(f"{name} — 표는 {want}, torch 는 {got}")
    assert not wrong, (
        "`_TAKES_OUT` 이 낡았다:\n  " + "\n  ".join(wrong) + "\n\n"
        "그 이름을 표에서 빼라 — 우리가 torch 보다 관대하면, 여기서 도는 코드가\n"
        "자기 컴퓨터에서 멈춘다. 관대한 것도 갈리는 것이다."
    )


def test_the_out_table_is_not_short():
    """torch 가 `out=` 을 받고 우리에게도 있는 이름은 표에 있어야 한다."""
    missing = []
    for name in _candidates():
        if not hasattr(borch, name):
            continue
        if name in borch._TAKES_OUT or name in borch._TAKES_OUT_TUPLE:
            continue
        if _classify(name) in ("단일", "여럿"):
            missing.append(name)
    assert not missing, (
        "`out=` 을 받아야 하는데 표에 없다: " + ", ".join(missing) + "\n\n"
        "표에 넣어라. 빠진 이름은 `_no_out` 이 거절하는데, torch 에서는 되므로\n"
        "**교재의 그 줄이 여기서만 멈춘다.**"
    )
