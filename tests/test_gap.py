"""빈자리 표가 **거짓말을 하지 않는지** 본다.

`tests/torch_gap.py` 의 표는 이름 하나를 적을 때마다 우리 비율이 올라가거나(`NOT_API`)
할 일 목록이 줄어드는(`SKIPPED`) 표다. 그런 표는 시간이 가면 조용히 넓어진다 —
막는 것은 사람의 의지가 아니라 여기 적힌 검사다.

세 가지를 본다.

- **죽은 줄이 없는가.** 어떤 torch 이름도 안 걸리는 줄은 torch 가 바뀌었다는 뜻이고,
  그때 그 사유는 더 이상 무엇에 대한 것도 아니다.
- **사유가 있는가.** 빈 사유는 "그냥 안 함" 이고, 그러면 표가 하는 일이 없다.
- **모순이 없는가.** 우리가 이미 구현한 이름을 `NOT_API` 가 "API 가 아니다" 라고
  적고 있으면 둘 중 하나는 거짓이다.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

torch = pytest.importorskip("torch")

from torch_gap import (  # noqa: E402
    DELIBERATE, NOT_API, SKIPPED, _look, _public, _spaces,
)


def _every_torch_name():
    """torch 쪽 이름 전부. 자리마다 모으고 **자리 붙은 꼴도** 함께 낸다."""
    got = set()
    for space, theirs, _ours in _spaces():
        for name in _public(theirs):
            got.add(name)
            got.add(f"{space}.{name}")
    return got


def test_no_table_entry_matches_nothing():
    """**죽은 줄이 없어야 한다.**

    torch 가 이름을 지우거나 바꾸면 그 줄은 아무것도 안 걸르면서 남는다. 남은 사유는
    다음에 읽는 사람에게 "이건 안 하기로 했다" 로 읽히는데, 그 이름은 이제 없다.
    """
    names = _every_torch_name()
    dead = []
    for table, label in ((NOT_API, "NOT_API"), (SKIPPED, "SKIPPED")):
        for key in table:
            if not any(_look({key: "x"}, n, n) for n in names):
                dead.append(f"{label}['{key}']")
    assert not dead, (
        "표에 아무 이름도 안 걸리는 줄이 있다:\n  " + "\n  ".join(dead) +
        "\n\ntorch 가 바뀌었거나 처음부터 오타다. 지우거나 고쳐라 — 안 걸리는 줄의 "
        "사유는 무엇에 대한 것도 아니다.")


def test_no_deliberate_prefix_matches_nothing():
    """이름 공간 쪽도 같다.

    **`_spaces()` 만 봐서는 안 된다.** 이 표의 앞머리는 대부분 우리가 아예 안 세는
    하위 모듈(`torch.jit`·`torch.distributed`)을 가리키므로, 여덟 자리만 훑으면 전부
    죽은 줄로 보인다 — 처음에 그렇게 적어서 열셋이 한꺼번에 걸렸다. torch 가 실제로
    그 이름을 갖고 있는지를 본다.
    """
    dead = [key for key in DELIBERATE
            if not hasattr(torch, key.split(".")[0])
            and not any(n.startswith(key) for n in _every_torch_name())]
    assert not dead, (
        f"`DELIBERATE` 에 아무것도 안 걸리는 앞머리가 있다: {dead}\n"
        "  torch 가 그 자리를 없앴거나 처음부터 오타다.")


def test_every_reason_says_something():
    """사유가 비었거나 한 글자면 없는 것과 같다."""
    thin = []
    for table, label in ((DELIBERATE, "DELIBERATE"), (NOT_API, "NOT_API"),
                         (SKIPPED, "SKIPPED")):
        for key, reason in table.items():
            if not reason or len(reason.strip()) < 4:
                thin.append(f"{label}['{key}'] = {reason!r}")
    assert not thin, (
        "사유가 없는 줄이 있다:\n  " + "\n  ".join(thin) +
        "\n\n사유를 못 적겠으면 그것은 빈자리다 — 표에서 빼라.")


def test_not_api_does_not_claim_what_we_implement():
    """**모순을 잡는다.**

    우리가 그 이름을 이미 만들어 두었다면 그것은 API 다. `NOT_API` 가 동시에
    "API 가 아니다" 라고 적고 있으면 둘 중 하나가 거짓이고, 어느 쪽이든 표를 믿을 수
    없게 된다.
    """
    clashes = []
    for space, _theirs, ours in _spaces():
        for name in _public(ours):
            full = f"{space}.{name}"
            reason = _look(NOT_API, name, full)
            if reason:
                clashes.append(f"{full} — '{reason}'")
    assert not clashes, (
        "우리가 만든 이름을 `NOT_API` 가 API 가 아니라고 적고 있다:\n  " +
        "\n  ".join(clashes) +
        "\n\n만들었으면 API 다. 표에서 빼거나, 만든 것을 지워라.")


def test_skipped_does_not_claim_what_we_actually_do():
    """**`안 하기로 함` 에 적힌 이름이 실제로 돌면 그 줄은 거짓이다.**

    `NOT_API` 는 위에서 모순을 잡는데 `SKIPPED` 는 안 잡고 있었다. 잡을 필요가
    없어 보였던 까닭이 있다 — 분류기는 **우리에게 없는 이름**에만 사유를 묻는다.
    그래서 우리가 만든 이름의 사유는 아무도 안 읽고, **수는 맞은 채로 문서만
    거짓이 된다.** 낡은 수는 다시 재면 드러나는데 낡은 사유는 안 드러난다.

    실제로 열둘이 그랬다. `complex`·`real`·`imag`·`conj` 계열의 사유가 여태
    "복소수 dtype 이 없다" 였는데 `complex64` 를 넣은 지 오래다. 그 줄을 읽고
    "복소수는 못 하는구나" 로 이해한 사람은 **틀린 것을 배운다.**

    ## 이름이 있다고 다 도는 것은 아니다

    `q_scale`·`int_repr` 처럼 **거절하려고 이름만 둔** 자리가 있다. 그쪽은 사유가
    여전히 참이다. `hasattr` 로 가르면 그것까지 걸리므로 **불러 본다** — 우리 거절은
    인자와 무관하게 `BorchError` 로 멈추므로 그것으로 가른다.

    못 불러 본 이름은 **판정하지 않는다.** 모르는 것을 아는 척하면 이 표가 다시
    거짓말을 시작한다.
    """
    import numpy as np

    import borch
    from borch import BorchError

    probe = borch.tensor(np.array([1.0, 2.0], dtype=np.float32))

    def verdict(fn):
        """`True`=돈다 · `False`=거절한다 · `None`=못 불러 봤다."""
        seen_type_error = False
        for args in ((), (probe,), (probe, probe)):
            try:
                fn(*args)
            except BorchError:
                return False
            except TypeError:
                seen_type_error = True
            except Exception:                                   # noqa: BLE001
                continue
            else:
                return True
        return None if seen_type_error else None

    alive = []
    for space, _theirs, ours in _spaces():
        for name in _public(ours):
            full = f"{space}.{name}"
            reason = _look(SKIPPED, name, full)
            if not reason:
                continue
            got = getattr(ours, name, None)
            if not callable(got):
                continue
            if verdict(got) is True:
                alive.append(f"{full} — '{reason}'")
    assert not alive, (
        "`SKIPPED` 가 **우리가 실제로 하는 것**을 안 한다고 적고 있다:\n  " +
        "\n  ".join(sorted(alive)) +
        "\n\n그 줄을 지워라. 사유는 수와 달리 다시 재는 사람이 없어서, 한 번 낡으면\n"
        "읽는 사람에게 계속 거짓을 준다.")


def test_the_tool_still_runs():
    """도구 자체가 도는가. 표를 고치다 문법이 깨지면 여기서 멈춘다."""
    import torch_gap

    assert torch_gap.main([]) == 0
