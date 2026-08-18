"""**모듈 자리의 이름이 torch 와 같은 종류인가.**

`borch/__init__.py` 는 Tensor 의 메서드를 모듈 함수로도 낸다 — torch 가 `x.sum()` 과
`torch.sum(x)` 를 둘 다 주기 때문이고, 목록을 손으로 적으면 다음에 메서드를 하나 늘릴
때 빠지므로 고리로 만든다.

그 고리가 **torch 에서 함수가 아닌 이름까지 가져갔다.** 우리 쪽에 함수가 앉으면 그
이름을 원래 용도로 쓰는 코드가 **한 칸 밀린 자리에서** 멈춘다:

    zeros(2, dtype=torch.float)   →  'function' object has no attribute 'np'

형이 있어야 할 자리에 함수가 있다는 말인데, 그 문구는 오타와 구별이 안 된다. 형 여덟
(`float`·`double`·`int`·`bool`·`half`·`short`·`bfloat16`·`chalf`)이 그렇게 걸렸고,
**세 번에 걸쳐 손으로 뺐다.** 손으로 빼는 동안은 규칙이 없으므로 네 번째가 온다.

## 이 검사가 묻는 두 가지

- **짧지 않은가** — 고리가 만든 이름 중 torch 에서 함수가 아닌 것이 있으면 빨개진다.
  새 메서드가 늘어나 그런 이름을 덮으면 그날 걸린다.
- **낡지 않은가** — `_NOT_OURS` 에 적힌 이름이 정말 torch 에서 함수가 아닌지 본다.
  torch 가 그 이름을 함수로 바꾸면 우리가 빼 둘 까닭이 사라진다.

코어는 torch 에 기대지 않으므로 표는 소스에 적고 **대조는 여기서** 한다. 이 저장소의
`KNOWN_ABSENT`·`NOT_PORTED` 와 같은 꼴이다 — 목록이 있고, 그 목록이 드리프트하면 운다.
"""

import inspect
import types

import torch

import borch

LOOP_MADE = "_as_function.<locals>.call"


def _torch_kind(name):
    """torch 에서 그 이름이 무엇인가. 함수면 None(정상)."""
    value = getattr(torch, name)
    if isinstance(value, torch.dtype):
        return "dtype"
    if isinstance(value, types.ModuleType):
        return "이름 공간"
    if inspect.isclass(value):
        return "클래스"
    if not callable(value):
        return type(value).__name__
    return None


def test_the_loop_does_not_claim_a_name_torch_gives_as_something_else():
    """고리가 만든 이름은 torch 에서도 **함수**여야 한다."""
    stolen = []
    for name in sorted(dir(borch)):
        if name.startswith("_") or not hasattr(torch, name):
            continue
        if getattr(getattr(borch, name), "__qualname__", "") != LOOP_MADE:
            continue
        kind = _torch_kind(name)
        if kind is not None:
            stolen.append(f"borch.{name} — torch 에서는 {kind}")
    assert not stolen, (
        "메서드에서 만든 함수가 **torch 의 다른 종류**를 가리고 있다:\n  "
        + "\n  ".join(stolen) + "\n\n"
        "`borch/__init__.py` 의 `_NOT_OURS` 에 까닭과 함께 적거나, 그 이름이 형이면\n"
        "고리보다 **위에서** 진짜 형을 놓아라. 함수가 앉으면 오류가 한 칸 밀려서 난다."
    )


def test_the_not_ours_table_does_not_rot():
    """빼 둔 이름이 정말 torch 에서 함수가 아닌지 본다."""
    stale = []
    for name, why in sorted(borch._NOT_OURS.items()):
        if not hasattr(torch, name):
            stale.append(f"{name} — torch 에 그 이름이 없다 ({why})")
        elif _torch_kind(name) is None:
            stale.append(f"{name} — torch 에서 이제 함수다 ({why})")
    assert not stale, (
        "`_NOT_OURS` 가 낡았다:\n  " + "\n  ".join(stale) + "\n\n"
        "그 줄을 지워라 — 까닭이 사라진 목록은 다음 사람에게 근거처럼 읽힌다."
    )
