"""**결속이 borch.ts 대신 채워 넣는 자리를 센다.**

이 저장소의 골든은 세 구현을 같은 기대값에 대조한다고 적혀 있다. 그런데 케이스가
`borch_webgpu` 를 지날 때, 그쪽이 borch.ts 위에 무언가를 **스스로 조립하면** 그
케이스는 borch.ts 를 안 묻는다. 표는 초록이고, 없는 것은 TypeScript 로 쓰는 쪽뿐이다.

한 세션에서 이 모양이 일곱 번 나왔다:

- `trapezoid`·`cumulative_trapezoid` — 조각을 자르고 더하는 몇 줄이 파이썬에 있었다.
  주석에는 "여기 하나 더 만들면 조립이 두 벌이 된다" 고 적혀 있었는데, **한 벌이
  아니라 파이썬 쪽에만 있었다.**
- `bernoulli`·`normal`·`poisson`·`binomial` — numpy 로 만들고 있었다.
- `ldl_factor_ex` — `_Fields` 를 손으로 세워 `info` 자리에 스칼라를 끼웠다.

그리고 `half`·`float`·`long` 같은 형 바꾸기 열넷이 같은 자리였다. 스물하나가 전부
"골든은 초록인데 borch.ts 에 이름이 없다" 였고, 전부 **사람이 `--show` 로 한 묶음씩
펴 보다가** 찾았다. 그것은 규율이고, 규율은 샌다.

## 무엇을 신호로 쓰는가

**torch 에 있는 이름을 결속이 구현하는데 borch.ts 에 그 이름이 없으면** 결속이
채워 넣고 있는 것이다. 위의 스물하나가 전부 여기 걸린다.

몸통을 읽어 "조립인가" 를 판정하지 않는다 — `ldl_factor_ex` 는 borch.ts 를 부르면서
동시에 조립했고, `trapezoid` 는 borch.ts 메서드를 여럿 불렀다. 부르느냐가 아니라
**그 이름이 저쪽에 있느냐**가 갈림이다.

## 왜 목록이 아니라 수인가

목록만 뽑는 도구는 사람이 돌려 봐야 하는 규율이라, 이 검사가 막으려는 그 문제를
그대로 되풀이한다. 그래서 아래 표에 **까닭과 함께 적힌 것만** 통과시키고, 새로
생긴 것은 이름을 대며 터진다. `NOT_PORTED`·`KNOWN_ABSENT` 와 같은 꼴이다.

표시:  파이썬 = 파이썬 표면의 것이라 TS 에 대응물이 있을 수 없다
       설계 = borch.ts 에 두지 **않기로** 한 것 (까닭이 있어야 한다)
"""

import ast
import json
import pathlib
import re

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent

# borch.ts 에 없는 것이 **맞는** 이름들. 하나하나 까닭이 있어야 한다.
#
# 여기 이름을 더하는 것은 "borch.ts 에 안 만든다" 는 결정이다. 밀린 일이면 여기가
# 아니라 borch.ts 에 넣어라 — 이 표는 대기열이 아니다.
# **파이썬 표면이라 저쪽에 있을 수 없는 것들.** 난수 줄기의 직렬화, numpy 왕래,
# 저장소·희소 들여다보기, 파이썬 형 이름 — 전부 TypeScript 에 대응물이 없다.
PYTHON_SIDE = """
as_tensor can_cast dense_dim from_numpy get_default_dtype get_device
get_rng_state initial_seed is_contiguous is_distributed is_grad_enabled
is_inference is_inference_mode_enabled is_storage numpy promote_types
set_rng_state share_memory_ sparse_dim to_dense tolist typename
asarray resize_as_ storage_offset type values
""".split()
# `is_floating_point`·`is_signed`·`is_nonzero` 가 여기 있었다. 앞의 둘은 "dtype 속성"
# 으로, 뒤의 하나는 "파이썬 표면" 으로 적혀 있었는데 **셋 다 torch 의 이름이고 저쪽에
# 없던 것**이다. `dtype::묻는것::` 아홉을 옮기려다 드러났다 — 케이스를 옮기는 일이
# 이름의 결손을 찾는 방법이기도 하다는 것이 이 세션에서 여러 번 반복됐다.

# **아직 판정 안 한 것들.** 이 검사가 처음 돌면서 한꺼번에 나온 자리이고, 하나하나
# 두 갈래 중 하나다: borch.ts 에 **다른 철자로 있다**(그러면 결손이 아니다), 또는
# **없다**(그러면 옮길 일이다).
#
# 수를 얼려 두는 것이 이 목록의 일이다. 새로 생긴 채움은 여기 없으므로 곧바로
# 터지고, 옛것은 **이름으로** 남아 있어 세다 만 것이 아니라는 게 보인다.
# 판정한 것은 이 목록에서 빼서 위로 올리거나 borch.ts 에 넣어라.
# **다른 철자로 저쪽에 있다** — 결손이 아니다. 옆에 borch.ts 쪽 이름을 적어 둔다.
# 적어 두지 않으면 다음 사람이 같은 확인을 다시 한다.
ALIASED = {
    "broadcast_to": "expand",
    "grid_sampler": "gridSample",
    "is_same_size": "shape 비교",
    "max_pool1d_with_indices": "maxPoolWithIndices",
    "moveaxis": "movedim",
    "numel": "size",
    "scatter": "scatterSet",
    "swapdims": "swapaxes",
    "take": "indexSelect(평평하게 편 뒤)",
    "take_along_dim": "gather",
    "trapz": "trapezoid",
    "vdot": "vecdot",
    "t": "transpose (2 차원 전치)",
}
# `is_tensor` 가 여기 한 줄 있었다. 선언 목록이 `index.ts` 를 안 훑어서 `isTensor` 가
# 공개 이름인데도 안 잡히던 것이었고 — 결손이 아니라 **목록의 사각지대**였다 —
# 생성기가 그 파일을 목록에 넣으면서 사라졌다.
#
# 그 사각지대는 **밖에서 보다가** 잡혔다. 생성기 쪽 검사는 "목록이 선언 파일과
# 같은가" 를 묻지 "선언 파일을 다 봤는가" 는 안 묻는다. 자기 입력을 자기가 검산하는
# 검사에는 늘 그만한 크기의 사각지대가 남고, 그것은 다른 각도에서만 보인다.

# `UNJUDGED` 가 여기 있었다 — 처음 돌렸을 때 62 개였고, 62 → 29 → 11 → 2 → 0 으로
# 갈렸다. 마흔셋은 borch.ts 에 넣었고 열넷은 다른 철자로 이미 있었으며 다섯은
# 파이썬 표면이었다. **비면 목록을 지운다** — 안 지우면 다음 사람이 아직 밀린 일이
# 있는 줄로 읽는다.
FILLED_ON_PURPOSE = set(PYTHON_SIDE) | set(ALIASED)

# 결속이 노출하지만 torch 이름이 아닌 것들 — 애초에 후보가 아니다.
_PRIVATE = re.compile(r"^_")


def _flat(name):
    """철자 차이를 지운다 — 대조하려는 것은 **이름의 존재**이지 철자가 아니다.

    torch 는 `searchsorted`·`logsumexp` 처럼 밑줄 없이 붙여 쓰고 borch.ts 는
    `searchSorted`·`logSumExp` 로 쓴다. 밑줄만 보고 camel 로 바꾸면 그 둘이 안
    만나서, **있는 이름이 없다고 나온다** — 첫 판이 그렇게 예순 건을 허수로 냈다.
    """
    return name.replace("_", "").lower()


def _ts_surface():
    """borch.ts 가 **선언하는** 이름 전부.

    처음에는 소스를 정규식으로 훑었다. 주석에는 "borch.ts 가 내는 이름 전부" 라고
    적어 놓고, 코드는 **들여쓴 뒤에 여는 괄호가 따라오는 아무 낱말**을 세고 있었다 —
    지역 변수 `const inner = …` 가 공개 이름 `inner` 로 잡혔다. 정규식이 1323 개를
    셌고 실제 선언은 845 개였다.

    그래서 **못 보는 자리가 생겼다**: `torch.inner` 는 borch.ts 에 없는데 이 검사가
    있다고 답했다. 결손을 찾는 검사가 결손을 가리는 꼴이고, 그 원인이 README 의
    일곱째 항목 그대로다 — 주석이 말하는 것과 코드가 묻는 것이 달랐다.

    이제 `site/assets/api.json` 을 만드는 생성기와 **같은 목록**을 읽는다. 그것은
    `tsc` 가 낸 선언 파일에서 나오므로 지역 변수가 못 샌다. 그 파일이 낡으면 이
    검사도 낡는데, 그 자리는 `tests/test_site.py` 가 이미 지킨다 — 검사 둘이
    한 파일에 걸리는 대신 각자 자기 몫만 본다.
    """
    index = ROOT / "site" / "assets" / "api-index.json"
    declared = json.loads(index.read_text(encoding="utf-8"))
    return {_flat(str(name).split(".")[-1]) for name in declared}


def _touches_the_ts_side(node):
    """이 함수가 borch.ts 손잡이를 **한 번이라도** 부르는가.

    **이름이 borch.ts 를 밑줄 꼴로 적지 않는다.** 파이썬은 점을 못 쓰므로 그렇게
    적고 싶어지는데, 그 철자가 곧 은퇴한 파이썬 패키지 이름이라 개명 도구가 그것을
    결속 이름으로 바꾼다 — 실제로 한 번 바뀌어서, 저쪽을 부르는지 묻는 함수가
    "결속을 부르는지" 로 읽히게 됐다. **코드는 일관되게 바뀌므로 안 터지고, 틀린
    것은 뜻뿐이라 아무 검사도 안 운다.** `rename.py` 주석에 같은 이야기가 있다.

    부르면 대개 개명이나 조합이다 — `scatter` 는 저쪽 `scatterSet` 이고 `take` 는
    `indexSelect` 다. 그 자리는 TypeScript 쪽에 **다른 철자로 있으므로** 결손이
    아니다. 한 번도 안 부르면 파이썬이 혼자 셈하고 있는 것이고, 그때가 저쪽에
    그 능력이 아예 없을 확률이 높다.

    **판정이 아니라 분류다.** `lerp` 는 `+`·`-`·`*` 로만 적혀 있어 손잡이가 안
    보이는데 저쪽에는 서명이 다른 `lerpFrom` 이 있고, `ldl_factor_ex` 는 손잡이를
    부르면서도 결손이었다. 그래서 이 갈래는 사람이 읽을 순서를 정할 뿐이다.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "handle":
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in ("numpy", "handle"):
            return True
    return False


def _binding_bodies():
    """이름 → borch.ts 손잡이를 부르는가."""
    out = {}
    for stem in ("_ops", "_base", "_nn", "_data", "_optim"):
        path = ROOT / "borch_webgpu" / f"{stem}.py"
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.setdefault(node.name, _touches_the_ts_side(node))
    return out


def _binding_names():
    """결속이 구현하는 이름 — 모듈 자리 함수와 `Tensor` 메서드."""
    names = set()
    for stem in ("_ops", "_base", "_nn", "_data", "_optim"):
        path = ROOT / "borch_webgpu" / f"{stem}.py"
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _PRIVATE.match(node.name):
                    names.add(node.name)
    return names


def _asked_by_golden():
    """케이스 표가 **철자 그대로 부르는** 이름들.

    골든이 안 묻는 이름은 이 검사의 관심 밖이다 — 그때 결속이 채워 넣는 것은 표를
    속이지 않는다. 그냥 어디에도 없는 것이고, 그건 다른 이야기다. **초록색을
    만들어 내는 자리**만 여기 남긴다.
    """
    text = (ROOT / "tests" / "cases.py").read_text(encoding="utf-8")
    return set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", text))


def _candidates():
    ts = _ts_surface()
    asked = _asked_by_golden()
    found = set()
    for name in _binding_names():
        if not hasattr(torch, name) and not hasattr(torch.Tensor, name):
            continue                      # torch 이름이 아니면 후보가 아니다
        if _flat(name) in ts:
            continue
        if name not in asked:
            continue
        found.add(name)
    return found


def test_binding_does_not_quietly_fill_in():
    """**결속이 채워 넣는 자리는 표에 적힌 것뿐이어야 한다.**

    새 이름이 여기서 터지면 두 갈래다. borch.ts 에 넣을 값이 있으면 넣어라 —
    골든이 그 이름을 묻기 시작한다. 넣지 않기로 했으면 `FILLED_ON_PURPOSE` 에
    **까닭과 함께** 적어라. 수만 올리는 것은 이 검사를 끄는 것과 같다.
    """
    bodies = _binding_bodies()
    surprise = sorted(_candidates() - set(FILLED_ON_PURPOSE))
    alone = [n for n in surprise if not bodies.get(n)]
    via = [n for n in surprise if bodies.get(n)]
    assert not surprise, (
        "결속이 borch.ts 대신 채우고 있다 — 골든은 결속을 지나므로 "
        f"이것을 못 본다:\n\n  파이썬이 혼자 셈한다 ({len(alone)}):\n    "
        + "\n    ".join(alone)
        + f"\n\n  손잡이를 부르며 조립한다 ({len(via)}):\n    " + "\n    ".join(via))


def test_the_table_has_no_stale_rows():
    """**다 옮긴 줄은 지워야 한다.** 안 지우면 다음 사람이 아직 없는 줄로 읽는다."""
    gone = sorted(set(FILLED_ON_PURPOSE) - _candidates())
    assert not gone, (
        "이 이름들은 이제 borch.ts 에 있다 — 표에서 지워라:\n  " + "\n  ".join(gone))
