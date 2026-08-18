"""**같은 이름의 `def` 가 모듈 자리에 둘 있으면 하나는 안 불린다.**

파이썬은 나중 것을 택하고 **아무 말도 안 한다.** 오류도 경고도 없다. 그래서 앞의
정의를 고치면 아무 일도 안 일어나고, 뒤의 정의를 안 보고 고친 사람은 자기 수정이
왜 안 먹는지를 한참 쫓는다.

## 잡은 것

`borch_webgpu/_ops.py` 에 `_dtype_name` 을 새로 쓰다가 물렸다. 같은 파일 아래쪽에
같은 이름이 이미 있었고(승격표가 쓰는 것), 파이썬이 그쪽을 택했다. 새로 쓴 것은
"이름만 있고 칸은 없는 형" 을 만나면 멈추게 되어 있었는데 **한 줄도 안 불렸고**,
`dtype=torch.int` 가 조용히 통과했다. 브라우저 대조가 "뜻밖의 성공" 으로 잡아 줄
때까지 몰랐다.

훑어보니 코어에 셋이 더 있었다. 둘(`_pair`·`matmul`)은 몸이 같아 해가 없었지만,
`vander` 는 **서로 다른 두 함수**였다 — 하나는 차수가 커지고 하나는 줄어든다.
그리고 그 갈림이 "클래스 몸이 먼저 정의된 쪽을 잡는다" 는 **정의 차례**에 기대고
있었다. 값은 맞았지만(torch 도 `vander` 와 `linalg.vander` 를 갈라 놓았다) 읽는
사람에게는 뒤가 앞을 덮는 것으로 보인다. 이름을 갈라 그 기댐을 없앴다.

## 왜 ruff 를 안 쓰나

`F811` 이 이것을 잡기는 한다. 그런데 지역 인자가 수입한 이름을 가리는 자리
(`def call(t, ...)`)와 재수입까지 같이 잡아서, 이 저장소에서 서른 건 넘게 나온다.
그만큼 걸리면 목록이 생기고 목록은 안 읽힌다. **위험한 것은 모듈 자리의 `def` 둘**
뿐이라 그것만 본다.
"""

import ast
import collections
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _sources():
    for pkg in ("borch", "borch_webgpu", "tests"):
        yield from sorted((ROOT / pkg).rglob("*.py"))


def test_no_module_level_name_is_defined_twice():
    """모듈 자리의 `def` 는 파일마다 이름이 한 번씩만 나와야 한다."""
    twice = []
    for path in _sources():
        seen = collections.defaultdict(list)
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen[node.name].append(node.lineno)
        for name, lines in sorted(seen.items()):
            if len(lines) > 1:
                where = "·".join(str(n) for n in lines)
                twice.append(f"{path.relative_to(ROOT)}:{where} — def {name}")
    assert not twice, (
        "모듈 자리에 같은 이름의 `def` 가 둘 이상이다:\n  " + "\n  ".join(twice) + "\n\n"
        "파이썬은 나중 것을 택하고 아무 말도 안 한다 — 앞의 것을 고치면 아무 일도\n"
        "안 일어난다. 이름을 가르거나 하나를 지워라. **몸이 같아 보여도** 지우기 전에\n"
        "클래스 몸이 앞의 것을 잡고 있지 않은지 보라(`vander` 가 그랬다)."
    )
