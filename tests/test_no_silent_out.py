"""**`out=` 을 조용히 삼키는 자리가 없어야 한다.**

torch 는 198 개 이름에서 `out=` 을 받아 미리 만든 텐서에 결과를 써 넣는다. 우리는
그것을 안 한다 — 우리가 하면 계산해서 사본을 넣게 되고, `out=` 의 존재 이유인
**할당을 안 하는 것**은 일어나지 않는다. 절약을 흉내 내면 없는 것을 배우게 된다.

안 하는 것 자체는 괜찮다. **문제는 어떻게 안 하느냐다.**

    torch.randint(0, 5, (4,), out=buf)    # torch: buf 에 쓴다
    borch.randint(0, 5, (4,), out=buf)    # 우리: buf 는 0 인 채로 남았다

여섯이 이랬다 — `range`·`randperm`·`randint`·`rand_like`·`randn_like`·
`searchsorted`. 오류가 없으니 다음 줄로 가고, 틀린 값은 한참 뒤에 드러난다.
나머지 190 여 자리는 `**kw` 가 없어서 파이썬이 `TypeError` 로 멈춰 준다 —
**우연히 안전했던 것**이고, `**kw` 를 하나 더 다는 날 그 우연이 끝난다.

## 이 검사가 묻는 것

torch 가 `out=` 을 받는 이름 중 우리에게도 있는 것마다: 그 함수가 `out` 을 이름으로
받지 않으면서 `**kw` 를 받는다면, 본문에 `_no_out` 이 있어야 한다. 없으면 삼킨다.

`out=` 을 실제로 지원하기로 결정하면 이 검사는 지운다. 그때까지는 **안 하는 방식이
일관되다**는 것을 지킨다.

## 목록을 docstring 에서 뽑는다 — 조금 넓다

torch 의 C 함수는 서명을 들여다볼 수 없어서 docstring 의 `out=None` 으로 고른다.
`rand_like`·`randn_like` 는 거기 적혀 있는데 **실제 오버로드는 안 받는다** — 케이스를
쓰다가 torch 쪽 `TypeError` 로 알았다. 그래서 이 검사는 몇 자리를 더 잡는데, 그쪽으로
넓은 것은 안전하다: 문을 하나 더 다는 것뿐이고 torch 도 거절하는 인자다. 반대로
좁으면 삼키는 자리가 남는다.
"""

import ast
import inspect
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _torch_names_taking_out():
    names = []
    for name in sorted(dir(torch)):
        if name.startswith("_"):
            continue
        fn = getattr(torch, name)
        if not callable(fn) or inspect.isclass(fn):
            continue
        if "out=None" in (fn.__doc__ or ""):
            names.append(name)
    return set(names)


def _functions_with_varkw(path):
    """파일 안 모듈 자리 함수 → (`**kw` 이름, `_no_out` 을 부르는가)."""
    found = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.args.kwarg is None or any(a.arg == "out" for a in node.args.args):
            continue
        guarded = any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_no_out"
                      for n in ast.walk(node))
        found[node.name] = guarded
    return found


def test_no_function_can_swallow_out():
    """`**kw` 로 `out=` 을 받는 자리에는 반드시 문이 있어야 한다."""
    taking_out = _torch_names_taking_out()
    naked = []
    for pkg in ("borch", "borch_webgpu"):
        for path in sorted((ROOT / pkg).rglob("*.py")):
            for name, guarded in _functions_with_varkw(path).items():
                # `range_top` 은 모듈에서 `range` 로 나간다 — torch 의 그 이름이다
                torch_name = "range" if name == "range_top" else name
                if torch_name in taking_out and not guarded:
                    naked.append(f"{path.relative_to(ROOT)} — {name}(**kw)")
    assert not naked, (
        "`out=` 을 조용히 삼킬 수 있는 자리:\n  " + "\n  ".join(sorted(naked)) + "\n\n"
        "본문 첫 줄에 `_no_out(kw)` 를 넣어라. 삼키면 오류 없이 목적지가 안 바뀌고,\n"
        "그 값은 한참 뒤에 엉뚱한 자리에서 드러난다."
    )
