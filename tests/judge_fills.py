"""판정 보조 — `UNJUDGED` 62 개마다 borch.ts 쪽 **가까운 이름**을 뽑아 준다.

검사가 아니다. `test_binding_fills_in.py` 의 목록을 사람이 갈라내는 동안, 이름마다
"저쪽에 비슷한 것이 있는가" 를 한 화면에 놓는 일회용 도구다. 판정이 끝나면 지운다.

    uv run --with torch python tests/judge_fills.py
"""

import difflib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_binding_fills_in import UNJUDGED, _flat, _ts_surface  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _ts_spellings():
    """평평한 이름 → 실제 철자들. 대조는 평평하게 하지만 사람은 철자를 읽는다."""
    import re

    out = {}
    src = ROOT / "borch-ts" / "src"
    method = re.compile(
        r"^\s+(?:public\s+|private\s+|protected\s+|static\s+|async\s+|get\s+|set\s+"
        r"|readonly\s+|\*)*([A-Za-z_]\w*)\s*[(<:=]")
    top = re.compile(
        r"^export\s+(?:declare\s+)?(?:async\s+)?"
        r"(?:function|class|const|abstract\s+class)\s+([A-Za-z_]\w*)")
    for path in sorted(src.rglob("*.ts")):
        for line in path.read_text(encoding="utf-8").splitlines():
            for pattern in (method, top):
                hit = pattern.match(line)
                if hit:
                    out.setdefault(_flat(hit.group(1)), set()).add(hit.group(1))
    return out


def _built_from():
    """이름 → 결속이 그 몸통에서 부르는 **borch.ts 메서드 이름들**.

    `handle(x).foo` 와 `guarded(handle(x).foo, …)` 둘 다 잡는다. 이것이 곧
    "무엇으로 지었는가" 이고, 거기서 이 이름이 저쪽에 **다른 철자로 있는지**
    (개명·조합) 아니면 **없는지**(결손)가 갈린다.
    """
    import ast

    out = {}
    for stem in ("_ops", "_base", "_nn"):
        path = ROOT / "borch_webgpu" / f"{stem}.py"
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            used, numpy = [], False
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Attribute):
                    continue
                base = sub.value
                if (isinstance(base, ast.Call)
                        and getattr(base.func, "id", "") == "handle"):
                    used.append(sub.attr)
                if sub.attr == "numpy":
                    numpy = True
            out.setdefault(node.name, (used, numpy))
    return out


def main():
    spellings = _ts_spellings()
    flat = list(_ts_surface())
    built = _built_from()
    for name in sorted(UNJUDGED):
        used, numpy = built.get(name, ([], False))
        if used:
            how = "짓는다: " + " ".join(dict.fromkeys(used))
        elif numpy:
            how = "**numpy**"
        else:
            near = difflib.get_close_matches(_flat(name), flat, n=3, cutoff=0.7)
            shown = []
            for key in near:
                shown.extend(sorted(spellings.get(key, {key})))
            how = "가까운 이름: " + (" · ".join(shown[:4]) or "(없음)")
        print(f"{name:26s} {how}")


if __name__ == "__main__":
    main()
