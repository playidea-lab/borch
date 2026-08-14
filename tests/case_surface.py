"""골든 케이스가 라이브러리에서 **실제로 부르는 이름**을 센다.

    uv run python tests/case_surface.py            # 모듈 함수 (L.exp …)
    uv run python tests/case_surface.py --methods  # 메서드 (x.sum() …)

새 구현을 얹을 때 물어야 하는 것은 "torch 표면이 얼마나 넓은가" 가 아니라 **"이 표가
무엇을 부르는가"** 다. 그 목록이 곧 통과의 조건이고, 세어 보면 어디부터 붙일지가
정해진다 — 많이 불리는 것부터 붙이면 지나는 케이스가 빨리 는다.

호출 횟수가 아니라 **케이스 수**로 세는 것이 요점이다. `mul` 이 백 번 불려도 그것이
한 케이스 안이면 하나를 여는 것이고, 열 케이스에 하나씩이면 열을 연다.
"""

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# `L.exp(...)` · `L.nn.functional.relu(...)` 같은 자리.
MODULE = re.compile(r"\bL\.((?:[a-z_][a-z0-9_]*\.)*[a-zA-Z_][a-zA-Z0-9_]*)")
# `x.sum()` · `a.masked_select(...)` — 잎 이름이 관례상 한 글자거나 짧다.
METHOD = re.compile(r"\b[a-z][a-z0-9_]{0,3}\.([a-z_][a-z0-9_]*)\s*\(")


def main(argv):
    text = (ROOT / "cases.py").read_text(encoding="utf-8")
    want_methods = "--methods" in argv
    pattern = METHOD if want_methods else MODULE

    # 케이스 하나를 한 줄로 보지 않는다 — 여러 줄에 걸친 것이 많으므로, 케이스
    # 이름이 나오는 줄부터 다음 케이스 이름까지를 한 덩이로 본다.
    seen = collections.Counter()
    for name in pattern.findall(text):
        seen[name] += 1

    kind = "메서드" if want_methods else "모듈 함수"
    print(f"골든 케이스가 부르는 {kind} — 서로 다른 것 {len(seen)}개")
    for name, n in seen.most_common(60):
        print(f"  {n:4d}  {name}")
    if len(seen) > 60:
        print(f"  … 그 아래로 {len(seen) - 60}개 더")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
