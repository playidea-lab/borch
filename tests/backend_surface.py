"""자매가 TF.js 에 **실제로 기대는 표면**을 센다.

    uv run python tests/backend_surface.py

자매를 borch.ts 위로 옮길 수 있는가 — 그 답은 "몇 줄이냐" 가 아니라 **밑바닥에
무엇을 요구하느냐**다. 5,296 줄 중 대부분은 파이썬 쪽 살림이고, 갈아끼울 때 실제로
문제가 되는 것은 `_tf.` 로 부르는 이름들이다. 그 목록이 곧 borch.ts 가 채워야 할
계약이고, 목록을 세어 보지 않고 "된다/안 된다" 를 말하는 것은 추측이다.
"""

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CALL = re.compile(r"_tf\.([A-Za-z][A-Za-z0-9_.]*)")


def main(argv):
    seen = collections.Counter()
    for path in sorted((ROOT / "borch_webgpu").glob("*.py")):
        for name in CALL.findall(path.read_text(encoding="utf-8")):
            seen[name] += 1

    print(f"자매가 부르는 TF.js 이름 — 서로 다른 것 {len(seen)}개, "
          f"호출 {sum(seen.values())}곳")
    for name, n in seen.most_common():
        print(f"  {n:4d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
