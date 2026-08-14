"""아직 안 물은 케이스를 뽑는다.

    uv run --with playwright python borch-ts/test/missing.py

러너가 세는 수(안 물은 케이스)는 **개수**뿐이라, 그것이 무엇인지 알려면 이름이
필요하다. 남은 것을 눈으로 훑어 "없더라" 하는 대신 목록으로 받는다.
"""

import collections
import json
import pathlib
import sys

import run as runner

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv):
    doc = json.loads((ROOT / "tests" / "golden.json").read_text(encoding="utf-8"))
    report = runner.run()
    if "error" in report:
        print(f"돌지 못했다: {report['error']}", file=sys.stderr)
        return 1
    # 러너는 등록된 이름만 안다. 골든 전체에서 그것을 빼면 남은 것이 나온다.
    asked = set(report.get("asked", []))
    missing = [n for n in doc["cases"] if n not in asked]

    by_group = collections.Counter(
        n.split("::")[0] if "::" in n else "(접두사없음)" for n in missing)
    print(f"안 물은 것 {len(missing)}건")
    for group, count in by_group.most_common():
        print(f"  {count:4d}  {group}")
    if "--names" in argv:
        for n in missing:
            print(f"    {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
