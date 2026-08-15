"""실패한 골든 케이스를 **사유별로** 묶는다.

    uv run --with playwright python tests/browser/why_failing.py --lib borch_webgpu --headed

새 구현을 얹는 중에 필요한 것은 실패 목록이 아니라 **실패의 종류**다. 617 줄을 눈으로
읽으면 어디부터 손대야 하는지가 안 보이고, 종류로 묶으면 "이 하나를 고치면 몇 건이
열리는가" 가 보인다.

`run.py --lib …` 이 이미 대조를 하므로 그것을 그대로 쓰고 결과만 다시 센다.
"""

import argparse
import collections
import re
import sys

import run as runner

# `이름: 종류 — 설명` 또는 `이름: 기대 여야 하는데 실제`.
KIND = re.compile(r"^(?P<name>.+?): (?P<rest>.*)$", re.S)
ERRORS = re.compile(
    r"AttributeError — (?:'(?P<obj>[^']+)' object has no attribute '(?P<attr>[^']+)'"
    r"|borch\.ts (?:텐서)?에 `(?P<js>[^`]+)`)")


def bucket(why):
    """실패 하나를 한 낱말로 줄인다. 같은 원인은 같은 낱말이 되어야 한다."""
    m = KIND.match(why)
    rest = m.group("rest") if m else why
    hit = ERRORS.search(rest)
    if hit:
        if hit.group("js"):
            return f"borch.ts 에 없다: {hit.group('js')}"
        return f"파이썬 결속에 없다: {hit.group('obj')}.{hit.group('attr')}"
    for probe, label in (
        ("최대차", "값이 갈렸다"),
        ("모양", "모양이 갈렸다"),
        ("여야 하는데", "답이 갈렸다(문자열)"),
        ("JsException", "borch.ts 가 던졌다"),
        ("TypeError", "파이썬 쪽 형이 안 맞는다"),
    ):
        if probe in rest:
            return label
    return rest.split("—")[0].strip()[:40]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="borch_webgpu")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--samples", type=int, default=0,
                    help="사유마다 본보기를 몇 줄 보여줄지")
    args = ap.parse_args(argv)

    result, _ = runner.run(args.lib, args.headed)
    if result.get("error"):
        print("러너가 터졌다:\n" + result["error"], file=sys.stderr)
        return 1

    bad = result["bad"]
    total = result["total"]
    print(f"{args.lib} — {total - len(bad)}/{total} 통과, {len(bad)} 실패\n")

    groups = collections.defaultdict(list)
    for why in bad:
        groups[bucket(why)].append(why)

    print("실패 사유별:")
    for label, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(items):4d}  {label}")
        # **본보기를 몇 개 보여준다.** 종류만 세면 "RuntimeError 84 건" 에서 멈추고,
        # 그 84 건이 한 가지 원인인지 여덟 가지인지를 모른다.
        if args.samples:
            for one in items[:args.samples]:
                print(f"          {one[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
