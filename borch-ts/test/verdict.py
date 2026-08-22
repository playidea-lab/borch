"""통과 여부를 **상태에서** 읽는다. 보고 문장에서 읽지 않는다.

여섯 러너가 같은 모양으로 판정하고 있었다:

    return 0 if "전부 통과" in result["text"] else 1

문장을 훑는 판정은 세 가지로 조용히 틀린다.

1. **문구가 바뀌면 답이 바뀐다.** 페이지 쪽이 "전부 통과" 를 다른 말로 적으면 러너는
   모든 실행을 실패로 읽고, 반대로 실패 줄에 그 낱말이 우연히 들어가면 통과로 읽는다.
2. **부분 통과가 통과로 읽힌다.** `readme.py` 가 그랬다. 판정 낱말이 `"그대로 돌고"`
   였는데 그 말은 두 예시의 성공 문장 **양쪽**에 들어 있어서, 첫 예시가 실패하고
   LBFGS 만 통과해도 러너는 0 을 냈다. 손실이 안 내려가는 예시를 문서에 둔 채로.
3. **같은 글을 두 곳에서 다르게 읽는다.** `cost.html` 은 `!text.includes("갈렸다")`
   로, `cost.py` 는 `"전부 통과" in text` 로 판정했다 — 한 보고에 두 판정이 있고,
   갈리면 어느 쪽이 맞는지 말해 주는 것이 없다.

그래서 페이지가 `checks`(`{name, ok, note}`)를 그대로 넘기고, 여기서 센다. 문장은
사람이 읽는 그림자로 남는다.

**`checks` 가 없으면 통과가 아니라 오류다.** 없는 것을 "실패 0 건" 으로 읽으면 이
파일이 없애려는 바로 그 침묵이 다른 이름으로 돌아온다.
"""

import sys


def failures(result, what):
    """실패한 검사 목록. `checks` 를 못 찾으면 그 자리에서 멈춘다.

    Args:
        result: 페이지가 넘긴 객체. `checks` 를 들고 있어야 한다.
        what: 무엇을 재는 러너인지 — 오류 문장에 쓴다.

    Returns:
        실패한 검사들의 리스트. 전부 통과면 빈 리스트.

    Raises:
        SystemExit: `checks` 가 없거나 비었을 때. 판정할 상태가 없다는 뜻이고,
            그것은 통과가 아니다.
    """
    checks = result.get("checks")
    if not isinstance(checks, list) or not checks:
        raise SystemExit(
            f"{what}: 페이지가 `checks` 를 안 넘겼다 — 판정할 상태가 없다.\n"
            f"  받은 열쇠: {sorted(result)}\n"
            "  `report()` 가 {text, checks} 를 돌려주고 html 이 그것을 실어야 한다.")
    return [c for c in checks if not c.get("ok")]


def verdict(result, what, stream=sys.stderr):
    """러너의 종료 코드. 실패가 있으면 **무엇이** 실패했는지 이름을 댄다."""
    bad = failures(result, what)
    if not bad:
        return 0
    print(f"\n**{what} — {len(bad)}건 실패**", file=stream)
    for c in bad:
        note = f" — {c['note']}" if c.get("note") else ""
        print(f"  ✗ {c.get('name', '(이름 없음)')}{note}", file=stream)
    return 1
