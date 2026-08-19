"""**사용자가 보는 오류 문구는 영어다.** 주석은 한국어로 남는다.

## 왜 이 검사가 있는가

영문 독자가 이 라이브러리에서 **처음 무언가 깨질 때** 만나는 것이 오류 문구다.
문서와 사이트가 전부 영어가 된 뒤에도 문구의 81% 가 한국어였고, 그것이 남은 가장
큰 한국어 표면이었다(실측: 셋 합쳐 던지는 자리 303 곳).

한 번 고치는 것으로는 안 된다. 새 커널·새 층이 들어올 때마다 한국어 문구가 하나씩
같이 들어오고, 그때 아무도 안 본다. 그래서 규칙을 검사로 둔다.

## 무엇을 안 막는가

- 주석과 docstring — 한국어가 규칙이다
- `repr`·`describe` 처럼 값을 찍는 문자열 — 오류가 아니다

세 라이브러리를 같이 본다. 플레이그라운드가 한 페이지에서 JS 와 파이썬을 나란히
돌리므로, 한쪽만 영어면 **같은 오류가 두 언어로** 보인다.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")

# **던지는 자리는 `raise` 만이 아니다.** 헬퍼에 문구를 넘기고 그 안에서 던지는
# 자리가 있고(`_unsupported("텐서 지수")`), 처음에 그것을 안 봐서 "다 고쳤다" 가
# 거짓이었다 — 골든이 대신 잡아 줬다. 검사가 안 보는 자리는 규칙이 없는 자리다.
_PY_RAISE = r"(?:raise \w*(?:Error|Exception)|_unsupported|_absent_here|_absent_dtype)\("

SURFACES = (
    ("borch-ts/src", "*.ts", re.compile(r"throw new \w*Error\(")),
    ("borch", "*.py", re.compile(_PY_RAISE)),
    ("borch_webgpu", "*.py", re.compile(_PY_RAISE)),
)


def _sites(text, opener):
    """여는 자리부터 괄호가 균형을 이룰 때까지 — 여러 줄짜리 문구까지 한 자리로."""
    found = []
    for m in opener.finditer(text):
        start = m.end() - 1
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    found.append((text[:start].count("\n") + 1, text[start:i + 1]))
                    break
    return found


def test_error_messages_are_english():
    bad = []
    for folder, glob, opener in SURFACES:
        for path in sorted((ROOT / folder).glob(glob)):
            for line, site in _sites(path.read_text(), opener):
                if HANGUL.search(site):
                    first = HANGUL.search(site)
                    snippet = site[max(0, first.start() - 30):first.start() + 40]
                    bad.append(f"{path.relative_to(ROOT)}:{line}  …{snippet.strip()}…")
    assert not bad, (
        f"오류 문구 {len(bad)} 곳이 한국어다. 사용자가 처음 만나는 표면이므로 영어여야 한다"
        " — 주석은 한국어 그대로 둔다.\n  " + "\n  ".join(bad[:40])
        + (f"\n  … 그리고 {len(bad) - 40} 곳 더" if len(bad) > 40 else ""))
