"""이름 바꾸기를 손이 아니라 기계가 한다.

    uv run python tests/rename.py --check     # 남은 자리를 센다
    uv run python tests/rename.py --apply     # 바꾼다

**손으로 41 개 파일을 훑으면 조용히 하나가 남는다.** 그리고 남은 하나는 임포트
오류로 즉시 터지는 것이 아니라, 브라우저 러너의 `?lib=` 질의처럼 문자열로만 쓰이는
자리에서 런타임에야 드러난다. 이 저장소가 반복해서 잡아온 결함의 모양 그대로다.

바꾸는 규칙은 셋이고 **순서가 중요하다** — 긴 이름부터 바꿔야 한다. `browsertorch`
를 먼저 바꾸면 `browsertorch_webgpu` 가 `borch_webgpu` 가 아니라 `borch_webgpu` 로
가긴 하는데, `browsertorch_vision` 이 `borch_vision` 이 되는 것과 겹쳐서 무엇이
무엇으로 갔는지 확인할 수 없게 된다. 길이 내림차순이 그것을 막는다.

`--check` 를 남겨 두는 이유는 이 파일이 일회용이 아니기 때문이다. 다음에 이름이
또 바뀔 때 같은 일을 다시 손으로 하지 않는다.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 긴 것부터. 짧은 것을 먼저 바꾸면 긴 이름의 앞부분이 먼저 잡힌다.
RULES = [
    ("browsertorch_webgpu", "borch_webgpu"),
    ("browsertorch_vision", "borch_vision"),
    ("browsertorch", "borch"),
]

# `.claude`·`.mcp.json` 은 이 기계의 설정이지 프로젝트가 아니다 — 저장소 경로가
# 적혀 있어서 걸리는데, 바꾸면 남의 도구 설정을 건드리는 것이 된다.
SKIP_DIRS = {".git", "node_modules", "vendor", "dist", "__pycache__", ".venv", ".claude"}
# 잠금 파일은 손대지 않는다 — 도구가 다시 만든다. 여기서 고치면 해시가 안 맞는다.
# 이 파일 자신도 뺀다. 규칙표가 `("borch_webgpu", "borch_webgpu")` 가 되면
# 다음 실행이 아무것도 못 찾고 그 사실을 알려주지도 않는다.
SKIP_FILES = {"uv.lock", "package-lock.json", ".mcp.json", pathlib.Path(__file__).name}
SUFFIXES = {".py", ".ts", ".js", ".html", ".json", ".md", ".yml", ".toml", ".cfg", ".txt"}


def targets():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in SUFFIXES:
            continue
        if p.name in SKIP_FILES or SKIP_DIRS & set(p.relative_to(ROOT).parts):
            continue
        yield p


def main(argv):
    apply = "--apply" in argv
    touched, remaining = [], 0
    for path in targets():
        text = path.read_text(encoding="utf-8")
        new = text
        for old, fresh in RULES:
            new = new.replace(old, fresh)
        if new == text:
            continue
        hits = sum(text.count(old) for old, _ in RULES)
        remaining += hits
        touched.append((path.relative_to(ROOT), hits))
        if apply:
            path.write_text(new, encoding="utf-8")

    verb = "바꿨다" if apply else "바꿀 것이 있다"
    print(f"{verb} — 파일 {len(touched)}개, 자리 {remaining}곳")
    for rel, hits in touched:
        print(f"  {hits:4d}  {rel}")
    # 확인 모드에서 남은 것이 있으면 종료 코드로 알린다 — CI 가 이것을 볼 수 있다.
    return 1 if (not apply and touched) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
