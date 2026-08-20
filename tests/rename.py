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
#
# 지금 표에 든 것은 **결속이 자매의 이름을 물려받는** 자리다. TF.js 판 자매를 지우고
# 그 이름을 borch.ts 위의 결속에게 넘겼다 — 사용자에게 그 이름이 뜻하는 것(브라우저,
# GPU)은 그대로이고 밑바닥만 바뀌었기 때문이다.
#
# **밑줄과 붙임표와 점은 다른 것이다.** `borch_ts`(파이썬 패키지)만 바뀌고
# `borch-ts`(TypeScript 디렉토리)와 `borch.ts`(글에서 부르는 이름)는 그대로 남는다.
# 이 셋이 눈으로는 비슷해 보이는 것이 애초에 이 개명을 부른 이유다.
#
# **그래서 은퇴한 토큰을 새 이름에 쓰면 안 된다.** 파이썬은 점을 못 쓰므로 사람이
# `borch.ts` 를 식별자로 적을 때 `borch_ts` 라고 쓰게 되는데, 그것이 정확히 은퇴한
# 패키지 이름이다. 실제로 `_touches_borch_ts`(borch.ts 를 부르는가)가 이 도구를 지나
# `_touches_borch_webgpu`(결속을 부르는가)가 됐다 — **코드는 일관되게 바뀌어 안
# 터지고, 틀린 것은 뜻뿐이라 아무 검사도 안 운다.** 새 이름에는 `ts` 나 `the_ts_side`
# 처럼 표에 없는 낱말을 쓴다.
RULES = [
    ("borch_ts", "borch_webgpu"),
    # **`torchvision` 에는 밑줄이 없다.** 이 프로젝트의 요점이 torch 의 구조를 그대로
    # 두는 것인데, `borch_vision` 은 대응하는 자리에 없는 밑줄을 하나 넣고 있었다.
    # `import borchvision as torchvision` 이 되어야 그 요점이 지켜진다.
    #
    # 배포 이름(`[project] name`)은 안 건드린다 — 그것은 별개의 문제이고, 지금
    # PyPI 의 `borch` 는 남의 것이다(Desupervised, 확률 프로그래밍). 모듈 이름과
    # 배포 이름을 같이 움직이면 무엇이 무엇 때문에 바뀌었는지 안 보인다.
    ("borch_vision", "borchvision"),
    # **철자가 다르면 규칙도 따로다.** `browsertorch` → `borch` 를 돌렸을 때 이
    # 이름은 안 걸렸다 — 규칙이 소문자였고 클래스는 `BrowserTorchError` 였다.
    # 소문자에는 낱말 경계가 없으므로(`browsertorch`) 도구가 `BrowserTorch` 를
    # 유추할 길이 없다. 그래서 **자동으로 만들지 않고 여기 적는다**, 그리고 못
    # 만드는 대신 `RETIRED` 가 남은 자리를 대문자 무시하고 세어 준다.
    ("BrowserTorch", "Borch"),
]

# **다시는 어느 철자로도 나오면 안 되는 이름들.** `RULES` 가 못 잡은 갈래를 여기서
# 잡는다 — 대소문자를 무시하고 세되 **고치지는 않는다.** 무엇으로 바꿀지는 철자마다
# 다르고(`BrowserTorchError` 는 `BorchError` 이지 `Borchtorcherror` 가 아니다),
# 그 판단을 기계가 하면 조용히 이상한 이름이 생긴다.
#
# 이 자리가 있는 까닭: 개명을 돌리고 나서 `BrowserTorchError` 가 37 곳에 남아
# 있었는데 도구는 "바꿀 것이 없다" 고 답했다. **도구 자신의 규칙 밖에서** 일어난
# 일이라 도구가 못 봤고, 그것이 이 도구가 막으려던 바로 그 일이다.
RETIRED = ["browsertorch"]

# **옛 이름을 일부러 적는 자리.** 역사를 이야기하는 문장은 옛 이름을 그대로 불러야
# 한다 — `test_docs.py` 가 "지난 수를 현재 수로 고치는 것은 역사를 위조하는 것" 이라고
# 적은 것과 같은 자리다. 그것까지 잡으면 검사가 늑대를 부르고, 늑대를 부르는 검사는
# 꺼진다.
#
# 파일마다 **까닭을 적는다.** 적을 까닭이 없으면 그것은 역사가 아니라 잔재다.
HISTORY = {
    "tests/test_site.py": "개명 때 사이트 링크가 왜 안 깨진 채 낡았는지를 적은 문장",
}

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


def survivors():
    """은퇴한 이름이 **어느 철자로든** 남아 있는 자리. 세기만 하고 안 고친다.

    `RULES` 를 다 돌린 뒤에도 남는 것이 여기 걸린다. 대소문자를 무시하는 이유는
    놓치는 갈래가 늘 대소문자였기 때문이다 — 소문자 규칙 하나로 개명을 돌리고
    `BrowserTorchError` 37 곳을 남긴 것이 그 예다.
    """
    found = []
    for path in targets():
        if path.relative_to(ROOT).as_posix() in HISTORY:
            continue
        text = path.read_text(encoding="utf-8")
        for old, fresh in RULES:
            text = text.replace(old, fresh)
        low = text.lower()
        spellings = set()
        for gone in RETIRED:
            at = low.find(gone)
            while at != -1:
                spellings.add(text[at:at + len(gone)])
                at = low.find(gone, at + 1)
        if spellings:
            found.append((path.relative_to(ROOT), spellings))
    return found


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

    left = survivors()
    if left:
        print("\n✘ 은퇴한 이름이 남아 있다 — 규칙이 못 잡은 철자다:")
        for rel, spellings in left:
            print(f"  {rel}: {' · '.join(sorted(spellings))}")
        print("  바꿀 이름을 정해 `RULES` 에 철자 그대로 적어라 — 무엇으로 바꿀지는\n"
              "  철자마다 다르므로 도구가 정하지 않는다.")

    # 확인 모드에서 남은 것이 있으면 종료 코드로 알린다 — CI 가 이것을 볼 수 있다.
    # **은퇴한 이름은 `--apply` 뒤에도 실패다** — 바꾸지 못한 것이 남았다는 뜻이라
    # 성공으로 끝내면 그 순간이 정확히 지난번 놓친 자리가 된다.
    return 1 if left or (not apply and touched) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
