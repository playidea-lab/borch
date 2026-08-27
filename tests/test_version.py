"""판 번호가 적힌 **세 곳**이 같은 수를 말하는지 본다.

브라우저에서 `package.json` 을 읽을 수 없어 판 번호가 두 곳에 적힌다. 두 곳에 적히는
것은 갈리기 마련이고, **갈린 판 번호는 조용하다** — 받는 쪽이 매니페스트의 범위를
대조할 때 틀린 수로 대조하고, 그러면 못 돌 것을 돌리거나 돌 것을 막는다.

그래서 갈릴 수 없게 여기서 붙잡는다.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_typescript_version_matches_package_json():
    declared = json.loads((ROOT / "package.json").read_text())["version"]
    source = (ROOT / "borch-ts" / "src" / "version.ts").read_text()
    found = re.search(r'VERSION\s*=\s*"([^"]+)"', source)
    assert found, "version.ts 에서 VERSION 을 못 찾았다"
    assert found.group(1) == declared, (
        f"borch-ts/src/version.ts 는 {found.group(1)!r}, package.json 은 {declared!r} —\n"
        "  판을 올릴 때 두 곳을 같이 고쳐라."
    )


def test_the_lockfile_version_matches_package_json():
    """잠금 파일의 루트 항목도 같은 판을 말하는지 본다.

    번호가 세 곳에 적혀 있다 — `package.json`, `version.ts`, 그리고 잠금 파일의
    루트다. 앞의 둘은 위 검사가 붙잡고 있었고, 잠금은 아무도 안 보고 있었다.

    **`npm ci` 는 이 불일치로 안 멈춘다**(실측). 그래서 CI 는 초록이고, 다음 사람이
    `npm install` 을 하는 날 잠금이 조용히 바뀌어 자기 변경과 무관한 diff 로 올라온다.
    그 사람은 자기가 무엇을 건드렸는지 모른 채 그것을 커밋하거나, 되돌리느라 시간을
    쓴다.

    0.2.4 를 낼 때 실제로 이렇게 됐다 — `package.json` 과 `version.ts` 는 올렸고
    잠금은 0.2.3 에 남았다. 앞의 둘은 검사가 잡았고 이것은 사람이 눈으로 찾았다.
    """
    declared = json.loads((ROOT / "package.json").read_text())["version"]
    lock = json.loads((ROOT / "package-lock.json").read_text())
    for where, got in (("루트", lock.get("version")),
                       ('packages[""]', lock.get("packages", {}).get("", {}).get("version"))):
        assert got == declared, (
            f"package-lock.json 의 {where} 는 {got!r}, package.json 은 {declared!r} —\n"
            "    판을 올릴 때 세 곳을 같이 고쳐라: package.json · version.ts · 잠금."
        )
