"""`VERSION` 상수가 `package.json` 과 같은 수를 말하는지 본다.

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
