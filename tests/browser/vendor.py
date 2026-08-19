"""바깥에서 받아오는 것을 한 번만 받아 두고, 바이트가 그대로인지 확인한다.

CDN 은 **테스트 시점에 살아 있어야 하는 의존**이다. 실제로 한 번 끊겨서 검증이 멈췄다
(`ERR_QUIC_PROTOCOL_ERROR`). 그리고 `@4.22.0` 이 언제까지 같은 바이트를 준다는 것은
정책이지 계약이 아니다 — 그 버전이 사라지거나 바뀌면 오늘의 골든을 재현할 수 없다.

**받은 것은 이제 저장소에 있다**(`vendor/pyodide/`). 오래 "해시만 둔다" 였고 근거는
크기였는데, 재 보니 여섯 파일이 팩에서 8.4MB 다 — 이 저장소는 `tests/golden.json`
하나에 이력 23.9MB 를 쓰고 있다. 그 옆에서 한 번 넣고 안 바뀌는 8.4MB 는 큰 값이
아니었다. 판올림마다 같은 만큼 영구히 붙는 것이 이 결정이 치르는 값이고, 0.27.2 에서
올릴 계획이 없어서 치르기로 했다.

바뀐 것은 **어디에 두는가**뿐이고 잠금 파일은 그대로 쓴다. 오히려 이제야 제 일을 한다 —
파일과 잠금이 **둘 다 커밋돼 있으므로** 대조가 네트워크 없이 CI 에서 돈다. 예전에는
신선한 러너에 파일이 없어서 `fetch` 가 받아 온 것으로 잠금을 새로 썼고, 그 뒤 대조는
자기 자신과 비교하는 것이었다.

    uv run python tests/browser/vendor.py check   # 있는 것과 잠금을 대조한다
    uv run python tests/browser/vendor.py fetch   # 판올림할 때만. 잠금이 다르면 멈춘다
    uv run python tests/browser/vendor.py fetch --bump   # 잠금을 새로 쓴다

여기서 받는 것들의 라이선스는 [THIRD-PARTY.md](../../THIRD-PARTY.md) 에 있다.
**Pyodide 는 MPL-2.0 이다** — 우리 코드로 번지지는 않지만, 커밋한 이상 **이 저장소도
재배포자다.** 그래서 THIRD-PARTY.md 에 소스를 구할 길을 적어 둔다.
"""

import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
VENDOR = ROOT / "vendor"
LOCK = pathlib.Path(__file__).resolve().parent / "assets.lock"

PYODIDE_VERSION = "0.27.2"
PYODIDE_BASE = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
# Pyodide 가 뜨면서 실제로 부르는 것들. 빠지면 브라우저 콘솔에 404 로 드러난다.
PYODIDE_FILES = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm",
                 "python_stdlib.zip", "pyodide-lock.json"]
PYODIDE_PACKAGES = ["numpy"]          # 파일 이름은 lock 에서 찾는다

# **TF.js 를 받던 자리다.** `@tensorflow/tfjs@4.22.0` 두 파일을 CDN 에서 받아
# `vendor/` 에 두었다. 그 위에 서던 구현을 손으로 쓴 WGSL 로 갈아치우면서 필요가
# 없어졌다 — 이제 밖에서 받아 오는 것은 Pyodide 하나뿐이다.
#
# 남은 벤더가 하나여도 이 파일은 남긴다. CDN 은 테스트 시점에 살아 있어야 하는
# 의존이고 실제로 한 번 끊겼다.


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _get(url):
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def _targets():
    """(저장 경로, 주소) 목록. numpy 는 lock 을 읽어야 이름을 알 수 있다."""
    return [(pathlib.Path("pyodide") / name, PYODIDE_BASE + name)
            for name in PYODIDE_FILES]


def _package_targets(lock_bytes):
    packages = json.loads(lock_bytes)["packages"]
    found = []
    for want in PYODIDE_PACKAGES:
        entry = packages.get(want)
        if entry is None:
            raise SystemExit(f"pyodide-lock.json 에 {want} 가 없습니다")
        name = entry["file_name"]
        found.append((pathlib.Path("pyodide") / name, PYODIDE_BASE + name))
    return found


def fetch(bump=False):
    """받아서 `vendor/` 에 둔다. **잠금이 이미 있으면 대조하고, 다르면 멈춘다.**

    오래 여기서 잠금을 **덮어썼다.** 그러면 받아 온 것이 곧 정답이 되어, 잠금이 지키는
    것은 "이미 파일을 가진 기계" 뿐이고 하나도 없는 기계에서는 아무것도 안 지켰다.
    신선한 CI 러너가 정확히 후자라, 커밋된 해시 여섯 개가 거기서는 조용히 새 잠금이
    됐다 — 검사가 자기 입력을 자기가 정하는 그 모양이다.

    판올림은 `--bump` 로 명시한다. 잠금이 바뀌는 것이 커밋에 남아야 하는 일이라,
    받는 김에 슬쩍 바뀌면 안 된다.
    """
    VENDOR.mkdir(exist_ok=True)
    lock, total = {}, 0
    targets = _targets()
    lock_bytes = None
    for rel, url in targets:
        data = _get(url)
        if rel.name == "pyodide-lock.json":
            lock_bytes = data
        dst = VENDOR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        lock[str(rel)] = _sha(data)
        total += len(data)
        print(f"  {rel}  {len(data) / 1e6:.2f} MB")

    for rel, url in _package_targets(lock_bytes):
        data = _get(url)
        (VENDOR / rel).write_bytes(data)
        lock[str(rel)] = _sha(data)
        total += len(data)
        print(f"  {rel}  {len(data) / 1e6:.2f} MB")

    text = "".join(f"{h}  {p}\n" for p, h in sorted(lock.items()))
    old = _read_lock()
    if old is not None and not bump:
        moved = sorted(p for p, h in lock.items() if old.get(p) != h)
        gone = sorted(p for p in old if p not in lock)
        if moved or gone:
            raise SystemExit(
                "받아온 것이 잠금과 다릅니다 — 덮어쓰지 않았습니다.\n  "
                + "\n  ".join([f"{p}: 바이트가 다르다" for p in moved]
                               + [f"{p}: 이제 안 받는다" for p in gone])
                + "\n\n판올림이라면 `fetch --bump` 로 잠금을 새로 쓰십시오.")
    LOCK.write_text(text, encoding="utf-8")
    print(f"\n받았다 — {len(lock)}개 · {total / 1e6:.1f} MB → {VENDOR}")
    print(f"잠금 파일: {LOCK}" + (" (새로 씀)" if bump or old is None else " (그대로)"))
    return 0


def _read_lock():
    if not LOCK.exists():
        return None
    entries = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.split("  ", 1)
            entries[path] = digest
    return entries


def check(quiet=False):
    """잠금과 대조한다. (문제 목록)"""
    entries = _read_lock()
    if entries is None:
        return ["잠금 파일이 없다 — 먼저 `vendor.py fetch` 를 돌려라"]
    bad = []
    for path, digest in entries.items():
        f = VENDOR / path
        if not f.exists():
            bad.append(f"{path}: 없다")
        elif _sha(f.read_bytes()) != digest:
            bad.append(f"{path}: **바이트가 다르다** — 받아온 것이 잠금과 어긋난다")
    if not bad and not quiet:
        print(f"벤더 대조 — {len(entries)}개 전부 일치")
    return bad


def ensure():
    """없으면 받고, 있으면 대조한다. 러너가 시작할 때 부른다."""
    if _read_lock() is None or check(quiet=True):
        missing = check(quiet=True)
        if missing and _read_lock() is not None:
            drifted = [m for m in missing if "다르다" in m]
            if drifted:
                raise SystemExit("벤더 파일이 잠금과 다릅니다:\n  " + "\n  ".join(drifted))
        print("벤더 파일을 받는다(처음 한 번)…")
        fetch()


def main(argv):
    what = argv[1] if len(argv) > 1 else "check"
    if what == "fetch":
        return fetch(bump="--bump" in argv)
    if what == "check":
        bad = check()
        for why in bad:
            print(f"  ✗ {why}")
        return 1 if bad else 0
    print("쓰는 법: vendor.py [check | fetch [--bump]]")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
