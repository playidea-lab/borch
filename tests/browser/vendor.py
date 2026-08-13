"""바깥에서 받아오는 것을 한 번만 받아 두고, 바이트가 그대로인지 확인한다.

CDN 은 **테스트 시점에 살아 있어야 하는 의존**이다. 실제로 한 번 끊겨서 검증이 멈췄다
(`ERR_QUIC_PROTOCOL_ERROR`). 그리고 `@4.22.0` 이 언제까지 같은 바이트를 준다는 것은
정책이지 계약이 아니다 — 그 버전이 사라지거나 바뀌면 오늘의 골든을 재현할 수 없다.

받은 것은 `vendor/` 에 두고 **저장소에는 넣지 않는다**(Pyodide 만 12MB 다. 이 프로젝트가
파는 것은 17KB 짜리 휠이고, 그 옆에 12MB 바이너리를 이력에 쌓을 이유가 없다).
대신 **해시를 저장소에 기록**한다. 그러면 오프라인·CDN 장애·바이트 변경이 전부 잡히고
저장소는 안 커진다. 골든과 같은 모양이다 — 한 번 굳히고, 그 뒤로는 대조한다.

    uv run python tests/browser/vendor.py fetch   # 받아서 잠금 파일을 쓴다
    uv run python tests/browser/vendor.py check   # 있는 것과 잠금을 대조한다

여기서 받는 것들의 라이선스는 [THIRD-PARTY.md](../../THIRD-PARTY.md) 에 있다.
**Pyodide 는 MPL-2.0 이다** — 우리 코드로 번지지는 않지만, 페이지에 실어 배포하면
소스를 구할 길을 알려야 한다. 이 저장소는 `vendor/` 를 커밋하지 않으므로 재배포하지
않지만, 브라우저에 띄우는 쪽은 재배포하게 된다.
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

TFJS_VERSION = "4.22.0"
TFJS = {
    "tf.min.js":
        f"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@{TFJS_VERSION}/dist/tf.min.js",
    "tf-backend-webgpu.min.js":
        f"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@{TFJS_VERSION}"
        "/dist/tf-backend-webgpu.min.js",
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _get(url):
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def _targets():
    """(저장 경로, 주소) 목록. numpy 는 lock 을 읽어야 이름을 알 수 있다."""
    out = [(pathlib.Path(name), url) for name, url in TFJS.items()]
    out += [(pathlib.Path("pyodide") / name, PYODIDE_BASE + name)
            for name in PYODIDE_FILES]
    return out


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


def fetch():
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

    LOCK.write_text("".join(f"{h}  {p}\n" for p, h in sorted(lock.items())), encoding="utf-8")
    print(f"\n받았다 — {len(lock)}개 · {total / 1e6:.1f} MB → {VENDOR}")
    print(f"잠금 파일: {LOCK}")
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
        return fetch()
    if what == "check":
        bad = check()
        for why in bad:
            print(f"  ✗ {why}")
        return 1 if bad else 0
    print("쓰는 법: vendor.py [fetch|check]")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
