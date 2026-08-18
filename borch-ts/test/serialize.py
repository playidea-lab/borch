"""체크포인트가 실제 브라우저에서 왕복하고, 재개가 이어지는지 본다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/serialize.py [--headed]

**왕복만 보면 부족하다.** 저장했다 읽어서 값이 같은지는 코덱만 묻는 것이다. 진짜
물음은 그 뒤다 — *끊었다 이은 학습이 안 끊고 돌린 학습과 같은가.* 모멘텀 하나, 스텝
계수기 하나, 스케줄러의 에폭 하나만 빠져도 왕복은 초록인 채로 재개만 갈린다.

전부 결정론적이라 **비트 단위로 같아야 한다.** 그래서 이 러너에는 허용 오차가 없다.

가중치만 되돌리고 나머지를 버린 경로도 같이 돌린다 — 그쪽이 **갈려야** 위의 동등성
검사가 무언가를 재고 있다는 뜻이다.
"""

import sys

import run as runner
from launch import browser as browser_of

PAGE = "/borch-ts/test/serialize.html"
TIMEOUT_MS = 300_000


def main(argv):
    # **낡은 방출물도 없는 것만큼 나쁘다** — 소스를 고치고 빌드를 잊으면 옛 코드를
    # 재게 된다. `require_fresh_dist` 가 그 자리를 본다(`run.py`).
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "serialize.js"
    if not dist.exists():
        print(f"방출물이 없다: {dist}\n  먼저: npm run build:ts", file=sys.stderr)
        return 2

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        # **닫는 것도 `with` 가 한다** — 마지막 줄에 두면 그 앞에서 예외가 날 때
        # 안 닫히고, 남은 크로미엄이 다른 측정을 망가뜨린다.
        with sync_playwright() as p, \
                browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  [브라우저] {m.text}")
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            page.goto(f"http://127.0.0.1:{port}{PAGE}")
            page.wait_for_function("window.__borchSerialize !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchSerialize")
    finally:
        stop()

    if "error" in result:
        print(f"**체크포인트 점검이 터졌다**\n{result['error']}", file=sys.stderr)
        return 1
    print(f"어댑터: {result.get('adapter', '(모름)')}")
    print(result["text"])
    ok = "전부 통과" in result["text"]
    return 0 if ok and cross_language(result.get("sample")) else 1


def cross_language(sample):
    """브라우저가 쓴 파일을 **numpy 로만** 뜯는다. borch 코드는 한 줄도 안 쓴다.

    우리 코덱이 우리 코덱과 왕복하는 것은 자체 형식으로도 된다. safetensors 를 든
    값어치는 **남이 읽는다**는 데 있고, 그 주장이 참인지는 여기서만 확인된다.
    """
    import json
    import struct

    import numpy as np

    if not sample:
        print("표본이 없다 — 페이지가 sample() 을 안 내보냈다", file=sys.stderr)
        return False

    blob = bytes(bytearray(sample))
    (head_len,) = struct.unpack_from("<Q", blob, 0)
    header = json.loads(blob[8:8 + head_len])
    body = blob[8 + head_len:]

    got = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        # dtype 은 언제나 F32 다. borch 의 int64·bool 은 이름표라 머리에만 적힌다 —
        # 몸에 I64 라고 써 놓으면 4 바이트짜리와 어긋나 이 줄이 깨진다.
        assert entry["dtype"] == "F32", entry["dtype"]
        got[name] = np.frombuffer(body[begin:end], dtype="<f4").reshape(entry["shape"])

    want = {
        "fc.weight": np.array([[1.5, -2.25, 0.5], [7.0, -0.125, 3.0]], dtype="<f4"),
        "fc.labels": np.array([3.0, 1.0, 4.0], dtype="<f4"),
    }
    for name, expected in want.items():
        if name not in got or not np.array_equal(got[name], expected):
            print(f"**numpy 가 읽은 값이 다르다** — {name}: {got.get(name)}",
                  file=sys.stderr)
            return False

    labels = header["__metadata__"].get("borch.dtype:fc.labels")
    if labels != "int64":
        print(f"**형 이름표가 안 실렸다** — {labels}", file=sys.stderr)
        return False

    print(f"  ✓ numpy 가 같은 파일을 읽는다 — 텐서 {len(got)}개, "
          f"이름표 fc.labels={labels}")
    return True


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
