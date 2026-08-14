"""borch.ts 의 CIFAR-10 정확도를 브라우저에서 잰다.

    npm run build:ts
    uv run --with playwright python borch-ts/test/accuracy.py --headed [--epochs 10]

`tests/browser/run.py --accuracy` 와 같은 잣대이고, 같은 데이터 파일을 쓴다:
저장소 루트에 `cifar-batch1.bin` 과 `cifar-batch-test.bin` 이 있어야 한다.

**창을 띄워야 한다.** 헤드리스는 소프트웨어 어댑터를 주고 그러면 이 측정이 몇 시간이
된다 — 페이지가 어댑터 이름을 먼저 찍으므로 결과에 그것이 같이 남는다.
"""

import sys

import run as runner
from launch import launch, refuse_if_software

TIMEOUT_MS = 7_200_000


def main(argv):
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "accuracy.js"
    if not dist.exists():
        print(f"방출물이 없다: {dist}\n  먼저: npm run build:ts", file=sys.stderr)
        return 2
    for name in ("cifar-batch1.bin", "cifar-batch-test.bin"):
        if not (runner.ROOT / name).exists():
            print(f"데이터가 없다: {runner.ROOT / name}\n"
                  "  CIFAR-10 바이너리를 저장소 루트에 두어라.", file=sys.stderr)
            return 2

    def opt(flag, default):
        return argv[argv.index(flag) + 1] if flag in argv else default

    query = f"?epochs={opt('--epochs', '10')}"
    if "--images" in argv:
        query += f"&images={opt('--images', '0')}"
    if "--augment" in argv:
        query += f"&augment={opt('--augment', 'off')}"

    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = launch(p, headed="--headed" in argv)
            page = browser.new_page()
            page.set_default_timeout(0)
            # 긴 측정은 도중에 죽으면 반환값이 통째로 사라진다. 진행 줄을 흘려보낸다.
            page.on("console", lambda m: print(f"  {m.text}")
                    if m.text.startswith("[accuracy]") or m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"  [브라우저 예외] {e}"))
            page.goto(f"http://127.0.0.1:{port}/borch-ts/test/accuracy.html{query}")
            page.wait_for_function("window.__borchAccuracy !== undefined",
                                   timeout=TIMEOUT_MS)
            result = page.evaluate("window.__borchAccuracy")
            browser.close()
    finally:
        stop()

    if "error" in result:
        print(f"재지 못했다: {result['error']}", file=sys.stderr)
        return 1
    print(result["text"])
    # 정확도 자체는 장치가 안 바꾼다. 그런데 **에폭 시간이 같이 적히고**, 소프트웨어
    # 어댑터에서는 그 절반이 거짓이 된다. 한 보고 안에 유효한 수와 무효한 수를 섞어
    # 내보내느니 다시 재라고 하는 편이 낫다.
    if refuse_if_software(result.get("adapter"), "에폭 시간"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
