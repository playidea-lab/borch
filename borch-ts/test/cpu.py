"""The CPU device against the WebGPU device, in a real browser.

    npm run build:ts
    uv run --with playwright python borch-ts/test/cpu.py [--headed] [--model=imagenet-resnet18]

Two hub checkpoints (EfficientNet-B0, ResNet-18) go through both devices on the same
seeded image; the page compares the logits and times both. Needs the network for the
checkpoints (68 MB the first time; the hub caches them) and a real adapter for the GPU
half to mean anything — on a software adapter the values still compare, the timings do
not, and the runner says so.
"""
import sys

import run as runner
from launch import browser as browser_of, warn_if_software
from verdict import verdict

PAGE = "/borch-ts/test/cpu.html"
TIMEOUT_MS = 900_000


def main(argv):
    runner.require_fresh_dist()
    dist = runner.ROOT / "borch-ts" / "dist" / "test" / "cpu.js"
    if not dist.exists():
        print(f"no emit: {dist}\n  first: npm run build:ts", file=sys.stderr)
        return 2
    only = next((a.split("=", 1)[1] for a in argv if a.startswith("--model=")), None)
    port, stop = runner.serve(runner.ROOT)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p, browser_of(p, headed="--headed" in argv) as browser:
            page = browser.new_page()
            page.set_default_timeout(0)
            page.on("console", lambda m: print(f"  {m.text[6:]}", flush=True) if m.text.startswith("[cpu] ") else None)
            page.on("pageerror", lambda e: print(f"  [browser exception] {e}", flush=True))
            page.goto(f"http://127.0.0.1:{port}{PAGE}" + (f"?model={only}" if only else ""))
            page.wait_for_function("window.__borchCpu !== undefined", timeout=TIMEOUT_MS, polling=1000)
            result = page.evaluate("window.__borchCpu")
    finally:
        stop()
    if "error" in result:
        print(result["error"])
        return 1
    adapter = result["text"].split("\n", 1)[0].replace("gpu adapter: ", "")
    warn_if_software(adapter, "the gpu half's timings")
    return verdict(result, "the cpu device")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
