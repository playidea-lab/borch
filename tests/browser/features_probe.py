"""What the adapter offers — **features and subgroup matrix configurations, by flag set.**

    uv run --with playwright python tests/browser/features_probe.py [--extra=--flag,--flag] [--headless]

Opens a blank page with the probes' Chrome flags (plus any `--extra`), asks the adapter
for its features and `info`, and prints them. For the question "does this Chrome on this
card expose `chromium-experimental-subgroup-matrix`, and under which flags" — the answer
decides whether the products run on the hardware's matrix units or on the scalar tile.
"""
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, serve  # noqa: E402
from launch import _headed  # noqa: E402

PROBE = """async () => {
  if (!navigator.gpu) return { error: "no navigator.gpu" };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return { error: "no adapter" };
  const info = adapter.info || {};
  const configs = [];
  for (const c of (info.subgroupMatrixConfigs || [])) configs.push({ M: c.M, N: c.N, K: c.K, component: c.componentType, result: c.resultComponentType });
  return { features: [...adapter.features].sort(), vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
           subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize, subgroupMatrixConfigs: configs, isFallback: adapter.isFallbackAdapter };
}"""


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headless" not in argv)
    extra = next((a.split("=", 1)[1] for a in argv if a.startswith("--extra=")), "")
    flags = list(FLAGS) + ([f for f in extra.split(",") if f] if extra else [])
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    # A served page, not about:blank — `navigator.gpu` is absent there.
    port, shutdown = serve(ROOT)
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(tempfile.mkdtemp(prefix="borch-feat-"), headless=not headed, channel=channel, args=flags, timeout=60_000)
            try:
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{port}/tests/browser/kernel_bench.html?bench=none", wait_until="commit")
                got = page.evaluate(PROBE)
            finally:
                context.close()
    finally:
        shutdown()
    print("flags: " + " ".join(flags))
    print(json.dumps(got, indent=1, ensure_ascii=False))
    return 0 if got and not got.get("error") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
