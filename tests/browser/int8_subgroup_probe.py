"""Does an **int8 subgroup-matrix GEMM** compile and run on this adapter, and how fast?

    uv run --with playwright python tests/browser/int8_subgroup_probe.py [--size=1024] [--headless]

The RTX 5080 through Chrome 151 / Vulkan exposes `chromium-experimental-subgroup-matrix`
with **int8 configurations only** — `i8 × i8 → i32` at 16 × 16 × 32 — and no f32 8 × 8 × 8
(`docs/SCALE-MEASURED.md`, 2026-09-20), so every subgroup kernel in the tree is off there.
Before an int8 path is planned, the one question that decides it: on that configuration,
does a GEMM written against the proposal's WGSL compile, give the right integers, and beat
the scalar f32 tile? This page asks the adapter for the configuration, requests a device
with the feature, tries the load syntax the proposal allows (packed `array<i32>`, then
`array<u32>`), checks four entries against a CPU reference, and prints TOPS (2·N³ per
second) over thirty timed dispatches. The scalar baseline to hold it against is
`kernel_bench --bench=mm` on the same card.
"""
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, serve  # noqa: E402
from launch import _headed  # noqa: E402

PROBE = r"""async (SZ) => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return { error: "no adapter" };
  const info = adapter.info || {};
  const cfgs = [...(info.subgroupMatrixConfigs || [])].map((c) => ({ M: c.M, N: c.N, K: c.K, comp: c.componentType, res: c.resultComponentType }));
  const cfg = cfgs.find((c) => c.comp === "i8" && c.res === "i32" && c.M === 16 && c.N === 16);
  if (!cfg) return { error: "no i8 16x16 configuration", cfgs, adapter: `${info.vendor} / ${info.architecture}` };
  const feats = ["subgroups", "chromium-experimental-subgroup-matrix"];
  const device = await adapter.requestDevice({ requiredFeatures: feats });
  const { M, N, K } = cfg;
  const W = SZ / 4;                      // packed words a row
  const source = (arr) => `enable subgroups;
enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> A: array<${arr}>;
@group(0) @binding(1) var<storage, read> B: array<${arr}>;
@group(0) @binding(2) var<storage, read_write> C: array<i32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
  let col0 = wid.x * 32u;
  let row0 = wid.y * 32u;
  var c00: subgroup_matrix_result<i32, ${N}, ${M}>;
  var c01: subgroup_matrix_result<i32, ${N}, ${M}>;
  var c10: subgroup_matrix_result<i32, ${N}, ${M}>;
  var c11: subgroup_matrix_result<i32, ${N}, ${M}>;
  for (var k = 0u; k < ${SZ}u; k = k + ${K}u) {
    let a0 = subgroupMatrixLoad<subgroup_matrix_left<i8, ${K}, ${M}>>(&A, (row0 * ${SZ}u + k) / 4u, false, ${W}u);
    let a1 = subgroupMatrixLoad<subgroup_matrix_left<i8, ${K}, ${M}>>(&A, ((row0 + ${M}u) * ${SZ}u + k) / 4u, false, ${W}u);
    let b0 = subgroupMatrixLoad<subgroup_matrix_right<i8, ${N}, ${K}>>(&B, (k * ${SZ}u + col0) / 4u, false, ${W}u);
    let b1 = subgroupMatrixLoad<subgroup_matrix_right<i8, ${N}, ${K}>>(&B, (k * ${SZ}u + col0 + ${N}u) / 4u, false, ${W}u);
    c00 = subgroupMatrixMultiplyAccumulate(a0, b0, c00);
    c01 = subgroupMatrixMultiplyAccumulate(a0, b1, c01);
    c10 = subgroupMatrixMultiplyAccumulate(a1, b0, c10);
    c11 = subgroupMatrixMultiplyAccumulate(a1, b1, c11);
  }
  subgroupMatrixStore(&C, row0 * ${SZ}u + col0, c00, false, ${SZ}u);
  subgroupMatrixStore(&C, row0 * ${SZ}u + col0 + ${N}u, c01, false, ${SZ}u);
  subgroupMatrixStore(&C, (row0 + ${M}u) * ${SZ}u + col0, c10, false, ${SZ}u);
  subgroupMatrixStore(&C, (row0 + ${M}u) * ${SZ}u + col0 + ${N}u, c11, false, ${SZ}u);
}`;
  const tried = [];
  let pipeline = null, used = null;
  for (const arr of ["i32", "u32"]) {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code: source(arr) });
    const msgs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    let p = null;
    if (msgs.length === 0) {
      try { p = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } }); } catch (e) { msgs.push(String(e)); }
    }
    const err = await device.popErrorScope();
    if (err) msgs.push(err.message);
    tried.push({ array: arr, errors: msgs.slice(0, 3) });
    if (msgs.length === 0 && p) { pipeline = p; used = arr; break; }
  }
  if (!pipeline) return { error: "no variant compiled", tried, cfg, adapter: `${info.vendor} / ${info.architecture}` };
  // Operands: int8 in [-8, 8), packed four to a word, row-major.
  const words = SZ * SZ / 4;
  const packA = new Int32Array(words), packB = new Int32Array(words);
  const a8 = new Int8Array(SZ * SZ), b8 = new Int8Array(SZ * SZ);
  let s = 12345 >>> 0;
  const next = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s; };
  for (let i = 0; i < SZ * SZ; i++) { a8[i] = (next() % 16) - 8; b8[i] = (next() % 16) - 8; }
  new Int8Array(packA.buffer).set(a8); new Int8Array(packB.buffer).set(b8);
  const mk = (data, usage) => { const b = device.createBuffer({ size: data.byteLength, usage }); device.queue.writeBuffer(b, 0, data); return b; };
  const bufA = mk(packA, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const bufB = mk(packB, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const bufC = device.createBuffer({ size: SZ * SZ * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bufA } }, { binding: 1, resource: { buffer: bufB } }, { binding: 2, resource: { buffer: bufC } }] });
  const run = (n) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    for (let i = 0; i < n; i++) pass.dispatchWorkgroups(SZ / 32, SZ / 32, 1);
    pass.end();
    device.queue.submit([enc.finish()]);
    return device.queue.onSubmittedWorkDone();
  };
  device.pushErrorScope("validation");
  await run(5);
  const fault = await device.popErrorScope();
  if (fault) return { error: "dispatch faulted: " + fault.message, used, cfg };
  const ITERS = 30;
  const t0 = performance.now();
  await run(ITERS);
  const ms = (performance.now() - t0) / ITERS;
  const tops = 2 * SZ * SZ * SZ / (ms / 1000) / 1e12;
  // Four entries against a CPU reference, exact integers.
  const stage = device.createBuffer({ size: SZ * SZ * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(bufC, 0, stage, 0, SZ * SZ * 4); device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const C = new Int32Array(stage.getMappedRange().slice(0));
  stage.unmap();
  const checks = [];
  for (const [r, c] of [[0, 0], [17, 33], [SZ - 1, SZ - 1], [513, 257]]) {
    let acc = 0;
    for (let k = 0; k < SZ; k++) acc += a8[r * SZ + k] * b8[k * SZ + c];
    checks.push({ r, c, cpu: acc, gpu: C[r * SZ + c], ok: acc === C[r * SZ + c] });
  }
  return { adapter: `${info.vendor} / ${info.architecture}`, cfg, used, size: SZ, msPerGemm: Math.round(ms * 1000) / 1000, tops: Math.round(tops * 100) / 100, checks, tried };
}"""


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headless" not in argv)
    size = int(next((a.split("=", 1)[1] for a in argv if a.startswith("--size=")), "1024"))
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    port, shutdown = serve(ROOT)
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(tempfile.mkdtemp(prefix="borch-i8sg-"), headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{port}/tests/browser/kernel_bench.html?bench=none", wait_until="commit")
                got = page.evaluate(PROBE, size)
            finally:
                context.close()
    finally:
        shutdown()
    print(json.dumps(got, indent=1, ensure_ascii=False))
    ok = bool(got) and not got.get("error") and all(c["ok"] for c in got.get("checks", []))
    print("**the int8 subgroup GEMM compiles, is exact, and runs**" if ok else "**no int8 subgroup GEMM on this adapter — see above**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
