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
  // **The load semantics are the unknown.** The first run compiled and ran at 26 TOPS and
  // every entry was wrong (2026-09-20): the proposal's text leaves the offset and stride of
  // an 8-bit load — words of the packed array, or components — and the order of the type's
  // dimensions to be read carefully, and a wrong reading is a fast wrong answer. So the
  // probe sweeps the readings on a small size against a CPU reference, then times the
  // reading that is exact.
  const variants = [];
  for (const arr of ["i32", "u32"]) for (const units of ["words", "elems"]) for (const order of ["KM", "MK"]) variants.push({ arr, units, order });
  const source = (v, SZ) => {
    const div = v.units === "words" ? "/ 4u" : "";
    const stride = v.units === "words" ? SZ / 4 : SZ;
    const L = v.order === "KM" ? `subgroup_matrix_left<i8, ${K}, ${M}>` : `subgroup_matrix_left<i8, ${M}, ${K}>`;
    const R = v.order === "KM" ? `subgroup_matrix_right<i8, ${N}, ${K}>` : `subgroup_matrix_right<i8, ${K}, ${N}>`;
    const Cm = v.order === "KM" ? `subgroup_matrix_result<i32, ${N}, ${M}>` : `subgroup_matrix_result<i32, ${M}, ${N}>`;
    return `enable subgroups;
enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read> A: array<${v.arr}>;
@group(0) @binding(1) var<storage, read> B: array<${v.arr}>;
@group(0) @binding(2) var<storage, read_write> C: array<i32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
  let col0 = wid.x * 32u;
  let row0 = wid.y * 32u;
  var c00: ${Cm};
  var c01: ${Cm};
  var c10: ${Cm};
  var c11: ${Cm};
  for (var k = 0u; k < ${SZ}u; k = k + ${K}u) {
    let a0 = subgroupMatrixLoad<${L}>(&A, (row0 * ${SZ}u + k) ${div}, false, ${stride}u);
    let a1 = subgroupMatrixLoad<${L}>(&A, ((row0 + ${M}u) * ${SZ}u + k) ${div}, false, ${stride}u);
    let b0 = subgroupMatrixLoad<${R}>(&B, (k * ${SZ}u + col0) ${div}, false, ${stride}u);
    let b1 = subgroupMatrixLoad<${R}>(&B, (k * ${SZ}u + col0 + ${N}u) ${div}, false, ${stride}u);
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
  };
  const compile = async (code) => {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code });
    const msgs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    let p = null;
    if (msgs.length === 0) {
      try { p = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } }); } catch (e) { msgs.push(String(e)); }
    }
    const err = await device.popErrorScope();
    if (err) msgs.push(err.message);
    return { pipeline: msgs.length === 0 ? p : null, errors: msgs.slice(0, 2) };
  };
  const operands = (SZ) => {
    const a8 = new Int8Array(SZ * SZ), b8 = new Int8Array(SZ * SZ);
    let s = 12345 >>> 0;
    const next = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s; };
    for (let i = 0; i < SZ * SZ; i++) { a8[i] = (next() % 16) - 8; b8[i] = (next() % 16) - 8; }
    const mk = (bytes, usage) => { const b = device.createBuffer({ size: bytes.byteLength, usage }); device.queue.writeBuffer(b, 0, bytes); return b; };
    return { a8, b8,
      bufA: mk(a8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      bufB: mk(b8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      bufC: device.createBuffer({ size: SZ * SZ * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }) };
  };
  const dispatch = (pipeline, op, SZ, n) => {
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: op.bufA } }, { binding: 1, resource: { buffer: op.bufB } }, { binding: 2, resource: { buffer: op.bufC } }] });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    for (let i = 0; i < n; i++) pass.dispatchWorkgroups(SZ / 32, SZ / 32, 1);
    pass.end();
    device.queue.submit([enc.finish()]);
    return device.queue.onSubmittedWorkDone();
  };
  const readC = async (op, SZ) => {
    const stage = device.createBuffer({ size: SZ * SZ * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(op.bufC, 0, stage, 0, SZ * SZ * 4); device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const C = new Int32Array(stage.getMappedRange().slice(0));
    stage.unmap(); stage.destroy();
    return C;
  };
  const check = (op, C, SZ) => {
    const out = [];
    for (const [r, c] of [[0, 0], [17, 33], [SZ - 1, SZ - 1], [SZ / 2 + 1, SZ / 4 + 1]]) {
      let acc = 0;
      for (let k = 0; k < SZ; k++) acc += op.a8[r * SZ + k] * op.b8[k * SZ + c];
      out.push({ r, c, cpu: acc, gpu: C[r * SZ + c], ok: acc === C[r * SZ + c] });
    }
    return out;
  };
  // The sweep, small: which reading is exact.
  const SMALL = 128;
  const small = operands(SMALL);
  const tried = [];
  let exact = null;
  for (const v of variants) {
    const { pipeline, errors } = await compile(source(v, SMALL));
    if (!pipeline) { tried.push({ ...v, errors }); continue; }
    device.pushErrorScope("validation");
    await dispatch(pipeline, small, SMALL, 1);
    const fault = await device.popErrorScope();
    if (fault) { tried.push({ ...v, fault: fault.message.slice(0, 120) }); continue; }
    const checks = check(small, await readC(small, SMALL), SMALL);
    const ok = checks.every((c) => c.ok);
    tried.push({ ...v, exact: ok, sample: checks.slice(0, 2).map((c) => `${c.cpu}/${c.gpu}`) });
    if (ok && !exact) exact = v;
  }
  if (!exact) return { error: "no reading of the 8-bit load is exact", tried, cfg, adapter: `${info.vendor} / ${info.architecture}` };
  // The exact reading, timed at the asked size.
  const { pipeline } = await compile(source(exact, SZ));
  const op = operands(SZ);
  await dispatch(pipeline, op, SZ, 5);
  const ITERS = 30;
  const t0 = performance.now();
  await dispatch(pipeline, op, SZ, ITERS);
  const ms = (performance.now() - t0) / ITERS;
  const tops = 2 * SZ * SZ * SZ / (ms / 1000) / 1e12;
  const checks = check(op, await readC(op, SZ), SZ);
  return { adapter: `${info.vendor} / ${info.architecture}`, cfg, exact, size: SZ, msPerGemm: Math.round(ms * 1000) / 1000, tops: Math.round(tops * 100) / 100, checks, tried };
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
