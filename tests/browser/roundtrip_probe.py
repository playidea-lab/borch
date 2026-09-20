"""Where the milliseconds go between `submit` and a mapped readback — **the round trip.**

    uv run --with playwright python tests/browser/roundtrip_probe.py [--headless]

The captured ResNet-18 forward on the RTX 5080 is 4.32 ms for 2.9 ms of GPU time; on
metal-3 the same recording is 4.1 ms for 3.9 of GPU (`docs/BOOK.md`, 2026-09-20). The
difference is not a kernel: it is what a forward pays once, between the last dispatch and
the value in JavaScript. This page takes that apart with raw WebGPU, forty repetitions
each, p10 / p50 / p90 in ms — three quantiles because the first run showed the 5080
bimodal (a 0.06 ms minimum under a 0.5 ms median), and a median alone hides that:

- `submit` of an empty command buffer, then `onSubmittedWorkDone` — the queue's round
  trip through the GPU process, nothing to run;
- a 4-byte copy, `submit`, `mapAsync(READ)`, `getMappedRange` — the readback pattern;
- the same behind one tiny dispatch;
- `pushErrorScope` / `popErrorScope` — what an allocation's out-of-memory scope costs to
  drain, since `Device.read` drains those before it maps;
- forty and four hundred tiny dispatches with their own bind groups, submitted and waited
  on — the encoding cost through the wire with no GPU time to hide behind;
- about 3 ms of real work in front of the copy, and every way of waiting for it: the map
  alone, an error scope drained first, `onSubmittedWorkDone` first, both at once;
- and borch's own `Tensor.toArray()` — on one element, and behind a 2048³ matmul with and
  without `synchronize()` first — the numbers the library pays.

Two adapters give the same answer for the GPU's part and different ones for the trip;
the rows say which part it is. Refuses a software adapter: a CPU's round trip is nobody's.
"""
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from first_run import FLAGS, ROOT, serve  # noqa: E402
from launch import _headed, refuse_if_software  # noqa: E402

PROBE = r"""async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return { error: "no adapter" };
  const info = adapter.info || {};
  const device = await adapter.requestDevice();
  const REPS = 40;
  const stat = (xs) => { const s = [...xs].sort((a, b) => a - b); const q = (f) => +s[Math.min(s.length - 1, Math.floor(f * s.length))].toFixed(3); return { p10: q(0.1), p50: q(0.5), p90: q(0.9) }; };
  const time = async (fn) => { const t = []; for (let i = 0; i < 5; i++) await fn(); for (let i = 0; i < REPS; i++) { const t0 = performance.now(); await fn(); t.push(performance.now() - t0); } return stat(t); };
  const rows = {};
  // 1. The queue's round trip, nothing to run.
  rows["submit + onSubmittedWorkDone (empty)"] = await time(async () => {
    device.queue.submit([device.createCommandEncoder().finish()]);
    await device.queue.onSubmittedWorkDone();
  });
  // 2. The readback pattern: a copy, a submit, a map.
  const src = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(src, 0, new Float32Array([1]));
  const stage = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  rows["copy 4 B + submit + mapAsync"] = await time(async () => {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, stage, 0, 4);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    stage.getMappedRange(); stage.unmap();
  });
  // 3. The same behind one tiny dispatch.
  const module = device.createShaderModule({ code: `@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(1) fn main() { X[0] = X[0] + 1.0; }` });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: src } }] });
  rows["1 dispatch + copy + submit + mapAsync"] = await time(async () => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1); pass.end();
    enc.copyBufferToBuffer(src, 0, stage, 0, 4);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    stage.getMappedRange(); stage.unmap();
  });
  // 4. An error scope drained.
  rows["pushErrorScope + popErrorScope"] = await time(async () => {
    device.pushErrorScope("out-of-memory");
    await device.popErrorScope();
  });
  // 5. Both awaited in series, as Device.read does eagerly, and concurrently.
  rows["popErrorScope then mapAsync (series)"] = await time(async () => {
    device.pushErrorScope("out-of-memory");
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, stage, 0, 4);
    device.queue.submit([enc.finish()]);
    await device.popErrorScope();
    await stage.mapAsync(GPUMapMode.READ);
    stage.getMappedRange(); stage.unmap();
  });
  rows["popErrorScope and mapAsync (concurrent)"] = await time(async () => {
    device.pushErrorScope("out-of-memory");
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, stage, 0, 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([device.popErrorScope(), stage.mapAsync(GPUMapMode.READ)]);
    stage.getMappedRange(); stage.unmap();
  });
  // 6. Forty tiny dispatches, each with its own bind group — the encoding cost the
  //    captured forward pays through the wire, without GPU time to hide behind.
  const bufs = Array.from({ length: 40 }, () => device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }));
  const binds = bufs.map((b) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: b } }] }));
  const many = (n) => { const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); pass.setPipeline(pipeline); for (let i = 0; i < n; i++) { pass.setBindGroup(0, binds[i % 40]); pass.dispatchWorkgroups(1); } pass.end(); return enc; };
  rows["40 dispatches + submit + onSubmittedWorkDone"] = await time(async () => { device.queue.submit([many(40).finish()]); await device.queue.onSubmittedWorkDone(); });
  rows["400 dispatches + submit + onSubmittedWorkDone"] = await time(async () => { device.queue.submit([many(400).finish()]); await device.queue.onSubmittedWorkDone(); });
  // 7. Real work in front of the readback — about the forward's 3 ms — so the trip is
  //    measured the way a forward pays it: with the GPU still busy at submit.
  const heavyMod = device.createShaderModule({ code: `@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) { var a = f32(g.x); for (var i = 0u; i < ITERS; i = i + 1u) { a = a * 0.999 + 0.5; } X[g.x] = a; }`.replace("ITERS", "40000u") });
  const heavyPipe = device.createComputePipeline({ layout: "auto", compute: { module: heavyMod, entryPoint: "main" } });
  const heavyBuf = device.createBuffer({ size: 256 * 1024 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const heavyBind = device.createBindGroup({ layout: heavyPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: heavyBuf } }] });
  const heavy = (enc) => { const pass = enc.beginComputePass(); pass.setPipeline(heavyPipe); pass.setBindGroup(0, heavyBind); pass.dispatchWorkgroups(1024); pass.end(); };
  rows["heavy + submit + onSubmittedWorkDone"] = await time(async () => { const enc = device.createCommandEncoder(); heavy(enc); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); });
  rows["heavy + copy + submit + mapAsync"] = await time(async () => { const enc = device.createCommandEncoder(); heavy(enc); enc.copyBufferToBuffer(heavyBuf, 0, stage, 0, 4); device.queue.submit([enc.finish()]); await stage.mapAsync(GPUMapMode.READ); stage.getMappedRange(); stage.unmap(); });
  rows["heavy + copy + submit + popErrorScope then mapAsync"] = await time(async () => { device.pushErrorScope("out-of-memory"); const enc = device.createCommandEncoder(); heavy(enc); enc.copyBufferToBuffer(heavyBuf, 0, stage, 0, 4); device.queue.submit([enc.finish()]); await device.popErrorScope(); await stage.mapAsync(GPUMapMode.READ); stage.getMappedRange(); stage.unmap(); });
  rows["heavy + copy + submit + onSubmittedWorkDone then mapAsync"] = await time(async () => { const enc = device.createCommandEncoder(); heavy(enc); enc.copyBufferToBuffer(heavyBuf, 0, stage, 0, 4); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); await stage.mapAsync(GPUMapMode.READ); stage.getMappedRange(); stage.unmap(); });
  rows["heavy + copy + submit + (onSubmittedWorkDone, mapAsync) concurrent"] = await time(async () => { const enc = device.createCommandEncoder(); heavy(enc); enc.copyBufferToBuffer(heavyBuf, 0, stage, 0, 4); device.queue.submit([enc.finish()]); await Promise.all([device.queue.onSubmittedWorkDone(), stage.mapAsync(GPUMapMode.READ)]); stage.getMappedRange(); stage.unmap(); });
  // 8. borch's own readback on one element — the library's number.
  let borch = null;
  try {
    const bt = await import("/borch-ts/dist/src/index.js");
    await bt.init();
    const t = bt.Tensor.from(new Float32Array([1]), [1]);
    rows["borch Tensor.toArray() (1 element, eager)"] = await time(async () => { await t.add(t).toArray(); });
    const k = bt.keepAlive(bt.Tensor.from(new Float32Array([2]), [1]));
    rows["borch keepAlive tensor toArray() (no alloc)"] = await time(async () => { await k.toArray(); });
    // The same with about 3 ms of borch work in front: a matmul, then one number out.
    const A = bt.keepAlive(bt.Tensor.randn([2048, 2048])), B = bt.keepAlive(bt.Tensor.randn([2048, 2048]));
    const dev = bt.device();
    rows["borch matmul 2048³ + synchronize()"] = await time(async () => { A.matmul(B); await dev.synchronize(); });
    rows["borch matmul 2048³ .sum().toArray()"] = await time(async () => { await A.matmul(B).sum().toArray(); });
    rows["borch matmul 2048³ then keepAlive toArray()"] = await time(async () => { A.matmul(B); await k.toArray(); });
    rows["borch matmul 2048³, synchronize(), keepAlive toArray()"] = await time(async () => { A.matmul(B); await dev.synchronize(); await k.toArray(); });
    borch = String(bt.Device.adapterInfo);
  } catch (e) { rows["borch"] = { error: String(e).slice(0, 200) }; }
  return { adapter: `${info.vendor} / ${info.architecture}`, borch, reps: REPS, rows };
}"""


def main(argv):
    from playwright.sync_api import sync_playwright
    headed = _headed("--headless" not in argv)
    channel = os.environ.get("BORCH_CHROME_CHANNEL") or None
    port, shutdown = serve(ROOT)
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(tempfile.mkdtemp(prefix="borch-rt-"), headless=not headed, channel=channel, args=list(FLAGS), timeout=60_000)
            try:
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{port}/tests/browser/kernel_bench.html?bench=none", wait_until="commit")
                got = page.evaluate(PROBE)
            finally:
                context.close()
    finally:
        shutdown()
    if not got or got.get("error"):
        print(json.dumps(got, indent=1)); return 1
    if refuse_if_software(got.get("adapter"), "the round trip"):
        return 1
    print(f"adapter: {got['adapter']} · {got['reps']} repetitions, p10 / p50 / p90 ms")
    for name, v in got["rows"].items():
        if "error" in v: print(f"  {name:<46} error: {v['error']}")
        else: print(f"  {name:<64} {v['p10']:7.3f} / {v['p50']:7.3f} / {v['p90']:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
