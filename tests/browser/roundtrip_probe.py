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
- a table: work of five sizes (one tiny dispatch, forty with their own bind groups, a
  loop kernel at three lengths), its GPU time by timestamp query, and its wall time under
  four ways of waiting for the readback — `onSubmittedWorkDone`, the map alone, an error
  scope drained then the map, and the map with the wire kicked by cheap round trips until
  it resolves. The second 5080 run put forty tiny dispatches and a 1 ms kernel at the same
  3.0 ms wall, and the scope-then-map wait *under* the kernel's `onSubmittedWorkDone`; that
  reads as the GPU process polling its fences on a backoff, and this table is the test;
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
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [] });
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
  // 6. **The GPU's time against the wall's, by how the wait is done.** Work of five sizes
  //    (one tiny dispatch, forty of them with their own bind groups, and a loop kernel at
  //    three lengths) with a timestamp query around the pass, then the same work timed by
  //    the wall under four ways of waiting for its readback: `onSubmittedWorkDone`; the map
  //    alone; an error scope drained, then the map; and the map with the wire kicked by a
  //    cheap round trip until it resolves. If the GPU process polls its fences on a
  //    backoff, the wall will step and the kick will pull it back to the GPU's time.
  const bufs = Array.from({ length: 40 }, () => device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE }));
  const binds = bufs.map((b) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: b } }] }));
  const loopSrc = (iters) => `@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) { var a = f32(g.x); for (var i = 0u; i < ${iters}u; i = i + 1u) { a = a * 0.999 + 0.5; } X[g.x] = a; }`;
  const heavyBuf = device.createBuffer({ size: 256 * 1024 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const loopPipe = (iters) => { const m = device.createShaderModule({ code: loopSrc(iters) }); const pl = device.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } }); const b = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: heavyBuf } }] }); return (pass) => { pass.setPipeline(pl); pass.setBindGroup(0, b); pass.dispatchWorkgroups(1024); }; };
  const works = {
    "1 dispatch": (pass) => { pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1); },
    "40 dispatches, 40 bind groups": (pass) => { pass.setPipeline(pipeline); for (let i = 0; i < 40; i++) { pass.setBindGroup(0, binds[i]); pass.dispatchWorkgroups(1); } },
    "loop 10k": loopPipe(10000), "loop 40k": loopPipe(40000), "loop 160k": loopPipe(160000),
  };
  const hasTs = device.features.has("timestamp-query");
  const qs = hasTs ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const qBuf = hasTs ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const qStage = hasTs ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  const encode = (work, ts) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(ts && hasTs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    work(pass); pass.end();
    if (ts && hasTs) { enc.resolveQuerySet(qs, 0, 2, qBuf, 0); enc.copyBufferToBuffer(qBuf, 0, qStage, 0, 16); }
    enc.copyBufferToBuffer(src, 0, stage, 0, 4);
    return enc;
  };
  const gpuMs = async (work) => {
    if (!hasTs) return null;
    const t = [];
    for (let i = 0; i < 10; i++) {
      device.queue.submit([encode(work, true).finish()]);
      await qStage.mapAsync(GPUMapMode.READ);
      const q = new BigUint64Array(qStage.getMappedRange().slice(0)); qStage.unmap();
      t.push(Number(q[1] - q[0]) / 1e6);
    }
    return stat(t);
  };
  const mapDone = async () => { await stage.mapAsync(GPUMapMode.READ); stage.getMappedRange(); stage.unmap(); };
  const kick = async () => { device.pushErrorScope("validation"); await device.popErrorScope(); };
  const waits = {
    wsd: async (work) => { device.queue.submit([encode(work).finish()]); await device.queue.onSubmittedWorkDone(); },
    map: async (work) => { device.queue.submit([encode(work).finish()]); await mapDone(); },
    "scope, map": async (work) => { device.pushErrorScope("out-of-memory"); device.queue.submit([encode(work).finish()]); await device.popErrorScope(); await mapDone(); },
    "map + kicks": async (work) => { device.queue.submit([encode(work).finish()]); let done = false; const m = mapDone().then(() => { done = true; }); let n = 0; while (!done) { await kick(); n++; } await m; kicks.push(n); },
  };
  let kicks = [];
  const table = [];
  for (const [name, work] of Object.entries(works)) {
    const row = { work: name, gpu: await gpuMs(work) };
    for (const [w, fn] of Object.entries(waits)) { kicks = []; row[w] = await time(() => fn(work)); if (w === "map + kicks") row.kicks = stat(kicks); }
    table.push(row);
  }
  rows.__table = table;
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
    const cal = bt.Device.kickCalibration;
    borch = `${bt.Device.adapterInfo} · readbackKicks ${bt.Device.readbackKicks} (calibration plain ${cal.plainMs.toFixed(2)} / kicked ${cal.kickedMs.toFixed(2)} ms)`;
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
    if got.get("borch"): print(f"borch: {got['borch']}")
    for name, v in got["rows"].items():
        if name == "__table":
            print("  work · GPU ms by timestamp · wall ms by how the readback is waited for (p10 / p50 / p90)")
            for r in v:
                q = lambda k: "   —   " if r.get(k) is None else f"{r[k]['p10']:5.2f}/{r[k]['p50']:5.2f}/{r[k]['p90']:5.2f}"
                print(f"    {r['work']:<30} gpu {q('gpu')} · wsd {q('wsd')} · map {q('map')} · scope,map {q('scope, map')} · map+kicks {q('map + kicks')} (kicks {r['kicks']['p50']:.0f})")
            continue
        if "error" in v: print(f"  {name:<64} error: {v['error']}")
        else: print(f"  {name:<64} {v['p10']:7.3f} / {v['p50']:7.3f} / {v['p90']:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
