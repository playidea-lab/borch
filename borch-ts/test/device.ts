/**
 * Device handling — negotiation, availability and placement, in a browser.
 *
 * **The golden does not catch this.** The golden is an instrument for asking whether a
 * value equals torch's, and what is asked here is not a value but *where it is* and *what
 * is said when it is not there.* Those are different questions, so the runner is separate.
 *
 * It does not refuse on a software adapter — placement follows the same rules on any.
 */

import {
  currentDevice,
  device,
  init,
  isAvailable,
  keepAlive,
  nn,
  noGrad,
  probe,
  scope,
  Tensor,
  optim,
  onnx,
  peft,
} from "../src/index.js";
import { bindingAccess, Device } from "../src/device.js";
import { tiledConfigFits as K_fits } from "../src/kernels.js";

const CROSS_DEVICE = "Expected all tensors to be on the same device";

interface Check {
  name: string;
  ok: boolean;
  note: string;
}

/**
 * `checks` is the authority in this report. `text` is the shadow a person reads.
 *
 * **The runner used to judge by scanning a sentence.** That way of judging changes its
 * answer quietly when the wording changes, and in `readme.py` it did — with one of the two
 * examples failing, the word it looked for was still sitting on another line, so it
 * returned 0. Hand the state over as it is and the runner can count, and can say for
 * itself which thing failed.
 */
export interface Report { text: string; checks: Check[] }

const checks: Check[] = [];

function want(name: string, ok: boolean, note = ""): void {
  checks.push({ name, ok, note });
}

/** Where it has to throw. **Not throwing is the failure** — pass quietly and the value
 * is wrong. */
function wantThrow(name: string, fragment: string, body: () => unknown): void {
  try {
    body();
    want(name, false, "it did not throw — it passed quietly");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    want(name, message.includes(fragment), message.includes(fragment)
      ? "" : `different wording: ${message}`);
  }
}

function same(a: Float32Array, b: Float32Array): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export async function report(): Promise<Report> {
  // ── Before it attaches ──────────────────────────────────────────────
  // **The order matters.** Asked after `init()`, the unattached state is never seen.
  want("currentDevice() is null before init", currentDevice() === null);

  // How many times the page asks the browser for an adapter — the number the Linux
  // NVIDIA driver charges seconds for. Counted from the outside, on the API itself.
  let asked = 0;
  const askedBefore = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = function (...a) { asked += 1; return askedBefore.apply(this, a); };

  const first = await probe();
  want("probe() finds an adapter", first.ok,
    first.ok ? "" : `${first.why}: ${first.message}`);
  want("probe() gives the adapter's name",
    first.ok && first.adapter.length > 0, first.ok ? first.adapter : "");
  want("isAvailable() is true", await isAvailable());

  // Does a reason come out when something absent is asked for. An environment where it
  // genuinely is absent cannot be built here, so a software adapter is forced and all that
  // is looked at is **whether the same path returns an adapter.**
  const fallback = await probe({ forceFallbackAdapter: true });
  want("asking for a fallback adapter gives an answer with a reason",
    fallback.ok || fallback.why === "no-adapter",
    fallback.ok ? fallback.adapter : fallback.why);

  // **`software` is the field a caller acts on, and the name is trivia.**
  // Knowing that `swiftshader`, `llvmpipe` and `lavapipe` mean the CPU is knowledge
  // this library has and its callers should not need — three files were each keeping
  // their own copy of that list before it moved here.
  want("probe() says whether the adapter is software — a GPU is not",
    first.ok && first.software === false, first.ok ? String(first.software) : "");
  want("probe() says so when the software adapter was asked for",
    !fallback.ok || fallback.software === true,
    fallback.ok ? `${fallback.adapter} → software=${fallback.software}` : fallback.why);
  want("and the two answers differ, so the field is reading the adapter",
    !fallback.ok || first.ok && first.software !== fallback.software);

  // Called in the form the README writes down — code in a document rots unless it runs.
  await init({ powerPreference: "high-performance" });

  // **A value read after a fault is a value of nothing — so the readback throws.** A
  // pipeline that will not compile is dispatched; the browser reports it asynchronously
  // and drops the command. The first readback after that must throw with the fault's
  // words, and the one after it must read again (a page may choose to go on).
  {
    const d = device();
    const before = d.faults.count;
    const junk = d.alloc(4);
    d.run1d(d.pipeline("device-test:not-wgsl", () => "this is not WGSL"), [junk], 4);
    await new Promise((resolve) => setTimeout(resolve, 150));
    let thrown = "";
    try {
      await Tensor.full([4], 1).toArray();
    } catch (err) {
      thrown = err instanceof Error ? err.message : String(err);
    }
    want("a readback after a fault throws, naming the fault", thrown.includes("fault"),
      thrown ? "" : `faults grew by ${d.faults.count - before} and nothing threw`);
    // **The browser reports one bad pipeline as several events** — the module, the
    // pipeline, its layout — and they arrive over time: one run in three saw the third
    // land after the first readback had already thrown and reset the count, so the
    // "reads again" readback threw on it (2026-09-21, metal-3). The contract is that a
    // readback after the faults have all been seen reads again; so the late ones are
    // given a moment and swallowed, and only then is the clean readback asked for.
    await new Promise((resolve) => setTimeout(resolve, 150));
    try { await Tensor.full([4], 2).toArray(); } catch { /* a late event of the same fault */ }
    const again = await Tensor.full([4], 2).toArray();
    want("the readback after that reads again", again[0] === 2, `got ${again[0]}`);
  }
  // **A learning rate that is not a number is refused at the door**, torch's sentence.
  // Let through, it reached WGSL as `[object Object]` and every step's pipeline was
  // invalid (measured, tests/browser/envelope.html).
  wantThrow("an options object in torch's positional lr seat is refused", "Invalid learning rate",
    () => new optim.SGD([Tensor.zeros([1])], { lr: 0.05 } as unknown as number));
  wantThrow("a NaN learning rate is refused", "Invalid learning rate",
    () => new optim.SGD([Tensor.zeros([1])], Number.NaN));
  // Three probes above (`probe()`, `isAvailable()`, the fallback) asked three times;
  // `init()` consumed the GPU adapter the second of them held, so the count stays at
  // three instead of reaching four. The fallback probe in between must not have evicted
  // it — the hold is per option set. On the RTX 5080 the request `init()` no longer
  // makes was 2,953 ms, the whole of the click.
  want("init() consumed the adapter probe() obtained — one request fewer",
    asked === 3, `requestAdapter was called ${asked} times across three probes and init()`);
  want("currentDevice() is webgpu after init", currentDevice() === "webgpu");
  want("the device is alive", device().alive && device().lost === null);

  // ── Placement ───────────────────────────────────────────────────────
  const values = [1, 2, 3, 4];
  const g = Tensor.from(values, [2, 2]);
  want("the default placement is webgpu", g.device === "webgpu");

  const c = await g.cpu();
  want("after cpu() it is on the cpu", c.device === "cpu");
  want("cpu() carries the values unchanged",
    same(await c.toArray(), Float32Array.from(values)));
  want("a cpu tensor keeps its shape and dtype too",
    c.shape.length === 2 && c.shape[0] === 2 && c.dtype === g.dtype);

  const one = await Tensor.from([7], [1]).cpu();
  want("item() runs on a cpu tensor", (await one.item()) === 7);
  want("repr() runs on a cpu tensor", (await one.repr()).includes("7"));

  // ── Devices that have parted throw ──────────────────────────────────
  wantThrow("an operation on a cpu tensor stops with torch's wording", CROSS_DEVICE,
    () => c.sum());
  wantThrow("mixing it with a gpu tensor stops too", CROSS_DEVICE, () => g.add(c));
  wantThrow("the cpu tensor on the left stops too", CROSS_DEVICE, () => c.add(g));

  // ── Bringing it back up ─────────────────────────────────────────────
  const back = c.webgpu();
  want("after webgpu() it is on the gpu", back.device === "webgpu");
  want("arithmetic runs again on what came back", (await back.sum().item()) === 10);
  want("the values survive the round trip",
    same(await back.toArray(), Float32Array.from(values)));

  // Already in place, it does nothing. Another round trip would be waste.
  want("cpu() is harmless on a cpu tensor", (await c.cpu()) === c);
  want("webgpu() is harmless on a gpu tensor", g.webgpu() === g);

  // ── Put on the host from the start ──────────────────────────────────
  const source = Float32Array.from([9, 8]);
  const host = Tensor.from(source, [2], { device: "cpu" });
  want("made with device: 'cpu' it is on the cpu", host.device === "cpu");
  source[0] = 0;
  want("changing the array afterwards does not change the tensor",
    (await host.toArray())[0] === 9);
  const copy = await host.toArray();
  copy[1] = 0;
  want("toArray() hands back a copy", (await host.toArray())[1] === 8);

  // A scope handles GPU buffers only. `keepAlive(await t.cpu())` must not catch on the
  // guard.
  want("keepAlive() does not refuse a cpu tensor", keepAlive(c) === c);
  let scoped: Tensor | null = null;
  await scope(async () => { scoped = await g.cpu(); });
  want("a cpu tensor still reads after leaving the scope",
    scoped !== null && same(await (scoped as Tensor).toArray(),
      Float32Array.from(values)));

  // ── Gradients ───────────────────────────────────────────────────────
  const leaf = Tensor.from([1, 2], [2], { requiresGrad: true });
  const dropped = await leaf.cpu();
  want("cpu() cuts the graph", !dropped.requiresGrad);

  // ── The re-tiled scalar GEMM against the tile as it was ─────────────────────
  // `docs/GEMM.md` Step 4: with subgroup matrices off (so the scalar path is taken on
  // every adapter) a product on the adapter's configurations must match the old tile to
  // a rounding, on a shape the big tile divides and on one only the 64 × 64 fits, and
  // then on a shape neither divides — which must fall through to the old tile unchanged.
  {
    const sgm = Device.subgroupMatrix; const cfgs = Device.gemmConfigs;
    Device.subgroupMatrix = false;
    const shapes: [number, number, number][] = [[256, 512, 128], [64, 320, 192], [100, 96, 40]];
    const configs = cfgs.length > 0 ? cfgs : [{ TM: 128, TN: 64, RM: 8, RN: 4, KT: 16, vec4: true, dbuf: false }, { TM: 64, TN: 64, RM: 4, RN: 4, KT: 16, vec4: true, dbuf: false }];
    for (const [M, K, N] of shapes) {
      const a = keepAlive(Tensor.randn([M, K])), b = keepAlive(Tensor.randn([K, N]));
      Device.gemmConfigs = [];
      const old = await a.matmul(b).toArray();
      Device.gemmConfigs = configs;
      const pipelinesBefore = device().pipelineCount;
      const fresh = await a.matmul(b).toArray();
      let err = 0, scale = 0;
      for (let i = 0; i < old.length; i++) { const o = old[i] ?? 0, f = fresh[i] ?? 0; err = Math.max(err, Math.abs(o - f)); scale = Math.max(scale, Math.abs(o)); }
      want(`re-tiled GEMM ${M}x${K}x${N} matches the old tile to a rounding`, err / scale < 1e-5, `rel ${(err / scale).toExponential(1)}`);
      const fits = configs.some((c) => K_fits(M, K, N, c, Device.workgroupStorage));
      want(`re-tiled GEMM ${M}x${K}x${N} ${fits ? "took a new pipeline" : "stayed on the old tile"}`, (device().pipelineCount > pipelinesBefore) === fits, `pipelines ${pipelinesBefore} → ${device().pipelineCount}`);
    }
    Device.subgroupMatrix = sgm; Device.gemmConfigs = cfgs;
  }

  // ── The fused attention against the composed chain ─────────────────────────
  // `Tensor.fusedAttention` (the inference kernel) must give what the split / permute /
  // bmm / mask / softmax / bmm / merge chain gives, to a rounding: an odd token count
  // with padded keys the mask zeroes, a head of 64 and one of 32, with the projection's
  // bias added in the kernel; and it must refuse a tensor that wants a gradient.
  {
    // A head of 128 as well (2026-09-24 review): the kernel needs 256·D + 8 KiB of workgroup
    // storage — the merge stages 64·D floats in the key block — and on a 32 KiB device it
    // halved its key block to 16 and merged past the end of it. Where it does not fit it
    // must refuse, naming the storage; where it fits it must match.
    const cases: [number, number, number, number, number][] = [[2, 37, 3, 64, 33], [1, 200, 6, 32, 197], [1, 50, 2, 128, 50]];
    for (const [B, N, H, D, keyLen] of cases) {
      if (256 * D + 8192 > Device.workgroupStorage) {
        wantThrow(`fused attention with a head of ${D} refuses a device with ${Device.workgroupStorage} bytes of workgroup storage`,
          "workgroup storage", () => noGrad(() => Tensor.fusedAttention(keepAlive(Tensor.randn([B, N, 3 * H * D])), H)));
        continue;
      }
      const width = 3 * H * D;
      const qkv = keepAlive(Tensor.randn([B, N, width]));
      const bias = keepAlive(Tensor.randn([width]));
      const ref = await noGrad(() => {
        const parts = qkv.add(bias).reshape([B, N, 3, H, D]).permute([2, 0, 3, 1, 4]);
        const fold = [B * H, N, D];
        const q = parts.select(0, 0).reshape(fold).mul(Tensor.owned([], 1 / Math.sqrt(D)));
        const k = parts.select(0, 1).reshape(fold);
        const v = parts.select(0, 2).reshape(fold);
        let scores = q.bmm(k.transpose(-2, -1));
        if (keyLen < N) {
          const mask = new Float32Array(N);
          for (let j = keyLen; j < N; j++) mask[j] = -1e30;
          scores = scores.add(Tensor.from(mask, [1, 1, N]));
        }
        return scores.softmax(-1).bmm(v).reshape([B, H, N, D]).permute([0, 2, 1, 3]).reshape([B, N, H * D]);
      }).toArray();
      const got = await noGrad(() => Tensor.fusedAttention(qkv, H, { keyLen, bias })).toArray();
      let err = 0, scale = 0;
      for (let i = 0; i < ref.length; i++) { const r = ref[i] ?? 0, g = got[i] ?? 0; err = Math.max(err, Math.abs(r - g)); scale = Math.max(scale, Math.abs(r)); }
      want(`fused attention [${B}, ${N}, ${H} heads of ${D}] with ${keyLen} keys matches the composed chain to a rounding`, err / scale < 1e-5, `rel ${(err / scale).toExponential(1)}`);
    }
    const wants = keepAlive(Tensor.randn([1, 8, 3 * 2 * 16]));
    wants.requiresGrad = true;
    wantThrow("fused attention refuses an input that wants a gradient", "inference kernel", () => Tensor.fusedAttention(wants, 2));
  }

  // ── Advice: the traps named once, and nothing on correct code ─────────────────
  // `docs/FIRST.md` 3. Silent here (`Device.advice = false`); `Device.advised` says what
  // would have been printed.
  {
    const dv = device();
    Device.advice = false;
    Device.advised.delete("unscoped"); Device.advised.delete("eval-grad"); Device.advised.delete("eights");
    dv.unscoped = 0;
    const lin = new nn.Linear(16, 16).eval();
    for (const p of lin.parameters()) keepAlive(p);
    const before = Device.advised.size;
    await scope(async () => { await noGrad(() => lin.call(Tensor.randn([8, 16]))).toArray(); });
    want("advice: a scoped noGrad forward of an eval model draws none", Device.advised.size === before, `${Device.advised.size - before} new`);
    await scope(async () => { await lin.call(Tensor.randn([8, 16])).toArray(); });
    want("advice: an eval() model run under gradient mode is told once", Device.advised.has("eval-grad"));
    await scope(async () => { await lin.call(Tensor.randn([8, 16])).toArray(); });
    want("advice: told once, not twice", Device.advised.size === before + 1, `${Device.advised.size - before} new`);
    // Two thousand small tensors outside any scope — the loop a first page writes.
    dv.unscoped = 0;
    for (let i = 0; i < Device.UNSCOPED_ADVICE_AT + 10; i++) Tensor.randn([4]);
    want("advice: an inference loop without a scope is told", Device.advised.has("unscoped"));
    if (Device.subgroupMatrix) {
      await scope(async () => { await Tensor.randn([200, 300]).matmul(Tensor.randn([300, 400])).toArray(); });
      want("advice: a large matmul off the eights is told", Device.advised.has("eights"));
    }
    Device.advice = true;
  }

  // ── Synchronising ───────────────────────────────────────────────────
  // It has to be possible to wait for completion without reading a value. Without this
  // the bench mixes the readback into its measurement.
  const before = device().submits;
  Tensor.from([1, 2, 3], [3]).sum();
  await device().synchronize();
  want("synchronize() sends what piled up and waits", device().submits > before);

  // ── One shader per shape, not per offset ────────────────────────────────────
  // A grouped convolution slices its input once per group and pads once per group, and
  // every slice starts at a different offset. When the offset was baked into the shader,
  // one EfficientNet-B4 forward compiled 19,533 pipelines, 19,249 of them these two kinds
  // (#121). The offset and the padding width now arrive in a parameter word, so the
  // second slice of a shape must find the first slice's pipeline — measured here rather
  // than trusted, because a key that quietly grows again is exactly what this was.
  const wide = Tensor.from(Array.from({ length: 24 }, (_, i) => i), [4, 6]);
  const baked = device().pipelineCount;
  await wide.narrow(1, 0, 2).toArray();
  const afterFirstSlice = device().pipelineCount;
  await wide.narrow(1, 3, 2).toArray();
  want("a second slice of the same shape bakes no new shader",
    device().pipelineCount === afterFirstSlice,
    `pipelines ${baked} → ${afterFirstSlice} → ${device().pipelineCount}`);
  const narrowed = wide.narrow(1, 0, 2);
  await narrowed.pad(1, 0, 4).toArray();
  const afterFirstPad = device().pipelineCount;
  await narrowed.pad(1, 4, 0).toArray();
  want("a second pad to the same width bakes no new shader",
    device().pipelineCount === afterFirstPad,
    `pipelines ${afterFirstSlice} → ${afterFirstPad} → ${device().pipelineCount}`);
  // And the values still come from the right place — sharing a shader must not mean
  // sharing an answer.
  want("the offset still reaches the shader",
    same(await wide.narrow(1, 3, 2).toArray(), Float32Array.from([3, 4, 9, 10, 15, 16, 21, 22])));
  want("the padding width still reaches the shader",
    same(await narrowed.pad(1, 4, 0).toArray().then((a) => a.slice(0, 6)),
         Float32Array.from([0, 0, 0, 0, 0, 1])));

  // **What a kernel does to each binding, read off its WGSL** (`bindingAccess`,
  // docs/COMPILER.md Step 0). The recording's liveness and the fusion graph stand on
  // this; a binding called write-only that the kernel reads would let a planner alias
  // a live buffer, so the scan must say `rw` wherever it cannot see the whole use.
  const access = bindingAccess(`
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read_write> Acc: array<f32>;
@group(0) @binding(3) var<storage, read_write> Ptr: array<f32>;
@group(0) @binding(4) var<storage, read_write> Cmp: array<f32>;
@group(0) @binding(5) var<uniform> U: vec4<u32>;
@group(0) @binding(6) var<storage, read_write> Nest: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  Out[i] = X[i] * 2.0;
  Acc[i] += X[i];
  let n = arrayLength(&Ptr);
  if (Cmp[i] == 0.0) { Out[i] = 1.0; }
  Nest[select(0u, 1u, X[i] > 0.0)] = X[U.x];
}`);
  want("a read binding and a uniform are reads", access[0] === "r" && access[5] === "r", access.join(","));
  want("a binding only ever assigned is a write", access[1] === "w" && access[6] === "w", access.join(","));
  want("a compound assignment is read-and-write", access[2] === "rw", access.join(","));
  want("a binding taken by address is read-and-write", access[3] === "rw", access.join(","));
  want("a binding compared is a read", access[4] === "rw", access.join(","));

  // **Two defects that answered wrongly with no error** (the 2026-09-24 review), held
  // by value rather than by counters — the pack-cache checks in capture.ts counted
  // repacks after a write that changed nothing, and passed with both defects in.
  {
    const dv = device();
    // A second bias on the same weight: the pack bakes the bias in, and was keyed on the
    // weight alone — `conv2d(x, w, b2)` answered with b1. The shape is one the bias-baking
    // pack takes (`sgfFits`: a row in whole eights, more than 128 channels, so not direct).
    const xin = Tensor.from(Float32Array.from({ length: 16 * 256 * 64 }, (_, i) => Math.sin(i * 0.37)), [16, 256, 8, 8]);
    const w = Tensor.from(Float32Array.from({ length: 256 * 256 * 9 }, (_, i) => Math.cos(i * 0.11) * 0.02), [256, 256, 3, 3]);
    const b2v = Float32Array.from({ length: 256 }, (_, i) => -1 + i * 0.01);
    const b1 = Tensor.from(new Float32Array(256).fill(0.5), [256]);
    const b2 = Tensor.from(b2v, [256]);
    const lookups0 = dv.packHits + dv.packMisses;
    const o1 = await scope(async () => noGrad(() => xin.conv2d(w, b1, 1, 1)).toArray());
    const o2 = await scope(async () => noGrad(() => xin.conv2d(w, b2, 1, 1)).toArray());
    let worst = 0;
    for (let i = 0; i < o1.length; i++) {
      const c = Math.floor(i / 64) % 256;
      worst = Math.max(worst, Math.abs(((o2[i] ?? 0) - (o1[i] ?? 0)) - ((b2v[c] ?? 0) - 0.5)));
    }
    // The same weight at another batch size reuses its pack: the pack is the weight's, not
    // the input's, and it was keyed on the whole convolution — every batch size a page ran
    // kept another weight-sized buffer, for as long as the weight lived (2026-09-24 review).
    const xin2 = Tensor.from(Float32Array.from({ length: 4 * 256 * 64 }, (_, i) => Math.cos(i * 0.21)), [4, 256, 8, 8]);
    const misses0 = dv.packMisses;
    const reused = await scope(async () => noGrad(() => xin2.conv2d(w, b2, 1, 1)).toArray());
    const made = dv.packMisses - misses0;
    // The pack cache serves `noGrad` alone; with the tape on the weight is packed afresh.
    const fresh = await scope(async () => xin2.conv2d(w, b2, 1, 1).toArray());
    let dPack = 0;
    for (let i = 0; i < fresh.length; i++) dPack = Math.max(dPack, Math.abs((reused[i] ?? 0) - (fresh[i] ?? 0)));
    want("eager pack cache: the same weight at another batch size makes no second pack, and answers as a fresh pack does",
      made === 0 && dPack < 1e-5, `${made} packs made · max |Δ| against a fresh pack ${dPack.toExponential(1)}`);
    want("eager pack cache: the same weight with a second bias answers with that bias",
      worst < 1e-3, `max |Δ| ${worst.toExponential(1)} · ${dv.packHits + dv.packMisses - lookups0} pack lookups`);

    // A frozen parameter under AdamW's arena: torch leaves a parameter with no gradient
    // alone — weight, moments and decay. The arena stepped it anyway.
    nn.manualSeed(7);
    const first = new nn.Linear(4, 4), head = new nn.Linear(4, 2);
    first.weight.requiresGrad_(false);
    const opt = new optim.AdamW([...first.parameters(), ...head.parameters()], 1e-2, [0.9, 0.999], 1e-8, 0.1);
    const frozen0 = await first.weight.toArray(), bias0 = await first.bias?.toArray();
    const xs = Tensor.from(Float32Array.from({ length: 32 }, (_, i) => Math.sin(i)), [8, 4]);
    for (let step = 0; step < 3; step++) {
      await scope(async () => {
        opt.zeroGrad();
        head.call(first.call(xs)).sum().backward();
        opt.step();
        await dv.synchronize();
      });
    }
    const frozen1 = await first.weight.toArray(), bias1 = await first.bias?.toArray();
    let moved = 0, biasMoved = 0;
    for (let i = 0; i < frozen0.length; i++) moved = Math.max(moved, Math.abs((frozen1[i] ?? 0) - (frozen0[i] ?? 0)));
    for (let i = 0; i < (bias0?.length ?? 0); i++) biasMoved = Math.max(biasMoved, Math.abs((bias1?.[i] ?? 0) - (bias0?.[i] ?? 0)));
    want("AdamW arena: a frozen weight is not stepped or decayed, and its neighbours still train",
      moved === 0 && biasMoved > 0, `frozen weight moved ${moved.toExponential(1)} · its bias moved ${biasMoved.toExponential(1)}`);
  }

  // **The arena and the per-parameter path hold one state** (2026-09-24 review). The Adam
  // arena kept m and v, and its own copy of the weights, to itself: a switch to the
  // per-parameter path (a second param group) restarted the moments from zero with the
  // bias correction already at step N, and a weight written from outside between steps
  // (loading a checkpoint into the model) was overwritten by the arena's copy. Each case
  // is run twice — arena on, and arena suppressed throughout — and must end equal.
  {
    const dv = device();
    const run = async (arena: boolean, outside: boolean): Promise<Float32Array> => {
      dv.suppressArena = !arena;
      try {
        nn.manualSeed(21);
        const a = new nn.Linear(6, 6), b = new nn.Linear(6, 3);
        const extra = new nn.Linear(3, 3);
        const opt = new optim.Adam([...a.parameters(), ...b.parameters()], 1e-2);
        const xs = Tensor.from(Float32Array.from({ length: 48 }, (_, i) => Math.cos(i)), [8, 6]);
        const stepOnce = async (withExtra: boolean): Promise<void> => {
          await scope(async () => {
            opt.zeroGrad();
            const h = b.call(a.call(xs));
            (withExtra ? extra.call(h) : h).sum().backward();
            opt.step();
            await dv.synchronize();
          });
        };
        for (let i = 0; i < 3; i++) await stepOnce(false);
        if (outside) noGrad(() => { a.weight.copy_(Tensor.from(new Float32Array(36).fill(0.25), [6, 6])); });
        else opt.addParamGroup({ params: [...extra.parameters()] });
        for (let i = 0; i < 2; i++) await stepOnce(!outside);
        const out = await a.weight.toArray();
        const tail = await b.weight.toArray();
        return Float32Array.from([...out, ...tail]);
      } finally {
        dv.suppressArena = false;
      }
    };
    for (const [outside, name] of [[false, "a second param group mid-training"], [true, "a weight written from outside between steps"]] as const) {
      const withArena = await run(true, outside), without = await run(false, outside);
      let d = 0;
      for (let i = 0; i < withArena.length; i++) d = Math.max(d, Math.abs((withArena[i] ?? 0) - (without[i] ?? 0)));
      want(`Adam arena and per-parameter path agree across ${name}`, d < 1e-6, `max |Δ| ${d.toExponential(1)}`);
    }
  }

  // Two more from the 2026-09-24 review. A reflect-padded Conv2d wrapped by LoRA became
  // zero-padded at every border, nothing said — refused now. And tracing an eval model for
  // ONNX runs it with gradients on on purpose: the library's own export said the
  // "eval model under gradient mode" advice to the page that called it.
  {
    wantThrow("LoRAConv2d refuses a Conv2d that pads with reflect — the adapter pads with zeros", "pads with zeros",
      () => peft.LoRAConv2d.fromConv2d(new nn.Conv2d(3, 4, 3, 1, 1, 1, 1, true, "reflect")));
    const wasAdvice = Device.advice;
    Device.advice = false;
    const had = Device.advised.has("eval-grad");
    Device.advised.delete("eval-grad");
    const model = new nn.Sequential(new nn.Linear(4, 3), new nn.ReLU(), new nn.Linear(3, 2));
    model.eval();
    await scope(async () => { onnx.exportOnnx(model, Tensor.randn([1, 4])); });
    want("ONNX export of an eval model does not tell the page it ran an eval model under gradient mode",
      !Device.advised.has("eval-grad"));
    if (had) Device.advised.add("eval-grad");
    Device.advice = wasAdvice;
  }

  const failed = checks.filter((c) => !c.ok);
  const lines = checks.map((c) =>
    `  ${c.ok ? "✓" : "✗"} ${c.name}${c.note ? ` — ${c.note}` : ""}`);
  lines.push(
    failed.length === 0
      ? `all ${checks.length} device-handling checks passed`
      : `**${failed.length} failed** / ${checks.length}`,
  );
  return { text: lines.join("\n"), checks };
}
