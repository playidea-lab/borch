/**
 * The WebGPU device, its buffers and the pipeline cache.
 *
 * ## Why the pipeline cache lives here
 *
 * Baking the shape into the shader is this library's premise (see `kernels.ts`). That
 * makes one operation several shaders as the shapes differ, and compiling on every pass
 * through a layer makes a fast kernel meaningless. So **shape signature → pipeline**
 * enters as a data structure. It is not an optimisation but the price of keeping the
 * speed the baking bought.
 *
 * ## The limits arrive quietly
 *
 * Past a buffer size or a dispatch limit, WebGPU **does not throw; it does not do it.**
 * The bench stepped on this twice — 240,000 GFLOPS above 128MB, and "144%" above 65,535
 * dispatches. Both are numbers that would have been believed if the values had not been
 * looked at, so here the limits are **measured in advance and exceeding one throws.**
 */

import { type Elementwise, grid1d, type Reduce, reduceParts, reduceSum, setConvTilesPreferred, setDirectWeightBytes, type TiledConfig, WORKGROUP } from "./kernels.js";
import { fuseRecords } from "./fuse.js";
import { planRecords, touchesOf } from "./plan.js";

const BYTES_PER_F32 = 4;

/** How many validation errors to print. The first is the cause and the rest are its
 *  wake. */
const MAX_REPORTED_ERRORS = 3;

/**
 * Where a tensor is.
 *
 * Where torch's `'cuda'` and `'cpu'` go. **There is no index** — WebGPU
 * gives no way to enumerate adapters, so there is nothing for `'webgpu:1'`
 * to point at. A string is enough.
 */
export type DeviceKind = "webgpu" | "cpu";

/**
 * How the adapter is chosen. Where torch's `CUDA_VISIBLE_DEVICES` goes.
 */
export interface InitOptions {
  /**
   * It defaults to `"high-performance"` because this is a **library that
   * measures.** The browser default may pick the integrated GPU on a
   * laptop, and then the same code gives different numbers on the same
   * machine — a number where you do not know what was measured.
   */
  powerPreference?: GPUPowerPreference;
  /**
   * Asks for the **software** adapter — Chrome's SwiftShader — rather than a GPU.
   *
   * ## This is the device axis, and it is not the Python one
   *
   * There are two axes here and they get confused because one of them is usually
   * empty:
   *
   * |            | CPU                  | GPU              |
   * |------------|----------------------|------------------|
   * | Python     | `borch` (numpy)      | `borch_webgpu`   |
   * | TypeScript | **this**             | `borch-ts`       |
   *
   * Sending someone to `borch` when their GPU will not come up is answering a
   * **device** question with a **language** one: their TypeScript does not run there.
   * This flag fills the cell, and SwiftShader is what makes it cheap — it is WebGPU's
   * own CPU implementation, so the API, the kernels and the code are the same and only
   * the device changes.
   *
   * ## What it does not change
   *
   * **Nothing was ever refused.** `init()` has always attached to whatever adapter came
   * back, software included — every SwiftShader golden run in this repository is proof
   * of that, and there are a lot of them. So this is not permission; it is a way to
   * **ask on purpose**, and to know from `probe().software` that you got it.
   *
   * The rule that matters is not *do not run on the CPU*. It is **a number measured
   * there must not be read as a GPU's**, and that is kept where it belongs: the
   * benchmark and accuracy runners refuse outright, the site's badge goes dark and says
   * so, and every score line prints the adapter.
   */
  forceFallbackAdapter?: boolean;
}

/**
 * The adapter names that mean **this is the CPU**.
 *
 * WebGPU does not report whether an adapter is software — the only signal is the name,
 * so the list is a list of names. `swiftshader` is Chrome's, `llvmpipe` and `lavapipe`
 * are Mesa's, and `software` catches what spells itself out.
 *
 * **It lives here so there is one copy of it that JavaScript can reach.** The judgement
 * had three homes — this library's callers, `site/assets/home.js` and
 * `site/assets/playground.js` — and three copies of a four-name list is the shape that
 * drifts, quietly, in whichever direction nobody reports. The site imports this now.
 * One copy remains outside, in `tests/browser/launch.py`, because Python cannot import
 * it; `test_the_software_adapter_rule_says_the_same_thing_in_every_copy` holds the two
 * together.
 */
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|software/i;

/**
 * Whether an adapter name is a CPU implementation.
 *
 * Takes the name rather than the adapter, because that is what survives: `probe()`
 * hands back a string, a score line carries a string, and a log read a week later is a
 * string.
 */
export function isSoftwareAdapter(adapter: string): boolean {
  return SOFTWARE.test(adapter);
}

/**
 * Whether WebGPU can be used. **It answers with a value, not an
 * exception.**
 *
 * `why` is what makes it worth having — `no-api` (the browser is too old,
 * or this is not a secure context) and `no-adapter` (driver blocklist,
 * virtual machine, headless with no GPU) leave the user with entirely
 * different things to do, and folding them into one exception erases that
 * split.
 */
export type Availability =
  | { ok: true; adapter: string; software: boolean }
  | { ok: false; why: "no-api" | "no-adapter"; message: string };

// **Naming the version is not enough.** Somebody received this message on Safari 18.6,
// already on 18+, on localhost, in a secure context — they had done everything the
// message tells you to do and got the same message. Then they go and check the browser
// version, learn only that it is not that, and come back **still not knowing what to do
// next.**
//
// On that Safari the remaining cause was the feature flag being off. Guidance is usually
// saying something true and then becomes **wrong for exactly one person**, and that one
// person is the one reading it. So the place to switch it on is written out.
// **The message a visitor sees when their browser has none, so it has to name their
// browser.** It said "Chrome/Edge 113+ or Safari 18+" and was wrong twice: Firefox has
// had WebGPU since 141 and was not named at all, so a Firefox reader was told to go and
// get another browser; and Safari turned it on by default in 26, not 18 — 18 through 25
// have it behind the flag this names. Versions from MDN's browser-compat-data,
// `api/GPU.json`, read 2026-09-10; the same table is on the setup page, and
// `test_site.py` holds the two to each other.
// **No markdown in a runtime message.** The asterisks around one clause were written for
// a reader of this file and reached a visitor's screen as characters — Safari 18 on the
// Korean home page, 2026-09-12. Nothing that renders this is a markdown renderer.
const NO_API =
  "There is no WebGPU here. It is in Chrome and Edge from 113, Firefox from 141, and " +
  "Safari from 26 — Safari 18 to 25 have it behind Settings → Advanced → Feature Flags → " +
  "WebGPU. Seeing this on a version that has it means it is switched off: on Linux " +
  "Chrome, Unsafe WebGPU in chrome://flags. It has to be https or localhost.";

const NO_ADAPTER =
  "No WebGPU adapter could be obtained — a driver blocklist, a virtual machine, or a " +
  "headless environment with no GPU.";

/** The work the kick calibration waits on: one dispatch of a loop kernel, this many
 *  iterations a thread — 0.12 ms of GPU on the RTX 5080, about 2 ms on metal-3. Forty
 *  tiny dispatches (0.08 ms) were tried first and the 5080 sometimes finished them
 *  before the GPU process's first look, which is the one wait that does not stall; a
 *  calibration of three said "no kicks" that way. Work the first look cannot catch
 *  always meets the stall where there is one. */
const KICK_PROBE_ITERS = 10000;
const KICK_PROBE_WORKGROUPS = 1024;
/** Plain waits in a row before any is measured — six, because a fresh GPU ramps its
 *  clock over its first waits (metal-3 ran the 2 ms kernel at 15 ms on the first, and a
 *  "slowest plain" rule turned kicks on there). Then `KICK_PROBE_PLAIN_REPS` plain waits
 *  measured, the median; then `KICK_PROBE_KICKED_REPS` kicked, the median. **Not
 *  interleaved**: a kicked wait resets the GPU process's polling, and the plain wait
 *  after it is the fast one (the 5080 read 0.9 ms interleaved against 2.5–3.0 in a row,
 *  and said "no kicks"). The stall is what a run of plain waits meets, so that is what
 *  is measured. */
const KICK_PROBE_WARMUPS = 6;
const KICK_PROBE_PLAIN_REPS = 5;
const KICK_PROBE_KICKED_REPS = 3;
/** The decision. With `timestamp-query`: the plain median **minus the kernel's GPU time**
 *  above `KICK_PROBE_FLOOR_MS` is the stall — on the 5080 2.1–3.0 ms of wall for 0.12 of
 *  GPU; on metal-3 the wall is the kernel plus 0.3–1.2 (the floor sits between the two,
 *  1.7: half a millisecond above metal-3's worst, a third under the 5080's best). This
 *  does not depend on how fast a
 *  kick is, which the ratio below does: three calibrations on the 5080 read the kicked
 *  median at 0.6–2.0 ms (a kick is 0.03 when the GPU process is idle and more when it is
 *  not) and said "no kicks" by the ratio while the plain waits stalled in plain sight.
 *  Without timestamps, the ratio: plain median this many times the kicked median and
 *  above the floor. Idle gaps between the waits were tried and dropped — after 4 ms of
 *  idle metal-3 pays ~0.7 ms of wake-up that kicks do not remove (`roundtrip:probe`, the
 *  idle rows), and a first gapped calibration read 9 ms there and turned kicks on, where
 *  they cost the eager forward a third. */
const KICK_PROBE_RATIO = 3;
const KICK_PROBE_FLOOR_MS = 1.7;
/** Rounds of the calibration, each after an idle of `KICK_PROBE_SETTLE_MS`; the stall
 *  in **any** round decides. The GPU process's polling has phases — right after
 *  `requestDevice`, after an idle, after a flush from elsewhere on the page — in which
 *  a run of plain waits is fast on work that stalls the rest of the time (the 5080 read
 *  0.23 ms for 0.12 of GPU in one calibration and 2.2–3.0 in the next four, on the same
 *  kernel), and one round in a phase says "no kicks" on a card that needs them. metal-3
 *  never shows the stall in any round (0.3–1.2 ms over the GPU across every calibration
 *  measured), so "any round" is one-sided the right way. */
const KICK_PROBE_ROUNDS = 4;
const KICK_PROBE_SETTLE_MS = 40;

/**
 * **Does this browser notice a finished fence on its own?** A short loop kernel behind
 * a 4-byte copy, mapped and waited for plainly in a row with a timestamp query around
 * it, then with the wire kicked by error-scope round trips until the map resolves; the
 * plain median against the kernel's GPU time, or against the kicked median where there
 * are no timestamps. See `Device.readbackKicks` for what it found and why it is
 * measured.
 */
async function calibrateKicks(device: GPUDevice, canTime: boolean): Promise<boolean> {
  const module = device.createShaderModule({ code:
    "@group(0) @binding(0) var<storage, read_write> X: array<f32>;\n" +
    "@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n" +
    `  var a = f32(g.x); for (var i = 0u; i < ${KICK_PROBE_ITERS}u; i = i + 1u) { a = a * 0.999 + 0.5; }\n` +
    "  X[g.x] = a; }" });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const buffer = device.createBuffer({
    size: KICK_PROBE_WORKGROUPS * 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
  // The staging buffer carries four bytes of the result and, with timestamps, the two
  // query values behind them — one map reads both.
  const STAGE_BYTES = 8 + 2 * 8;
  const stage = device.createBuffer({ size: STAGE_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const querySet = canTime ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const resolved = canTime ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const submit = (): void => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(querySet
      ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
      : {});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(KICK_PROBE_WORKGROUPS);
    pass.end();
    encoder.copyBufferToBuffer(buffer, 0, stage, 0, 4);
    if (querySet && resolved) {
      encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, stage, 8, 16);
    }
    device.queue.submit([encoder.finish()]);
  };
  const gpuTimes: number[] = [];
  const wait = async (kicks: boolean): Promise<number> => {
    const t0 = performance.now();
    submit();
    let done = false;
    const mapped = stage.mapAsync(GPUMapMode.READ).then(() => { done = true; });
    while (kicks && !done) {
      device.pushErrorScope("validation");
      await device.popErrorScope();
    }
    await mapped;
    const wall = performance.now() - t0;
    if (querySet && !kicks) {
      const q = new BigUint64Array(stage.getMappedRange(8, 16).slice(0));
      const start = q[0];
      const end = q[1];
      if (start !== undefined && end !== undefined && end > start) gpuTimes.push(Number(end - start) / 1e6);
    }
    stage.unmap();
    return wall;
  };
  const sample = async (kicks: boolean, reps: number): Promise<number[]> => {
    const t: number[] = [];
    for (let i = 0; i < reps; i++) t.push(await wait(kicks));
    return t;
  };
  const median = (t: number[]): number => t.sort((a, b) => a - b)[t.length >> 1] ?? 0;
  const stalls = (plain: number, gpu: number, kicked: number): boolean => gpu >= 0
    ? plain - gpu > KICK_PROBE_FLOOR_MS
    : plain > KICK_PROBE_FLOOR_MS && plain > KICK_PROBE_RATIO * kicked;
  await wait(false);                       // compile the pipeline, unmeasured
  let verdict = false;
  let worst = { plainMs: 0, kickedMs: 0, gpuMs: -1 };
  for (let round = 0; round < KICK_PROBE_ROUNDS; round++) {
    await new Promise((resolve) => setTimeout(resolve, KICK_PROBE_SETTLE_MS));
    await sample(false, KICK_PROBE_WARMUPS);
    gpuTimes.length = 0;                   // the warm-ups' GPU times carry the clock ramp
    const plain = median(await sample(false, KICK_PROBE_PLAIN_REPS));
    const gpu = gpuTimes.length > 0 ? median(gpuTimes) : -1;
    const kicked = median(await sample(true, KICK_PROBE_KICKED_REPS));
    if (plain - Math.max(gpu, 0) > worst.plainMs - Math.max(worst.gpuMs, 0)) worst = { plainMs: plain, kickedMs: kicked, gpuMs: gpu };
    if (stalls(plain, gpu, kicked)) { verdict = true; break; }
  }
  buffer.destroy();
  stage.destroy();
  resolved?.destroy();
  querySet?.destroy();
  Device.kickCalibration = worst;
  return verdict;
}

/**
 * **The re-tiled scalar GEMM's configurations for an adapter, best first** (`docs/GEMM.md`
 * Step 4). `matmul` takes the first that fits the shape (`tiledConfigFits`) and the
 * device's workgroup storage; a shape none fits stays on the tile as it was. Measured,
 * not chosen (`kernel_bench mm --sweep=gemm`, 2026-09-21), and **the same answer on both
 * adapters**, which is why one list serves every vendor:
 * - the 8 × 4 micro-tile on a 128 × 64 tile with `vec4` staging — metal-3 1.46× the old
 *   tile on 2048³ (3.55 → 2.43 ms), the RTX 5080 1.21× (0.812 → 0.671, 25.6 TFLOP/s) and
 *   1.33× on 1024³ and on the deep ResNet GEMM shapes (0.084 → 0.063);
 * - then 64 × 64 r4×4 `vec4` for shapes the first does not divide — 1.34× on metal-3,
 *   1.20× on the 5080, on its own.
 * The 8 × 8 micro-tile lost on both (Apple's register file; on the 5080 0.109 + 0.011
 * against 0.102 at 1024³ and 0.807 against 0.671 at 2048³), and double buffering lost
 * on both (+3–10 % on metal-3, +20–50 % on the 5080). A vendor this was not measured on
 * gets the same list: the old tile stays for every shape the list does not divide.
 */
function gemmConfigsFor(_vendor: string): readonly TiledConfig[] {
  // A bench or a bisection may switch the re-tiled GEMM off; nothing else sets this.
  if ((globalThis as { BORCH_NO_RETILE?: boolean }).BORCH_NO_RETILE) return [];
  // **Direct3D 12 is the third answer** (`docs/GEMM.md` ledger, 2026-09-21, an RTX 5050
  // Laptop through Chrome on Windows 11): there the 8 × 8 micro-tile on a 128 × 128 tile
  // wins — 2048³ 6.35 → 3.60 ms (1.77×) against the 8 × 4's 4.88 and the plain vec4's
  // 5.56; the deep ResNet GEMM shapes 0.43 → 0.26. The compiler under D3D12 keeps
  // sixty-four accumulators in registers where Metal's and Vulkan's do not. The API is
  // not in `GPUAdapterInfo`; Chrome on Windows reaches WebGPU through D3D12 by default,
  // so the platform stands for it.
  if (isWindows()) {
    return [
      { TM: 128, TN: 128, RM: 8, RN: 8, KT: 16, vec4: true, dbuf: false },
      { TM: 128, TN: 64, RM: 8, RN: 4, KT: 16, vec4: true, dbuf: false },
      { TM: 64, TN: 64, RM: 4, RN: 4, KT: 16, vec4: true, dbuf: false },
    ];
  }
  return [
    { TM: 128, TN: 64, RM: 8, RN: 4, KT: 16, vec4: true, dbuf: false },
    { TM: 64, TN: 64, RM: 4, RN: 4, KT: 16, vec4: true, dbuf: false },
  ];
}

/** Whether the page runs on Windows — where Chrome's WebGPU is Direct3D 12. Read from
 *  `userAgentData.platform` where it exists, `navigator.platform` otherwise. */
function isWindows(): boolean {
  const nav = globalThis.navigator as (Navigator & { userAgentData?: { platform?: string } }) | undefined;
  const platform = nav?.userAgentData?.platform ?? nav?.platform ?? "";
  return /^win/i.test(platform);
}

/** Which adapter, on one line. Empty fields are dropped — the browser hides most of
 *  them. */
function describe(adapter: GPUAdapter): string {
  const info: Partial<GPUAdapterInfo> = adapter.info ?? {};
  // **A repeat is not a field.** Chrome fills these with four different things and the
  // badge reads `apple / metal-3`; Safari 26 answers `apple` to all four and the badge
  // read `apple / apple / apple / apple` (measured, 2026-09-12). Saying the same word
  // four times is not more information than saying it once.
  const said = [info.vendor, info.architecture, info.device, info.description].filter(Boolean);
  return [...new Set(said)].join(" / ") || "(unknown)";
}

function askAdapter(options: InitOptions): Promise<GPUAdapter | null> {
  return navigator.gpu.requestAdapter({
    powerPreference: options.powerPreference ?? "high-performance",
    forceFallbackAdapter: options.forceFallbackAdapter ?? false,
  });
}

/**
 * The adapter the last `probe()` obtained, kept for the `init()` that follows it.
 *
 * **On Linux with the NVIDIA driver, every `requestAdapter` after the first costs one
 * to three seconds** — measured on an RTX 5080 (driver 580, Chrome 151): the page's
 * probe at load took 14 ms and the click's second request 2,953 ms, and on a revisit
 * both requests were slow. Apple answers both in tens of milliseconds, which is why
 * nobody saw it. A page that probes and then inits asked twice for the same thing;
 * now the probe's adapter is held and `Device.create` consumes it when the options
 * match, so the second request never happens.
 *
 * Held, not cached: a WebGPU adapter is consumed by its first `requestDevice`, and it
 * can go stale on its own, so the holder is cleared once used and `create()` falls back
 * to a fresh request if the held one refuses.
 */
// One per option set: a probe for the software adapter must not evict the one a
// probe for the GPU obtained — the device test does exactly that sequence.
const held = new Map<string, GPUAdapter>();

function optionsKey(options: InitOptions): string {
  return `${options.powerPreference ?? "high-performance"}|${options.forceFallbackAdapter ?? false}`;
}

async function adapterFor(options: InitOptions): Promise<GPUAdapter | null> {
  const key = optionsKey(options);
  const kept = held.get(key);
  if (kept) {
    held.delete(key);
    return kept;
  }
  return askAdapter(options);
}

/**
 * Asks whether it could attach, without attaching. It does not create a
 * device.
 *
 * **It does not stand in for `init()`** — `requestDevice` can still refuse
 * after this passes, and that still arrives as an exception from `init()`.
 * What this function answers reaches as far as "is there an adapter", and
 * since most of what actually blocks sits before that, it is worth having
 * on its own.
 */
export async function probe(options: InitOptions = {}): Promise<Availability> {
  // **The question is whether there is an API to call, not whether a key exists.** This
  // asked `"gpu" in navigator`, which is true of a property defined as undefined — and
  // then `askAdapter` read `.requestAdapter` off it and the visitor got
  // `Cannot read properties of undefined` where the sentence above was meant to go.
  // Safari with the flag off does remove the property, so this was only ever wrong for
  // the shapes a policy or an extension leaves; it is one test for all of them now.
  if (!navigator.gpu) return { ok: false, why: "no-api", message: NO_API };
  const adapter = await askAdapter(options);
  if (!adapter) return { ok: false, why: "no-adapter", message: NO_ADAPTER };
  held.set(optionsKey(options), adapter);
  const name = describe(adapter);
  // **`ok` and `software` are two answers, not one.** Folding them together is the
  // mistake this repository spent a day undoing at a larger scale: a software run is a
  // real run whose values are real, and calling it "not ok" would refuse work that
  // works. What it is not is a GPU's number, and that is what this field says.
  return { ok: true, adapter: name, software: isSoftwareAdapter(name) };
}

/**
 * Asks only whether WebGPU can be used. Where `torch.cuda.is_available()`
 * goes.
 *
 * **Unlike torch's, it is async** — obtaining an adapter is asynchronous
 * and there is no way around it. If you need to know why not, use
 * `probe()`.
 */
export async function isAvailable(options: InitOptions = {}): Promise<boolean> {
  return (await probe(options)).ok;
}

/** Numbers the shader's lines so the line an error names can be found. */
/**
 * Whether to ask each shader module how its compile went.
 *
 * Set `globalThis.BORCH_SHADER_DIAGNOSTICS = true` before building anything.
 * Read per call rather than cached so it can be switched on from a console
 * mid-session, which is exactly when someone wants it.
 */
function shaderDiagnostics(): boolean {
  return (globalThis as { BORCH_SHADER_DIAGNOSTICS?: boolean })
    .BORCH_SHADER_DIAGNOSTICS === true;
}

function numbered(code: string): string {
  return code
    .split("\n")
    .map((line, i) => `${String(i + 1).padStart(3)} | ${line}`)
    .join("\n");
}

interface SubgroupMatrixConfig {
  readonly componentType: string;
  readonly resultComponentType: string;
  readonly M: number;
  readonly N: number;
  readonly K: number;
}

/** The int8 subgroup-matrix configuration an adapter offers — `i8 × i8 → i32` at
 *  16 × 16 × 32 — or `null`. What the RTX 5080 through Chrome 151 / Vulkan has *instead
 *  of* the f32 one (`docs/INT8.md`); read from the configurations, never from the
 *  feature name. */
export interface SubgroupInt8Config { readonly M: number; readonly N: number; readonly K: number }
function subgroupMatrixInt8(adapter: GPUAdapter): SubgroupInt8Config | null {
  if (!adapter.features.has("chromium-experimental-subgroup-matrix" as GPUFeatureName)
    || !adapter.features.has("subgroups" as GPUFeatureName)) return null;
  const info = adapter.info as unknown as { subgroupMatrixConfigs?: Iterable<SubgroupMatrixConfig> };
  for (const c of info.subgroupMatrixConfigs ?? []) {
    if (c.componentType === "i8" && c.resultComponentType === "i32" && c.M === 16 && c.N === 16 && c.K === 32) return { M: c.M, N: c.N, K: c.K };
  }
  return null;
}

/** Whether the adapter offers subgroup matrices with the f32 8 × 8 × 8 configuration. */
function subgroupMatrixF32(adapter: GPUAdapter): boolean {
  if (!adapter.features.has("chromium-experimental-subgroup-matrix" as GPUFeatureName)
    || !adapter.features.has("subgroups" as GPUFeatureName)) return false;
  const info = adapter.info as unknown as { subgroupMatrixConfigs?: Iterable<SubgroupMatrixConfig> };
  for (const c of info.subgroupMatrixConfigs ?? []) {
    if (c.componentType === "f32" && c.resultComponentType === "f32" && c.M === 8 && c.N === 8 && c.K === 8) return true;
  }
  return false;
}

/**
 * One dispatch as recorded under a capture — or one buffer copy: every in-place
 * operation is a copy back into the original buffer (`copyInto`), and an optimizer's
 * weight decay is one. Left out of the recording, a replay silently skipped them: AdamW
 * trained a GPT to a loss 3e-6 away from eager while Adam's U-Net replayed bit for bit
 * (measured 2026-09-07). A copy has `copy` set and `buffers` as `[src, dst]`.
 */
/** What a dispatch does to one binding: reads it, writes it, or both. */
export type Access = "r" | "w" | "rw";

/** One way of running a kernel a rule chooses among — see `Device.choose`. `keys` are the
 *  pipeline keys its `run` dispatches under, which the profiler files times by. */
export interface TuneCandidate { readonly label: string; readonly keys: readonly string[]; readonly run: () => void }

/** Thrown by `Device.pipeline` while the tuner is pre-warming: the pipeline is being made
 *  asynchronously and the candidate is to be run again once every pending one is ready. */
class PrewarmMiss extends Error {
  constructor(signature: string) { super(`pipeline ${signature} compiling`); }
}

/** One decision of `Device.runTuning`. */
export interface TuneReport {
  readonly key: string; readonly prior: string; readonly priorMs: number;
  readonly chosen: string; readonly chosenMs: number; readonly candidates: readonly string[];
}

export interface Recorded {
  readonly pipeline?: GPUComputePipeline;
  readonly bindGroup?: GPUBindGroup;
  readonly copy?: { readonly bytes: number; readonly srcOff?: number; readonly dstOff?: number };
  readonly groups: readonly [number, number, number];
  /** The buffers behind the bind group, in binding order — what a fusion pass reads. */
  readonly buffers: readonly BindSlot[];
  /** For an elementwise dispatch, what it computes — see `Elementwise`; for a reduction,
   *  what it reads once — see `Reduce`. */
  readonly meta?: Elementwise | Reduce;
  /**
   * Per binding, what the kernel does to it — read off the WGSL when the pipeline was
   * built (`bindingAccess`). A recipe (`meta`) says the same more precisely for the
   * kernels that carry one; every other kernel used to be taken to read and write
   * everything it binds, which made `liveIns` and the fusion graph conservative by
   * exactly that much (`docs/COMPILER.md` Step 0). Absent only for a pipeline this
   * device did not build.
   */
  readonly access?: readonly Access[];
  /**
   * A window refill: the host bytes a streamed weight was placed from, written into
   * `buffers[0]` (its slot) again on every replay. Not a dispatch and not a copy the
   * device can re-encode — the bytes come from the host through a staging map, which is
   * why a recording with refills replays through `replayAsync`. `docs/COMPILER.md` Step 7.
   */
  readonly refill?: { readonly win: Window; readonly data: Float32Array | Uint16Array | Uint32Array };
  /** The pipeline's signature at the time of recording — what the profiler files a
   *  replayed dispatch under. Without it a replay is one kind: the last signature set. */
  readonly sig?: string;
}

/**
 * What a dispatch binds at one slot: a whole buffer, or a `{buffer, offset, size}`
 * sub-range of one. The **object itself is the identity** the fusion pass and the
 * capture's live-in analysis key on — a whole buffer is its own key (as it always was),
 * and two tensors sharing one arena buffer at different offsets are two different keys
 * because they are two different objects. A tensor caches its slot so the key is stable.
 */
export type BindSlot = GPUBuffer | { readonly buffer: GPUBuffer; readonly offset: number; readonly size: number };

/** The GPUBuffer behind a slot — for a usage or pool check, never for identity. */
export function bufOf(s: BindSlot): GPUBuffer {
  return s instanceof GPUBuffer ? s : s.buffer;
}

const BINDING_DECL = /@group\(0\)\s*@binding\((\d+)\)\s*var<(storage|uniform)(?:\s*,\s*(read|read_write))?>\s*([A-Za-z_][A-Za-z0-9_]*)/g;

/**
 * What a kernel does to each of its bindings, read off its WGSL. `var<uniform>` and
 * `var<storage, read>` are reads. A `read_write` binding is a write when every use of
 * its name in the body is an assignment (`Name[i] = …`) — the common output — and
 * read-and-write when any use is a read, a compound assignment (`+=`), or the name is
 * taken by address (`&Name`: a pointer, an atomic, `arrayLength`), where the scan
 * cannot follow it and says the conservative thing. Never less than the kernel does.
 */
export function bindingAccess(code: string): Access[] {
  const out: Access[] = [];
  const bodyStart = code.indexOf("fn main");
  const body = bodyStart >= 0 ? code.slice(bodyStart) : code;
  for (const m of code.matchAll(BINDING_DECL)) {
    const index = Number(m[1]);
    const name = m[4] as string;
    if (m[2] === "uniform" || m[3] === "read") { out[index] = "r"; continue; }
    out[index] = writeOnly(body, name) ? "w" : "rw";
  }
  return out;
}

/** True when every use of `name` in `body` is a plain assignment to an element of it. */
function writeOnly(body: string, name: string): boolean {
  const uses = new RegExp(`(^|[^A-Za-z0-9_.])(&?)${name}(?![A-Za-z0-9_])`, "g");
  for (const m of body.matchAll(uses)) {
    if (m[2] === "&") return false;                       // a pointer — cannot follow
    let i = (m.index ?? 0) + m[0].length;
    if (body[i] !== "[") return false;                    // whole-array use (a copy, a length)
    let depth = 0;
    for (; i < body.length; i++) {                        // skip the index expression
      const c = body[i];
      if (c === "[") depth++;
      else if (c === "]") { depth--; if (depth === 0) { i++; break; } }
    }
    while (i < body.length && (body[i] === " " || body[i] === "\t")) i++;
    // `= ` assigns; `==`, `+=`, `-=`, `*=`, `/=` and anything else reads.
    if (body[i] !== "=" || body[i + 1] === "=") return false;
  }
  return true;
}

/**
 * A recorded step. `replay()` issues its dispatches again, in order, into the current
 * batch — the values land in the same buffers, so a tensor made during the capture (the
 * loss, the parameters) reads the new step's result. `dispose()` returns the memory.
 */
export class Capture {
  constructor(
    private readonly dev: Device,
    private records: readonly Recorded[],
    private readonly pinned: Set<GPUBuffer>,
    /** The buffers `upload` made under the capture — a tensor from the CPU: an input
     *  copied in, a constant, an optimizer's counter made on its first step. */
    private readonly uploaded: Set<GPUBuffer> = new Set(),
  ) {}

  /** How many dispatches one replay issues. */
  get dispatches(): number {
    return this.records.length;
  }

  /**
   * The recording as a list: each dispatch's pipeline key, its grid, and its buffers as
   * small integers (the same buffer → the same number). What a fusion pass, or a person
   * asking where the dispatches go, reads.
   */
  describe(): { key: string; groups: readonly [number, number, number]; buffers: number[]; sizes: number[] }[] {
    const ids = new Map<BindSlot, number>();
    const id = (b: BindSlot): number => {
      let n = ids.get(b);
      if (n === undefined) { n = ids.size; ids.set(b, n); }
      return n;
    };
    return this.records.map((r) => ({
      key: r.copy ? "copy" : r.refill ? "refill" : (r.pipeline && this.dev.keyOf(r.pipeline)) || "?", groups: r.groups,
      buffers: r.buffers.map(id), sizes: r.buffers.map((b) => bufOf(b).size),
    }));
  }

  replay(): void {
    if (this.hasRefills) throw new Error("this recording refills a window from the host — replay it with replayAsync()");
    this.dev.replayRecorded(this.records);
  }

  /** Whether the recording carries window refills (`Recorded.refill`). */
  get hasRefills(): boolean {
    return this.records.some((r) => r.refill !== undefined);
  }

  /**
   * The replay for a recording that streams: the dispatches between two refills are
   * encoded as one run and submitted, then the refill writes the block's bytes into the
   * same slot it had — the queue keeps submit order, so the dispatches already submitted
   * read the old bytes and the ones after read the new — and the walk goes on. The
   * window's own bookkeeping is not touched: the slots a replay writes are the slots the
   * recording chose, and they were free then exactly as they are now.
   */
  async replayAsync(): Promise<void> {
    let from = 0;
    for (let i = 0; i < this.records.length; i++) {
      const r = this.records[i] as Recorded;
      if (!r.refill) continue;
      if (i > from) this.dev.replayRecorded(this.records.slice(from, i));
      this.dev.flush();
      // eslint-disable-next-line no-await-in-loop
      await r.refill.win.refill(r.buffers[0] as BindSlot, r.refill.data);
      from = i + 1;
    }
    if (from < this.records.length) this.dev.replayRecorded(this.records.slice(from));
  }

  /**
   * How well the recording knows what its dispatches touch — the gate of
   * `docs/COMPILER.md` Step 0. `exact`: a recipe says which binding is read and which
   * written. `declared`: read off the kernel's WGSL. `guessed`: neither — taken to read
   * and write everything it binds. `copies` are exact by construction. `readWrite` counts
   * the declared bindings the scan could not call write-only (a pointer, a compound
   * assignment) — conservative, and a place to look when a planner wants more.
   */
  coverage(): { dispatches: number; exact: number; declared: number; guessed: number; copies: number; readWrite: number; refills: number } {
    let exact = 0, declared = 0, guessed = 0, copies = 0, readWrite = 0, refills = 0;
    for (const r of this.records) {
      if (r.copy) copies++;
      else if (r.refill) refills++;
      else if (r.meta) exact++;
      else if (r.access) { declared++; for (const a of r.access) if (a === "rw") readWrite++; }
      else guessed++;
    }
    return { dispatches: this.records.length, exact, declared, guessed, copies, readWrite, refills };
  }

  /**
   * What a horizontal fusion pass would find — **counted before any pass is written**
   * (`docs/COMPILER.md` Step 3). A run is consecutive elementwise dispatches with the same
   * recipe (expression, prelude, operand count and layout — the element count may differ)
   * and no dependence between them: none writes what another in the run reads or writes.
   * `buffersMax` is the most distinct buffers one run binds; a kernel that merged the run
   * would bind them all, and the device allows ten storage buffers a stage.
   */
  horizontal(): { runs: number; dispatches: number; largest: number; buffersMax: number } {
    let runs = 0, dispatches = 0, largest = 0, buffersMax = 0;
    let key: string | null = null;
    let len = 0;
    const reads = new Set<GPUBuffer>(), writes = new Set<GPUBuffer>(), bufs = new Set<GPUBuffer>();
    const close = (): void => {
      if (len >= 2) { runs++; dispatches += len; largest = Math.max(largest, len); buffersMax = Math.max(buffersMax, bufs.size); }
      key = null; len = 0; reads.clear(); writes.clear(); bufs.clear();
    };
    for (const r of this.records) {
      const m = r.meta;
      if (!m || !("expr" in m)) { close(); continue; }
      const k = `${m.prelude ?? ""}|${m.expr}|${m.inputs.map((i) => `${i.binding}:${i.strides ? "s" : "c"}`).join(",")}|${m.out}`;
      const rIn = m.inputs.map((i) => bufOf(r.buffers[i.binding] as BindSlot));
      const rOut = bufOf(r.buffers[m.out] as BindSlot);
      const dependent = rIn.some((b) => writes.has(b)) || writes.has(rOut) || reads.has(rOut);
      if (k !== key || dependent) close();
      key = k; len++;
      for (const b of rIn) { reads.add(b); bufs.add(b); }
      writes.add(rOut); bufs.add(rOut);
    }
    close();
    return { runs, dispatches, largest, buffersMax };
  }

  /**
   * The recording as a person reads it: one line a dispatch — its index, the pipeline's
   * key, the workgroup grid, the buffers it binds as small ids with their bytes — and,
   * given a profile (`Device.nsByKind` after a replay run under `profile`), the GPU time
   * of that kind divided among its dispatches here. `head` limits the lines.
   */
  explain(ns?: ReadonlyMap<string, number>, head = Infinity): string {
    const rows = this.describe();
    const count = new Map<string, number>();
    for (const r of this.records) if (r.sig) count.set(r.sig, (count.get(r.sig) ?? 0) + 1);
    const lines: string[] = [];
    rows.forEach((row, i) => {
      if (i >= head) return;
      const rec = this.records[i] as Recorded;
      const bufs = row.buffers.map((id, k) => `${id}:${Math.round((row.sizes[k] as number) / 1024)}K`).join(" ");
      const time = ns && rec.sig && ns.has(rec.sig)
        ? `  ${((ns.get(rec.sig) as number) / 1e6 / (count.get(rec.sig) ?? 1)).toFixed(3)} ms` : "";
      lines.push(`#${String(i).padStart(4)}  ${row.key.slice(0, 48).padEnd(48)}  grid ${row.groups.join("×").padEnd(12)}  [${bufs}]${time}`);
    });
    if (rows.length > head) lines.push(`… ${rows.length - head} more`);
    return lines.join("\n");
  }

  /**
   * The buffers the recording reads before it writes them — the step's inputs and its
   * state: parameters, the optimizer's moments and counters, running statistics. A
   * replay starts from what they hold; snapshot them and the step can be run again from
   * the same place. A dispatch without a recipe is taken to read every buffer it binds,
   * so activations such a kernel writes are counted too — more than needed, never less.
   */
  liveIns(): GPUBuffer[] {
    const { live } = this.flows();
    return [...live].map(bufOf).filter((b) => this.external(b));
  }

  /**
   * Whether the recording writes any buffer that is not its own — a parameter, an
   * optimizer's state, a batch norm's running statistic. A recording that does not is a
   * pure function of its inputs (an inference forward) and can be made again without
   * the world moving; one that does is a step, and is made once.
   */
  mutatesState(): boolean {
    const { written } = this.flows();
    return [...written].map(bufOf).some((b) => this.external(b));
  }

  /** A buffer the recording did not make (or made as an upload) and can read back. */
  private external(b: GPUBuffer): boolean {
    return (!this.pinned.has(b) || this.uploaded.has(b)) && (b.usage & GPUBufferUsage.COPY_SRC) !== 0;
  }

  /** The buffers read before any write in the recording (`live`), and every written one. */
  private flows(): { live: Set<BindSlot>; written: Set<BindSlot> } {
    const written = new Set<BindSlot>();
    const live = new Set<BindSlot>();
    for (const r of this.records) {
      const reads: BindSlot[] = [];
      const writes: BindSlot[] = [];
      if (r.copy) {
        reads.push(r.buffers[0] as BindSlot); writes.push(r.buffers[1] as BindSlot);
      } else if (r.refill) {
        writes.push(r.buffers[0] as BindSlot);
      } else if (r.meta && "expr" in r.meta) {
        for (const inp of r.meta.inputs) reads.push(r.buffers[inp.binding] as BindSlot);
        writes.push(r.buffers[r.meta.out] as BindSlot);
      } else if (r.meta && "input" in r.meta) {
        const input = r.meta.input;
        r.buffers.forEach((b, k) => { if (k === input) reads.push(b); else writes.push(b); });
      } else if (r.access) {
        r.buffers.forEach((b, k) => {
          const a = r.access?.[k] ?? "rw";
          if (a !== "w") reads.push(b);
          if (a !== "r") writes.push(b);
        });
      } else {
        reads.push(...r.buffers); writes.push(...r.buffers);
      }
      for (const b of reads) if (!written.has(b)) live.add(b);
      for (const b of writes) written.add(b);
    }
    // A buffer allocated under the capture is the step's own — an activation, a
    // gradient — unless it was uploaded from the CPU: an input, a constant, a counter
    // an optimizer made on its first step. The step's own buffers are written by the
    // recording and count for nothing; a kernel without a recipe would have counted
    // them as read (measured: the q, k, v slices an attention cuts from its weight,
    // rewritten by the replay and left alone by an eager rerun, read as differences).
    // A buffer that cannot be read back — the one-word offsets a gather carries — is a
    // constant of the recording, not state. `external` says which.
    return { live, written };
  }

  /**
   * Merges elementwise dispatches that feed each other into single kernels, and into the
   * reductions that read them — see `fuse.ts`. The recording is rewritten in place;
   * `replay` runs the fused list. Returns how many dispatches there were and are, and
   * how many intermediates the fused kernels leave unwritten.
   *
   * `held` — the buffers of every tensor the caller still holds. Given, an intermediate
   * made with autograd off that nobody holds and nothing later reads is never written;
   * left out, only autograd's own intermediates are.
   */
  fuse(held?: Iterable<GPUBuffer>): { before: number; after: number; fused: number; unwritten: number } {
    const before = this.records.length;
    const { records, fused, unwritten } = fuseRecords(this.dev, this.records, held ? new Set(held) : undefined);
    this.records = records;
    return { before, after: records.length, fused, unwritten };
  }

  /** Buffers a hoisted dispatch wrote — constants of the recording now, never moved by
   *  the plan and never released before `dispose`. */
  private readonly frozen = new Set<GPUBuffer>();

  /**
   * **Runs the replay-invariant dispatches once and takes them out of the replay.** A
   * dispatch whose every read is a constant of the recording — a buffer no record writes,
   * not an input uploaded under the capture (the caller rewrites those before each
   * replay), not a window slot (refilled) — and whose writes are pure (no binding read
   * and written) produces the same bytes on every replay; it ran when the step was
   * recorded, its outputs are pinned for the capture's life, and a replay has no reason
   * to run it again. Found to a fixed point, so a chain of such dispatches hoists whole.
   *
   * What it is in practice: the eval forward's weight repacks (`tmw`, the subgroup
   * conv's tap-major copy of a frozen weight — 0.7 ms a forward on ResNet-18's 512-channel
   * layer, measured), a padded constant, a block of ones. In a training step the weights
   * are written by the optimizer and nothing is hoisted — the fixed point finds that on
   * its own. `docs/INFER.md` Step 2 and `docs/COMPILER.md`. Run after `fuse`, before
   * `plan`.
   */
  hoist(): { hoisted: number; bytes: number } {
    const refilled = new Set<GPUBuffer>();
    for (const r of this.records) if (r.refill) refilled.add(bufOf(r.buffers[0] as BindSlot));
    const pure = (r: Recorded): boolean => {
      if (r.copy || r.refill || !r.pipeline) return false;
      if (r.meta) return true;                       // a recipe's output is written whole, never read
      if (!r.access) return false;
      return r.access.every((a) => a !== "rw");
    };
    let remaining = [...this.records];
    let hoisted = 0, bytes = 0, changed = true;
    while (changed) {
      changed = false;
      const written = new Set<GPUBuffer>();
      for (const r of remaining) for (const b of touchesOf(r).writes) written.add(b);
      const next: Recorded[] = [];
      for (const r of remaining) {
        const t = touchesOf(r);
        const constant = pure(r) && t.reads.length > 0
          && t.reads.every((b) => !written.has(b) && !this.uploaded.has(b) && !refilled.has(b));
        if (!constant) { next.push(r); continue; }
        hoisted++; changed = true;
        for (const b of t.writes) { this.frozen.add(b); bytes += this.dev.bytesOf(b); }
      }
      remaining = next;
    }
    this.records = remaining;
    return { hoisted, bytes };
  }

  /**
   * Lays the step's intermediates into arenas so that buffers whose lives do not overlap
   * share bytes, and releases the buffers they replace — see `plan.ts`. Run after
   * `fuse()` (a fused tree leaves intermediates untouched, and those are released here
   * too). `held` — the buffers of every tensor the caller still holds, as for `fuse`;
   * a held buffer stays where it is. Returns what moved and what it saved.
   */
  plan(held?: Iterable<GPUBuffer>): { moved: number; released: number; bytesBefore: number; bytesAfter: number; arenas: number; kept: { liveIn: number; subRange: number } } {
    const heldSet = new Set(held ?? []);
    const movable = (b: GPUBuffer): boolean => this.pinned.has(b) && !this.uploaded.has(b) && !heldSet.has(b) && !this.frozen.has(b);
    const plan = planRecords({
      records: this.records,
      movable,
      // A hoisted dispatch's output is touched by no record left in the recording and
      // must not read as an untouched intermediate to release.
      candidates: [...this.pinned].filter((b) => !this.frozen.has(b)),
      sizeOf: (b) => this.dev.bytesOf(b),
      align: this.dev.offsetAlignment,
      arenaMax: this.dev.maxBinding,
    });
    const arenas = plan.arenas.map((bytes) => {
      const buf = this.dev.allocOwned(bytes / BYTES_PER_F32);
      this.pinned.add(buf);
      return buf;
    });
    const slotOf = (b: BindSlot): BindSlot => {
      if (!(b instanceof GPUBuffer)) return b;
      const p = plan.placements.get(b);
      return p ? { buffer: arenas[p.arena] as GPUBuffer, offset: p.offset, size: this.dev.bytesOf(b) } : b;
    };
    this.records = this.records.map((r) => {
      if (!r.buffers.some((b) => b instanceof GPUBuffer && plan.placements.has(b))) return r;
      const buffers = r.buffers.map(slotOf);
      if (r.copy) return { ...r, buffers };
      const pipeline = r.pipeline as GPUComputePipeline;
      return { ...r, buffers, bindGroup: this.dev.bindGroupFor(pipeline, buffers) };
    });
    let bytesBefore = 0;
    const release: GPUBuffer[] = [];
    for (const b of plan.placements.keys()) { bytesBefore += this.dev.bytesOf(b); release.push(b); }
    for (const b of plan.untouched) { bytesBefore += this.dev.bytesOf(b); release.push(b); }
    for (const b of release) this.pinned.delete(b);
    this.dev.unpin(release);
    const bytesAfter = plan.arenas.reduce((a, b) => a + b, 0);
    return { moved: plan.placements.size, released: plan.untouched.length, bytesBefore, bytesAfter, arenas: arenas.length, kept: plan.kept };
  }

  dispose(): void {
    this.dev.unpin(this.pinned);
    this.pinned.clear();
  }
}

export class Device {
  private readonly device: GPUDevice;
  private readonly limits: GPUSupportedLimits;
  /** Signature including the shape → pipeline. */
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  /**
   * Pipeline → bind group layout.
   *
   * `getBindGroupLayout` **makes a new object on every call** — the specification
   * promises no cache. Called per dispatch, that is one made and thrown away each time,
   * seven hundred times a step.
   */
  private readonly layouts = new WeakMap<GPUComputePipeline, GPUBindGroupLayout>();
  /** Pipeline → what it does to each binding, read off its WGSL once at build. */
  private readonly accesses = new WeakMap<GPUComputePipeline, readonly Access[]>();
  /**
   * The **idle ones** among the staging buffers used for reading back. Several per size.
   *
   * At first there was one per size, reused, and two overlapping reads mapped the same
   * buffer twice and blew up with "Buffer already has an outstanding map pending". It
   * appeared the moment `equal` read two tensors through `Promise.all` — overlapping
   * reads are ordinary, so one is not enough.
   */
  private readonly stagingFree = new Map<number, GPUBuffer[]>();

  private constructor(device: GPUDevice) {
    this.device = device;
    this.limits = device.limits;
  }

  static async create(options: InitOptions = {}): Promise<Device> {
    if (!navigator.gpu) throw new Error(NO_API);          // see probe(): the key can be there and undefined
    const adapter = await adapterFor(options);
    if (!adapter) throw new Error(NO_ADAPTER);
    // **A measured number means something only once you know which device it came
    // from.** A headless browser sometimes hands back a software adapter instead of a
    // real GPU, and that is an adapter too, so nothing is raised — then the wall clock
    // runs perfectly well and all that is left is the conclusion "it is slow". Whoever
    // is measuring has to see this.
    Device.adapterInfo = describe(adapter);
    Device.adapterFeatures = [...adapter.features].sort().join(" ");
    // Rather than taking the default limits, it requests the maximum the adapter
    // offers. The default maxStorageBufferBindingSize is 128MB, and above it a quietly
    // wrong answer comes out.
    const want: Record<string, number> = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      // A fused kernel binds one buffer per leaf and per node of its tree; the guaranteed
      // eight would hold three or four nodes. Apple offers 31, NVIDIA far more.
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    };
    Device.storageBuffersPerStage = adapter.limits.maxStorageBuffersPerShaderStage;
    // **`timestamp-query` is taken when it is there.** Requested and unused it costs
    // nothing, and switching it on later means building the device again — which whoever
    // is measuring cannot know at that moment. Requesting it on an adapter without it
    // makes `requestDevice` refuse, so it goes in only when present.
    const canTime = adapter.features.has("timestamp-query");
    // **Subgroup matrices are taken when the adapter has them — with the f32 8×8×8
    // configuration, which is the one the kernels are written for.** Chrome exposes the
    // feature as `chromium-experimental-subgroup-matrix` on Metal and Vulkan (measured on
    // the M4 Max, 2026-09-06: a GEMM on them reaches 11 TFLOP/s, torch's own number,
    // against 4.5 for the scalar tile). D3D12, Safari and Firefox do not have it, and the
    // scalar kernels stay as the path for them — this flag only opens the other one.
    const sgm = subgroupMatrixF32(adapter);
    // **The int8 configuration is taken where it is the one the adapter has** — the 5080
    // exposes the feature with int8 configurations only, and the f32 kernels stay off
    // there; `matmulInt8` (`docs/INT8.md` Step 1) is what runs on it.
    const sgi8 = subgroupMatrixInt8(adapter);
    const features: GPUFeatureName[] = [];
    if (canTime) features.push("timestamp-query");
    // Subgroups on their own are wider than subgroup matrices: Vulkan without the
    // matrix extension still has them, and a row reduction (softmax) is 5× faster on
    // them than through workgroup memory and a barrier tree.
    const sg = adapter.features.has("subgroups" as GPUFeatureName);
    if (sg) features.push("subgroups" as GPUFeatureName);
    if (sgm || sgi8) features.push("chromium-experimental-subgroup-matrix" as GPUFeatureName);
    // **`shader-f16` is requested where the adapter has it — and it is not everywhere.**
    // Apple's Metal offers it; a recent NVIDIA card (RTX 5080) through Chrome on
    // Linux/Vulkan offers none (measured, `docs/SCALE-MEASURED.md`). So it is an optional
    // capability read per device, never assumed. The kernels that use it (a half-precision
    // weight operand, `docs/SCALE.md` Step 4) ask `Device.f16` before choosing their path,
    // and a device without it stays on the f32 kernels. Requesting a feature the adapter
    // lacks makes `requestDevice` refuse, so it goes in only when present.
    const f16 = adapter.features.has("shader-f16" as GPUFeatureName);
    if (f16) features.push("shader-f16" as GPUFeatureName);
    const descriptor = {
      requiredLimits: want,
      requiredFeatures: features,
    };
    Device.subgroupMatrix = sgm;
    Device.subgroupInt8 = sgi8;
    Device.subgroups = sg;
    Device.f16 = f16;
    Device.workgroupStorage = adapter.limits.maxComputeWorkgroupStorageSize;
    setDirectWeightBytes(Device.workgroupStorage);
    Device.gemmConfigs = gemmConfigsFor(String((adapter.info as Partial<GPUAdapterInfo> | undefined)?.vendor ?? ""));
    // **The convolutions' implicit GEMM keeps the tiles as they were.** The 8 × 4
    // micro-tile that wins the plain product loses there on the 5080 (`kernel_bench fwd
    // --sweep=tiles`, 2026-09-21: 512 → 512 at 4 × 4, batch 16, 0.100 → 0.114 ms; batch 1
    // 0.029 → 0.034) and is a wash on metal-3 — its B side is a gather, scalar and
    // bounds-checked per element, and that, not the micro-tile, is what the kernel waits
    // on; a bigger tile only costs occupancy. The mechanism stays for a kernel that stages
    // the gather (`docs/GEMM.md` §3, the conv verdict); nothing prefers a tile today.
    setConvTilesPreferred([]);

    let device: GPUDevice;
    try {
      device = await adapter.requestDevice(descriptor);
    } catch (err) {
      // A held adapter can have gone stale between the probe and the click. One fresh
      // request, then the error stands.
      const fresh = await askAdapter(options);
      if (!fresh) throw err;
      device = await fresh.requestDevice(descriptor);
    }
    // Validation errors do not arrive as exceptions either. Uncaught, a badly built
    // pipeline quietly does nothing, and we see that result only as "the values are
    // wrong".
    //
    // Only the first few are emitted — one broken shader raises the same error again on
    // every dispatch after it, pushing the real cause (the first line) off the top of the
    // scroll. That happened, measured. **This trims rather than swallows.** How many were
    // folded away is written at the end.
    //
    // **They are counted and exposed.** Printed alone, whoever is measuring does not see
    // them — the ResNet bench produced a ms/step while holding an invalid command buffer,
    // and that number was not a measurement but the wall clock of a state where nothing
    // was learning. Whoever measures has to be able to see this count and refuse the
    // result.
    const made = new Device(device);
    const seen = made.faults;
    device.addEventListener("uncapturederror", (event) => {
      seen.count += 1;
      const err = (event as GPUUncapturedErrorEvent).error;
      // **The kind is read, not assumed.** Every one of these used to print as a
      // *validation error*, which is the wording for a command the device would not
      // run — and an allocation it could not make arrives on the same event. Calling
      // the second one the first sends the reader looking for a bug in a kernel when
      // what happened was that the memory was not there.
      //
      // `instanceof` against a global that may not exist: the class is in the WebGPU
      // spec and Chrome has it, but a runtime that does not would throw here rather
      // than in the code that made the error, which is the worst place to find out.
      const isOom = typeof GPUOutOfMemoryError !== "undefined"
        && err instanceof GPUOutOfMemoryError;
      if (isOom) seen.outOfMemory += 1;
      const kind = isOom ? "out-of-memory error" : "validation error";
      if (seen.first === "") seen.first = err.message;
      if (seen.count <= MAX_REPORTED_ERRORS) {
        console.error(`[borch.ts] WebGPU ${kind} ${seen.count}: ${err.message}`);
      } else if (seen.count === MAX_REPORTED_ERRORS + 1) {
        console.error(
          `[borch.ts] more than ${MAX_REPORTED_ERRORS} validation errors — ` +
            "no more will be printed. The cause is the first one above.",
        );
      }
    });
    device.lost
      .then((info) => {
        // **Printing is not enough.** Losing the device empties every tensor and every
        // number after it of meaning, and whoever is measuring does not read the log —
        // for the same reason `faults` is exposed, this has to be a state that can be
        // asked about. That is what lets a bench refuse its result.
        made.lost = { reason: String(info.reason), message: info.message };
        console.error(`[borch.ts] the WebGPU device was lost: ${info.reason} — ${info.message}`);
      })
      .catch(() => {
        /* lost is not rejected, and even if it were there is nothing more to do here */
      });
    Device.readbackKicks = await calibrateKicks(device, canTime);
    Device.canTime = canTime;
    Device.loadTune();
    return made;
  }

  /**
   * **Whether this browser has to be kicked to notice a finished fence.** Measured
   * once at `create` by `calibrateKicks`, and read by `kicked`, which is what every wait
   * for the GPU in this class goes through.
   *
   * Chrome on Linux over Vulkan (RTX 5080, 2026-09-20, `roundtrip:probe`): forty tiny
   * dispatches are 0.08 ms of GPU and **2.6 ms of wall** under `onSubmittedWorkDone` or
   * `mapAsync`; a 1.57 ms kernel is 2.5. Everything past a few hundred microseconds
   * lands at the same 2.5 — the GPU process polls its fences on a backoff, and a fence
   * that signals between polls waits for the next one. A cheap round trip on the wire
   * (`pushErrorScope`/`popErrorScope`, 0.04 ms) makes it look, and a loop of them until
   * the wait resolves brings the wall to the GPU's time plus 0.1. metal-3 needs none of
   * it: the wall follows the GPU within 0.2 ms whatever the wait, and a loop of kicks
   * there is only chatter — which is why it is measured rather than assumed.
   */
  static readbackKicks = false;

  /** What `calibrateKicks` measured in its worst round, ms — the plain median, the
   *  kernel's GPU time under it (−1 without `timestamp-query`) and the kicked median — so
   *  a table can print the numbers the decision came from rather than only the decision. */
  static kickCalibration: { plainMs: number; kickedMs: number; gpuMs: number } =
    { plainMs: 0, kickedMs: 0, gpuMs: -1 };

  /** The kick decision and its numbers on one line, for the adapter line of a table. */
  static get readbackNote(): string {
    const c = Device.kickCalibration;
    const gpu = c.gpuMs >= 0 ? ` for ${c.gpuMs.toFixed(2)} of GPU` : "";
    return `readbackKicks ${Device.readbackKicks} (plain ${c.plainMs.toFixed(2)}${gpu} / kicked ${c.kickedMs.toFixed(2)} ms)`;
  }

  /**
   * Waits for `pending` — a map or `onSubmittedWorkDone` — kicking the wire until it
   * resolves where `readbackKicks` says the browser needs it. Each kick is a round trip
   * awaited in turn, so the loop yields to the page between them; it is not a spin.
   *
   * **Not kicked where the calibration said no, and not re-checked later.** A detector
   * that waited on an empty submit after each plain wait was tried (2026-09-20): a fence
   * wait covers everything queued, so under a bench that queues the next forward before
   * reading the last it waited for that too — the fused eager forward on metal-3 went
   * 2.9 → 6.1 ms and a 2.25 ms outlier turned kicks on, after which the same forward
   * was 7.7 (kicks cost the eager path on Chrome's Metal backend; the captured and the
   * training step not at all). The calibration at `create` is the decision.
   */
  private async kicked<T>(pending: Promise<T>): Promise<T> {
    if (!Device.readbackKicks) return pending;
    let done = false;
    const watched = pending.then((v) => { done = true; return v; }, (e) => { done = true; throw e; });
    while (!done) {
      this.device.pushErrorScope("validation");
      await this.device.popErrorScope();
    }
    return watched;
  }

  /**
   * The story, if the device was lost; otherwise `null`.
   *
   * There is no counterpart in torch — a CUDA context lives with the
   * process. In a browser another tab or the driver can reclaim our device,
   * and no exception is raised when it happens.
   */
  lost: { reason: string; message: string } | null = null;

  /**
   * Whether it is still usable. Somewhere a long training loop looks at
   * every step.
   */
  get alive(): boolean {
    return this.lost === null;
  }

  /** How many faults the last readback had already reported — `read` throws when the
   *  count has grown since. */
  private faultsReported = 0;

  /**
   * **Out-of-memory scopes waiting to be read.** `createBuffer` never throws on OOM —
   * the failure arrives asynchronously, and the only witness was the uncaptured-error
   * handler surfacing at the next readback (the day that cost, `faults` above). So a
   * fresh allocation is wrapped in `pushErrorScope("out-of-memory")` and the pop's
   * promise parked here; {@link drainAllocations} awaits them at the next readback and
   * folds any real OOM into `faults`, where the existing throw already lives. Only the
   * `out-of-memory` filter is pushed, so a *validation* error still travels to the
   * uncaptured handler unchanged. After warm-up the pool serves the repeats and this
   * list is empty every step — the cost is paid where allocations are actually made
   * (a load, a shape change), not in the training loop.
   */
  private oomPending: Promise<GPUError | null>[] = [];
  /**
   * Errors the device reported and nobody caught.
   *
   * **Whoever is measuring has to look at this.** An invalid command buffer
   * throws nothing and simply does no work, so the wall clock keeps running
   * in that state and numbers come out — something that looks like a
   * measurement comes out.
   *
   * **`outOfMemory` is counted apart, because it is a different thing to be
   * told.** WebGPU raises `GPUOutOfMemoryError` for an allocation it could not
   * make and `GPUValidationError` for a command it would not run, and both
   * arrive here. Folded into one number they read as one fault, and the answer
   * to *the model returns zeros* is different in the two cases: too large to
   * fit is a smaller batch, and invalid is a bug in a kernel.
   *
   * That distinction cost a day the last time it was needed. A batch too large
   * to submit was returning zeros with **this counter at 0 throughout**, and
   * ruling out the allocation half was done by hand because the counter could
   * not say. It still cannot say *this was an allocation*; it can now say *this
   * was not*.
   *
   * `count` stays the total, so everything already reading it is unchanged.
   */
  faults: { count: number; first: string; outOfMemory: number } =
    { count: 0, first: "", outOfMemory: 0 };

  /**
   * Dispatches issued so far.
   *
   * Stopping at "it is slow" leaves you with no next move. Knowing
   * dispatches per step separates whether the slow part is **the kernel
   * itself or the number of calls** — this design currently builds and
   * submits a fresh command encoder per operation, so a large count points
   * there.
   */
  dispatches = 0;

  /**
   * Dispatches by kernel kind.
   *
   * A total alone does not say what to fix next. Whether 1,636 is twenty
   * convs or five hundred BatchNorm assemblies calls for different work —
   * this exists to measure that split.
   */
  readonly byKind = new Map<string, number>();

  /** Which kernel is being called right now. `pipeline` leaves the head of the
   *  signature here. */
  private current = "?";
  /** The **whole signature** of the pipeline being baked. The profiler accumulates by
   *  it. */
  private currentSig = "?";

  /**
   * The commands not yet submitted.
   *
   * **Submitting per operation is wrong.** At first a command encoder was built and
   * submitted per dispatch, and multiplying the batch by 4 raised the time by only 2.1× —
   * fitted to a line, the fixed cost independent of the batch was 5.2 seconds per step
   * and 7.4ms per dispatch. One submission cannot cost that, so that *was* the cost of
   * the number of submissions.
   *
   * Now they accumulate in one encoder and go out once **when something is read.** WebGPU
   * inserts the barriers between dispatches within a pass itself, so the order is kept.
   */
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  /**
   * Submissions actually issued so far. Whoever is measuring whether
   * batching works looks here.
   */
  submits = 0;
  /**
   * Bytes handed out since the last submission.
   *
   * The encoder holds every buffer its commands name, so this grows with the
   * whole un-submitted batch, not with what is live.
   */
  private sinceSubmit = 0;

  /**
   * It does not swallow shader compilation errors — **when asked to look.**
   *
   * **A failed WGSL compile does not arrive as an exception.**
   * `createShaderModule` simply returns, and dispatching with that pipeline
   * does nothing at all — the result buffer stays zero and all the screen
   * says is "the values differ". A reduction kernel in this very runner
   * returned zeros that way, with no error visible anywhere. So the
   * diagnostics are pulled out deliberately.
   *
   * ## Why the asking is now opt-in
   *
   * It used to run for every pipeline, unawaited. That is fine at a hundred
   * shaders and **not fine at twenty thousand**, which is what one
   * EfficientNet builds:
   *
   *     resnet18      66 pipelines
   *     resnet152     72
   *     vit_base     121
   *     efficientnet_b4  **19,531**
   *
   * The count is that high because the pipeline key bakes in the spatial
   * dims and the channel counts, and a depthwise stack changes both at
   * every block. Twenty thousand `getCompilationInfo()` promises are then
   * in flight at once, each holding the full WGSL source in its closure for
   * a message it will almost never print — and the device is lost partway
   * through with `Instance dropped error in getCompilationInfo`.
   *
   * So it is off unless `BORCH_SHADER_DIAGNOSTICS` is set on `globalThis`.
   * The zero-returning kernel that motivated this is a **development**
   * failure: it happens while writing a kernel, and that is when the switch
   * is on. Nothing about a shipped model needs it.
   */
  pipeline(signature: string, source: () => string): GPUComputePipeline {
    // The first segment of the signature is the kernel kind (`cnt:...`, `u:relu:...`).
    // Counting the shape too gives hundreds of kinds and hides where the weight is.
    this.current = signature.split(":")[0] ?? "?";
    // **While profiling it uses the whole signature.** The kind alone reaches "gb is
    // 94%" and stops at the next question (which rule, which shape) — which is where it
    // actually stopped. It accumulates only while switched on, so it costs nothing
    // otherwise.
    this.currentSig = signature;
    const hit = this.pipelines.get(signature);
    if (hit) return hit;
    const code = source();
    const module = this.device.createShaderModule({ code });
    // **Pre-warming (the tuner): the pipeline is made asynchronously, so the GPU process
    // compiles candidates side by side instead of one after another on its main thread.**
    // Measured before this path (capture:ts, 2026-09-21): the tuner's warm wave was 349 ms
    // on the RTX 5080 (Vulkan) and 3,595 ms on the RTX 5050 Laptop (D3D12) for the fifteen
    // pipelines the rule's picks had not compiled — a serial compile each, DXC's at ~200 ms.
    if (this.prewarming) {
      const t0 = performance.now();
      this.prewarmPending.push(this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } })
        .then((pipeline) => {
          this.pipelines.set(signature, pipeline); this.accesses.set(pipeline, bindingAccess(code));
          this.tuneCompiles.push({ sig: signature, ms: performance.now() - t0, bytes: code.length });
        }));
      throw new PrewarmMiss(signature);
    }
    if (shaderDiagnostics()) {
      void module.getCompilationInfo().then((info) => {
        for (const m of info.messages) {
          if (m.type !== "error" && m.type !== "warning") continue;
          console.error(
            `[borch.ts] ${signature} shader ${m.type} ${m.lineNum}:${m.linePos} — ` +
              `${m.message}\n${numbered(code)}`,
          );
        }
      });
    }
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.pipelines.set(signature, pipeline);
    this.accesses.set(pipeline, bindingAccess(code));
    return pipeline;
  }

  /**
   * Shaders baked so far. Tests look at it to see whether the cache works.
   */
  get pipelineCount(): number {
    return this.pipelines.size;
  }

  /**
   * The scopes currently open. What `alloc` builds is written here and released when the
   * scope closes.
   *
   * **Without it nothing trains.** One ResNet step builds thousands of intermediate
   * buffers, and JavaScript's garbage collector does not release a GPU buffer in time —
   * the handle disappears and the memory stays. The sister library holds a `scope()` for
   * the same reason.
   */
  private readonly scopes: Set<GPUBuffer>[] = [];
  /** What survives a scope closing — the parameters and the optimiser state. */
  private readonly kept = new WeakSet<GPUBuffer>();
  /**
   * Reuses released buffers by size.
   *
   * `createBuffer` goes through the driver to claim GPU memory, and one training step
   * does it hundreds of times. And **the same sizes repeat every step**, because the
   * shapes are the same every time. Reusing rather than destroying and rebuilding reduces
   * that to once.
   */
  private readonly spare = new Map<number, GPUBuffer[]>();
  /**
   * Whether {@link auditInvariants} runs at each scope and capture boundary. Off in
   * production — the scan is O(pool) and a training step closes a scope every time — and
   * turned on by the invariants probe, which trains and captures with it watching.
   */
  auditPool = false;
  /**
   * The buffers currently sitting in `spare`, mirrored here so a double-return is O(1) to
   * catch — added on the push into the pool, removed on the pop back out. Returning a
   * buffer that is already pooled would hand the same memory to two allocations (the
   * "9,9,9,9" silent read); this is the one pool invariant cheap enough to check always,
   * not only under {@link auditPool}.
   */
  private readonly inPool = new WeakSet<GPUBuffer>();
  /** How many bytes a buffer actually is. Which pool it returns to comes from here. */
  private readonly sizes = new WeakMap<GPUBuffer, number>();

  /**
   * **Which life a buffer is on.** It rises by one every time the buffer returns to the
   * pool.
   *
   * ## Why it is needed — measured
   *
   * When a scope closes a buffer is not destroyed but returned to the pool (that is what
   * the pool is for). And if a tensor pointing at that buffer leaked out of the scope,
   * that tensor still holds the same `GPUBuffer`, and **the next allocation takes it out
   * and overwrites it.**
   *
   * It was measured. Leak a tensor holding `[1,2,3,4]` out of a scope, take four more
   * allocations of the same size, read it back, and out comes **`9,9,9,9`** — somebody
   * else's values, with no exception. This repository's opening sentence is that it stops
   * loudly rather than quietly producing a different value, and the opposite was
   * happening in the core training loop.
   *
   * A tensor records this number when it is born and compares on reaching its value. A
   * mismatch means that tensor is **already dead** and it stops there. The golden cannot
   * see this — each case has a clean page, so the pool is never stirred.
   */
  private readonly ages = new WeakMap<GPUBuffer, number>();

  /**
   * This buffer's current life. A tensor compares this number at birth
   * against its value at use.
   */
  age(buffer: GPUBuffer): number {
    return this.ages.get(buffer) ?? 0;
  }

  /**
   * Raises the life by one — **every tensor pointing at that buffer dies at this moment.**
   *
   * It rises **on return to the pool** rather than when the buffer is actually taken out
   * again. Raising it on the way out creates a window where "nobody has taken it yet, so
   * it still reads", and code appears that passes only inside that window — which is how
   * a defect whose reproduction depends on allocation order gets made.
   */
  private retire(buffer: GPUBuffer): void {
    this.ages.set(buffer, this.age(buffer) + 1);
  }

  beginScope(): void {
    this.scopes.push(new Set());
  }

  // ── Capture and replay ──────────────────────────────────────────────────────────
  //
  // **A training step is the same dispatches with the same buffers every time.** The
  // Python side builds the autograd graph, allocates, and issues 262 dispatches a step,
  // and on the M4 Max that CPU-side work is 3.4 ms of a 17.7 ms step (timestamps against
  // the wall clock, 2026-09-07). While a capture is open, `run` records each dispatch —
  // pipeline, bind group, grid — and every buffer allocated is pinned rather than
  // returned to the pool when its scope closes, so the recorded bind groups keep
  // pointing at live memory. `replay` re-encodes the list: no Python, no allocation, no
  // bind-group creation. What has to stay the same is the caller's business: the input
  // tensors (write the next batch into them), and anything a step varies on the CPU
  // side — Adam's bias correction moved to a kernel for exactly this.
  private recording: Recorded[] | null = null;
  private pinned: Set<GPUBuffer> | null = null;
  private uploaded: Set<GPUBuffer> | null = null;
  /** Every buffer some live capture owns — a scope closing after the capture ended must
   *  still leave them alone (measured: the loss returned from a compiled step "belonged to
   *  a closed scope" the moment the caller's scope closed). */
  private readonly owned = new Set<GPUBuffer>();

  /** **The single-parameter-group optimizer arena, held off.** A captured step takes the
   *  per-parameter path (its momentum lives in the optimizer's own buffers); the arena
   *  keeps momentum in a separate slab it fills lazily. `torch.compiled(check=True)`
   *  reruns the step eagerly to compare against the replay — and that rerun, not being
   *  captured, would otherwise take the arena and leave the per-parameter momentum buffer
   *  the replay wrote untouched, so the two disagree on a buffer that in fact holds the
   *  same values in different places (measured: single-group `SGD(momentum=…)` raised a
   *  spurious "the replay is not the eager step"). The verify rerun sets this so it takes
   *  the same per-parameter path the recording did. */
  suppressArena = false;

  beginCapture(): void {
    if (this.recording) throw new Error("a capture is already open");
    this.recording = [];
    this.pinned = new Set();
    this.uploaded = new Set();
  }

  endCapture(): Capture {
    if (!this.recording || !this.pinned) throw new Error("no capture is open");
    const capture = new Capture(this, this.recording, this.pinned, this.uploaded ?? new Set());
    this.recording = null;
    this.pinned = null;
    this.uploaded = null;
    return capture;
  }

  /** What a pipeline does to each binding, read off its WGSL at build (`bindingAccess`) —
   *  for a record made outside `run`, such as a fused kernel's. */
  accessOf(pipeline: GPUComputePipeline): readonly Access[] | undefined {
    return this.accesses.get(pipeline);
  }

  /** The key a pipeline was built under, or undefined for one this device did not build. */
  keyOf(pipeline: GPUComputePipeline): string | undefined {
    for (const [key, p] of this.pipelines) if (p === pipeline) return key;
    return undefined;
  }

  /** Whether a capture is open. */
  get capturing(): boolean {
    return this.recording !== null;
  }

  /** Encodes recorded dispatches again, in order. Called by `Capture.replay`. */
  /**
   * Runs `fn` with the open capture set aside: nothing it dispatches is recorded and
   * nothing it allocates is pinned. The window's refill copy takes this — what the
   * recording keeps of a refill is the host bytes (`Recorded.refill`), not the copy from
   * a staging buffer whose contents will be another block's by the time of a replay.
   */
  unrecorded<T>(fn: () => T): T {
    const recording = this.recording, pinned = this.pinned, uploaded = this.uploaded;
    this.recording = null; this.pinned = null; this.uploaded = null;
    try {
      return fn();
    } finally {
      this.recording = recording; this.pinned = pinned; this.uploaded = uploaded;
    }
  }

  /** Records a window refill — see `Recorded.refill`. A no-op outside a capture. */
  recordRefill(win: Window, slot: BindSlot, data: Float32Array | Uint16Array | Uint32Array): void {
    this.recording?.push({ refill: { win, data }, groups: [0, 0, 0], buffers: [slot] });
  }

  replayRecorded(records: readonly Recorded[]): void {
    for (const r of records) {
      if (r.refill) throw new Error("a window refill cannot be re-encoded — replay through replayAsync()");
      if (r.copy) {
        // A copy is recorded over whole buffers; after `Capture.plan` either end may be
        // a slot of an arena, whose offset is added to the copy's own.
        const [s, d] = r.buffers as [BindSlot, BindSlot];
        const sOff = s instanceof GPUBuffer ? 0 : s.offset;
        const dOff = d instanceof GPUBuffer ? 0 : d.offset;
        if (r.copy.srcOff !== undefined || r.copy.dstOff !== undefined || sOff || dOff) {
          this.copyRange(bufOf(d), dOff + (r.copy.dstOff ?? 0), bufOf(s), sOff + (r.copy.srcOff ?? 0), r.copy.bytes);
        } else {
          this.copyInto(bufOf(d), bufOf(s), r.copy.bytes / BYTES_PER_F32);
        }
        continue;
      }
      if (!r.pipeline || !r.bindGroup) throw new Error("a recorded dispatch without a pipeline");
      this.currentSig = r.sig ?? "replay";
      const pass = this.openPass();
      pass.setPipeline(r.pipeline);
      pass.setBindGroup(0, r.bindGroup);
      pass.dispatchWorkgroups(r.groups[0], r.groups[1], r.groups[2]);
      this.dispatches += 1;
      // A replay writes what the recording wrote — an optimiser's step into a parameter
      // among them — and the eager pack cache has to know.
      this.noteWrites(r.buffers, r.access);
    }
  }

  /**
   * **The one door a buffer goes back to the spare pool through.** `endScope` and `unpin`
   * both release buffers, and each used to carry its own copy of this decision — which is
   * how they drifted: `unpin` pooled a `kept` scalar constant that `endScope` would have
   * spared, and the next use of that cached value read a dead buffer (the compiled
   * small-CNN's eval forward threw after training). With the decision in one place no
   * release path can miss part of it:
   *
   * - A **kept** buffer (the scalar cache's own) is permanent — never pooled, never
   *   retired, so its age stays put for the tensors that share it.
   * - Pooling **retires** the buffer (bumps its age), so a tensor that outlived its scope
   *   fails `refuseIfDead` rather than quietly reading what the next allocation wrote.
   * - A buffer `alloc` did not size is not ours to pool; it is destroyed once pending
   *   commands that might point at it have gone out.
   */
  private returnToPool(buf: GPUBuffer): void {
    if (this.kept.has(buf)) return;
    this.dropPacks(buf);
    if (this.inPool.has(buf)) {
      throw new Error(
        "a buffer was returned to the pool while already in it — the same memory would be " +
          "handed to two allocations (a release path ran twice for one buffer).",
      );
    }
    this.retire(buf);
    const size = this.sizes.get(buf);
    if (size === undefined) { this.flush(); buf.destroy(); return; }
    let pool = this.spare.get(size);
    if (!pool) { pool = []; this.spare.set(size, pool); }
    pool.push(buf);
    this.inPool.add(buf);
  }

  /**
   * **The pool's invariants, checked rather than trusted.** {@link returnToPool} is the one
   * door a buffer takes back to `spare`; this is the assertion that the door was not
   * bypassed. A buffer sitting in the pool must be:
   *
   *  - **not kept** — a permanent scalar-cache buffer that reached the pool would be handed
   *    out and overwritten (the class the compiled-training regression was in);
   *  - **not owned** — a buffer an open capture still needs would be reused under it;
   *  - **in the bucket its size names** — a wrong bucket hands back a buffer of the wrong
   *    length, and WebGPU writes only part of it;
   *  - **in the pool once** — the same buffer pooled twice is handed to two allocations,
   *    the "9,9,9,9" silent read the age guard exists to stop.
   *
   * Thrown loudly with the boundary that found it, so a future release path that skips part
   * of `returnToPool`'s decision is caught where it happened rather than as a wrong value a
   * hundred dispatches later. Returns how many pooled buffers it checked — a caller that
   * expects a stirred pool can refuse a vacuous zero. Off the hot path (`auditPool`).
   */
  auditInvariants(where: string): number {
    const seen = new Set<GPUBuffer>();
    for (const [size, bucket] of this.spare) {
      for (const buf of bucket) {
        if (this.kept.has(buf)) throw new Error(`pool invariant (${where}): a kept buffer is in the pool — it would be handed out and overwritten`);
        if (this.owned.has(buf)) throw new Error(`pool invariant (${where}): a buffer an open capture owns is in the pool — the replay would reuse it`);
        const actual = this.sizes.get(buf);
        if (actual !== size) throw new Error(`pool invariant (${where}): a ${actual}-byte buffer sits in the ${size}-byte bucket`);
        if (seen.has(buf)) throw new Error(`pool invariant (${where}): the same buffer is in the pool twice — it would be handed to two allocations`);
        seen.add(buf);
      }
    }
    return seen.size;
  }

  /** Hands a capture's pinned buffers back to the pool. Called by `Capture.dispose`. */
  unpin(buffers: Iterable<GPUBuffer>): void {
    for (const buf of buffers) {
      this.owned.delete(buf);
      // A buffer made under a capture is in the scope that was open then as well as
      // pinned; `endScope` skips it while it is owned. Released here while that scope is
      // still open — `compiled` records inside the caller's scope, and `Capture.plan`
      // releases what it laid into arenas — the scope's close would return it a second
      // time (measured: "returned to the pool while already in it"). It leaves every
      // open frame as it goes to the pool.
      for (const frame of this.scopes) frame.delete(buf);
      this.returnToPool(buf);
    }
    if (this.auditPool) this.auditInvariants("unpin");
  }

  /**
   * A buffer a capture owns from the start — an arena `Capture.plan` lays intermediates
   * into. Made outside any scope (a scope closing must not pool it) and marked owned, as
   * `alloc` under an open capture does; the capture adds it to its pinned set and
   * `dispose` returns it.
   */
  allocOwned(count: number): GPUBuffer {
    const buf = this.alloc(count, true);
    this.scopes[this.scopes.length - 1]?.delete(buf);
    this.owned.add(buf);
    return buf;
  }

  /** The bytes a buffer was allocated with — the pool's bucket, and a slot's size. */
  bytesOf(buf: GPUBuffer): number {
    return this.sizes.get(buf) ?? buf.size;
  }

  /** `minStorageBufferOffsetAlignment` — what a bound sub-range's offset must be a multiple of. */
  get offsetAlignment(): number {
    return this.limits.minStorageBufferOffsetAlignment;
  }

  /** `maxStorageBufferBindingSize` — the most one binding, and so one arena, can be. */
  get maxBinding(): number {
    return this.limits.maxStorageBufferBindingSize;
  }

  /**
   * Closes the scope and releases what was made inside it.
   *
   * @param keep what to keep alive. With an enclosing scope it is handed
   *   there — unhanded, nobody releases it when the outer one closes.
   * @returns the number released and **the number that survived**. Both are
   *   given — the survivors are what this scope let out, and in a training
   *   loop a non-zero count means something accumulates every step.
   */
  endScope(keep: readonly GPUBuffer[] = []): { freed: number; survived: number } {
    const frame = this.scopes.pop();
    if (!frame) return { freed: 0, survived: 0 };
    const spare = new Set(keep);
    const outer = this.scopes[this.scopes.length - 1];
    let freed = 0;
    let survived = 0;
    for (const buf of frame) {
      if (spare.has(buf) || this.kept.has(buf)) {
        outer?.add(buf);
        survived += 1;
        continue;
      }
      // **They die here.** If a tensor holding this buffer leaked out, using it stops
      // from now on — otherwise it quietly reads what the next allocation overwrote.
      // Pinned by an open capture: neither pooled nor passed outward — the capture owns it.
      if (this.owned.has(buf)) continue;
      // Retired and pooled, or destroyed if `alloc` did not build it — the one release
      // path `unpin` shares, so the two cannot decide it differently again.
      this.returnToPool(buf);
      freed += 1;
    }
    if (this.auditPool) this.auditInvariants("endScope");
    // **The last count is kept.** There was a place calling `beginScope`/`endScope`
    // directly rather than `scope()` because it needed this value — the bench, measuring
    // leaks. A recommended path that hides something is a recommendation nobody keeps.
    this.lastScope = { freed, survived };
    return this.lastScope;
  }

  /**
   * The tally of the most recently closed scope. It stays here even when
   * closed via `scope()`.
   *
   * **A non-zero `survived` means something accumulates every step** — in a
   * training loop that is the leak.
   */
  lastScope: { freed: number; survived: number } = { freed: 0, survived: 0 };

  /**
   * The count and bytes of buffers currently held.
   *
   * A benchmark measuring leaks has to be able to ask this from outside.
   * The sister project's benchmark called `js.tf.memory()` directly, which
   * ties the instrumentation to TF.js and makes the same benchmark
   * unrunnable against another implementation — and that is exactly why it
   * could not be run.
   *
   * What sits in `spare` is excluded. A buffer back in the pool waiting for
   * the next step is held, but it **is not leaking** — counting it reads
   * something that is not a leak as one.
   */
  get memory(): { tensors: number; bytes: number } {
    const { count, bytes } = this.pooled;
    return { tensors: this.made - count, bytes: this.madeBytes - bytes };
  }

  /**
   * Buffers in the pool waiting for the next step. **What `memory`
   * deliberately excludes.**
   *
   * That one asks "is it leaking" and this one asks "how much is held". Two
   * different questions need two numbers, and the second one was missing —
   * so **nobody could ask about the real footprint.**
   *
   * The pool grows when shapes change. It is split by size, so a buffer
   * that ran at batch 16 cannot serve batch 32 and simply stays. A
   * benchmark running three batch sizes in one pass leaves the first two
   * sizes' worth sitting in the pool, and `memory` does not count it.
   */
  get pooled(): { count: number; bytes: number } {
    let count = 0;
    let bytes = 0;
    for (const [size, pool] of this.spare) {
      count += pool.length;
      bytes += size * pool.length;
    }
    return { count, bytes };
  }

  /**
   * Empties the pool. Where `torch.cuda.empty_cache()` goes.
   *
   * **The pool does not shrink on its own.** With repeating shapes, as in a
   * training loop, that is right — remaking them each time is the cost. But
   * when the shape **changes**, the old shape's buffers stay forever. In a
   * browser, where GPU memory is shared between tabs, that costs more than
   * it does on a desktop.
   *
   * A returned buffer may still be referenced by commands not yet
   * submitted, so the release happens **after** submitting.
   */
  emptyCache(): { count: number; bytes: number } {
    const freed = this.pooled;
    if (freed.count === 0) return freed;
    this.flush();
    for (const pool of this.spare.values()) {
      for (const buf of pool) buf.destroy();
    }
    this.spare.clear();
    // Subtracted from what was built — otherwise `memory` goes on counting dead
    // buffers.
    this.made -= freed.count;
    this.madeBytes -= freed.bytes;
    return freed;
  }

  // `sizes` is a WeakMap and cannot be counted — making it countable would keep the
  // buffers alive. So it counts at build time. **There is nowhere to subtract**, because
  // a buffer `alloc` built returns to the pool rather than being destroyed (`endScope`).
  // The two places that do destroy (a buffer from outside `alloc`, and a read staging
  // buffer) were never counted here in the first place.
  private made = 0;
  private madeBytes = 0;

  // ── Write epochs and the pack cache ───────────────────────────────────────────────
  //
  // **A weight that has not been written since is not repacked again.** The subgroup
  // and staged convolutions read the weight tap-major (`tmw`, `tmwc`), and an eager
  // forward laid that out on every call — a dispatch and a weight-sized write per deep
  // layer per forward, three of the ResNet-18's twenty at 0.12–0.15 ms each of a 1.1 ms
  // batch-1 GPU total (`compare:ts`, 2026-09-21). ORT prepacks at session creation. A
  // recording hoists its own repacks (`Capture.hoist`); this is the eager forward's.
  //
  // What decides "written since" is not `Tensor.version` (only `mutate` bumps it — an
  // optimiser's fused kernel writes a parameter without it) but the device: every
  // dispatch bumps an epoch on each buffer its pipeline may write (`accesses`; all of
  // them where the access is not declared), and so do copies, `writeWords` and a
  // replay's records. A pack is kept beside its source's epoch and remade into the
  // same buffer when the epoch has moved; it dies with its source.

  private readonly writeEpoch = new WeakMap<GPUBuffer, number>();
  private readonly packs = new WeakMap<GPUBuffer, Map<string, { epochs: number[]; buf: GPUBuffer }>>();
  /** `packed` calls that returned a pack as it was, and that made or remade one. */
  packHits = 0;
  packMisses = 0;

  /** The write epoch of `buffer` — how many times the device has been told it was written. */
  epochOf(buffer: GPUBuffer): number {
    return this.writeEpoch.get(buffer) ?? 0;
  }

  private noteWrites(buffers: readonly BindSlot[], access: readonly Access[] | undefined): void {
    buffers.forEach((b, i) => {
      if (access && access[i] === "r") return;
      const buf = b instanceof GPUBuffer ? b : b.buffer;
      this.writeEpoch.set(buf, (this.writeEpoch.get(buf) ?? 0) + 1);
    });
  }

  private noteWrite(buffer: GPUBuffer): void {
    this.writeEpoch.set(buffer, (this.writeEpoch.get(buffer) ?? 0) + 1);
  }

  /**
   * A repack of `srcs[0]` (with the others, a bias say) under `key`, `count` words long,
   * made by `make(dst)`. The first time, and whenever any source has been written since,
   * `make` runs into a buffer kept outside every scope; otherwise the pack is returned as
   * it is. **Under a capture the pack is made into a scope buffer as before** — the
   * recording hoists what is replay-invariant and must not read a buffer it did not
   * record the making of.
   */
  packed(srcs: readonly GPUBuffer[], key: string, count: number, make: (dst: GPUBuffer) => void): GPUBuffer {
    const src = srcs[0];
    if (src === undefined) throw new Error("packed: no source");
    if (this.capturing) { const dst = this.alloc(count); make(dst); return dst; }
    let m = this.packs.get(src);
    if (!m) { m = new Map(); this.packs.set(src, m); }
    const epochs = srcs.map((b) => this.epochOf(b));
    const hit = m.get(key);
    if (hit && hit.epochs.length === epochs.length && hit.epochs.every((e, i) => e === epochs[i])) {
      this.packHits += 1;
      return hit.buf;
    }
    this.packMisses += 1;
    let dst = hit?.buf;
    if (dst === undefined) {
      dst = this.alloc(count);
      this.rehome(dst, 0);
      this.keep(dst);
    }
    make(dst);
    m.set(key, { epochs, buf: dst });
    return dst;
  }

  /** The packs made from `src` go with it. */
  private dropPacks(src: GPUBuffer): void {
    const m = this.packs.get(src);
    if (!m) return;
    this.packs.delete(src);
    for (const { buf } of m.values()) this.unkeep(buf);
  }

  /**
   * Keeps something alive regardless of scope. Parameters and optimizer
   * state use it.
   */
  keep(buffer: GPUBuffer): void {
    this.kept.add(buffer);
  }

  /**
   * The inverse of {@link keep}, for a caller that owns a kept buffer's whole life and is
   * done with it — the SGD arena on a resume, where the next step rebuilds it. It leaves
   * `kept` and is destroyed; a pending command may still bind it, so the destroy waits
   * behind a flush, the order {@link emptyCache} uses. The `made`/`madeBytes` tally is
   * corrected as `emptyCache` does, so a released buffer stops counting as held.
   */
  unkeep(buffer: GPUBuffer): void {
    if (!this.kept.delete(buffer)) return;
    this.dropPacks(buffer);
    this.flush();
    const size = this.sizes.get(buffer);
    if (size !== undefined) {
      this.made -= 1;
      this.madeBytes -= size;
      this.sizes.delete(buffer);
    }
    buffer.destroy();
  }

  /**
   * Moves a buffer `alloc` just filed under the innermost scope to the frame `depth`
   * scopes deep — `0` is outside every scope. `Tensor.ensureOwned` uses it so an owned
   * copy lives as long as the tensor it replaces, not as long as the scope that
   * happened to be open when the write came.
   */
  rehome(buffer: GPUBuffer, depth: number): void {
    this.scopes[this.scopes.length - 1]?.delete(buffer);
    if (depth > 0) this.scopes[depth - 1]?.add(buffer);
  }

  /**
   * The depth of open scopes. Tests look at it for balance.
   */
  get scopeDepth(): number {
    return this.scopes.length;
  }

  /**
   * **Uploads must not come from the pool.** `writeBuffer` runs at the
   * queue's current position, whereas we stack commands and submit them
   * later — if a dispatch not yet submitted is about to read a buffer taken
   * from the pool, we would overwrite it. Allocating fresh removes the
   * situation entirely.
   *
   * @param recycle whether something from the pool may be taken.
   */
  alloc(count: number, recycle = true): GPUBuffer {
    const bytes = count * BYTES_PER_F32;
    const max = this.limits.maxStorageBufferBindingSize;
    if (bytes > max) {
      // Run past the limit, WebGPU quietly writes only some of it. Stopping here is
      // better.
      throw new Error(
        `buffer exceeds the limit: ${(bytes / 1048576).toFixed(1)}MB > ` +
          `${(max / 1048576).toFixed(0)}MB (maxStorageBufferBindingSize)`,
      );
    }
    const size = Math.max(bytes, BYTES_PER_F32);
    // Under a capture nothing is recycled: a pooled buffer may still be bound by a
    // recorded dispatch of this very step.
    const reused = recycle && !this.pinned ? this.spare.get(size)?.pop() : undefined;
    if (reused) this.inPool.delete(reused);   // out of the pool — no longer a double-return risk
    let buf: GPUBuffer;
    if (reused) {
      buf = reused;
    } else {
      // **A soft budget, checked before the buffer is made.** `held` is what `memory`
      // reports — built minus pooled, the live footprint. Over budget it throws here,
      // naming the number and the request, rather than letting the allocation fail
      // asynchronously and read back as zeros. It does **not** reclaim: `emptyCache`
      // flushes, and a flush inside a forward would add a submit and break the one-submit
      // step. Reclaiming is the caller's to do between steps (the streaming window does).
      // `budget = 0` is off, which it is unless someone measured a ceiling and set it.
      if (Device.budget > 0) {
        const held = this.madeBytes - this.pooled.bytes;
        if (held + size > Device.budget) {
          throw new Error(
            `allocation would cross the budget: ${((held + size) / 1048576).toFixed(1)}MB ` +
              `> ${(Device.budget / 1048576).toFixed(0)}MB (Device.budget). ` +
              "Free something (emptyCache, or drop a window slot) before allocating, or " +
              "raise the budget if the device has the memory.",
          );
        }
      }
      // **Catch the OOM the API will not throw.** Only this allocation is inside the
      // scope; the pop's promise is drained at the next readback (`drainAllocations`).
      this.device.pushErrorScope("out-of-memory");
      buf = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.oomPending.push(this.device.popErrorScope());
    }
    if (this.pinned) { this.pinned.add(buf); this.owned.add(buf); }
    if (!reused) {
      this.made += 1;
      this.madeBytes += size;
    }
    this.sizes.set(buf, size);
    this.sinceSubmit += size;
    this.scopes[this.scopes.length - 1]?.add(buf);
    return buf;
  }

  upload(data: Float32Array): GPUBuffer {
    const buf = this.alloc(data.length, false);
    this.device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
    // Under a capture, remembered as an upload — see `Capture.liveIns`.
    this.uploaded?.add(buf);
    return buf;
  }

  /**
   * One `u32` for a kernel to read at run time — an offset, a padding width.
   *
   * **Cached by value, for the life of the device.** The first version allocated a fresh
   * four-byte buffer per dispatch; a grouped convolution issues one such dispatch per
   * group, and an EfficientNet-B4 forward asked for 83,724 of them — which is why the
   * word is not a pooled buffer either. Slices repeat their offsets far more than they
   * vary them (11,042 distinct against 55,794 asked, measured on that model), so a map
   * from value to buffer is small, and a buffer that lives outside the scopes cannot be
   * released under a dispatch that has not run yet.
   *
   * It cannot be one buffer rewritten per dispatch: the dispatches are recorded now and
   * submitted later, and `writeBuffer` lands before the submission — every dispatch
   * would read the last value written.
   *
   * A storage buffer like the rest, so the shader declares `array<u32>` and no new usage
   * flag or pool appears.
   */
  word(value: number): GPUBuffer {
    const hit = this.words.get(value);
    if (hit) return hit;
    const buf = this.device.createBuffer({
      size: BYTES_PER_F32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, new Uint32Array([value]));
    this.words.set(value, buf);
    return buf;
  }

  private readonly words = new Map<number, GPUBuffer>();

  /**
   * One kernel. `groups` is the **workgroup count**, and the per-axis limit
   * is rechecked here — `kernels.ts` folds the grid, but a hand-called path
   * may appear.
   */
  run(
    pipeline: GPUComputePipeline,
    buffers: readonly BindSlot[],
    groups: readonly [number, number, number],
    meta?: Elementwise | Reduce,
  ): void {
    const cap = this.limits.maxComputeWorkgroupsPerDimension;
    for (const [axis, count] of groups.entries()) {
      if (count > cap) {
        throw new Error(
          `dispatch on axis ${axis} exceeds the limit: ${count} > ${cap}. ` +
            "WebGPU does not throw for this — it silently does nothing.",
        );
      }
    }
    const bindGroup = this.bindGroupFor(pipeline, buffers);
    const pass = this.openPass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    this.dispatches += 1;
    this.noteWrites(buffers, this.accesses.get(pipeline));
    if (this.recording) {
      const access = this.accesses.get(pipeline);
      this.recording.push({ pipeline, bindGroup, groups: [groups[0], groups[1], groups[2]], buffers: [...buffers], sig: this.currentSig,
        ...(meta ? { meta } : {}), ...(access ? { access } : {}) });
    }
    // **A batch that grows too large is dropped, and nothing says so.**
    //
    // Commands accumulate in one encoder and go out when something is read. The
    // encoder holds every buffer its commands name, and a forward that never reads
    // releases nothing along the way — so the batch grows with the whole model.
    //
    // Past some size Metal stops running it. There is no exception, no validation
    // error and no lost device: every output is exactly 0, which reads as a model
    // that answers zero rather than as work that never happened. Measured on
    // EfficientNet-B4, one submission, 140,445 dispatches either way:
    //
    //     288x288   134 GB handed out   correct
    //     296x296   190 GB              every logit 0
    //     304x304   247 GB              every logit 0
    //
    // Submitting once in the middle fixes it, so what bounds a batch is its size and
    // not its dispatch count — the count is identical across all three.
    //
    // **The cheap threshold is the safe one.** Anything that runs today stays in one
    // submission and pays nothing; a ResNet-18 forward hands out far less than this.
    // What crosses it is work that currently fails outright, and there a few extra
    // submissions cost milliseconds against an answer that was zero.
    //
    // Not near the measured edge, deliberately. That edge belongs to the driver and
    // was found on one machine; a threshold that only just fits here is one that
    // fails elsewhere, silently, in the way this comment is about.
    if (this.sinceSubmit > Device.MAX_BATCH_BYTES) this.flush();
    this.byKind.set(this.current, (this.byKind.get(this.current) ?? 0) + 1);
  }

  /**
   * Runs one-dimensional work spread over a grid. Paired with the indexing
   * in `kernels.ts`.
   */
  run1d(pipeline: GPUComputePipeline, buffers: readonly BindSlot[], n: number, meta?: Elementwise | Reduce): void {
    const g = grid1d(n);
    this.run(pipeline, buffers, [g.x, g.y, 1], meta);
  }

  /** Writes `words` at the start of `buffer` — a seed, a counter. */
  writeWords(buffer: GPUBuffer, words: Uint32Array<ArrayBuffer>): void {
    this.device.queue.writeBuffer(buffer, 0, words);
    this.noteWrite(buffer);
  }

  /** A bind group for `pipeline` over `buffers`, in binding order. */
  bindGroupFor(pipeline: GPUComputePipeline, buffers: readonly BindSlot[]): GPUBindGroup {
    let layout = this.layouts.get(pipeline);
    if (!layout) {
      layout = pipeline.getBindGroupLayout(0);
      this.layouts.set(pipeline, layout);
    }
    return this.device.createBindGroup({
      layout,
      entries: buffers.map((b, binding) => {
        if (!(b instanceof GPUBuffer) && b.offset % this.limits.minStorageBufferOffsetAlignment !== 0) {
          // A storage binding offset must be a multiple of the device alignment (256 on
          // the cards here). An unaligned sub-range is a WebGPU validation error the
          // uncaptured-error handler would swallow, leaving the output silently zero — so
          // stop loudly here. A view whose offset is not aligned must materialise before
          // it is bound.
          throw new Error(
            `bind offset ${b.offset} is not a multiple of ${this.limits.minStorageBufferOffsetAlignment} ` +
              "— an unaligned sub-range cannot be bound; materialise the view first.",
          );
        }
        return { binding, resource: b instanceof GPUBuffer ? { buffer: b } : { buffer: b.buffer, offset: b.offset, size: b.size } };
      }),
    });
  }

  /**
   * A full sum. It calls the same kernel again until the partial sums come
   * down to one.
   *
   * **It uses no atomics** — floating-point addition changes value when the
   * order changes, and then the same seed run twice gives different
   * training. The slower way is the one that reproduces.
   */
  sumAll(input: GPUBuffer, n: number): GPUBuffer {
    let src = input;
    let count = n;
    let owned: GPUBuffer | null = null;
    while (count > 1) {
      const parts = reduceParts(count);
      const dst = this.alloc(parts);
      const size = count;
      // The first pass reads the input once each; a fusion pass may inline its producer.
      this.run1d(
        this.pipeline(`reduceSum:${size}`, () => reduceSum(size)),
        [src, dst],
        size,
        src === input ? { n: size, input: 0, serial: 1, make: (source) => reduceSum(size, source) } : undefined,
      );
      // **It must not be released here.** The commands accumulate and go out later, so
      // the dispatch just issued is still about to read this buffer. It returns to the
      // pool when the scope closes.
      owned = dst;
      src = dst;
      count = parts;
    }
    if (owned) return owned;
    // With one element there is nothing to fold. Handing the input straight back would
    // have the caller destroy somebody else's buffer, so a copy is given.
    //
    // **It has to ride the accumulated queue.** Building a separate encoder here and
    // submitting immediately copied the value the unsubmitted commands were going to make
    // **before** they made it, and 0 came out — the value is computed later and the copy
    // has already left. Not an exception and not a NaN but **simply 0**, which is where
    // the loss quietly became 0 when `x.mean()`'s `x` held one element. Folding a
    // one-element tensor is rare, so 1,399 golden cases went by green.
    const copy = this.alloc(1);
    this.copyInto(copy, input, 1);
    return copy;
  }

  /**
   * Overwrites one buffer with another. In-place operations use it.
   *
   * It is a copy, not a kernel — the result is made in a new buffer and
   * then moved back to the original slot. Reading and writing the original
   * at once leaves the threads unordered and the values mixed.
   */
  copyInto(dst: GPUBuffer, src: GPUBuffer, count: number): void {
    const bytes = Math.max(count * BYTES_PER_F32, BYTES_PER_F32);
    // A copy cannot go inside a compute pass. Closing the pass and riding the same
    // encoder keeps the order and still submits once.
    this.openEncoder().copyBufferToBuffer(src, 0, dst, 0, bytes);
    this.noteWrite(dst);
    // Under a capture the copy is part of the step — see `Recorded`.
    this.recording?.push({ copy: { bytes }, groups: [0, 0, 0], buffers: [src, dst] });
  }

  /**
   * A copy between sub-ranges — `srcOff`/`dstOff` bytes in, `bytes` long. `copyInto` is
   * the whole-buffer case; this is what an arena's gather and scatter ride (a byte offset
   * must be a multiple of 4, which every f32 slice is). Recorded with its offsets so a
   * captured step replays the same slices.
   */
  copyRange(dst: GPUBuffer, dstOff: number, src: GPUBuffer, srcOff: number, bytes: number): void {
    this.openEncoder().copyBufferToBuffer(src, srcOff, dst, dstOff, bytes);
    this.noteWrite(dst);
    this.recording?.push({ copy: { bytes, srcOff, dstOff }, groups: [0, 0, 0], buffers: [src, dst] });
  }

  /**
   * Measures GPU time per kernel. **Off by default.**
   *
   * ## Why it exists
   *
   * A wall clock can only measure a whole step. Asking which of 429 dispatches was
   * expensive turned out to have no way to be asked — there were **counts** per kind and
   * no **time** per kind, and the counts stay the same as the batch grows, so they
   * pointed at nothing.
   *
   * ## What changes when it is on
   *
   * Normally every dispatch shares **one** compute pass (and one submission per step).
   * Timestamps are per pass, so in that state only the whole pass's start and end can be
   * stamped. So switching it on **opens a pass per dispatch.**
   *
   * **The absolute numbers then come out larger than usual**, because opening a pass
   * costs. What is wanted here is not absolute time but **which kernel holds the largest
   * share**, and that ratio survives. For absolute numbers, switch it off and use the
   * bench.
   */
  private profiling = false;
  /**
   * GPU time in nanoseconds accumulated per kernel kind, when enabled.
   */
  readonly nsByKind = new Map<string, number>();
  /** Dispatches counted per kernel kind, when enabled — the pair to `nsByKind`. */
  readonly countByKind = new Map<string, number>();
  private querySet: GPUQuerySet | null = null;
  private queryUsed = 0;
  private queryKinds: string[] = [];
  /**
   * Dispatches **not measured** for want of room. Whoever calls has to
   * report this alongside.
   *
   * Non-zero means `nsByKind` holds only part of the step while still
   * reading like a total. Not writing down what was cut reads as
   * "everything was measured", and that is one of the kinds of lie this
   * repository has been counting.
   */
  profileDropped = 0;
  /** The query set's size. More than this in one submission and the rest go
   *  unmeasured. */
  /**
   * How many bytes one submission may hand out before it goes early.
   *
   * 4 GB against a measured failure at 190 GB — a wide margin on purpose,
   * because the real edge belongs to the driver. See `run`.
   */
  private static readonly MAX_BATCH_BYTES = 4 * 1024 * 1024 * 1024;

  private static readonly MAX_QUERIES = 4096;

  /**
   * Runs `body` while measuring. **It always turns off afterwards —
   * including on the way out through an exception.**
   *
   * Turning it on and off must not be left to the caller. While profiling,
   * each dispatch opens a pass and the time inflates, and if it leaks out
   * still on, **every measurement after it comes out quietly inflated.** A
   * benchmark measures several batches, so an exception in one batch makes
   * the next batch's ms/step a profiled number rather than a measurement —
   * and it prints on screen looking exactly the same. There should be one
   * door, and the door should clean up.
   */
  async profile<T>(body: () => Promise<T>): Promise<T> {
    this.profiling = true;
    this.nsByKind.clear();
    this.countByKind.clear();
    this.queryUsed = 0;
    this.queryKinds = [];
    this.profileDropped = 0;
    try {
      return await body();
    } finally {
      this.profiling = false;
      await this.collectProfile();
    }
  }

  /** Opens a compute pass. Normally one is shared; while profiling one is opened per
   *  dispatch. */
  private openPass(): GPUComputePassEncoder {
    if (!this.profiling) {
      if (!this.pass) this.pass = this.openEncoder().beginComputePass();
      return this.pass;
    }
    // While profiling — close the previous pass and open a new one with timestamps.
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    const encoder = this.encoder ?? (this.encoder = this.device.createCommandEncoder());
    this.querySet ??= this.device.createQuerySet({
      type: "timestamp", count: Device.MAX_QUERIES,
    });
    if (this.queryUsed + 2 > Device.MAX_QUERIES) {
      // With no room it simply opens as usual — **what was not measured must not be
      // counted as 0.** It is counted, though: uncounted, a truncated table looks exactly
      // like a complete one.
      this.profileDropped += 1;
      this.pass = encoder.beginComputePass();
      return this.pass;
    }
    const at = this.queryUsed;
    this.queryUsed += 2;
    this.queryKinds.push(this.currentSig);
    this.pass = encoder.beginComputePass({
      timestampWrites: {
        querySet: this.querySet,
        beginningOfPassWriteIndex: at,
        endOfPassWriteIndex: at + 1,
      },
    });
    return this.pass;
  }

  /**
   * Reads the stamped timestamps and sums them per kind. **It has to be called after
   * submission.**
   *
   * The resolve buffer and the read buffer are built and thrown away as needed — the
   * profile is a rarely travelled path, so a pool is not worth it, and a pool would have
   * the measuring apparatus touch what it measures.
   */
  private async collectProfile(): Promise<void> {
    if (!this.querySet || this.queryUsed === 0) return;
    const count = this.queryUsed;
    const kinds = this.queryKinds;
    this.queryUsed = 0;
    this.queryKinds = [];
    const bytes = count * 8;
    const resolved = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const stage = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.resolveQuerySet(this.querySet, 0, count, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, stage, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await this.kicked(stage.mapAsync(GPUMapMode.READ));
    const times = new BigUint64Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    resolved.destroy();
    for (const [i, kind] of kinds.entries()) {
      const start = times[i * 2];
      const end = times[i * 2 + 1];
      if (start === undefined || end === undefined || end <= start) continue;
      this.nsByKind.set(kind, (this.nsByKind.get(kind) ?? 0) + Number(end - start));
      this.countByKind.set(kind, (this.countByKind.get(kind) ?? 0) + 1);
    }
  }

  /** Opens the encoder. An open compute pass is closed — a copy has to be outside
   *  it. */
  private openEncoder(): GPUCommandEncoder {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    this.encoder ??= this.device.createCommandEncoder();
    return this.encoder;
  }

  /**
   * Submits the stacked commands.
   *
   * It has to be passed before values are read — reading the result of
   * unsubmitted commands returns the old value.
   */
  flush(): void {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
    if (!this.encoder) return;
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    this.sinceSubmit = 0;
    this.submits += 1;
  }

  /**
   * Waits until the submitted work has **actually finished.** Where
   * `torch.cuda.synchronize()` goes.
   *
   * `flush()` returns having only put things on the queue — time it with
   * that and the wall clock has already stopped while the GPU is still
   * working. Until now the way this repository forced completion was to
   * read one value (`item()`), and that **mixes the readback round trip
   * into the measurement.** It is the place where "am I measuring the
   * kernel or the bus" gets blurred, and this function is what separates
   * them.
   */
  async synchronize(): Promise<void> {
    if (Device.readbackKicks) {
      // **Where the browser needs kicking, `onSubmittedWorkDone` does not answer to
      // kicks — a mapped four-byte copy does.** Measured on the 5080 (`roundtrip:probe`,
      // 2026-09-20): a 2048³ matmul followed by this method on the queue's promise was
      // 3.0–3.2 ms with kicks for 1.1 of GPU, while a one-element `toArray()` after the
      // same matmul was 1.09. So the wait here is the same wait a readback makes: a copy
      // of the first word of a scratch buffer into staging, on the encoder the work is
      // on, submitted with it, mapped and kicked.
      const free = this.stagingFree.get(BYTES_PER_F32) ?? [];
      this.stagingFree.set(BYTES_PER_F32, free);
      const stage = free.pop() ?? this.device.createBuffer({
        size: BYTES_PER_F32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.openEncoder().copyBufferToBuffer(this.syncScratch(), 0, stage, 0, BYTES_PER_F32);
      this.flush();
      await this.kicked(stage.mapAsync(GPUMapMode.READ));
      stage.unmap();
      free.push(stage);
    } else {
      this.flush();
      await this.device.queue.onSubmittedWorkDone();
    }
    await this.drainAllocations();
  }

  /** One word the kicked `synchronize` copies from; made on first use, never read. */
  private scratch: GPUBuffer | null = null;
  private syncScratch(): GPUBuffer {
    this.scratch ??= this.device.createBuffer({ size: BYTES_PER_F32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    return this.scratch;
  }

  /**
   * **Reads the out-of-memory scopes parked by `alloc`.** A non-null one is an
   * allocation the device could not make — folded into `faults` exactly as the
   * uncaptured handler would have (count, `outOfMemory`, and `first` if it is the first
   * word of a fault), so the throw already in `read` surfaces it and every existing
   * reader of `faults` is unchanged. Awaiting an empty list is a resolved `Promise.all`,
   * which is what every step after warm-up hits.
   */
  private async drainAllocations(): Promise<void> {
    if (this.oomPending.length === 0) return;
    const pending = this.oomPending;
    this.oomPending = [];
    const errors = await Promise.all(pending);
    for (const err of errors) {
      if (err === null) continue;
      this.faults.count += 1;
      this.faults.outOfMemory += 1;
      if (this.faults.first === "") this.faults.first = err.message;
    }
  }

  async read(buffer: GPUBuffer, count: number): Promise<Float32Array> {
    // **A lost device stops here.**
    //
    // Commands issued to a lost device throw nothing and simply do not run (the WebGPU
    // specification says so). So the training loop keeps going, the loss does not move,
    // and `ms/step` comes out perfectly well — **the same screen** as a validation error,
    // and that place was already blocked with `faults`. The same reasoning applied here
    // and only this side was empty.
    //
    // It sits where a value goes out. Checked per dispatch it would be checked 429 times,
    // and more to the point, **this is the moment a number becomes one a person
    // believes.**
    if (this.lost) {
      throw new Error(
        `the WebGPU device was lost (${this.lost.reason}) — nothing after this means ` +
          `anything.\n  ${this.lost.message}\n` +
          "  Reload the page to get a device again.",
      );
    }
    // Reading an empty tensor has to give something empty. A buffer claims at least one
    // cell, and reading that as it is brings along an element that does not exist.
    if (count === 0) return new Float32Array(0);
    const bytes = Math.max(count * BYTES_PER_F32, BYTES_PER_F32);
    let free = this.stagingFree.get(bytes);
    if (!free) {
      free = [];
      this.stagingFree.set(bytes, free);
    }
    const stage = free.pop() ?? this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      // The accumulated commands ride the same encoder and go out **once, here.**
      this.openEncoder().copyBufferToBuffer(buffer, 0, stage, 0, bytes);
      this.flush();
      // Read the parked OOM scopes before this value is trusted — a failed allocation
      // upstream makes this readback a value of nothing, and the throw below is what
      // says so. Draining here folds it into `faults` so that throw fires.
      await this.drainAllocations();
      await this.kicked(stage.mapAsync(GPUMapMode.READ));
      // Mapped memory disappears on unmap. It is always copied before going out.
      const out = new Float32Array(stage.getMappedRange().slice(0));
      stage.unmap();
      // **A value read after a fault is a value of nothing.** An invalid pipeline or an
      // allocation that failed throws no exception; the command is dropped and the
      // buffer read holds whatever the pool last held. Three times in one day that came
      // out as a number — a forward of "0.5 ms", a loss of 0.000, a gate that passed —
      // and only `faults.count` said otherwise. So the first readback after a new fault
      // throws, with the first fault's words (the catch below destroys this staging
      // buffer, as for any failure), and reading resumes after: a page may choose to go
      // on, and the count on `device().faults` keeps the whole story.
      if (this.faults.count > this.faultsReported) {
        const fresh = this.faults.count - this.faultsReported;
        this.faultsReported = this.faults.count;
        throw new Error(
          `${fresh} WebGPU fault(s) since the last readback — this value is of nothing.\n` +
            `  first: ${this.faults.first}\n` +
            "  An invalid pipeline or a failed allocation drops its command without an " +
            "exception; the count is on device().faults.");
      }
      // **Only a success returns it.** A failed buffer's mapping state is unknown, and
      // putting it in the pool hands the broken state to the next caller, moving the
      // cause one step further away.
      free.push(stage);
      return out;
    } catch (err) {
      stage.destroy();
      throw err;
    }
  }

  /** The workgroup size. The kernels and the device have to see the same value. */
  static readonly workgroup = WORKGROUP;

  /**
   * Which adapter it attached to. A value anyone measuring performance must
   * record alongside.
   */
  static adapterInfo = "(not attached yet)";

  /**
   * Optional features the adapter offers. **`timestamp-query` has to be
   * here for per-kernel timing.**
   *
   * A wall clock can only measure the whole step, and then there is no way
   * to ask which of 429 dispatches is expensive — which is exactly where
   * this got stuck.
   */
  static adapterFeatures = "";

  /** Whether the device was built with subgroup matrices (f32, 8 × 8 × 8) — see
   *  `create`. `matmul` asks this before choosing its kernel. */
  static subgroupMatrix = false;
  /** The int8 subgroup-matrix configuration the device was built with (`i8 × i8 → i32`,
   *  16 × 16 × 32), or `null` — see `create`. `matmulInt8` runs on it. */
  static subgroupInt8: SubgroupInt8Config | null = null;
  /** Whether the device has subgroup operations (`subgroupAdd`, `subgroupMax`). */
  static subgroups = false;
  /**
   * Whether the device was built with `shader-f16` — half-precision arithmetic and storage
   * in WGSL. **Optional and not universal**: Apple Metal has it, a recent NVIDIA card
   * through Chrome on Linux/Vulkan does not (measured, `docs/SCALE-MEASURED.md`). A kernel
   * with a half-precision path asks this first; without it, the f32 kernel runs and the
   * program answers the same, only larger and slower. `docs/SCALE.md` Step 4.
   */
  static f16 = false;

  /** The adapter's workgroup storage in bytes — 16 KB is the guaranteed floor, Apple
   *  gives 32 KB. A kernel that stages more than the floor asks this first. */
  static workgroupStorage = 16384;

  /** The re-tiled scalar GEMM's configurations for this adapter, best first — see
   *  `gemmConfigsFor`. Empty is the tile as it was. Settable, so a test can force a path. */
  static gemmConfigs: readonly TiledConfig[] = [];

  // ── Kernel selection by measurement (`docs/COMPILER.md` Step 5) ─────────────────────
  //
  // Where a hand rule chooses among kernels — the convolution's path and tile, the
  // product's tile — the choice is a `TuneCandidate` list with the rule's pick first, and
  // `choose` decides: a cached decision for this adapter and key, if there is one; else,
  // while a `compiled` step is collecting, the list is queued and the rule's pick runs;
  // else the rule's pick. `runTuning` then times every queued list with the timestamp
  // profiler and caches the fastest by `(adapter, key)` — in memory and in
  // `localStorage`, so the next page load pays nothing. The hand rules stay the prior
  // and the whole answer where timestamps are not available.

  /** Whether the device has `timestamp-query` — the autotune's instrument. */
  static canTime = false;
  /** Decisions by `adapter|key` → the candidate's label. */
  static tune = new Map<string, string>();
  private static readonly TUNE_STORE = "borch-ts.tune.v1";
  /** What a `compiled` step is doing with choices: collecting candidates, or nothing. */
  tuneMode: "collect" | null = null;
  /** While the tuner pre-warms, `pipeline` makes misses asynchronously and throws `PrewarmMiss`. */
  private prewarming = false;
  private prewarmPending: Promise<void>[] = [];
  /** The last `runTuning`'s warm wave in ms — the candidates' pipelines compiling, and one
   *  wait. Reported apart from the timing, since it is the platform's compile cost. */
  tuneWarmMs = 0;
  /** The pipelines the warm waves compiled so far — the tuner's candidates and, for a pure
   *  `compiled` step, the step's own on its first call — each with its wall time (they
   *  compile side by side, so the times overlap; the order says which are dear) and its
   *  WGSL size. */
  tuneCompiles: { sig: string; ms: number; bytes: number }[] = [];

  /** Whether `err` is `pipeline` refusing a miss while compiling ahead — the caller runs
   *  its step again after `awaitCompiles`. */
  static isCompileMiss(err: unknown): boolean {
    return err instanceof PrewarmMiss;
  }

  /** Turns compiling-ahead on or off: on, a pipeline miss is made asynchronously and
   *  thrown as a miss (`isCompileMiss`); off, misses compile as they come. */
  compileAhead(on: boolean): void {
    this.prewarming = on;
  }

  /** Waits for every pipeline a wave started. */
  async awaitCompiles(): Promise<void> {
    const pending = this.prewarmPending;
    this.prewarmPending = [];
    await Promise.all(pending);
  }

  /** The candidate lists queued since the last take — a recording's, for its tuner. */
  takeTuneQueue(): Map<string, readonly TuneCandidate[]> {
    const q = new Map(this.tuneQueue);
    this.tuneQueue.clear();
    return q;
  }
  private readonly tuneQueue = new Map<string, readonly TuneCandidate[]>();

  /** Loads the decisions saved by earlier sessions on this adapter. */
  static loadTune(): void {
    try {
      const raw = globalThis.localStorage?.getItem(Device.TUNE_STORE);
      if (!raw) return;
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed !== "object" || parsed === null) return;
      for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) if (typeof v === "string") Device.tune.set(k, v);
    } catch { /* no storage, or none of ours — the rules decide */ }
  }

  private static saveTune(): void {
    try { globalThis.localStorage?.setItem(Device.TUNE_STORE, JSON.stringify(Object.fromEntries(Device.tune))); } catch { /* not persisted */ }
  }

  /** The candidate that runs for `key`: the cached decision, or the rule's pick (first). */
  choose(key: string, candidates: readonly TuneCandidate[]): TuneCandidate {
    const first = candidates[0];
    if (first === undefined) throw new Error(`choose(${key}): no candidates`);
    if (candidates.length === 1) return first;
    const k = `${Device.adapterInfo}|${key}`;
    const cached = Device.tune.get(k);
    if (cached !== undefined) {
      const hit = candidates.find((c) => c.label === cached);
      if (hit) return hit;
    }
    if (this.tuneMode === "collect" && !this.tuneQueue.has(k)) this.tuneQueue.set(k, candidates);
    return first;
  }

  /**
   * Times every queued candidate list and caches the fastest. Returns one line per
   * decision, for the test that holds "nothing tuned is slower than the rule's pick" and
   * the report that says what changed.
   *
   * **The cost is round trips and shader compiles, not GPU time — so both are paid
   * once, not per candidate.** Measured on metal-3 (2026-09-21, the ResNet-18 step's ten
   * candidates): a profiled round was 0.3–0.5 ms of readback when its pipeline was
   * compiled and 4–7 ms when it was not, and a round per candidate per repetition made
   * 45–50 ms of the first call. So every candidate runs once first, unprofiled, with
   * one wait — the pipelines compile there, side by side in the GPU process — and then
   * each of `TUNE_ROUNDS` rounds is one profiled pass over every candidate of every
   * decision, `TUNE_REPS` repetitions each, one readback for the lot; a candidate's
   * time is the minimum over rounds of its kinds' summed nanoseconds. Two candidates
   * whose kernel kinds overlap cannot share a pass (their times would add), so the
   * candidates are dealt into passes with no kind repeated.
   */
  async runTuning(queue: Map<string, readonly TuneCandidate[]> = this.takeTuneQueue()): Promise<TuneReport[]> {
    const out: TuneReport[] = [];
    if (queue.size === 0) return out;
    const all: { k: string; i: number; cand: TuneCandidate }[] = [];
    for (const [k, candidates] of queue) candidates.forEach((cand, i) => all.push({ k, i, cand }));
    // Warm: every candidate once, one wait. The compiles happen here — asynchronously,
    // in waves: a candidate whose pipeline is missing throws `PrewarmMiss` from `pipeline`
    // and runs again after every pending compile has resolved (a candidate can miss more
    // than once — its convolution, then its split sum).
    const tw = performance.now();
    this.prewarming = true;
    try {
      let pending = all.map((e) => e.cand);
      for (let wave = 0; pending.length > 0 && wave < 8; wave++) {
        const missed: TuneCandidate[] = [];
        for (const cand of pending) {
          try { cand.run(); } catch (err) { if (err instanceof PrewarmMiss) missed.push(cand); else throw err; }
        }
        await Promise.all(this.prewarmPending);
        this.prewarmPending = [];
        pending = missed;
      }
    } finally {
      this.prewarming = false;
      this.prewarmPending = [];
    }
    this.flush();
    await this.synchronize();
    this.tuneWarmMs = performance.now() - tw;
    // Passes with no kernel kind repeated.
    const passes: { k: string; i: number; cand: TuneCandidate }[][] = [];
    for (const entry of all) {
      let placed = false;
      for (const pass of passes) {
        const used = new Set(pass.flatMap((e) => e.cand.keys));
        if (entry.cand.keys.some((key) => used.has(key))) continue;
        pass.push(entry); placed = true; break;
      }
      if (!placed) passes.push([entry]);
    }
    const best = new Map<string, number>();   // `${k}#${i}` → ms
    for (let round = 0; round < Device.TUNE_ROUNDS; round++) {
      for (const pass of passes) {
        await this.profile(async () => {
          for (const { cand } of pass) for (let r = 0; r < Device.TUNE_REPS; r++) cand.run();
          this.flush();
          await this.synchronize();
        });
        for (const { k, i, cand } of pass) {
          const ns = cand.keys.reduce((a, key) => a + (this.nsByKind.get(key) ?? 0), 0);
          const ms = ns / 1e6 / Device.TUNE_REPS;
          const id = `${k}#${i}`;
          best.set(id, Math.min(best.get(id) ?? Infinity, ms));
        }
      }
    }
    for (const [k, candidates] of queue) {
      const ms = candidates.map((_, i) => best.get(`${k}#${i}`) ?? Infinity);
      let pick = 0;
      for (let i = 1; i < ms.length; i++) if ((ms[i] ?? Infinity) < (ms[pick] ?? Infinity)) pick = i;
      const chosen = candidates[pick];
      if (chosen) Device.tune.set(k, chosen.label);
      out.push({ key: k, prior: candidates[0]?.label ?? "", priorMs: ms[0] ?? 0, chosen: chosen?.label ?? "", chosenMs: ms[pick] ?? 0, candidates: candidates.map((c, i) => `${c.label} ${(ms[i] ?? 0).toFixed(3)}`) });
    }
    if (out.length) Device.saveTune();
    return out;
  }
  private static readonly TUNE_ROUNDS = 2;
  private static readonly TUNE_REPS = 5;

  /** How many storage buffers one compute stage may bind — 8 is the guaranteed floor.
   *  The fusion pass sizes its trees by this. */
  static storageBuffersPerStage = 8;

  /**
   * **A soft allocation budget in bytes — `0` is off, the default.** When set, `alloc`
   * throws before making a buffer that would carry the live footprint past it, rather
   * than letting WebGPU fail the allocation silently and read back as zeros (there is no
   * synchronous OOM signal — see `oomPending`). It does not reclaim; freeing is the
   * caller's, so the check never adds a submit to a step. A ceiling probe
   * (`tests/browser/ceiling.py`) measures what a device can hold; a caller that wants a
   * guard rail sets this from that number.
   */
  static budget = 0;

  /** The storage-binding offset alignment (256 on the cards here). A window's slots must
   *  start on a multiple of it, or `bindGroupFor` refuses the sub-range. */
  get storageAlign(): number {
    return this.limits.minStorageBufferOffsetAlignment;
  }

  /** A host-writable staging buffer (`MAP_WRITE | COPY_SRC`), for filling a window through
   *  `copyRange` rather than `writeBuffer`. Not pooled — the window owns and destroys it. */
  stagingBuffer(bytes: number): GPUBuffer {
    return this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    });
  }

  /**
   * **A frozen-weight window** — `docs/SCALE.md` Step 3, ADR-003. One large STORAGE buffer
   * that holds many weights end to end, each bound as an offset slice; filled through a
   * host-writable staging buffer and `copyRange` (never `writeBuffer`, which jumps the
   * queue and would overwrite a slot a pending dispatch still reads). The window lives
   * outside the pool and is `keep`-ed, so a scope close does not reclaim it; `free()`
   * returns it. `bytes` is capped at the device's binding tier — a window sized to one
   * adapter's tier (Apple's 4 GiB) would fail to bind on another (the RTX 5080's 2 GiB),
   * so a caller reads the tier per device.
   */
  window(bytes: number): Window {
    const cap = Math.min(bytes, this.limits.maxStorageBufferBindingSize);
    const buffer = this.device.createBuffer({
      size: cap,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.keep(buffer);
    return new Window(this, buffer, cap);
  }
}

/**
 * A window's fill cursor and staging. Kept small on purpose — the scheduling (which block
 * is resident, prefetch, eviction) is the caller's; this only owns the buffer, lays slots
 * end to end on the alignment, and fills them correctly-ordered.
 */
export class Window {
  /** Next free byte offset, always a multiple of the storage alignment. */
  private cursor = 0;
  /** One host-writable staging buffer, grown to the largest slot seen. `place` maps it,
   *  copies through it, and waits — a ring would let fills overlap, which the scheduler can
   *  add when a measured need appears. */
  private staging: GPUBuffer | null = null;
  private stagingBytes = 0;
  /**
   * **The generation of each slot offset.** Bumped every time an offset is (re)written, so
   * a tensor that recorded the generation it was placed at can tell whether its slot still
   * holds its weight. The window is one buffer with many slots, so the whole-buffer age
   * guard is too coarse — evicting one block would kill every windowed tensor; this is
   * per-slot. `docs/SCALE.md` Step 3, ADR-003 decision 2 (eviction gates on liveness).
   */
  private readonly gens = new Map<number, number>();
  private tick = 0;
  /**
   * **Freed regions available for reuse**, each `{offset, size}` with `size` the aligned
   * reservation. `evict` returns a slot's region here; `place` takes one (first fit) before
   * growing the cursor. This is what bounds the window: streaming a hundred blocks through a
   * window sized for three reuses the same bytes, so the buffer never grows past the few
   * blocks resident at once. No coalescing — the streaming case is equal-sized blocks, and a
   * freed region is taken whole. `docs/SCALE.md` Step 3 ④.
   */
  private readonly freed: { offset: number; size: number }[] = [];
  /** The reserved (aligned) size of each live slot, by offset — what `evict` returns to the
   *  free list. */
  private readonly reserved = new Map<number, number>();

  constructor(
    private readonly dev: Device,
    readonly buffer: GPUBuffer,
    readonly capacity: number,
  ) {}

  /** The high-water mark of the cursor, in bytes — the most the window ever grew to. With
   *  eviction and reuse this stays near the resident set, not the total streamed. */
  get used(): number {
    return this.cursor;
  }

  /** Bytes currently placed and not evicted — the live resident set. */
  get live(): number {
    let n = 0;
    for (const size of this.reserved.values()) n += size;
    return n;
  }

  /**
   * Places `data` at the next aligned slot and returns the binding for it. The copy rides
   * a staging buffer (`MAP_WRITE`), so it is ordered against pending dispatches; the wait
   * afterwards is what lets the one staging buffer serve the next `place`.
   */
  async place(data: Float32Array | Uint16Array | Uint32Array): Promise<BindSlot> {
    const bytes = data.byteLength;
    const align = this.dev.storageAlign;
    const need = Math.ceil(bytes / align) * align;   // aligned reservation
    // Reuse a freed region first (first fit), so a stream of blocks does not grow the
    // window past the few resident at once; only grow the cursor when nothing fits.
    let offset: number;
    const hit = this.freed.findIndex((r) => r.size >= need);
    if (hit >= 0) {
      offset = (this.freed[hit] as { offset: number }).offset;
      this.freed.splice(hit, 1);
    } else {
      offset = this.cursor;
      if (offset + need > this.capacity) {
        throw new Error(
          `window is full: ${offset} + ${need} > ${this.capacity} bytes, and no freed ` +
            "region fits. Evict a slot, or size the window larger (up to the binding tier).",
        );
      }
      this.cursor = offset + need;
    }
    await this.write(offset, data);
    const slot: BindSlot = { buffer: this.buffer, offset, size: bytes };
    // Under a capture the recording keeps the host bytes and the slot, not the staging
    // copy (`Recorded.refill`); a replay writes the same bytes into the same slot.
    this.dev.recordRefill(this, slot, data);
    this.reserved.set(offset, need);
    this.tick += 1;
    this.gens.set(offset, this.tick);
    return slot;
  }

  /** The bytes of a weight into the slot a recording placed it in — a replay's refill. */
  async refill(slot: BindSlot, data: Float32Array | Uint16Array | Uint32Array): Promise<void> {
    if (slot instanceof GPUBuffer || slot.buffer !== this.buffer) throw new Error("a refill names a slot of this window");
    await this.write(slot.offset, data);
  }

  /** Host bytes into the window at `offset`, through the staging map; the copy is never
   *  recorded (see `recordRefill`), and it is submitted but not waited for. */
  private async write(offset: number, data: Float32Array | Uint16Array | Uint32Array): Promise<void> {
    const bytes = data.byteLength;
    if (this.staging === null || this.stagingBytes < bytes) {
      this.staging?.destroy();
      this.staging = this.dev.stagingBuffer(bytes);
      this.stagingBytes = bytes;
    }
    await this.staging.mapAsync(GPUMapMode.WRITE);
    // Copy raw bytes, so an f32 weight and a half-precision (Uint16Array) one take the
    // same path — the window does not care which, only how many bytes.
    new Uint8Array(this.staging.getMappedRange(0, bytes)).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    this.staging.unmap();
    const staging = this.staging;
    this.dev.unrecorded(() => this.dev.copyRange(this.buffer, offset, staging, 0, bytes));
    // **Submit the copy, but do not block on it.** It only has to be *submitted* here so the
    // staging buffer is not held by an open encoder; it does not have to be *done*, because
    // the block that reads this slot runs in a later submit and the queue keeps submit order
    // — the bytes are in place before they are read. The wait moves to the next `place`'s
    // `mapAsync`, which needs only this staging buffer free, and by then the block between
    // has run: the upload overlaps the compute instead of stalling the GPU every slot. (A
    // ring of staging buffers would overlap two uploads too; this removes the full stall
    // with none of that machinery.)
    this.dev.flush();
  }

  /** The current generation of the slot at `offset`, `-1` if nothing was ever placed there.
   *  A windowed tensor records this at creation and compares on every use. */
  genOf(offset: number): number {
    return this.gens.get(offset) ?? -1;
  }

  /**
   * **Evicts a slot** — bumps its generation so any tensor still pointing at it throws on
   * next use (its `weightBinding` sees the generation moved), rather than reading whatever
   * is placed there next. This is the liveness gate ADR-003 decision 2 rests on: eviction
   * is safe because a stale read is loud, not silent. The region returns to the free list,
   * so the next `place` reuses it — this is what keeps the window bounded while a stream of
   * blocks passes through it.
   */
  evict(slot: BindSlot): void {
    if (slot instanceof GPUBuffer) return;
    this.tick += 1;
    this.gens.set(slot.offset, this.tick);
    const size = this.reserved.get(slot.offset);
    if (size !== undefined) {
      this.reserved.delete(slot.offset);
      this.freed.push({ offset: slot.offset, size });
    }
  }

  /** Returns the window and its staging to the driver. The slots' bindings are dead after. */
  free(): void {
    this.staging?.destroy();
    this.staging = null;
    this.dev.unkeep(this.buffer);
  }
}
