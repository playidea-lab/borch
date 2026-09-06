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

import { grid1d, reduceParts, reduceSum, WORKGROUP } from "./kernels.js";

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
const NO_API =
  "There is no WebGPU. Chrome/Edge 113+ or Safari 18+ is required. " +
  "**Seeing this on a version that matches means it is switched off** — on Safari, " +
  "Settings → Advanced → Feature Flags → WebGPU; on Linux Chrome, " +
  "Unsafe WebGPU in chrome://flags. It has to be https or localhost.";

const NO_ADAPTER =
  "No WebGPU adapter could be obtained — a driver blocklist, a virtual machine, or a " +
  "headless environment with no GPU.";

/** Which adapter, on one line. Empty fields are dropped — the browser hides most of
 *  them. */
function describe(adapter: GPUAdapter): string {
  const info: Partial<GPUAdapterInfo> = adapter.info ?? {};
  return [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean).join(" / ") || "(unknown)";
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
  if (!("gpu" in navigator)) return { ok: false, why: "no-api", message: NO_API };
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

/** One dispatch as recorded under a capture. */
export interface Recorded {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly groups: readonly [number, number, number];
  /** The buffers behind the bind group, in binding order — what a fusion pass reads. */
  readonly buffers: readonly GPUBuffer[];
}

/**
 * A recorded step. `replay()` issues its dispatches again, in order, into the current
 * batch — the values land in the same buffers, so a tensor made during the capture (the
 * loss, the parameters) reads the new step's result. `dispose()` returns the memory.
 */
export class Capture {
  constructor(
    private readonly dev: Device,
    private readonly records: readonly Recorded[],
    private readonly pinned: Set<GPUBuffer>,
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
    const ids = new Map<GPUBuffer, number>();
    const id = (b: GPUBuffer): number => {
      let n = ids.get(b);
      if (n === undefined) { n = ids.size; ids.set(b, n); }
      return n;
    };
    return this.records.map((r) => ({
      key: this.dev.keyOf(r.pipeline) ?? "?", groups: r.groups,
      buffers: r.buffers.map(id), sizes: r.buffers.map((b) => b.size),
    }));
  }

  replay(): void {
    this.dev.replayRecorded(this.records);
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
    if (!("gpu" in navigator)) throw new Error(NO_API);
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
    };
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
    const features: GPUFeatureName[] = [];
    if (canTime) features.push("timestamp-query");
    if (sgm) features.push("subgroups" as GPUFeatureName, "chromium-experimental-subgroup-matrix" as GPUFeatureName);
    const descriptor = {
      requiredLimits: want,
      requiredFeatures: features,
    };
    Device.subgroupMatrix = sgm;
    Device.workgroupStorage = adapter.limits.maxComputeWorkgroupStorageSize;

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
    return made;
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

  beginCapture(): void {
    if (this.recording) throw new Error("a capture is already open");
    this.recording = [];
    this.pinned = new Set();
  }

  endCapture(): Capture {
    if (!this.recording || !this.pinned) throw new Error("no capture is open");
    const capture = new Capture(this, this.recording, this.pinned);
    this.recording = null;
    this.pinned = null;
    return capture;
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
  replayRecorded(records: readonly Recorded[]): void {
    for (const r of records) {
      const pass = this.openPass();
      pass.setPipeline(r.pipeline);
      pass.setBindGroup(0, r.bindGroup);
      pass.dispatchWorkgroups(r.groups[0], r.groups[1], r.groups[2]);
      this.dispatches += 1;
    }
  }

  /** Hands a capture's pinned buffers back to the pool. Called by `Capture.dispose`. */
  unpin(buffers: Iterable<GPUBuffer>): void {
    for (const buf of buffers) {
      const size = this.sizes.get(buf);
      if (size === undefined) { buf.destroy(); continue; }
      let pool = this.spare.get(size);
      if (!pool) { pool = []; this.spare.set(size, pool); }
      pool.push(buf);
    }
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
      if (this.pinned?.has(buf)) continue;
      this.retire(buf);
      // Returned to the pool rather than destroyed. The next step asks for the same
      // size again.
      const size = this.sizes.get(buf);
      if (size === undefined) {
        // What arrives here is a buffer `alloc` did not build. Unsubmitted commands may
        // point at it, so it is released after they go out.
        this.flush();
        buf.destroy();
      } else {
        let pool = this.spare.get(size);
        if (!pool) {
          pool = [];
          this.spare.set(size, pool);
        }
        pool.push(buf);
      }
      freed += 1;
    }
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

  /**
   * Keeps something alive regardless of scope. Parameters and optimizer
   * state use it.
   */
  keep(buffer: GPUBuffer): void {
    this.kept.add(buffer);
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
    const buf = reused ?? this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.pinned?.add(buf);
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
    buffers: readonly GPUBuffer[],
    groups: readonly [number, number, number],
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
    let layout = this.layouts.get(pipeline);
    if (!layout) {
      layout = pipeline.getBindGroupLayout(0);
      this.layouts.set(pipeline, layout);
    }
    const bindGroup = this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = this.openPass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    this.dispatches += 1;
    this.recording?.push({ pipeline, bindGroup, groups: [groups[0], groups[1], groups[2]], buffers: [...buffers] });
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
  run1d(pipeline: GPUComputePipeline, buffers: readonly GPUBuffer[], n: number): void {
    const g = grid1d(n);
    this.run(pipeline, buffers, [g.x, g.y, 1]);
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
      this.run1d(
        this.pipeline(`reduceSum:${size}`, () => reduceSum(size)),
        [src, dst],
        size,
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
    await stage.mapAsync(GPUMapMode.READ);
    const times = new BigUint64Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    resolved.destroy();
    for (const [i, kind] of kinds.entries()) {
      const start = times[i * 2];
      const end = times[i * 2 + 1];
      if (start === undefined || end === undefined || end <= start) continue;
      this.nsByKind.set(kind, (this.nsByKind.get(kind) ?? 0) + Number(end - start));
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
    this.flush();
    await this.device.queue.onSubmittedWorkDone();
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
      await stage.mapAsync(GPUMapMode.READ);
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

  /** The adapter's workgroup storage in bytes — 16 KB is the guaranteed floor, Apple
   *  gives 32 KB. A kernel that stages more than the floor asks this first. */
  static workgroupStorage = 16384;
}
