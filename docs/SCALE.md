# Scale — where the memory ceiling is, and the eight steps that raise it

> Plan, written 2026-09-18 from a code map and an external survey (both cited inline).
> The decision it implements is ADR-003 in [WEBGPU-DESIGN.md](../WEBGPU-DESIGN.md) §10.
> Everything here lives in `borch-ts` and its neighbours (`borch-hub`, `bimm-ts`); the
> numpy core is untouched — ADR-001 in [ROADMAP.md](../ROADMAP.md) stands.

## 0. What the ceiling is made of

borch trains and infers on WebGPU with **f32 storage only** and **whole-model residency**.
That is not one limit but four, each with its own wall:

| wall | where it stands today | number |
|---|---|---|
| bytes per parameter | `DType = "float32"\|"int64"\|"bool"\|"complex64"` (`src/dtype.ts:38`); every buffer is `count × 4` bytes (`src/device.ts:23`, `:1161`) | 4 B/param, no other width exists |
| residency | `hub.load` decodes the whole file and uploads every tensor (`borch-hub load.ts:509-513`, `src/serialize.ts:143-198`); nothing evicts (`grep offload\|evict\|prefetch\|stream` over `src/` — 0 hits) | peak = file + per-tensor host copies + full GPU residency, **simultaneously** |
| one buffer | `maxStorageBufferBindingSize`, requested at the adapter's maximum (`device.ts:467-475`); Dawn reports **tiers**, not values | 128 MiB → 256 MiB / 1 / 2 / 4 GiB−4 (tiered) |
| the tab | no API reports total GPU memory (gpuweb #5505, #6957 open); Metal working set ≈ 75 % of RAM on Apple; iOS Safari tab < 500 MB | unknown until probed |
| the cache | weights go to the Cache API only (`load.ts:25,274,312`); a 346 MB `put` failed (`load.ts:306-309`) | the bigger the model, the likelier the cache is lost |

And one wall that is not memory at all: **OOM cannot be caught.** There is no
`pushErrorScope`/`popErrorScope` in `src/` (grep, 0 hits); `createBuffer` never throws
on OOM; the only signal is the asynchronous uncaptured-error listener
(`device.ts:530-555`) surfacing as `faults.outOfMemory` at the *next readback*
(`:1652-1660`). A design that "tries an allocation and falls back" has nothing to
catch. That cost a day once ("returning zeros with this counter at 0 throughout",
`device.ts:594-611`).

**Why this enters at all** — the curriculum did not ask for scale, and the roadmap says
what it does not ask for does not go in. What asks is the workbench: its go/no-go is a
224 px pretrained backbone, a head and a neighbour ordering in two minutes
(`tests/browser/envelope.py`), and its frozen path already loads a complete backbone to
run one `no_grad` pass over it (`borch/_workbench.py:379-392`). The 346 MB cache failure
is that path meeting the ceiling. So the demand is the tool's, and this plan raises the
ceiling **only as far as a frozen backbone plus an adapter needs** — not to an LLM
(§10 says why it stops there).

## 1. What the survey settled (external, 2026-09)

- **`shader-f16`**: Chrome 120+, Safari 26 (all Apple devices), Firefox shipped. Measured
  availability (web3dsurvey): Apple 99.99 %, ARM 100 %, NVIDIA 83.8 %, **Qualcomm 71.1 %**
  — Adreno lacks 16-bit storage access (gpuweb #5006), so an f32 path stays mandatory.
  Throughput gain +25–50 % where it exists. Subgroups (Chrome 134) need both features
  requested together. No standard cooperative-matrix.
- **Limits**: spec defaults 256 MiB buffer / 128 MiB binding / 8 storage buffers per stage /
  16 KiB workgroup memory; Dawn tiers above. No total-VRAM API; the practice is probe
  allocations under an out-of-memory error scope.
- **What others do**: WebLLM, ORT Web and llama.cpp-web keep the **quantized model fully
  GPU-resident** — no VRAM offload anywhere (Llama-3.1-8B q4f16 = 4,598 MB). LlamaWeb
  (2026-05) streams OPFS → GPU through four 1 MB staging buffers, skipping the wasm heap:
  −49 % peak host memory. ORT Web is capped by the 4 GB wasm heap.
- **QLoRA** (paper): NF4 storage, 16-bit compute, dequantised **per layer in both the
  forward and the backward**. Memory is dominated by activation gradients, not the
  adapter (7B: LoRA 26 MB vs input gradients 567 MB, 18 MB/seq with checkpointing). No
  browser implementation exists; the nearest is llama.cpp's WebGPU backward kernels
  (PR #28269, 3.3–6.5× slower than CUDA).
- **Gradient checkpointing**: O(√n) activation memory for one extra forward (+20 %
  measured). No WebGPU or JS implementation exists, tfjs included.
- **Storage**: OPFS sync handles (Worker only) ≈ 1.1 GB/s (secondary source), 3–4× IDB.
  Chromium 60 % of disk per origin; Firefox 10 GiB without `persist()`; Safari evicts
  after 7 days unused. `navigator.storage.estimate()` before every large write.

## 2. What the code map settled (internal)

The load-bearing facts, each with the line that proves it. The plan's steps are cut
along these seams and nowhere else.

**Memory is the Device's, not the Tensor's.** Lifetime is scope frames
(`device.ts:752` stack, `:824` begin, `:991` end); a Tensor holds `gpu` but stamps an
`age` at birth (`tensor.ts:933`) and re-checks it on every `buffer`/`raw` access
(`refuseIfDead`, `tensor.ts:1024-1032`) — **every op passes through one of those two
getters**. Bumping a buffer's age (`retire`, `device.ts:820`) turns a stale read into a
loud throw. Explicit lifetime outside scopes already exists and is proven: `keep`
(`:1110`) / `unkeep` (`:1121`, flush-then-destroy, corrects the tally) — the SGD arena
lives on it (`optim.ts:716-721`).

**There is one allocation door, one binding door, two weight funnels.** `alloc`
(`device.ts:1160-1189`, pool hit at `:1174`, hard limit at `:1162-1170`); `bindGroupFor`
(`:1303-1326`) — and it already accepts a **slice binding** `BindSlot = GPUBuffer |
{buffer, offset, size}` (`:315`) that nothing in the package constructs yet. Weights are
bound in exactly two places: matmul (`tensor.ts:2099-2132`) and conv (`convForwardRun`,
`tensor.ts:779`). Embeddings via `indexSelect` (`kernels.ts:3303`).

**`writeBuffer` jumps the queue** (`device.ts:1152-1159`): it runs at the queue's
current position while dispatches are stacked for a later submit, so filling a *reused*
slot with it overwrites what an unsubmitted dispatch is still reading — silently. The
correctly ordered path is a staging buffer plus `copyRange` (`:1395`), which encodes into
the current encoder. No `MAP_WRITE` staging buffer exists yet. Every copy also ends the
open compute pass (`openEncoder`, `:1560-1567`) — a cost to measure, not assume.

**Tensor identity is sacred to the optimizer.** `Optimizer.params` is a flat array
captured once, state banks indexed by position (`optim.ts:150,169-172`). Storage may be
repointed in place — the precedent is `ensureOwned` (`tensor.ts:1250-1258`: alloc →
copy → rehome → `this.gpu = own; this.age = age(own)`) — but the Tensor object must
never be replaced.

**Capture/replay bakes buffer objects.** Replay re-issues baked bind groups
(`device.ts:900-903`; `fuse.ts:397/413`). Moving a weight to a new buffer under a live
capture means stale weights or a use-after-free read as zeros. Streaming and `compiled`
are therefore exclusive unless eviction **refills the same buffer object**.

**The arena is whole or nothing, SGD only.** `buildArena` (`optim.ts:708-734`) lays every
parameter into one slab and keeps four buffers; `arenaStep` (`:670-704`) is one dispatch
(2.8× measured); gate `paramGroups.length === 1 && !capturing && !suppressArena`
(`:660`). Adam/AdamW/RMSprop have none. It assumes every trainable parameter is
resident — which is fine, because **trainable parameters are the small half**.

**Kernels are strings with `f32` inside.** 162 generators in `kernels.ts`, `f32` hardcoded
441× (`vision.ts` 87, `tensor.ts` 36, `special.ts` 13), no scalar parameter. The pipeline
cache key is a hand-built `op:shape` string with **no dtype** (`device.ts:704-734`, 113
call sites); the profiler reads `signature.split(":")[0]` as the kind (`:707`), so a
precision tag belongs at the tail. `enable f16;` is module-level and has a precedent
(`kernels.ts:1681` prepends `enable chromium_experimental_subgroup_matrix;`). The fused
pass rewrites `array<f32>` → `array<vec4<f32>>` by string replace (`fuse.ts:374`).
`f32lit` (`kernels.ts:1011`) refuses inf/NaN and would need a 65504 guard.

**The subgroup GEMM cannot dequantise.** `subgroupMatrixLoad` reads storage directly with
offsets derived from `workgroup_id` alone (`kernels.ts:1650-1653,1676-1680`); there is no
place for a per-lane scale. The scalar tile stages A/B through workgroup memory
(`kernels.ts:2053-2056`) and is where int8 unpacks — at 4.5 TFLOP/s-class against the
subgroup path's 11.0 (`kernels.ts:1646`). **Quantisation buys memory and sells speed.**

**Readback is one line.** `read()` (`device.ts:1604-1670`) ends in
`new Float32Array(stage.getMappedRange().slice(0))` (`:1642`); `toArray`/`item` go
through it (`tensor.ts:13055-13061, 13292-13309`). `Tensor.from` converts *values*
(`tensor.ts:1150`), so f16 bits passed as `Uint16Array` become integers silently; the
per-dtype byte decoder precedent is `Tensor.frombuffer` (`:1368-1389`).

**The golden has one tolerance.** `atol = rtol = 1e-4`, defined once
(`tests/export_json.py:116`), frozen in `golden.json`, compared at
`borch-ts/test/golden.ts:78-82,186` — no per-case tolerance, and the runner may not set
its own (`:77`). f16 epsilon is 9.77e-4, five times over. Loosening the number would
blind the f32 regression check; f16 needs its own tier.

**Half precision was declined on stated conditions** (`tests/torch_gap.py:430-471`,
`src/index.ts:356-443`): an optional feature makes the same program answer differently
per machine and the golden loses an answer to freeze; the thing to re-measure is
"whether `shader-f16` has become effectively universal"; and a setter that accepts and
drops "is how a run trains in f32 believing it is in f16". §3 answers all three.

**The hub loads whole files.** Chunked transport with Range resume (`load.ts:344-474`),
concatenated into one `Uint8Array` (`:408-414`), whole-file sha256 before load
(`:322-335`, re-hashed on cache hits), Cache API only, then `decode` copies each tensor
(`serialize.ts:186-191`) and `loadStateDict` copies values into pre-allocated parameters
(`nn.ts:443-466`, never swapping the object). The safetensors header already carries
per-tensor `data_offsets` (`serialize.ts:170-196`) — lazy per-tensor loading needs **no
format change**. The manifest has no shards and no per-tensor index (`manifest.ts:98-104`).
Nothing in `borch-ts/src` calls `fetch`; byte acquisition is the caller's.

**bimm already has a GPU-free plan for every architecture**, and a second consumer of
it: `efficientnetPlan` / `resnetPlan` / `vitPlan` (`bimm-ts efficientnet.ts:113`,
`resnet50.ts:107`, `vit.ts:93`) are walked by the constructors *and* by `cpuGraphFor`
(`cpu.ts:150`), which builds an op graph from a flat state dict by timm key without
instantiating a module. A streaming loader is the **third consumer** of the same tables.
Friction: the plan functions are not re-exported from `bimm-ts` `index.ts`;
`forwardFeatures` calls `this.blocks.forward(h)` as one opaque call
(`efficientnet.ts:280`, `vit.ts:288`), so a scheduler wraps the `blocks` child.

**The workbench's frozen pass is the friendliest seam in the repository.** One `no_grad`
forward per batch, no tape, no backward, output pulled to host at once
(`borch/_workbench.py:388-392`). Evicting block k−1 mid-pass breaks nothing there. The
training path is the opposite: backward closures hold the weights.

**LoRA exists, for two layers, in TypeScript only.** `src/peft.ts` (215 lines):
`LoRALinear`, `LoRAConv2d` (`merge()` throws on grouped conv, `:191-194`), base frozen out
of `parameters()` by `registerBuffer` (`:59-61`), `adapterState()` (`:79-81`),
`fromLinear` (`:69-76`). No apply-to-model helper, no attention unit, no Python mirror
(`_ts.peft` has zero references in `borch_webgpu/`).

**Instrumentation is sufficient and needs nothing new.** `Device.memory` (leak question)
and `Device.pooled` (footprint question) are deliberately different numbers
(`device.ts:1043,1061`); `lastScope.survived` (`:1028`); `dispatches`, `submits`,
`profile()` (`:1465`). `borch-ts/test/cost.ts` freezes them per step on an
adapter-independent path and runs in CI.

## 3. The decisions (what ADR-003 fixes)

1. **Two memory regions with different owners.** *Trainable* parameters stay resident,
   scope-managed, arena-eligible — nothing about them changes. *Frozen* weights live in a
   **window**: one large buffer outside the pool, filled by staging + `copyRange`, bound
   as offset slices, evicted by age bump. The line between the two is
   `requiresGrad`, which `peft.ts` already draws.
2. **Eviction gates on liveness, not on a schedule.** The views lesson (ADR-001, 2026-09-08)
   was that a base's lifetime cannot be deferred while anything references it. A window
   slot is evictable only when no live Tensor points at it *and* no unsubmitted command
   reads it — `retire` + `flush` ordering, exactly what `unkeep` does today.
3. **Prefetch is one block ahead and never moves a value.** §3 of WEBGPU-DESIGN confines
   async to `.item()`; the window's only asynchrony is a staging map for the *next*
   block, resolved before that block's first dispatch is encoded. No pipelining that
   reorders arithmetic.
4. **Half precision is a request, not a default.** `autocast()` is an explicit scope; the
   f32 golden stays the frozen answer; f16 runs get a separate tolerance tier witnessed
   by hardware only. On a device without `shader-f16` the scope **throws** — it never
   runs f32 while claiming f16. The re-measurement the refusal asked for has been done:
   f16 is not universal (Qualcomm 71 %), which is why it can never be the default and
   why this is not a reversal of that refusal but the addition it left room for.
5. **Quantised weights are not Tensors.** Packed int8/int4 blobs stay raw `GPUBuffer`s
   inside the window and reach kernels through the two weight funnels. `DType` and
   `floatsPerElement` keep their invariant (`dtype.ts:50-60`); the numpy core never
   learns a narrow dtype.
6. **Streaming and `compiled` are exclusive** until refill-in-place is proven under a
   capture. Until then a window refuses `compiled` loudly, and vice versa.
7. **Measure the wall before building against it.** Step 0 ships a probe and its numbers
   before any window code, in the platform-claims style: a claim about hardware is
   witnessed only by hardware.

## 4. The steps

Ordered by dependency and by what each one is worth on its own. Every step ends with a
gate — the harness that decides it and the number it must show. "CI" means the
adapter-independent path (SwiftShader); "GPU" means the peer's headed run or the
nightly on a real adapter (`refuse_if_software` holds).

### Step 0 — Measure the ceiling  · size S · nothing depends on it, everything reads it

- **Build**: `tests/browser/ceiling.py` + `ceiling.html`. Reports, per machine: adapter
  name and tier of `maxBufferSize` / `maxStorageBufferBindingSize`; `shader-f16` and
  `subgroups` presence; the **largest single allocation** and the **largest total** that
  survive, found by doubling under `pushErrorScope("out-of-memory")` and confirmed by a
  fill-and-read round trip (a zero read is a fail, per `device.ts:594-611`);
  `navigator.storage.estimate()`; and the hub load peak for `imagenet-efficientnet-b0`
  and one ≥ 300 MB model — host bytes held (`performance.memory` where present, else the
  sum of live `Uint8Array`/`Float32Array` the loader reports) and `Device.memory.bytes`.
- **Gate (GPU)**: numbers recorded for at least two adapters (Apple, NVIDIA) into
  `docs/SCALE-MEASURED.md`. Wired into nightly (`--list`) in the same commit, per the
  entry-point census.
- **Why first**: the window's size, the shard size and the f16 decision all read from
  this table. Without it the plan is an estimate, and the project's rule is measured or
  nothing.
- **2026-09-18, Apple metal-3 measured** (`SCALE-MEASURED.md`): the 4 GiB tier holds as
  one buffer with its marker; 32 GiB of 256 MiB chunks hold together (the cap, half the
  machine's RAM — no wall found below it); `shader-f16` and `subgroups` present; storage
  quota 10 GiB on this profile. **And the hub finding that reorders Step 2**: loading
  EfficientNet-B0 (21 MB) leaves **+149 MB** resident on the GPU, ViT-B/16 (346 MB)
  **+1,193 MB** — 3.4–7× the file, with the pool at +0. Nothing was pooled because
  nothing was scoped: `borch-hub`'s `load` decodes every tensor and `verify` runs a 224 px
  forward with no `scope()` open (`load.ts:509-513`, `verify.ts:77`), so the decoded
  copies and the forward's intermediates are never returned. Host peak was exactly 2.0×
  the file (the chunk list plus its concatenation). NVIDIA is still to be measured
  (the 4090 through cq, or the peer's headed run).

### Step 1 — OOM becomes catchable; a budget exists  · size S · depends on 0

- **Build**: in `Device.alloc` (`device.ts:1160`), before the pool hit at `:1174`:
  `pushErrorScope("out-of-memory")` around the `createBuffer` at `:1176` and a
  `popErrorScope()` promise attached to the buffer; `flush()` (`:1284` path) awaits the
  outstanding scopes so an OOM surfaces at the next flush rather than the next readback —
  **deferred confirmation**, allocation stays synchronous. A soft `Device.budget` (bytes;
  default = Step 0's measured total × 0.75, else the binding-size tier) checked against
  `made − pooled`; over budget → `emptyCache()` first, then a `RuntimeError` naming the
  budget and the request. `faults.outOfMemory` keeps counting for the uncaught path.
- **Gate (CI)**: a `cost.ts`-style test that allocates past a tiny budget throws *before*
  any readback; `cost.ts` frozen numbers unmoved (dispatches 34, submits 1, survived 0).
- **Why**: every later step's fallback ("spill", "shrink the window") needs a failure it
  can catch. Today there is none.
- **2026-09-18, landed** (borch-ts `device.ts`): a fresh `alloc` wraps its `createBuffer`
  in `pushErrorScope("out-of-memory")` and parks the pop; `drainAllocations()` awaits the
  parked scopes at the next `read`/`synchronize` and folds a real OOM into `faults`, where
  the existing throw surfaces it — validation errors still travel to the uncaptured
  handler untouched (only the OOM filter is pushed). After warm-up the pool serves the
  repeats, so the parked list is empty every step and the drain is a resolved
  `Promise.all`. `Device.budget` (bytes, 0 = off) makes `alloc` throw *before* the buffer
  when the live footprint would cross it; it does **not** reclaim, because a flush inside a
  step would add a submit. `cost.ts` gained the budget-throw check and its frozen counts
  are unmoved. The reclaim rung (`emptyCache` on over-budget) is the caller's, and lands
  with the window in Step 3.

### Step 2 — Lazy tensors and a byte source in the hub  · size M · depends on 0

- **Build** (borch-hub, borch-ts `serialize.ts`):
  - **2a, first and separately: scope the load.** `decode` + `loadStateDict` inside one
    `scope()` (the values are copied into the kept parameters; the decoded buffers go
    back), `verify`'s forward and readback inside another, and `emptyCache()` after — a
    load is rare and the pool it leaves is shaped like weights, not activations. Gate:
    the ceiling probe's "gpu resident" for ViT-B/16 drops from +1,193 MB to within 10 %
    of the file (346 MB). This alone gives back 2.4× the weights on every model the
    workbench loads today.
  - `ByteSource { size, read(offset, length): Promise<Uint8Array> }` with three backings:
    in-memory (today's path, unchanged), **OPFS** (Worker sync handle; falls back to the
    async handle outside a Worker), and HTTP Range (the existing `pull` logic,
    `load.ts:435-474`, generalised from "from offset to end").
  - `decode` gains a lazy mode: it parses the header (`serialize.ts:170-196`), returns
    `{ tensors: Map<name, () => Promise<Tensor>> }` and does the slice + `Tensor.from`
    (`:186-193`) only when asked. **No format change** — `data_offsets` is already there.
  - Manifest v2 (optional, backward compatible): `weights.shards: [{url, sha256, bytes}]`
    of ≤ 64 MB each, so a shard can be hashed with the one-shot WebCrypto digest the hub
    already uses (`hash.ts`) without holding the whole file. v1 single-file manifests
    keep the whole-file path.
  - Cache policy: files above a threshold (Step 0's measured Cache API failure point,
    346 MB known) go to OPFS, with `estimate()` checked and Firefox's 10 GiB wall
    respected; the Cache API stays for small models. Verification stays two-layered:
    bytes (per shard) and behaviour (sample verify, `load.ts:527-540`).
- **Gate (GPU, `envelope.py` + `hub:py`)**: loading the ≥ 300 MB model holds
  **host peak ≤ 1.5 × file size** (today ≥ 2 × plus per-tensor copies) and survives a
  cold reload without re-downloading; `loadStateDict` reports `{missing, unexpected}`
  empty; sample verify passes. `weight_probe.py` unmoved (the landing page carries none
  of this).
- **Why**: the window (Step 3) needs bytes per tensor on demand; the Cache API failure is
  the workbench's live bug.

### Step 3 — The frozen-weight window  · size L · depends on 1, 2

- **Build** (borch-ts `device.ts`, `nn.ts`; bimm-ts):
  - `Device.window(bytes)`: one `STORAGE|COPY_DST` buffer of ≤ the binding-size tier,
    allocated outside the pool and `keep`-ed; a ring of `MAP_WRITE|COPY_SRC` staging
    buffers (LlamaWeb's four × 1 MB is the starting point; Step 0 sizes it); slots
    256-aligned (`minStorageBufferOffsetAlignment`); fill = map staging → `copyRange`
    (`:1395`) into the slot, **never** `writeBuffer`; evict = `retire` the slot's age so
    any Tensor still pointing at it throws through `refuseIfDead`.
  - A frozen Tensor gains a window-backed storage: `gpu` points at the window buffer and
    a cached `BindSlot` object per tensor (the object is the fusion/`liveIns` key,
    `device.ts:308-315`); the two weight funnels (`tensor.ts:2099`, `:779`) pass it
    through `bindGroupFor` unchanged. Repointing follows `ensureOwned`; the Tensor object
    is never replaced.
  - The scheduler is bimm's **third plan consumer**: walk `efficientnetPlan` /
    `resnetPlan` / `vitPlan` (upstream ask: re-export them), fetch block k's keys from
    the `ByteSource`, fill, run, evict, with block k+1's staging map in flight — one
    ahead, resolved before k+1's first encode. Installed by replacing the `blocks` child
    (`Sequential` names children by index, `nn.ts:563-570`, so checkpoint keys survive).
  - Hooks: `Module.call()` (`nn.ts:165-167`) is the one dispatch point per layer; there is
    no hook machinery today, so this is where prefetch-next / evict-previous attaches.
  - First landing: the workbench feature pass (`_workbench.py:388-392`) — no tape, no
    backward. `compiled` and a live window refuse each other by name.
  - Batch: streaming amortises a block's upload over the batch, so the workbench's
    `cfg["batch"]` default of 16 is re-measured upward, not downward.
- **Gate (GPU, `envelope.py`)**: a model whose f32 weights exceed the window (the
  ≥ 300 MB model in a 128 MB window, and any model in a window forced to 32 MB to prove
  the mechanism) runs the 224 px feature pass with **throughput ≥ 0.5 × resident**,
  `lastScope.survived` 0, `Device.memory.bytes` bounded by window + staging across 100
  batches, sample verify equal to the resident run within the f32 golden tolerance.
  The pass-boundary cost per block (`profile()`) recorded in `SCALE-MEASURED.md`.
- **Gate (CI, `cost.ts`)**: a `Small` variant with a 3-block windowed `Sequential`
  freezes dispatches, submits and "buffers held" — held must not grow with block count.
- **Why**: this is the step that changes what fits. Everything after it makes the window
  cheaper (4, 5) or lets a backward cross it (6, 7).
- **2026-09-18, primitive landed** (borch-ts `device.ts` `Device.window` + `Window`): one
  STORAGE buffer outside the pool, `keep`-ed; `place(data)` fills the next 256-aligned slot
  through a `MAP_WRITE` staging buffer and `copyRange` (never `writeBuffer`) and returns a
  `{buffer, offset, size}` BindSlot; `free()` returns it. `window_probe.py` (adapter-
  independent, in CI and nightly): two arrays land in two slots of one buffer at offsets 0
  and 256, a kernel reads each slice at its offset and the values return intact, a fresh
  window too, faults 0 — verified on apple/metal-3. Still to come on top of this primitive:
  the frozen-Tensor window storage with a cached BindSlot through the two weight funnels,
  age-gated eviction, the bimm plan-walking scheduler, and the f16 raw-buffer weight (Step
  4 folded in). The staging path waits after each fill (one staging buffer); a ring is a
  later optimisation when a measured need appears.
- **2026-09-18, weight funnel integrated.** A `Tensor` may carry a `windowSlot`;
  `Tensor.inWindow(win, data, shape)` fills a slot and returns a frozen
  (`requiresGrad=false`) tensor bound to it. The matmul funnel binds `mat2.weightBinding()`
  (`windowSlot ?? raw`), so a windowed weight is read as its slice in place — and every
  generic op refuses a windowed tensor at the `buffer` getter (reading the window from
  offset 0 would be another slot's values). `window_probe.py` extended: `x @ W` with W
  windowed is **bit-identical** to W as an ordinary tensor on both the subgroup path
  (16×16×16) and the scalar tile (6×10×7), a windowed weight refuses `.add`, faults 0 —
  verified on apple/metal-3, golden 4057/0 unmoved.
- **2026-09-18, conv funnel integrated.** `convForwardRun` takes `w: BindSlot` now, and
  every conv path that reads the weight — the direct/tiled forward, the depthwise kernel,
  the subgroup conv's `tapMajorWeights`, and the input-gradient turns — binds
  `weight.weightBinding()`. So a windowed conv weight is read as its slice across all three
  forward kernels (direct, tiled, subgroup) and the backward. `window_probe.py`: `conv2d`
  with a windowed weight is bit-identical to a normal one on a small shape (3→4, direct)
  and a larger one (16→16, tiled/subgroup) — apple/metal-3, faults 0, golden 4057/0.
- **2026-09-18, eviction gate landed.** The window is one buffer with many slots, so the
  whole-buffer age guard is too coarse (evicting one block would kill every windowed
  tensor); instead each slot offset carries a generation (`Window.gens`), bumped on `place`
  and on `evict(slot)`. `Tensor.inWindow` returns `{weight, evict}`, and the weight's
  `weightBinding` runs a `windowLive` check that throws when its slot's generation has moved
  — a stale read after eviction is loud, ADR-003 decision 2. `window_probe.py`: a windowed
  weight works, then `evict()`, then using it throws (`/evicted/`); apple/metal-3 and the
  5080, faults 0.
- **2026-09-18, bounded streaming landed — the window now reuses its bytes.** `evict`
  returns a slot's aligned region to a free list; `place` takes a freed region (first fit)
  before growing the cursor. So a stream of blocks — place k, run, evict, place k+1 into
  the freed region — keeps the buffer at the resident set, not the total streamed.
  `window_probe.py`: **24 blocks stream through a window sized for 3** (matmul each,
  bit-identical to the normal weight), and `used` stays at one block's bytes (1 KB) against
  a 3 KB cap — a model larger than the window fits. apple/metal-3, faults 0. This is the
  bounded-memory mechanism Step 3 exists for.
- **2026-09-18, f16 raw weight landed (Step 3 ⑤).** `Tensor.inWindowF16(win, data, shape)`
  packs the f32 values to IEEE f16 on the host (`f32ToF16Bits`, `src/half.ts`) and places
  the bytes; on use, `weightBinding` unpacks the slot to an f32 scratch with `unpackHalf`
  (core WGSL `unpack2x16float` — **no `shader-f16` needed**, so the storage win is on every
  device; `Device.f16` only buys a later direct-f16 read). The maths stays f32, the resident
  bytes halve. `window_probe.py`: a matmul with an f16 window weight is bit-identical to the
  same weight rounded to f16 on the host, and the f16 slot is 512 B against the f32 slot's
  1024 B — apple/metal-3 and the 5080, faults 0, golden 4057/0. Step 4's storage win reached
  through the window as a raw buffer, not a `float16` dtype (ADR-003 decision 5).
- **2026-09-18, the scheduler landed (Step 3 ④, the abstraction).** `streamSequential(win,
  input, blocks)` (`src/stream.ts`) runs a stack block by block: place a block's weights in
  the window, run it in a scope that keeps only the output, evict, next — so only a few
  blocks are resident and a model larger than the window fits. `f16: true` stores the
  weights half-precision. `window_probe.py`: a **10-block Linear+ReLU network streamed
  through a window sized for 3 is bit-identical to the resident run**, residency bounded —
  apple/metal-3 and the 5080, faults 0, golden 4057/0. Prefetch-one-ahead is a later speed
  lever (needs the staging ring); this establishes correctness and bounded memory. Next:
  the bimm adapter — turn a real EfficientNet/ViT into `StreamBlock`s (needs the upstream
  `bimm-ts` plan-table re-export) and land it on the workbench feature pass.
- **2026-09-18, the real model streams (Step 3 ④b).** `stream_model_probe.{html,py}`: a
  bimm ResNet-18's `layer1` — a `Sequential` of `BasicBlock`s, the real conv/BN/ReLU/
  residual mix — turned into `StreamBlock`s whose weights are each block's conv kernels,
  streamed through a window sized for ~4 kernels. Each block's forward swaps the windowed
  kernels into its conv modules, runs the genuine `block.forward`, and restores the resident
  kernels so the model stays runnable. **The streamed layer output is bit-identical to the
  resident one** (max |Δ| 0.00, faults 0), on a window a fraction of the layer's conv bytes.
  This closes the abstraction against a real backbone: not a hand-built Linear stack but a
  shipped model's own blocks. Wired into the census (fifty-one entry points), nightly, and
  `stream-model:py`; needs `bimm-ts@0.12.0` from esm.sh, so it runs where the CDN is
  reachable.
- **2026-09-18, the adapter landed (Step 3 ④).** `streamSequence(win, input, modules)` and
  `streamBlock(module)` (`src/stream_module.ts`, exported) turn a real `Module` into
  `StreamBlock`s so a caller streams a shipped model without hand-building blocks: it
  snapshots the module's frozen weights, and the forward swaps the windowed tensors into the
  module's own parameter fields, runs the genuine `module.forward`, and restores them
  (non-consuming, so the model stays runnable). **A window slot is a weight operand only** —
  bound as an offset slice, as matmul's `mat2` and conv's kernel are read, and *not* how a
  generic op reads offset 0. The first draft streamed **every** parameter and batch-norm
  read its scale from a slot at offset 0 and threw; the fix is the default `select` — 4-D
  conv kernels stream, biases and BN affine stay resident (what a frozen eval wants anyway),
  and a caller streaming Linear `weight`s passes a `select` that admits 2-D `mat2` operands.
  `stream_model_probe` now checks **both** paths — hand-built blocks and the adapter — against
  the resident ResNet-18 `layer1`: bit-identical on apple/metal-3, faults 0. Remaining ④: a
  memory-window variant that frees the resident kernels instead of restoring them, for a model
  that will not fit resident at all.
- **2026-09-18, speed lever — direct f16 read (Step 4).** The scalar matmul tile gains a
  `weightF16` variant (`enable f16;`, `B: array<f16>`, `f32(B[…])`), and the matmul funnel
  takes it (`mat2.f16WeightBinding()`) where the weight is an f16 window weight **and** the
  device has `shader-f16` — reading the half-precision slot directly, no unpack pass, no f32
  scratch. It bypasses the subgroup matrix (whose loader cannot narrow), so it trades that
  path for the saved unpack; where f16 is absent the unpack fallback runs. `window_probe.py`:
  on apple/metal-3 the f16 matmul is **1 dispatch (direct read)**, on the 5080 **2 (unpack +
  matmul)** — both bit-identical to the host f16-rounded reference, golden 4057/0. The
  wall-clock trade (scalar-f16 vs subgroup-f32) is unmeasured; a `bench` on a real model
  through the window will settle it.
- **2026-09-18, speed lever — the fill no longer stalls the GPU.** `Window.place` used to
  `synchronize()` (submit **and** wait for the GPU) after every copy, so each block's upload
  stalled the device before the block could run. It now `flush()`es — submits the copy but
  does not block — because the block that reads the slot runs in a later submit and the
  queue keeps submit order, so the bytes are in place before they are read. The wait moves
  to the next `place`'s `mapAsync` (which needs only the one staging buffer free), and by
  then the block between has run: the upload overlaps the compute instead of stalling. No
  staging ring needed. `window_probe.py` still passes (correctness); the latency win is not
  measurable adapter-independently, so it, too, awaits a real-model `bench`.
  and the f16 raw weight.

### Step 4 — `shader-f16`: storage first, compute second  · size M+M · depends on 0; parallel to 2–3

- **4a, storage-only half (weights in the window are f16; arithmetic stays f32)**:
  - `Device.create` requests `"shader-f16"` when the adapter has it (`device.ts:488-499`);
    `Device.precision: {f16: boolean}` reported like `subgroupMatrix` (`:500`);
    `platform_claims.py` gains the row, abstaining on a CPU adapter as it does now.
  - `dtype.ts`: `bytesPerElement(dtype)` beside `floatsPerElement`; `"float16"` enters
    `DType` **for storage only** — a `float16` Tensor is legal in `borch-ts` and refused
    by the numpy core, exactly the split `int64`'s label already lives with. `Tensor.floats`
    → `bytes` at the five `BYTES_PER_F32` sites (`device.ts:1161,1171,1221,1381,1626`).
    Ingress: `Tensor.frombuffer` decodes f16 (`tensor.ts:1368-1389`); `Tensor.from` keeps
    converting values (so f16 bits are never smuggled in as integers). Safetensors `F16`
    accepted by `serialize.ts:338-343` when the target is a window; still refused for a
    trainable parameter.
  - **2026-09-18, scoped — 4a is bigger than "add a dtype", and here is why.** `float16`
    is not merely absent from `borch-ts`; it is a **deliberate, golden-frozen, cross-
    implementation refusal**. The numpy core makes it an `_AbsentDtype` gathered into
    float32 (`borch/_base.py:241`, `half = float16` at `:252`), `borch-ts` refuses it as a
    golden case (`weRefuse("half")`, `borch-ts/test/cases.ts:834`), and the same wording is
    frozen in `tests/test_dtype_throat.py`, `test_messages.py`, `test_why_failing.py`,
    `borch_webgpu/_base.py:1681` and the `torch_gap` ledger. borch's axiom is that the three
    implementations agree (one-name-one-list). So making `float16` a real storage dtype in
    `borch-ts` alone **breaks parity by construction** unless the change is coordinated: it
    is not the `int64` case (int64 is a *working* label everywhere), it is reversing a
    refusal asserted in ~10 places across core, binding and ts plus a golden regen. The
    foundation (`Device.f16`) landed at fccd287; the packing itself uses WGSL
    `pack2x16float`/`unpack2x16float`, which need **no** `shader-f16` (so the storage
    round-trip is adapter-independent — Mac, the 5080, and CI). The open design question is
    whether the core accepts `float16` as a label too (keeping parity) or borch-ts takes a
    documented, separately-tested divergence. Either way this is a focused, ledger-touching
    unit of its own, not a tail-of-session edit.
  - **2026-09-18, decided — do NOT make `float16` a Tensor dtype; the window holds it as a
    raw buffer.** Going one level deeper into "float16 as a real storage dtype across all
    three implementations" found it reverses the codebase's central axiom, stated at the top
    of both `dtype.ts` and `borch/_base.py`: **storage is float32, and only that.** The
    one-width-per-category promotion model, the `_requested_dtype` narrowing throat (which
    narrows even `float64` to float32 on purpose), and the golden parity all rest on that
    axiom; a second float width disturbs all of them. And it buys nothing in the core — the
    core has no GPU and no memory pressure, so an `np.float16` array there only breaks the
    axiom for no gain. **ADR-003 decision 5 already ruled this out**: quantised and f16
    weights are *not* Tensors — they are raw `GPUBuffer`s in the window, so `DType` and
    `floatsPerElement` keep their invariant and the numpy core never learns a narrow dtype.
    So Step 4a's "float16 enters DType" line is **withdrawn**: the f16 weight operand is a
    window-owned raw buffer (Step 3), packed with `pack2x16float` (no `shader-f16` needed to
    pack; the matmul's f16 *read* path uses `Device.f16` where present and upcasts to f32
    otherwise). What stays from Step 4 is `Device.f16` (landed) and, later, the matmul f16
    weight-read on a windowed buffer. Net: **Step 4 folds into Step 3** — there is no
    standalone f16-dtype refactor.
  - Kernels: the matmul and conv **weight operand only** typed `array<f16>` under a
    generator parameter `weight: "f32"|"f16"` on the four matmul generators and the two
    conv generators (six functions, not 162); `enable f16;` prepended like `:1681`; the
    pipeline key gets a `:h` tail inside `device.pipeline()` (one site, not 113) from the
    active precision, so a cached f32 pipeline is never returned for an f16 binding.
    `f32lit` guarded at 65504 for the f16 side. Subgroup GEMM loads f16 into an f32
    accumulator — the componentType probe (`device.ts:276-284`) gets an f16 sibling.
  - Readback: a cast dispatch f16→f32 immediately before `read()`; `read()` and the
    staging pool unchanged.
- **4b, compute half (`autocast()` scope: activations and arithmetic in f16, f32 master
  weights and reductions)** — only after 4a's numbers, and only if 4a's traffic win is
  not enough for the window's throughput gate. Touches the elementwise and reduction
  generators (`kernels.ts:1234,1262,1322`) via the same `scalar` parameter; the fused
  pass's `array<f32>` replace (`fuse.ts:374`) becomes precision-aware. Loss scaling as
  torch does it. On a device without f16 the scope throws (`index.ts:365`'s sentence is
  the rule).
- **Golden**: `golden.json` gains a `tolerance_f16` tier chosen by case dtype at
  `test/golden.ts:186`; the f32 tier is untouched. The ten autocast questions
  (`index.ts:356-443`) start answering from `Device.precision` instead of constants;
  `torch_gap.py`'s row is rewritten from "declined" to "opt-in, f32 default, see ADR-003".
- **Gate (GPU only — CI has no `shader-f16`)**: `kernel_bench` f16-weight matmul vs f32 on
  Apple and NVIDIA (report, no threshold — the 1e-4 WRONG check gets a per-variant
  tolerance); `accuracy.ts` top-1 on EfficientNet-B0 with f16 weights within **0.3 pt** of
  f32; Step 3's window gate re-run with half the bytes per block; on an adapter without
  f16 the same program runs f32 and says so.
- **Where these run** (measured 2026-09-18): the f16 path is developed and verified
  **headed on this Mac** — `apple / metal-3` offers `shader-f16` (a headed run gets the
  real adapter; only *headless* falls to SwiftShader). The **f32-fallback** path has a
  measured witness in the RTX 5080 (`nvidia / blackwell`, Chrome on Linux/Vulkan), which
  offers **no** `shader-f16` at all — so "the same program runs f32 and says so" is tested
  on real hardware, not assumed. That card's `maxStorageBufferBindingSize` is also 2048 MiB
  against Apple's 4096, so the window (Step 3) must read the binding tier per device, never
  hardcode Apple's.
- **Why storage first**: the window's cost is bytes moved per block; halving them is the
  whole win for a frozen backbone, and it changes no arithmetic the golden reasons about.
  Compute-f16 is a speed lever with a numerics bill; it waits for a number that says it
  is needed.

### Step 5 — int8 weight-only for the window  · size M · depends on 3; parallel to 4

- **Build**: per-channel symmetric int8 (scale per output row) packed as `array<u32>`
  raw `GPUBuffer`s in the window — not Tensors (decision 5); quantised offline into the
  hub (manifest `weights.quant: "int8-perchannel"`, shards as in Step 2), never on the
  client. Unpack in the **scalar** matmul tile at `kernels.ts:2053-2056`
  (`unpack4xI8(...) * scale`) and the tiled conv (`:3566`); the subgroup path is
  bypassed for quantised operands (decision from `:1650-1653`). int4 is the same
  generator with a different unpack and is not planned until int8's accuracy number is
  in.
- **Gate (GPU)**: `accuracy.ts` top-1 within **1.0 pt** of f32 on EfficientNet-B0 and the
  ≥ 300 MB model; `kernel_bench` records the scalar-tile speed honestly beside the
  subgroup f16 number (expect ≈ 4.5 vs 11 TFLOP/s-class); Step 3's window gate at a
  quarter of the bytes.
- **Why after f16, and optional**: f16 halves bytes at full GEMM speed; int8 quarters
  them at less than half the speed. It is taken only where a model still does not fit
  after Step 4 — and the ceiling this plan targets (a frozen vision backbone) may never
  need it.
- **2026-09-18, the int8 matmul weight landed.** `src/quant.ts` (`quantizeInt8PerChannel`:
  symmetric, one scale per output channel, four signed bytes per `u32`) + `Tensor.inWindowInt8`
  (packs to the window, scales in a small resident buffer). The scalar matmul tile gains a
  `weightInt8` mode (`kernels.ts`): `B: array<u32>`, a third `scale` binding, the byte read and
  **sign-extended by hand** (a top-aligned left shift then an arithmetic right shift — no
  `unpack4xI8`, which not every adapter has) then scaled. The matmul funnel routes a windowed
  int8 weight there when read transposed (`transB` — the Linear/conv weight case, where the
  scale's output channel is the matmul's `N`); every other consumer (a conv, a backward)
  dequantises to f32 scratch (`dequantInt8`, also hand-unpacked). The subgroup matrix is bypassed
  for the quantised operand. `window_probe`: **int8 window matmul == the host-reconstructed
  weight, bit-identical, in a quarter of the f32 slot's bytes** — on apple/metal-3 **and**
  google/swiftshader (the manual unpack has no feature gate). This is the matmul path — the
  ≥ 300 MB ViT's whole weight.
- **2026-09-18, the int8 accuracy gate passed** (`finetune.html`, `streamSequential({int8:true})`).
  The same ViT-Base fine-tune, its frozen blocks then streamed **as int8** for the held-out
  evaluation: **top-1 100 % = the f32 backbone's, within 1.0 pt**, on a window a quarter of the
  f32 bytes, apple/metal-3. So Step 5 holds for the matmul path.
- **2026-09-18, int8 conv landed — Step 5 complete.** A conv reads an int8 window weight through
  the **dequant fallback** in `weightBinding`: the slot is unpacked to an f32 scratch (a pooled,
  transient buffer freed at the scope) on the way into the conv funnel, so the conv needs no int8
  kernel of its own and both conv paths (direct and tiled) get it at once. `window_probe`: **an
  int8 window weight through both conv2d kernels equals the host-reconstructed weight bit for
  bit, at a quarter of the f32 slot's bytes** — on apple/metal-3 **and** google/swiftshader. The
  window (the resident cost) is quartered; the dequant scratch is one weight's f32 at a time. A
  *direct* int8 read in the tiled conv kernel (`convNDForwardTiled`, avoiding the dequant pass and
  the scratch) is a **perf optimisation not taken** — the memory win is already had by the
  quartered window, int8 is optional and its models (EfficientNet-B0, 21 MB) fit resident anyway,
  and it would thread a scale binding through every conv-kernel permutation for a compute saving
  on a path the plan never made hot. **Step 5 done: the whole int8 weight surface — matmul direct,
  conv via dequant — a quarter of the bytes, values within a rounding, on every adapter.**

### Step 6 — Gradient checkpointing  · size M · depends on nothing; parallel to 2–5

- **Build** (`borch-ts/src/`, must live inside because `flow`/`enableGrad`/`makeNode`
  are not exported): `checkpoint(fn, ...inputs)`:
  - forward under `noGrad` (`autograd.ts:81-89`) inside a **nested `scope()`**
    (`tensor.ts:13905-13938`), keeping only the output;
  - one graph node via `makeNode(out, shape, [inputs, ...params], backwardFn,
    "CheckpointBackward0")` (`tensor.ts:13983`);
  - `backwardFn` re-runs `fn` under `enableGrad` (`autograd.ts:99-107`) on `detach()`ed
    inputs (`tensor.ts:2421-2429`, shares the buffer) inside its own `scope()`, then
    `flow(roots, seeds, add)` (`autograd.ts:192-240`) into the segment; recomputed
    tensors re-stamp `savedVersions` (`tensor.ts:913-914,955-970`) so the version guard
    does not fire on the recompute;
  - a `Sequential.checkpointEvery(n)` convenience for the plan-walking case.
- **Gate (CI, `cost.ts`)**: a deep `Small` (12 blocks) with `checkpointEvery(4)` holds
  **≥ 40 % fewer buffers at the backward peak** at **≤ 1.35 × dispatches**; `survived` 0.
  **Golden**: checkpointed and plain runs agree within the f32 tolerance (the recompute
  uses the same kernels; the expectation is bit equality, the gate is 1e-4).
- **Why**: it is the half of QLoRA's memory story that is not about weights — activation
  gradients dominate, and checkpointing is what makes a backward through a *windowed*
  block self-contained (Step 7). No WebGPU implementation exists; this one will be the
  first, which is a reason to keep it small and measured.
- **2026-09-18, landed** (borch-ts `checkpoint.ts`, exported): `checkpoint(fn, ...inputs)`
  runs the forward under `noGrad` in a `scope()` that keeps only the output, and a node
  whose backward re-runs `fn` on detached leaves inside its own `scope()`, `flow`s the
  incoming gradient in, and keeps only the returned input grads. Gradient routes to the
  passed `inputs`; a grad-requiring tensor *closed over* rather than passed is refused
  (the recompute graph is walked for stray grad-requiring leaves), so a captured parameter
  is a loud throw, never a silent zero — Step 7 passes the adapter as an input, the frozen
  backbone does not require grad and is not flagged. `checkpoint_probe.py` (adapter-
  independent, in CI and nightly): the recomputed gradients are **bit-identical** to the
  taped ones (max |Δ| 0.00e0 across x, first/last W, b), a captured grad tensor is refused,
  and the buffers held after the forward fall (68 % at 8 blocks, the win growing with
  depth). `Sequential.checkpointEvery(n)` is the convenience left for Step 7, where the
  block's grad-requiring params are collected and passed.

### Step 7 — LoRA on a streamed frozen backbone  · size L · depends on 3, 6 (4 recommended)

- **Build**:
  - `peft.applyLora(model, {targets, r, alpha})` walking `namedModules()` and swapping
    `Linear`/`Conv2d` children by name (`nn.ts:213,230,240,286`) — the missing
    apply-to-model helper; attention projections covered because ViT's `qkv` is a
    `Linear`. The Python mirror (`borch_webgpu/_nn.py` factory reads `_ts.nn` only) so
    the workbench can call it.
  - Residency rule per block k during training: **forward** — block k's frozen weights
    in the window, output kept, intermediates dropped (checkpoint); **backward** —
    block k refilled (in reverse order, so the window walks the plan backwards),
    recompute, backward into LoRA A/B (resident, arena-eligible) and into the block
    input, evict. That is QLoRA's per-layer dequant in both passes, with "dequant"
    replaced by "refill". Weight gradients for the base are never formed
    (`requiresGrad = false` already stops them).
  - Optimizer: adapter parameters only (`parameters()` already returns just A/B); the
    SGD arena applies unchanged since every trainable tensor is resident.
  - Export: `adapterState()` (KB–MB) for borch-fed; `merge()` into a resident copy only
    when the merged layer fits, else refused by name (`LoRAConv2d.merge` already refuses
    grouped conv).
- **Gate (GPU, new `tests/browser/finetune.py` in the envelope's shape)**: the ≥ 300 MB
  backbone, LoRA r = 8 on all `Linear`s, three classes × 200 images at 224 px, five
  epochs, in a window of **≤ 256 MB**: completes; held-out accuracy **≥ the frozen-head
  baseline** of `envelope.py`; seconds per epoch and `Device.memory.bytes` peak reported;
  adapter export ≤ 2 MB; `lora.ts` invariants hold on the streamed layers.
- **Why last**: it is the payoff — fine-tuning a model that does not fit, in a tab, with
  a payload borch-fed can carry — and it needs every step before it.
- **2026-09-18, the apply-to-model helper landed** (borch-ts `peft.ts`, `peft.applyLora`).
  `applyLora(model, {targets, r, alpha})` walks `namedModules()` and swaps every matched
  `Linear` for a `LoRALinear.fromLinear` wrapping it — the base weight/bias become frozen
  buffers, so afterwards `parameters()` is the adapters alone and the optimiser touches only
  them. `targets` is a predicate or a name list (`["qkv"]` catches every
  `blocks.k.attn.qkv`); the default is every `Linear`, Step 7's gate set. `B` starts at zero,
  so the adapted forward equals the original until training, and the call is idempotent (a
  `LoRALinear` is not a `Linear`). **The swap is guarded**: field assignment does not reach a
  child held in a `Sequential`/`ModuleList` array, so `applyLora` reads the submodule back and
  throws rather than silently skipping it. `lora.ts` (nightly `lora`, adapter-independent)
  gains three checks: every `Linear` adapted with the forward unchanged and params reduced to
  adapters; a name target adapts only its match with a no-op second call; the Sequential-child
  guard throws. All 11 LoRA checks pass on apple/metal-3.
- **2026-09-18, conv targeting landed** (borch-ts `peft.ts`). `LoRAConv2d.fromConv2d(conv)`
  wraps a trained `Conv2d` as the frozen base, carrying its stride/padding/dilation/groups —
  read through a narrow typed view because those are private (torch stores them as tuples, so
  borch does not expose the single number as a public attribute) and are not recoverable from
  the weight shape. `applyLora`'s default target set is now every adaptable leaf (`Linear` or
  `Conv2d`), dispatching `fromLinear`/`fromConv2d` by type; the gate's Linear-only set is an
  explicit `targets`. `lora.ts` gains a check: adapting a `Conv2d` with non-default geometry
  (stride 2, padding 1) leaves the forward unchanged (a weights-only copy would not). All 12
  LoRA checks pass.
- **2026-09-18, the residency rule landed** (borch-ts `stream_train.ts`, `streamTrainStep`).
  The frozen base is resident only while a block runs, in both passes: forward streams each
  block through the window keeping only the boundary activation; backward walks the blocks in
  reverse, refills each block's weights, recomputes its forward with the tape on from the saved
  boundary, and flows the gradient into the resident adapters and the block input. It is **not**
  `checkpoint()` — that recomputes synchronously, but refilling the window is `async` (a staging
  map), so the reverse pass is written out and awaits each refill; the gradient arithmetic is the
  same `flow`. `stream_train_probe` (nightly `stream-train`, adapter-independent): a 4-block
  LoRA stack trained one step with the base streamed through a window sized for ~2 bases gives
  adapter gradients **bit-identical** to the fully-resident run (max |Δ| 0.00 on A and B), loss
  equal, window used a quarter of the backbone's base bytes, faults 0 — on apple/metal-3.
- **2026-09-18, the streaming-train adapter landed** (borch-ts `stream_train.ts`,
  `streamTrainSequence`/`trainBlock`). The model-facing form of `streamTrainStep`: `trainBlock`
  turns a real `applyLora`'d module into a `TrainBlock`, streaming its frozen weight buffers
  (rank ≥ 2 — conv kernels, linear weights; batch-norm stats stay resident) and harvesting its
  trainable parameters (the adapters) for gradients; the forward swaps the windowed buffers into
  the module's fields, runs `module.forward`, and restores. `streamTrainSequence(win, input,
  modules, loss)` streams a sequence of them. `stream_train_probe` now checks this adapter path
  beside the hand-built one: adapter gradients **bit-identical** to the resident run (max |Δ|
  0.00) on apple/metal-3. This is the bridge `finetune.py` will call.
- **2026-09-18, the Python mirror landed** (`borch_webgpu/_peft.py`, `torch.peft`). The
  individual layers bridge to `_ts.peft` (`LoRALinear`/`LoRAConv2d` are classes over there), but
  `apply_lora` cannot: the WebGPU binding composes models **Python-side** (its `nn.Sequential` is
  a Python class, not a `_ts` module), so a Python-built model has no TypeScript tree for the TS
  `applyLora` to walk. So the walk is Python — over `named_modules()` — swapping each matched
  `Linear`/`Conv2d` for its LoRA wrapper, whose base bridges from the leaf's `_ts` layer; the
  leaf kind is read from `describe()` (every wrapped leaf is the one Python class `Module`) and
  the swap is the same read-back-guarded replace. `peft_py.py` (nightly `peft`, wheel-based):
  `apply_lora` on a Python-composed model adapts every Linear with the forward unchanged and
  params reduced to adapters, a name target hits only its match, a Conv2d keeps its geometry, and
  an indexed leaf is refused — all pass through the wheel on apple/metal-3. Three source-tree
  loader lists (`runner.js`, `runner.html`, `scope_escape.html`, `onnx_binding.html`) carry the
  new module.
- **2026-09-18, the gate passed — Step 7 is complete** (`tests/browser/finetune.{html,py}`,
  nightly `finetune`). ViT-Base (**346 MB**, from the hub) LoRA-fine-tuned on 3 classes × 200
  images at 224 px, 5 epochs: the 12 transformer blocks `applyLora`'d (48 Linears, r=8), their
  frozen bases **offloaded** off the GPU (`trainBlock({offload:true})` frees the resident buffer
  after reading the bytes) and streamed through a **57 MB window** (≈ 2 blocks). Result on
  apple/metal-3: **held-out accuracy 100 % = the frozen-head (linear-probe) baseline**, and the
  **training GPU peak was 32 MB against a 346 MB backbone** — the model is never fully resident
  while it trains. 10 s/epoch, adapter 4.72 MB, faults 0. Two things this needed, both fixed
  here: `streamTrainStep` gained `lossParams` so the new 3-class head trains through the loss
  (grad bit-identical to resident, checked in `stream_train_probe`); and **`applyLora` was
  leaking a whole model's weight bytes** — `LoRALinear.fromLinear`/`fromConv2d` allocate a
  full-size random base in the constructor, then swap in the real one, and the throwaway stayed
  kept resident (691 MB for a 346 MB model). They now `unkeep` it, so offload actually frees the
  340 MB. Note: r=8 on 48 Linears is a 4.72 MB adapter — the plan's "≤ 2 MB" is a smaller target
  set; the gate keeps r=8 on all Linears (needed for every block weight to stream as a buffer)
  and reports the real size.

**Step 7 done. The plan's payoff — fine-tuning a model that does not fit, in a tab, with a
borch-fed-carriable adapter — is demonstrated end to end on a real 346 MB backbone.**

- **2026-09-18, wired into the workbench** (`torch.workbench.setup(..., finetune=True)`). The
  scale work reached the surface it was for. Three pieces landed on the Python side: the peft
  mirror's `apply_lora` now **delegates a wrapped (hub-loaded) backbone to the TS `applyLora`**
  (its tree lives on the TS side, invisible to a Python walk); `torch.streaming`
  (`borch_webgpu/_stream.py`) bridges `trainBlock`/`streamTrainStep`/`streamSequential` through
  JSPI (`run_sync`) with the loss as a `create_proxy` callback; and the workbench `Session` gains
  a `_fit_lora` path — freeze the backbone, `apply_lora` it, train the adapters and a new head
  through the backbone (resident for a budget-sized model; `torch.streaming` for one too large).
  `workbench_lora_py.py` (nightly `workbench-lora`, wheel + network): a folder of three colour
  classes fine-tuned through `setup(..., finetune=True).fit()` on ViT-Tiny trains, scores (1.00
  held-out), and exports a model — apple/metal-3, faults 0. This closes the workbench page's own
  stated gap ("a linear probe reads what the frozen backbone already sees; tissue is where a
  partial fine-tune starts to matter"). `streaming_py.py` and `peft_py.py` hold the bridges.
- **2026-09-18, the streaming tax measured** (`stream_bench.{html,py}`, a diagnostic). The same
  LoRA stack trained one step four ways — resident/streamed × forward-only/full — turns "streaming
  is slower" into a number. On apple/metal-3, a 12-block D=384 stack at batch 16: **a streamed step
  is ≈2.9× a resident one** (6.47 ms vs 2.25 ms; +100 dispatches, ≈ a whole extra forward). The
  breakdown: **recompute + backward streaming 57 %, placement + swap 18 %, forward compute 25 %.**
  So the tax is **the checkpointing recompute, not the placement** — a first, reasoned review had
  that backwards. And the recompute is **inherent to offload**: the frozen weights were freed, so
  the backward must rebuild them; the only way not to pay it is not to stream, which is right when
  the model fits — exactly what `_fit_lora` does (resident under budget, streamed above it). Two
  caveats keep this honest: the bench is **dispatch-bound** (a resident forward is 99 dispatches in
  1.5 ms ≈ 65 µs each — overhead, not FLOPs), so a real 224 px ViT with heavy per-block compute
  pays a **smaller fraction**; and each block here has one weight, so it does not exercise the one
  real optimisation — **coalescing a multi-weight block's placements into a single staging fill**
  (a ViT block has four: qkv/proj/fc1/fc2), which would cut most of the 18–25 % placement share but
  cannot touch the recompute. Conclusion: the design is right; streaming perf is not the lever to
  pull. `stream_bench.py` re-measures it after any streaming change.

### Step 8 — What this plan does not do

- **LLM scale (multi-GB)**: needs int4, multi-window tiers, Firefox's 10 GiB storage wall
  handled, and a kernel set tuned for decode. The survey found nobody offloads at that
  size; borch would not be first to the ceiling but to a worse one. Not planned; if the
  widget product ever asks for a language model in a frame, it will be a small one that
  fits, and this plan's steps 2–4 carry it unchanged.
- **Model parallel / multi-GPU**: WebGPU exposes one device per adapter and no
  peer-to-peer; the compute track is data-parallel across tabs (borch-fed), already
  measured there.
- **Compute-f16 by default, bf16 anywhere**: decision 4; bf16 has no WGSL substrate.
- **Anything in the numpy core**: ADR-001.

## 5. Order, parallelism, size

```
0 ──► 1 ──► 3 ──► 7
 │          ▲     ▲
 └──► 2 ────┘     │
 └──► 4a ─► 4b    │
            5 ────┘   (5 optional)
      6 ──────────────┘
```

| step | size | runs in parallel with | verified where |
|---|---|---|---|
| 0 ceiling probe | S (days) | — | GPU, two adapters |
| 1 OOM + budget | S (days) | 2, 4a, 6 | CI |
| 2 lazy tensors + byte source | M (1–2 wk) | 1, 4a, 6 | GPU |
| 3 window | L (3–4 wk) | 4a, 6 | GPU + CI |
| 4a f16 storage | M (1–2 wk) | 2, 3, 6 | GPU only |
| 4b f16 compute | M (2 wk), conditional | 5 | GPU only |
| 5 int8 window | M (2 wk), optional | 4b, 6 | GPU |
| 6 checkpointing | M (1–2 wk) | everything | CI + golden |
| 7 LoRA streamed | L (3–4 wk) | — | GPU |

Sizes are working-time ranges for one person who has the map above; they are not
promises. Two tracks can run at once without touching the same files: **memory**
(1 → 2 → 3 → 7, all in `device.ts` / hub / bimm) and **numerics** (4a → 4b, 6, in
`kernels.ts` / `dtype.ts` / autograd). Step 7 is where they meet.

## 6. Risks, and the sentence that retires each

| risk | what would show it | retirement |
|---|---|---|
| the pass-boundary cost of per-block copies eats the window's throughput | Step 3 gate < 0.5 × resident | measured in Step 3 before the scheduler is generalised; if it fails, the window grows (bigger blocks, fewer boundaries) before anything else |
| OOM error scopes make allocation effectively async and slow the hot loop | `cost.ts` submits > 1, or step time up | deferred confirmation at flush only; scopes off when no budget is set |
| f16 cannot be verified on this Mac (SwiftShader) or in CI | every 4x gate | GPU-only gates run by the peer's headed session and the nightly on a real adapter; recorded in `SCALE-MEASURED.md` with the adapter named |
| Qualcomm has no storage-f16 | a device where the window's f16 path fails | Step 4a keeps the f32 window path as the default; `Device.precision` decides per device, never per program |
| capture/replay (`compiled`) silently reads evicted weights | zeros in a compiled run | decision 6 — mutual refusal by name until refill-in-place is proven under a capture |
| a view or a backward closure references an evicted slot | `refuseIfDead` throws | that is the intended failure; the fix is the liveness gate, never a longer schedule |
| OPFS quota / Safari eviction loses the cache | re-download on next visit | `estimate()` before writes; the hub already tolerates a lost cache (verified bytes are in hand) |
| checkpoint recompute fires the version guard | spurious "modified in place" errors | re-stamp `savedVersions` on recompute; a test that in-place-modifies inside a checkpointed segment must still be refused |
| Tensor object replaced somewhere in the swap | optimizer steps a dead parameter | the `applyLora` helper swaps modules *before* the optimizer is built, and the window repoints `.gpu` only |

## 7. Trust layer (not code, but on the critical path to being used)

The cold-start check found `pyborch` for its exact query and could not tell whether it
was safe to depend on. Alongside the steps:

- PyPI and npm pages carry the measured tables (parity, golden, the envelope numbers) and
  the version that measured them — the same tables the landing page shows.
- `ROADMAP.md`'s "what will not be done" row splits "mixed precision" out of "CUDA,
  distributed" and points here, so a reader does not find a refusal that the code no
  longer matches.
- `SCALE-MEASURED.md` is the ledger every gate above writes into: adapter, date, commit,
  number. A claim about scale that is not in it is not made.

## 8. Reading order for whoever picks this up

1. §2 of this file, then `device.ts:1152-1159` and `:594-611` in the source — the two
   comments that explain most of the design.
2. `WEBGPU-DESIGN.md` §7 (memory), §10 ADR-001's 2026-09-08 note (the views lesson) and
   ADR-003.
3. `borch-ts/test/cost.ts`'s header — how a gate is frozen here.
4. Step 0. Do not start Step 3 without its table.
