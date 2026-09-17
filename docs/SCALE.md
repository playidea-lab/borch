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
