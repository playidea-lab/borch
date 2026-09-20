# Inference — where the forward loses to an engine, and the six steps that close it

> Plan, written 2026-09-20 from the same-page measurements of 2026-09-19 (`docs/BOOK.md`,
> "And the half this library loses") and a read of the forward's code path. It is the
> inference twin of `docs/SCALE.md`: every step has a gate that is a number, and a step
> whose gate fails is retired, not argued with. Nothing here touches the numpy core.

## 0. Where the forward stands

ResNet-18 (CIFAR), torch's weights, the logits within 1e-7 of torch's, mean of twenty
after three warm-ups, readback included — the same page as `compare:ts`:

| forward, ms | adapter | batch 1 | batch 16 |
|---|---|---|---|
| borch.ts fused (`fuse_conv_bn_eval` + `nn.intrinsic`), 2026-09-19 | `apple / metal-3` | **2.3–2.9** | 7.0–7.4 |
| ONNX Runtime Web 1.29.0, WebGPU EP | `apple / metal-3` | 4.0–4.1 | **5.3** |
| borch.ts fused, 2026-09-03 | `nvidia / lovelace` | **2.8** | 4.6 |
| ONNX Runtime Web 1.29.0, WebGPU EP | `nvidia / lovelace` | 3.4 | **3.5** |

At batch 1 the fused network is ahead on both adapters. At batch 16 it is 1.3–1.4×
behind on both. The training step is 2.4–16× ahead of every library that trains in a
browser (`compare-peers:ts`), so this is the one table where borch is second, and the
plan is about that column.

## 1. What the batch-16 forward is made of (apple/metal-3, 2026-09-19)

The unfused forward at batch 16: **10.06 ms wall, 5.3 ms of GPU time, 97 dispatches.**
Fused: **7.37 ms wall, 38 dispatches** — the fusion removed 59 dispatches and 2.7 ms,
and not one of those milliseconds was GPU time. The GPU time by kernel:

| kernel (unfused, batch 16) | ms | share |
|---|---|---|
| `cnt` 512→512 on a 4×4 plane (scalar tiled GEMM, K-split) | 1.4 | 26 % |
| `cnt` 256→256 on 8×8 (scalar tiled GEMM, K-split) | 1.4 | 26 % |
| `cnf` 64→64 on 32×32 (subgroup forward) | 0.5 | 9 % |
| `cnf` 128→128 on 16×16 (subgroup forward) | 0.4 | 8 % |
| `cnt` 256→512 stride 2, `cnd` direct convs, one `bna` | 0.8 | 15 % |
| everything else | 0.8 | 15 % |

So the forward is three buckets, and each has a different lever:

- **(a) 2–3 ms that is not GPU time.** Wall minus GPU: 7.37 − ~4.5 at batch 16, 2.89 −
  ~1.2 at batch 1. It is what it was in the training step before capture
  (`docs/SCALE.md`, "the step's 30 % overhead is CPU encode"): JavaScript building bind
  groups and encoding thirty-eight dispatches, an allocation per intermediate, and the
  readback's round trip. ORT pays the same kind of cost — at batch 1 it is *slower* than
  the fused network — but it pays it once per session, at graph build.
- **(b) 2.8 ms in two convolutions that never reach the subgroup kernel.** `sgfFits`
  (`kernels.ts`) takes the subgroup forward only when the output row is a multiple of
  eight, the reduction is unsplit and there is no epilogue. The 4×4 plane fails the first
  (a row of four), both deep layers take the K-split path (`convForwardSplit` > 1, checked
  before the subgroup path in `tensor.ts` `convNDForward`), and the fused network's
  relu/residual epilogue fails the third. So the two layers with the longest reductions in
  the network (K = 4,608 and 2,304) run on the scalar tiled GEMM — the kernel ORT's
  subgroup-matrix conv is "better fed" against, as the BOOK put it.
- **(c) the early wide-plane layers read more than they multiply.** 64 channels on 32×32:
  0.5 ms for a ninth of the FLOPs of the 512-channel layer. Bandwidth, not arithmetic.

And one cost that is in every bucket: **the subgroup forward repacks its weights every
call.** `convNDForward` allocates `turned` and dispatches `tmw` (tap-major weights) on each
forward, because in training the weights change every step. In `eval()` they do not, and
the repack is a dispatch plus a weight-sized write per layer per forward — ORT prepacks at
session creation and never again.

## 2. What ORT does that the forward does not — and which of it is ours to take

| ORT Web (WebGPU EP) | borch.ts today | take it? |
|---|---|---|
| static graph → the whole forward encoded once, replayed | eager: encoded every call | **yes** — `torch.capture()` exists and is measured on training; it has not been pointed at `eval()` |
| weights prepacked at session build | `tmw` every forward | **yes** — cache the turned weights on the parameter, invalidated by its version |
| subgroup-matrix conv for every shape, epilogues in-kernel | subgroup only for stride 1 / row % 8 / unsplit / no epilogue | **yes** — three lifts of `sgfFits`, §3 Step 3 |
| memory planned once, no allocation in the hot loop | pooled allocation per intermediate | comes free with capture (replay reuses its buffers) |
| layout chosen per graph (NHWC where the kernel wants it) | NCHW, torch's, everywhere | **no** — our kernels are ours and read NCHW at full speed on the subgroup path; a layout switch would be a second kernel set for a gain not yet shown |
| graph-level fusion beyond conv+bn+relu+add | `fuse_conv_bn_eval`, `nn.intrinsic` | not now — the remaining fusions (pool into conv, the final GEMM's bias) are single dispatches |
| f16 weights | f32 only | **maybe** — Step 4; a memory lever on metal-3 (`f16_stream_probe`), not a compute one |

## 3. The steps

### Step 0 — The instrument · size S

`compare.ts` prints, for the fused forward, wall time and dispatches. It does not print
its GPU time by kind, and it has no row for a captured forward. Add both, and the
`measured on:` line already there carries them.

- **Gate**: the table has a `fused + captured` row and a per-kind breakdown for the fused
  forward, on both adapters (the 4090 through cq — the wrapper's `script` mode runs any
  browser script of the checkout).

### Step 1 — Replay the eval forward · size S · bucket (a)

`torch.capture()` around `noGrad(() => model.forward(x))`, with `x` a pinned input
buffer the caller writes into before each replay (the `Capture.uploaded` set is exactly
this: buffers `upload` made under the capture). Nothing new in the device; the API
surface is `model.compile_for_inference()`-shaped (Step 6) but the measurement comes
first.

- **Predict**: batch 16 wall 7.0–7.4 → ~5.0 ms on metal-3 (GPU time + one submit + the
  readback); batch 1 2.3–2.9 → ~1.5. That alone puts batch 16 level with ORT's 5.3.
- **Gate**: fused + captured wall ≤ fused GPU time + 0.6 ms at batch 16, two runs; logits
  bit-identical to the eager fused forward (the capture probe's standard).
- **Retire if**: the replay is not within 1 ms of the GPU time — then the readback, not
  the encode, is the cost, and Step 1 becomes "double-buffer the readback", a different
  and smaller step.

### Step 2 — Prepack once · size S · every bucket

The turned weights (`tmw` output) cached on the `ConvND` parameter under `eval()`,
keyed by the weight buffer and its version counter (the in-place guard already stamps
one); `fuse_conv_bn_eval` and `nn.intrinsic` construction is the natural moment to fill
it, a first `noGrad` forward the fallback. Training mode never reads the cache.

- **Gate**: zero `tmw` dispatches in the steady-state eval forward; the eager fused
  wall drops by the `tmw` time (measure it first — it is in the per-kind breakdown Step 0
  adds). Under capture the replay would repeat `tmw` too, so this step is not made
  redundant by Step 1.

### Step 3 — The deep layers onto subgroup matrices · size M · bucket (b)

Three lifts of `sgfFits`, in this order, each measured alone:

1. **K-split on the subgroup forward** — the weight gradient already splits its reduction
   on subgroup matrices (`convGradWeightSubgroup`, the `cnwg` floor); the forward gets the
   same: pieces of whole eights of K into a `parted` slab, then `sumSplitsConv` with the
   epilogue, as the scalar path does today. This alone moves the 8×8 layer (row of eight
   fits) off `cnt`.
2. **Rows shorter than eight** — the pixel axis becomes `N·OH·OW` staged through
   workgroup memory (an implicit-GEMM tile gathered from the padded plane, then
   `subgroupMatrixLoad` from workgroup storage), so the 4×4 plane is 16 pixels per image
   and the tile spans images. The staging tile is bounded by
   `Device.workgroupStorage` (16 KiB on the smallest tier) — the tile shape is chosen
   from it, not fixed.
3. **Epilogue in-kernel** — relu and the residual add applied when the accumulator tile
   is stored, so the fused network's `ConvReLU2d` / `ConvAddReLU2d` take the subgroup path
   instead of falling back to `cnt`.

- **Predict**: the two layers 2.8 → ~0.8 ms at batch 16 (the 64-channel subgroup layer
  runs its FLOPs at ~4× this rate today); forward GPU time 5.3 → ~3.3.
- **Gate**: `cnt` absent from the ResNet-18 forward's top eight; batch-16 fused GPU time
  ≤ 3.5 ms; every conv shape in the parity suite still within 1e-6 of torch (the padding
  bug the fake probe hid in September is the reason this gate is the parity suite and
  not the bench).
- **This is the step that decides batch 16.** Steps 1–2 reach parity with ORT; Step 3
  is what would put the fused forward ahead at batch 16 on metal-3.

### Step 4 — The wide early layers · size M · bucket (c) · optional

The 64→64 conv on 32×32 reads each input pixel nine times from storage. Two candidates,
measured against each other on `kernel_bench` before either enters the tree:
(i) the input tile staged once in workgroup memory and read for all nine taps;
(ii) f16 storage for the weights and the padded input (`castF32ToF16`, bench-only today)
— on metal-3 it halved residency and did not speed compute, but this layer is bandwidth.

- **Gate**: `cnf` 64@32×32 0.5 → ≤ 0.3 ms at batch 16. Retire the candidate that does
  not reach it; if neither does, retire the step.

### Step 5 — NVIDIA · size S · measurement

On the 4090 through Chrome 143 / Vulkan the subgroup kernels are absent and every path
falls to the scalar ones (`docs/BOOK.md`, the compiler section). Steps 3–4 give that
card nothing until `chromium-experimental-subgroup-matrix` reaches Vulkan on it.
`tests/browser/features_probe.py` answers, per flag set, what this Chrome on this card
exposes; run it through cq at each Chrome update and record the line in
`docs/SCALE-MEASURED.md`. Steps 1–2 are layout- and feature-independent and are the
whole plan for that card meanwhile.

- **Gate**: the probe's line for the 4090, dated, in the ledger; the Step 1 gate re-run
  there.

### Step 6 — One call · size S

`model.eval()` + `fuse_conv_bn_eval` + `nn.intrinsic` + prepack + capture behind one
name — torch's shape is `torch.compile(model, mode="reduce-overhead")`, and that is the
name to borrow (`torch.compile` is on the "deliberately not supported" list for training,
where it would promise a graph compiler; for inference it promises exactly this). The
Python binding gets the same call. The workbench's frozen-backbone pass
(`_workbench.py`, the `no_grad` pass over the backbone) is its first user.

- **Gate**: one call; logits within 1e-7 of the eager forward; the `compare:ts` table
  gains the row and the BOOK's "for inference alone, use ORT Web" sentence is re-read
  against the new numbers — kept if they say so.

## 4. What this plan does not do

- **A layout switch (NHWC)** — §2; a second kernel set for an unmeasured gain.
- **A graph compiler of the IR kind** — what exists is half of one: `torch.capture()`
  records a run and replays it (CUDA-Graph-shaped), and its fusion pass folds elementwise
  trees into single kernels (`fuse:py`, 2165 → 1699 dispatches on ViT-tiny). What it is
  not: an IR with shape inference, algebraic rewrites, layout and memory planning, or
  per-shape kernel selection. Those are not built because the gains they would reach
  here are already taken by hand where they were measurable (`nn.intrinsic`, the
  matmul-epilogue fusion tried and refused at 0.72 → 1.69 ms) and the fusions that
  remain after conv+bn+relu+add are single dispatches each. LLM decode would want
  dynamic shapes (a growing KV cache) that a recording cannot hold — and decode is out
  of scope anyway (`docs/SCALE.md` Step 8), because its cost is weight bandwidth and
  int4 kernels, not the graph.
- **LLM decode** — int4, KV cache, decode-shaped kernels; `docs/SCALE.md` Step 8 says
  why not, and nothing here changes it.
- **Safari / Firefox** — unmeasured; the same-page harness runs in Chrome.
- **Beating ORT at every batch on every card** — the claim to make is the one the table
  supports on the day it is printed, with the adapter beside it.

## 5. Order and size

```
0 ──► 1 ──► 2 ──► 6        (overhead track: S · S · S · S)
 │
 └──► 3 ──► 4              (kernel track:   M · M-optional)
 5 at every Chrome update
```

The two tracks touch different files (`device.ts`/`nn.ts` against `kernels.ts`) and run
at once. Predicted end state on metal-3, batch 16: today 7.0–7.4 ms → Step 1 ~5.0 →
Step 2 ~4.8 → Step 3 ~3.0, against ORT's 5.3. Batch 1: 2.3–2.9 → ~1.5. The
prediction is written down so that the ledger can say which step was wrong.

## 6. Ledger

- **2026-09-20, Steps 0 and 1 landed — the instrument, and the eval forward replayed.**
  `compare.ts` prints the fused forward's GPU time by kind and a `fused + captured` row:
  `compiled` over the fused network's `noGrad` forward (`docs/COMPILER.md` Step 6's JS
  `compiled`), its logits held to the eager fused forward's bit for bit before the clock
  is read. **metal-3, twenty forwards after three warm-ups, readback included: batch 1 —
  fused 2.86 → captured 1.97 ms (ORT 3.83); batch 16 — fused 7.39 → captured 5.57 ms
  (ORT 5.32), 38 dispatches a replay, max |replay − eager| 0.** The prediction was ~5.0 at
  batch 16; 5.57 is the fused forward's profiled GPU time (5.4, a pass per dispatch while
  profiling) plus a submit and the readback, so the gate — wall ≤ GPU + 0.6 ms — is met
  and the non-GPU 2 ms is gone. At batch 16 the fused-and-captured forward is 1.05× ORT;
  at batch 1, 0.51×. What the instrument says about Step 3's order: in the *fused* forward
  the 64- and 128-channel layers run on the direct kernel with the epilogue (`cnd:…:b:ra`,
  ~2 ms of 5.4) — the subgroup forward has no epilogue, so `nn.intrinsic` sends them
  elsewhere — and the two deep layers stay on the scalar tiled GEMM (`cnt`, 2.7 ms). So
  lift 3 (epilogue in the subgroup kernel) comes first, then Step 2's prepack (only a
  subgroup layer repacks), then lifts 1–2 for the deep layers.

- **2026-09-20, Step 3, first attempt — measured slower, reverted.** A gathered subgroup
  forward (`cnfg`: one subgroup a workgroup, a 16-channel × 16-pixel tile, the 8 × 16 input
  block for each channel block and tap gathered by the lanes into workgroup memory with the
  `(n, oh, ow)` decode and the padded, strided source index computed per element, then
  `subgroupMatrixLoad` from workgroup storage; partial sums per K piece into a slab and
  `sumSplitsConv` for bias and epilogue — so any stride, any row length, any epilogue).
  Correct on every conv the golden asks (4,057 / 4,057, logits 7.45e-8 from torch's). **And
  slower**: metal-3 batch 16, 512 → 512 on 4 × 4 — 2.00 ms against `cnt`'s 1.38; 256 → 256
  on 8 × 8 — 1.30 against 1.35; the fused + captured forward 5.57 → 6.79 ms; the training
  step 21.3 → 24.1 (`cnfg` 4.1 ms where `cnt` had 2.8). Batch 1 was mixed (0.49 vs 0.57 on
  the 4 × 4 layer, 0.56 vs 0.30 on the 8 × 8) and the weight repack it needs (`tmw`, 0.4
  ms at batch 1) is a cost of its own. The diagnosis is arithmetic: per channel block and
  tap the workgroup gathers 128 input values and loads 128 weights for 2,048 multiply-adds
  — eight per load, with two barriers — where the scalar tiled GEMM's 64 × 16 tile with
  register blocking reuses each load more and runs many more threads; the hardware
  multiply is starved. The kernel that would win is the implicit GEMM proper: four
  subgroups a workgroup sharing a 32 × 64 staged input block, the weight block staged too
  and reused across pixel tiles, K walked in eights — a different size of work than this
  step was given. **Reverted** (the attempt is the commit before the revert); the plan's
  Step 3 stands with that design under it, and the number to beat is `cnt`'s 1.38 ms.

- **2026-09-20, Step 3, second attempt — the subgroup forward widened, and the forward is
  ahead of ORT at batch 16.** Three changes to `cnf`, each measured: (1) its gate carried the
  direct kernel's `C ≤ 128` through `directFits`, a bound this kernel never needed (it reads
  the weight block from storage per tap) — dropped, and the 256-channel 8 × 8 layer left the
  scalar GEMM: **1.4 → 0.7 ms at batch 16, 8.4 → 3.0 at batch 64**, the training step
  21.3 → 19.9 ms. (2) An epilogue at a staged store — the tiles through workgroup memory,
  the lanes adding the residual and clamping as they write — so the fused `ConvReLU2d` /
  `ConvAddReLU2d` take it: the fused 256 → 256 layer **1.35 → 0.46 ms**. (3) The tap-major
  repack `tmw` rewritten one thread a (row, column) writing every tap's slab — contiguous
  reads, coalesced writes — from 0.7 ms on the 512-channel weight to below the top eight.
  **Fused forward at batch 16: 7.39 → 6.46 ms eager, 5.57 → 4.95 captured, against ORT's
  5.30–5.39 — 0.93×, ahead; at batch 1, 1.66 against 4.07.** Golden 4,057 / 4,057, logits
  8.9e-8. What was measured and taken back out: a row shorter than the tile over a wider
  padded row (tiles of eight computing eight to keep four) put the 512-channel 4 × 4 layer
  on this kernel at **1.5 ms against `cnt`'s 1.4** — the waste ate the multiply — and cost
  the training step 19.9 → 21.0 with its repack; `sgfFits` keeps the row of eight and that
  layer keeps the scalar GEMM. It is now the largest single kernel of the fused forward
  (1.39 of 4.9 ms) and the one a proper implicit GEMM would still be for.

- **2026-09-20, Step 2 landed as a compiler pass, not a cache.** `Capture.hoist()`
  (`docs/COMPILER.md`): a dispatch whose every read is a constant of the recording — no
  record writes it, it is not an uploaded input, not a window slot — and whose writes are
  pure, gives the same bytes on every replay; it ran when the step was recorded and leaves
  the replay, its outputs pinned for the capture's life. Found to a fixed point. In the
  captured eval forward that is exactly the three `tmw` repacks of the subgroup layers:
  **41 → 38 dispatches a replay**, the repack paid once. In the training step nothing
  qualifies (the optimizer writes the weights) and the pass finds that itself — `capture:py`
  257 dispatches, bit for bit. Nothing is cached on a tensor, nothing can leak, and any
  future prepack (a constant's pad, a block of ones) hoists the same way.

- **2026-09-20, Step 3, third attempt — the last layer leaves the scalar GEMM.**
  `convForwardSubgroupSmall` (`cnfs`), for the plane the row-of-eight kernel cannot take:
  the padded planes of one channel block for the tile's images are **staged once in
  workgroup memory** (a 4 × 4 plane padded is thirty-six floats; eight channels of two
  images, 576), each tap's 8 × 32 input block is assembled from that staging — no gather
  from storage, no phantom column — and **four subgroups share it**, each owning sixteen
  output channels, so the staging is paid once for sixty-four and every weight block
  loaded from storage feeds thirty-two pixels. Partial sums per K piece, `sumSplitsConv`
  for bias and epilogue, as before. What the two lost attempts taught is in its shape: the
  first gathered every tap from storage (eight multiply-adds a load), the second computed
  phantom columns; this one loads each input value from storage once per channel block and
  computes only real pixels. **metal-3, the three 512 → 512 layers on 4 × 4 at batch 16:
  `cnt` 1.38 → `cnfs` 0.84 ms; at batch 64, 10.8 → 4.6.** Fused + captured forward at
  batch 16 **4.95 → 4.42 ms against ORT's 5.25–5.31 (0.83×)**; batch 1 **1.38** against
  3.07–4.97. The training step 20.0 → 19.5 ms at batch 16, 33.9 → 31.8 at 32, **59.7 → 54.1
  at 64**. Golden 4,057 / 4,057, logits 6.0e-8 from torch's; the replay bit for bit. With
  this, no convolution of ResNet-18 runs on the scalar tiled GEMM at stride one; what
  remains on it is the stride-2 shortcut and the stride-2 3 × 3 (`cnt:…|2,2|…:s`, 0.3 ms),
  and the early wide layers on the direct kernel (Step 4's question, still open).

- **2026-09-20, Step 4 — candidate (i) built, measured per shape, and kept where it wins.**
  The staged kernel of Step 3 generalised from whole planes to *bands*: a tile of thirty-two
  pixels is whole images (a plane dividing thirty-two) or a run of rows of one image (a row
  dividing thirty-two), and the staging holds the padded rows the tile touches plus the
  kernel's reach — 32 × 32: one row, 816 floats; 16 × 16: two rows, 576 — so the input is
  read from storage once per channel block and the nine taps read it from workgroup memory.
  Unsplit, it applies bias and epilogue at its own store (the seven `sumSplitsConv` passes it
  first added cost the forward what the kernel won: 4.42 → 4.52 ms, measured). Then each shape
  against the kernel it had, fused, batch 16, metal-3: **128 → 128 on 16 × 16: direct 0.81 →
  staged ~0.6 ms — kept; 64 → 64 on 32 × 32: direct 1.02 → staged 1.03 — a wash, kept for the
  dispatch it saves; 256 → 256 on 8 × 8: row-of-eight 0.75 against staged 0.85 — the
  row-of-eight kernel stays, by a rule on the row.** The gate as written (the 64-channel layer
  0.5 → ≤ 0.3) is not met: that layer is bandwidth on either kernel, and candidate (ii),
  half-precision storage of its input and weights, is the one left to it. What the forward
  gained anyway: **fused + captured at batch 16 4.42 → 4.14 ms (ORT 5.25–5.88, 0.70–0.79×),
  38 dispatches a replay; batch 1 1.38 → 1.10.** The training step 19.5 → 19.0 at batch 16.
  Golden 4,057 / 4,057, logits 6.0e-8, replay bit for bit.

- **2026-09-20, Step 4, candidate (ii) — retired by a measurement made before it was
  built.** The question was whether the 64-channel layer is bandwidth. The bytes say no
  (its weights are 147 KB a tile and cache; its staged input is 13 MB a layer — under
  0.2 ms of DRAM either way), so the test was the other suspect: the two barriers a tap.
  The staged kernel now assembles a kernel row's three taps at once and pays its barriers
  per row (six a channel block, not eighteen). **The 64-channel layer did not move: 1.03 →
  0.98 ms for its two convolutions, noise; the 512-channel 4 × 4 layer 0.84 → 0.77.** So
  the layer is neither DRAM-bound nor sync-bound; it runs at ~4.9 TFLOP/s, about half the
  subgroup GEMM's measured rate, and the other half is the staging and assembly work a
  convolution has and a matmul does not. Half-precision storage cuts bytes the layer is
  not waiting on; not built. The barrier schedule is kept (simpler, never slower). What
  would move this layer next is a different data path — B kept resident across taps by
  shifting the assembled block rather than rebuilding it — and that is the next plan's
  question, not this one's. The forward stands at **4.1–4.2 ms captured at batch 16
  (ORT 5.3–5.9) and 1.1 at batch 1**; golden 4,057 / 4,057, logits 6.0e-8.

- **2026-09-20, Step 6 landed — one call, under the name the library already has.**
  `torch.compiled(model)` (a module where a function was): eval mode, the folds, and the
  `noGrad` forward recorded — the repacks hoisted, the intermediates in arenas, every call
  after the first a replay. Not `torch.compile`: that name is on the list of what this
  library deliberately does not do, and a training-time graph compiler is still what it
  would promise; the inference form lives beside `compiled(fn)`, which it is. The fold is
  `fuseForInference`: a module that defines its own `fuse()` is asked (a residual block's
  add is in its forward, where no container can see it — the bench's ResNet-18); any other
  module folds its `Sequential`s (`Conv → BN` into one convolution by `fuseConvBnEval`,
  then `Conv → ReLU` into `ConvReLU2d`) and recurses. The binding's `compiled(model)`
  does the same through a model's `fuse()`, a borch.ts-backed child's fold, or its
  children. **Held: `capture:ts` — on the ResNet-18 the call gives the hand-fused eval
  forward bit for bit, 38 dispatches a replay; a plain `Sequential` folds to
  `ConvReLU2d → Conv2d`, replays in 3 dispatches where its eager eval forward took 7, and
  matches it to the output's scale. `capture:py` — the U-Net (a Python module of binding
  `Sequential`s) through `torch.compiled(m)`: the plain eval forward exactly (rel 0),
  replays identical, 52 dispatches.** Two mistakes on the way: the fold before `eval()`
  ("Fusion only for eval!" — order matters), and a per-element relative gate that read a
  1e-7 difference on a value near zero as 7e-4.

- **2026-09-20, Step 5 — the second adapter measured, through cq on the RTX 5080.**
  `features_probe` on `nvidia / blackwell` (driver 580, Chrome 151, Vulkan):
  `chromium-experimental-subgroup-matrix` **is** exposed, with configurations `u8`/`i8` →
  `u32`/`i32` at 16 × 16 × 32 and 16 × 8 × 32 — **no f32 8 × 8 × 8**. `subgroupMatrixF32`
  reads the configurations, not the feature name, so the device came up with
  `subgroupMatrix` off and every convolution on the scalar kernels; nothing faulted,
  nothing read as zeros. So Steps 3–4 give this card nothing today, as the plan said, and
  Steps 1–2 are its whole gain: **fused + captured forward 3.00 ms at batch 1 (ORT 3.98,
  0.75×) and 4.32 at batch 16 (ORT 3.58, 1.21×)**, GPU 2.9 ms of the 4.32 — the
  submit-and-readback round trip is 1.3–1.4 ms on Linux/Vulkan against 0.3 on metal-3,
  and that, not a kernel, is the batch-16 gap there (measured the same night — it is not
  a round trip, it is the GPU process's polling schedule; the entry below). The rest of the day's work held on
  the card: `capture:ts` 18 / 18 (ResNet-18 replay bit for bit, 13.4 → 11.6 ms; the plan
  415 → 161 MB in 4 arenas; the streamed chain; `compiled(model)`); training 13.5 / 24.4 /
  37.4 ms at batch 16 / 32 / 64 against TF.js's 75.1 / 123.8 / 235.7 (5.6× / 5.1× / 6.3×)
  and jax-js's 78.5 / 86.2 / 118.7. The 4090 stays off the bus; the 5080's cq worker was
  held by a run whose process had ended (the harness took a debug run regardless). What
  the card asks for next is an int8 subgroup path — the configuration it has — which is
  a plan of its own, not a lift of this one.

- **2026-09-20, the round trip — measured, and it was not a round trip.** The open item
  above: 4.32 ms of wall for 2.9 of GPU on the 5080, called "the submit-and-readback
  round trip, 1.3–1.4 ms on Linux/Vulkan against 0.3 on metal-3". `tests/browser/
  roundtrip_probe.py` (`npm run roundtrip:probe`, census 68) takes it apart with raw
  WebGPU on both adapters: work of five sizes with a timestamp query around it, then the
  same work under four ways of waiting for its readback. **On the 5080 every wait whose
  fence is not signalled at the GPU process's first look lands at the same 2.1–2.7 ms**:
  forty tiny dispatches are 0.08 ms of GPU and 2.6 of wall under `onSubmittedWorkDone`
  or `mapAsync`; a 0.12 ms kernel is 2.6; a 1.57 ms kernel is 2.5. That is a polling
  schedule, not a transfer — the GPU process looks at its fences on a period of about
  two milliseconds unless something makes it look. **A `pushErrorScope`/`popErrorScope`
  round trip (0.03 ms) makes it look**, and a loop of them until the map resolves brings
  the wall to the GPU's time plus 0.1: forty dispatches 0.20, the 1.57 ms kernel 1.66.
  metal-3 has none of it — wall = GPU + 0.2 under every wait, kicks included — and after
  an idle gap it pays ~0.7 ms of wake-up that kicks do not remove. So the library kicks
  where the browser needs it and nowhere else: `Device.readbackKicks`, decided once at
  `create` by `calibrateKicks`, and `Device.kicked()`, which `read()`, `synchronize()` and
  the timestamp readback go through. The calibration took seven versions to get right,
  each retired by a number: forty tiny dispatches (the 5080 sometimes finished them
  before the first look — "no kicks"); a slowest-of-eight rule (metal-3's clock ramp ran
  the kernel at 15 ms on its first waits — "kicks"); idle gaps between waits (metal-3
  read 9 ms once — "kicks"); interleaved pairs (a kicked wait resets the schedule and
  the plain wait after it is fast — "no kicks"); a runtime detector on an empty submit
  (a fence wait covers everything queued, and under a bench that queues the next
  forward it waited for that: the eager fused forward on metal-3 2.9 → 6.1 ms); a
  kicked-median ratio (a kick is 0.03 ms with the GPU process idle and 0.6–2.0 when it
  is not — "no kicks" three times on the card that stalls). What stands: a 0.12 ms loop
  kernel, timestamp-queried, waited for plainly in a run after six warm-ups; **wall
  minus GPU above 1.7 ms in any of four rounds** (40 ms idle before each) is the stall.
  metal-3 reads 0.3–1.2 over the GPU in its worst round across every calibration
  measured; the 5080 reads 2.0–2.9. Kicks are calibrated rather than assumed because on
  metal-3 they **cost** the eager fused forward — 2.9 → 5.8 ms at batch 1, 6.4 → 12.2 at
  batch 16 — while the captured and the training step do not move; the mechanism is
  not known, the cost is. The adapter line of every table now carries the decision and
  its numbers. **Held: `capture:ts` 18 / 18 on both adapters.** What it bought on the
  5080, `compare:ts` with kicks on: **fused + captured 0.99 ms at batch 1 (ORT 3.88,
  0.26×) and 2.66 at batch 16 (ORT 3.63–3.89, 0.68–0.73×) — borch ahead on the second
  adapter at both batches**, where the day started at 3.00 / 4.32 against 3.98 / 3.58;
  the training step 12.3 / 17.4 / 27.6 ms at batch 16 / 32 / 64 (was 13.5 / 24.4 / 37.4)
  against TF.js 75.2 / 123.4 / 234.3 — 6.1× / 7.1× / 8.5×. The batch-64 step lost ten
  milliseconds: it waited more than once. Two sentences retired: the one that called
  this a round trip, and `docs/INT8.md`'s premise that int8 was the remaining lever on
  this card. Open: a `synchronize()` right after a dropped matmul still reads 2.8 ms on
  the probe where the same call followed by a readback reads 1.2 — not understood, and
  no bench uses that path.

- **2026-09-21, the scalar kernels swept on the second adapter — Step 5's other half.**
  With the stall gone the 5080's captured forward was 2.61 ms of which 2.9 was GPU, and
  the two deep layers ran at 1.8 TFLOP/s on `cnt` where the same card's scalar GEMM does
  15.8: every constant in the scalar conv path had been chosen on the M4 Max. `kernel_bench
  fwd` gained the tiled kernel and a sweep (`--sweep=splits|tiles|direct|all`, the split
  and tile forced through globals only the bench sets), and the sweep decided three
  things, each a number on both adapters:
  - **The split policy** (`convForwardSplit`): 128 tiles wanted and pieces of 256 → **1024
    and 128**. 512 → 512 at 4 × 4, batch 16, on the 5080: split 1 0.635 · 2 0.323 · **4
    0.179 (the policy)** · 8 0.120 · 16 0.110 · 32 0.100 ms; metal-3 4 0.437 → 16 0.315.
    64 × 64 won every tile sweep. In the step, the two deep layers 0.68 → 0.38 each;
    forward 2.61 → 2.33, batch 1 0.98 → 0.85, training 13.1 / 17.4 / 28.2 → 11.6 / 15.0 /
    24.7.
  - **The direct kernel's weight slice** (`setDirectWeightBytes`): capped at WebGPU's
    16 KiB floor, a 128-channel layer's slice was three; at the device's 48 KiB it is
    eight — 128 → 128 at 16 × 16, batch 16: 0.146 → **0.093** on the 5080, a wash on
    metal-3's 32 KiB (0.264 / 0.272).
  - **Small direct grids go to the split GEMM** (`directGridFills`, 64 workgroups): at
    batch 1 the direct kernel's grid is 43 workgroups for 128 → 128 and the tiled kernel
    split sixteen is 0.027 against 0.042; the stride-2 128 → 256 layer at batch 16 (a grid
    of 32) 0.33 → 0.08 in the step.

  **Held**: `capture:ts` 18 / 18 on both adapters; `compare:ts` on the 5080: **fused +
  captured 0.65 ms at batch 1 (ORT 3.51, 0.19×) and 1.74 at batch 16 (ORT 3.62,
  0.48×)**; training 10.6 / 14.7 / 23.2 ms at batch 16 / 32 / 64 (TF.js 75.4 / 122.0 /
  236.7 — 7.1× / 8.3× / 10.2×). metal-3 unchanged (4.1 / 1.1; its deep layers are on the
  subgroup kernels). The prediction written before the sweep was "forward 2.61 → ~1.7";
  it landed at 1.74. What is left on the 5080's GPU time at batch 16 (1.9 ms) was first
  read as "the two deep layers at 0.37 each — 3.5× the bench's 0.107, a discrepancy to
  understand". **It was not a discrepancy: a kind's time in the tables is the sum over
  every dispatch of that kind, and the ResNet-18 has three 512 → 512 convolutions at
  4 × 4 and three 256 → 256 at 8 × 8.** 0.37 / 3 = 0.12 — the bench's number. The tables
  now print the count beside the time (`cnt:… 0.37×3`), so that a sum is never again read
  as a dispatch. Per dispatch, then, every convolution of the forward on the 5080 runs at
  0.08–0.12 ms — 10–13 TFLOP/s against the scalar tile's own 15.8 on 1024³ — and the
  1.9 ms is twenty of them. Nothing cheap is left on the scalar path; what would move it
  is a faster scalar GEMM (the tile itself is at 15.8 of a card that does ~56 f32) or
  the int8 configuration (`docs/INT8.md`, 3–3.6× on the GEMM core).

## 7. Risks, and the sentence that retires each

| risk | what would show it | retirement |
|---|---|---|
| a replayed forward reads a weight the user changed after capture | stale logits | the mutual refusal already in `SCALE.md` decision 6 (a captured network refuses in-place parameter writes by name); Step 6's call re-captures when a parameter's version moves |
| the input buffer under capture is re-allocated instead of re-written | replay reads the old input | the `Capture.uploaded` contract — the input is written into the pinned buffer; a test that changes the input between replays and checks the logits move |
| the workgroup-memory staging tile does not fit the smallest tier | shader compile refuses | tile chosen from `Device.workgroupStorage`; the parity suite runs at the 16 KiB tier in CI's SwiftShader |
| the subgroup K-split's slab sum costs what the split saves | Step 3 lift 1 slower than `cnt` | measured alone before lift 2; retired if it is |
| single-op timings swing 2× on the M4 Max | one run says a step won | every gate is two runs, fenced, adapter printed — the rule the Adam arena taught |
| Vulkan never gets subgroup matrices on the 4090 | Step 5's probe line | Steps 1–2 stand on their own there; Step 3's gain is Apple-only until the probe says otherwise |
