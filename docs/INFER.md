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

## 7. Risks, and the sentence that retires each

| risk | what would show it | retirement |
|---|---|---|
| a replayed forward reads a weight the user changed after capture | stale logits | the mutual refusal already in `SCALE.md` decision 6 (a captured network refuses in-place parameter writes by name); Step 6's call re-captures when a parameter's version moves |
| the input buffer under capture is re-allocated instead of re-written | replay reads the old input | the `Capture.uploaded` contract — the input is written into the pinned buffer; a test that changes the input between replays and checks the logits move |
| the workgroup-memory staging tile does not fit the smallest tier | shader compile refuses | tile chosen from `Device.workgroupStorage`; the parity suite runs at the 16 KiB tier in CI's SwiftShader |
| the subgroup K-split's slab sum costs what the split saves | Step 3 lift 1 slower than `cnt` | measured alone before lift 2; retired if it is |
| single-op timings swing 2× on the M4 Max | one run says a step won | every gate is two runs, fenced, adapter printed — the rule the Adam arena taught |
| Vulkan never gets subgroup matrices on the 4090 | Step 5's probe line | Steps 1–2 stand on their own there; Step 3's gain is Apple-only until the probe says otherwise |
