# Int8 — the subgroup path an int8-only adapter has, and what it is worth

> Plan, written 2026-09-20 from one measurement made before anything was built. The
> sibling plans are `docs/INFER.md` (inference) and `docs/COMPILER.md`; this one exists
> because the second adapter measured (`docs/SCALE-MEASURED.md`, RTX 5080 through Chrome
> 151 / Vulkan) exposes subgroup matrices in **int8 configurations only** — `i8 × i8 → i32`
> at 16 × 16 × 32 and 16 × 8 × 32, no f32 8 × 8 × 8 — so every subgroup kernel in the tree is
> off there and the card runs the scalar tile. Every step has a gate that is a number.

## 0. The measurement this plan stands on

`tests/browser/int8_subgroup_probe.py` (`npm run int8-sg:probe`), on the RTX 5080: a GEMM
on the `i8 → i32` 16 × 16 × 32 configuration, written against the proposal's WGSL, swept
over the readings the proposal leaves open, held to a CPU reference, then timed.

| what | measured |
|---|---|
| the load that is exact | operands in `array<i32>` (four `i8` a word; `array<u32>` is for `u8` only), **offset and stride in components, not words**, types `subgroup_matrix_left<i8, K, M>` / `right<i8, N, K>` / `result<i32, N, M>` — the same column-count-first order the f32 kernels use, which the 8 × 8 case had hidden |
| the reading that is fast and wrong | offsets in words: **26 TOPS and every entry wrong** — the first run; a wrong reading of an 8-bit load is a fast wrong answer, which is why the probe sweeps against a reference before it times |
| 1024 × 1024 × 1024, i8 → i32 | **0.078 ms · 27.7 TOPS** (thirty dispatches, 32 × 32 tile a workgroup, one subgroup) |
| the same GEMM, f32 on the scalar tile (`kernel_bench --bench=mm`, same card, same day) | 0.136 ms · **15.8 TFLOP/s** |
| the ratio | **1.75×** on the multiply — not the 10× the f32 subgroup path is over the scalar tile on Apple, because this card's scalar tile is already at 15.8 |
| **the same GEMM under the timestamp profiler, Step 1's kernel** (`kernel_bench --bench=mmi8`, the same night) | **0.038 ms · 56.6 TOPS · 3.59× the tile**; 2048³ 0.273 ms · 62.9 TOPS · 3.0×. The probe's 0.078 was a wall clock over thirty dispatches, and the wall on this card carried the GPU process's polling stall (`docs/INFER.md` ledger, the round trip) — so the ceiling above is **not 1.75× but 3–3.6×** on shapes that fill the card, and 1.0–1.3× on 256³ / 512 × 1024 × 256, where a dispatch is launch-bound |

Two things follow. **The ceiling is 1.75× on the GEMM core**, before any cost of getting
there (quantising activations is a pass per layer; dequantising the i32 result is the
epilogue's). And the forward it would serve — the fused ResNet-18 at batch 16 on this
card — is **2.9 ms of GPU time inside a 4.32 ms captured step**: a perfect int8 pass over
every convolution saves at most ~1.1 ms of GPU, and the 1.3–1.4 ms that is the
submit-and-readback round trip on Linux/Vulkan stays. So the plan's own arithmetic says:
**int8 alone takes this card from 4.3 to ~3.4 ms against ORT's 3.6, at an accuracy cost,
and the round trip is the larger and cheaper lever.** The plan is written so that either
can be taken up; the order is in §4.

## 1. What an int8 path is — and is not

- **Inference only, W8A8.** Weights quantised per output channel to int8 once
  (`quant.ts` has the per-channel quantiser from the int8 window), activations quantised
  per tensor before each int8 layer with a scale, the product accumulated in i32 and
  dequantised by `scale_a · scale_w` in the epilogue with the bias. Training stays f32:
  gradients have no int8 configuration to run on, and the loss curves are held bit for bit.
- **Not bit-for-bit with torch.** Every gate in the tree is exactness against torch's
  f32; an int8 forward is held instead to **accuracy** (the classifier's top-1 on the
  held-out set) and to a **logit tolerance** that is written down and wide. This is the
  first path in the library whose contract is "as good", not "the same", and the tables
  say so on every row it produces.
- **Only where it is the hardware's path.** The routing is by the adapter's
  configurations: an int8 configuration and no f32 one (`Device.subgroupInt8`, read the
  way `subgroupMatrixF32` reads). On metal-3, which has f32 8 × 8 × 8 and where the f32
  subgroup convs already win, nothing changes.

## 2. The steps

### Step 0 — The probe · done

§0. The gate was "compiles, exact, faster than the scalar tile": met, at 1.75×.

### Step 1 — The int8 GEMM in the tree · size S · **done 2026-09-20**

`kernels.ts` `matmulInt8`: the probe's kernel generalised — M, N, K in multiples of the
configuration, a 32 × 32 output tile a workgroup (four 16 × 16 results), K walked in
32s, `array<i32>` operands packed four to a word with offsets in components, the i32 result
dequantised at the store by a per-row weight scale and a per-tensor activation scale, plus
bias. `kernel_bench --bench=mmi8` runs it where the adapter has the configuration
(`Device.subgroupInt8`, read from the configurations the way `subgroupMatrix` is).

- **Gate**: exact against a CPU int32 reference on three shapes; ≥ 1.5× the scalar tile
  on 1024³ on the 5080 (the probe's 1.75× less the epilogue).
- **Measured** (RTX 5080, commit 64e1f78): **exact on every entry** of 256³, 512 × 1024 ×
  256, 1024³ and 2048³ (the raw i32 store), the dequantising store within 1.1e-7 of the
  scaled reference; 1024³ **0.038 ms, 3.59× the scalar tile** (0.136); 2048³ 0.273 ms,
  3.0× (0.821); the dequantising store costs 0.011–0.077 ms over the raw one. The small
  shapes are launch-bound at 1.03–1.27×. The gate is met with room; the ceiling of §0 was
  stall-inflated and is revised there.

### Step 2 — Activation quantisation · size S · **done 2026-09-21** (§6; its 10 % gate failed as written, one dispatch now)

Per tensor, dynamic: one reduction for the absolute maximum (the reduce kernels exist),
one elementwise pass to `round(x / scale)` packed four to a word. Two dispatches a layer;
under `compiled` the second fuses with whatever elementwise tree precedes it, and both
replay.

- **Gate**: the packed tensor round-trips within half a step of the scale; the two
  dispatches cost less than 10 % of the int8 layer they feed at batch 16.

### Step 3 — The convolution · size M · **done 2026-09-21** (§6; exact on nine shapes, 2.1–3.2× the f32 kernel)

The staged kernel's shape (`convForwardSubgroupSmall`): the padded band of eight input
channels in workgroup memory — but int8, so **four channels a word** and thirty-two
channels a K-step, the tap's block assembled as packed words; the tap-major weights
packed the same way once (hoisted under `compiled`). Bias and epilogue at the dequantising
store. Where the row and plane rules of the f32 kernels do not hold, the same staging
rules apply here.

- **Predict**: the fused ResNet-18 forward's convolutions 2.5 → ~1.5 ms of GPU at batch 16
  on the 5080.
- **Gate**: exact against a CPU int32 reference for every conv shape of ResNet-18; GPU
  time per layer below the scalar tile's on each.

### Step 4 — Accuracy · size S · the gate that decides whether it ships · **passed 2026-09-21** (§6: 92.70 → 92.75 %)

The bench's ResNet-18 with the exported weights, W8A8 through `compiled(model,
{ int8: true })`, against the f32 forward on the same held-out images.

- **Gate**: top-1 within **0.5 points** of f32 on the CIFAR test set; max |logit − torch|
  written into the table beside the time. A path that fails this is not routed to, by
  the same rule that keeps `check=True` honest.

### Step 5 — Routing and the one call · size S · **done 2026-09-21** (§6)

`Device.subgroupInt8` from the configurations; `compiled(model, { int8: true })` folds,
quantises the weights once, records the int8 forward. Off by default — the caller asks
for the accuracy trade — and refused by name on an adapter without the configuration.

- **Gate**: the `compare:ts` table gains an int8 row on the 5080 with its accuracy beside
  it; `capture:ts` holds the int8 replay to its eager int8 forward bit for bit.

### Step 6 — The measurement it was for · size S · **measured 2026-09-21** (§6)

The 5080, batch 1 and 16, int8 against f32 against ORT, same page.

- **Predict**: batch 16 4.3 → ~3.4 ms (ORT 3.6). If the round trip (§3) has been cut by
  then, ~2.0.

## 3. The lever this plan pointed at instead — taken, 2026-09-20

The captured forward on the 5080 was 4.32 ms for 2.9 ms of GPU; on metal-3 the same
recording is 4.1 ms for 3.9 of GPU. The plan called the difference "the submit-and-
readback round trip on Linux/Vulkan, 1.3–1.4 ms", and said to measure it before building
anything here. Measured (`roundtrip:probe`, `docs/INFER.md` ledger): **not a round trip —
the GPU process's polling schedule**, about two milliseconds between looks at its fences
unless something makes it look; a cheap wire round trip does, and a loop of them until
the readback resolves brings the wall to the GPU's time. With that in the library
(`Device.readbackKicks`, calibrated per adapter), the 5080's captured forward is **2.66 ms
at batch 16 against ORT's 3.63–3.89, and 0.99 at batch 1 against 3.88** — ahead at both,
with no int8 kernel. The premise of §0's arithmetic (4.3 → ~3.4 against ORT's 3.6) is
gone; the 2.9 ms of GPU that remains is now the whole wall, and int8's 3–3.6× on the GEMM
core (Step 1, measured under the profiler — the probe's 1.75× carried the same stall)
would apply to the convolutions' share of it.

## 4. Order and verdict

```
0 (done) ──► 1 ──► 2 ──► 3 ──► 4 ──► 5 ──► 6
```

Sizes S · S · M · S · S · S; one person, measured on the 5080 through cq at each gate.
**Verdict written before the work**: on the numbers of §0 this path is worth building only
after the round trip of §3 is measured, because on today's arithmetic it moves the
5080's batch-16 forward from behind ORT to level with it, at an accuracy cost, while the
round trip could move it ahead at none. If the round trip cannot be cut, int8 is the
remaining lever on that card and this plan is ready.

**The verdict's condition was met the same night** (§3): the stall was cut and the 5080
is ahead of ORT at both batches on f32 alone. Int8 is no longer what the card *needs*; it
is what would take 2.66 ms toward ~1.3 (the convolutions' ~2.5 ms of GPU at 3–3.6× on the
shapes that fill the card, less the quantise passes and the launch-bound small layers),
at the accuracy cost §1 names. Step 1 went ahead as planned — the GEMM in the tree, gated
on exactness and 1.5×, met at 3.59× — because it is the kernel any int8 path starts from
and its gate is a number on a card the tree can reach; Steps 2–6 wait for a caller who
wants the trade.

## 6. Ledger

- **2026-09-21, Steps 2 and 3 built and gated on the RTX 5080** (`kernel_bench --bench=cnvi8`).
  The activation is quantised on the GPU — `absMaxAtomic` (one dispatch: a workgroup tree,
  then `atomicMax` on the bit pattern) and `quantizeActivationInt8` (NCHW → four channels a
  word) — the weights on the host (`quantizeConvWeightInt8TapMajor`, `[tap][ci][co]`,
  a scale per output channel). The kernel, `convForwardInt8`: the small-plane staged
  kernel's shape with **the operand roles swapped** — the activation is the left operand
  (a pixel's 32 channels are eight consecutive words: a kernel row's three taps are copied
  from the staged band eight words a pixel, nothing repacked) and the weights the right,
  `i8 × i8 → i32` at 16 × 16 × 32, four subgroups × 32 output channels a workgroup, the
  reduction split to 128 workgroups with dequantised partials summed by `sumSplitsConv`.
  **Two gates, kept apart**: the quantiser to the host to ±1 on a near-tie only (the GPU's
  `1.0 / s` is not the correctly rounded quotient — one input at x·inv = −67.4999924 came
  out −68, and a day was nearly spent reading that as a wrong tap), and the convolution
  **exactly** to a CPU int32 reference built on the GPU's own bytes. Nine shapes, every
  output exact; ms, the minimum of five rounds:

  | shape | int8 kernel | f32 tiled (+ sum) | kernel alone | quantise passes (2 passes + zero) | with them |
  |---|---|---|---|---|---|
  | 128 → 128 at 16 × 16, b16 | 0.037 | 0.098 + 0.007 | **2.85×** | 0.015 | 2.03× |
  | 256 → 256 at 8 × 8, b16 | 0.037 + 0.002 | 0.099 + 0.007 | 2.86× | 0.016 | 1.91× |
  | 512 → 512 at 4 × 4, b16 | 0.034 + 0.002 | 0.100 + 0.007 | **3.15×** | 0.015 | 2.09× |
  | 64 → 64 at 32 × 32, b16 | 0.052 | 0.101 + 0.011 | 2.13× | 0.017 | 1.60× |
  | 512 → 512 at 4 × 4, b1 | 0.012 + 0.004 | 0.029 + 0.004 | 2.73× | 0.012 | 1.16× |
  | 256 → 256 at 8 × 8, b1 | 0.014 + 0.003 | 0.022 + 0.004 | 1.87× | 0.013 | 0.86× |
  | 128 → 128 at 16 × 16, b1 | 0.014 + 0.003 | 0.023 + 0.004 | 1.87× | 0.013 | 0.86× |

  (the quantise column is the three-dispatch first version; the one-dispatch `absMaxAtomic`
  replaced it the same night — its number follows.) Step 3's prediction — the deep layers
  2.5 → ~1.5 ms of GPU at batch 16 — reads, per layer, 0.10 → 0.037: **better than
  predicted on the kernel**, and Step 2's gate ("the two dispatches cost less than 10 % of
  the int8 layer") **failed** as written: three launch-bound dispatches are 40 % of a
  batch-16 layer and more than the whole kernel at batch 1. That is the quantiser's bill,
  and the reason a static scale (folded into the producing layer's epilogue) is the next
  thing to want — which needs calibration data, which needs Step 4's trained network.

- **2026-09-21, the one-dispatch quantiser, then Steps 5 and 6.** `absMaxAtomic` in
  place of two passes: the quantise passes 0.015–0.019 → **0.009–0.012 ms** a layer, and
  the int8 layer with them 2.29× / 2.32× / 1.71× the f32 kernel at batch 16 (128 → 128,
  512 → 512, 64 → 64) and 1.41× / 1.03× at batch 1 (512 → 512, 128 → 128). **Step 5**:
  `torch.compiled(model, { int8: true })` — `ConvND.quantizeInt8` once per layer (a
  readback, the packed words and scales kept), the intrinsic modules' forwards take
  `convNDFusedInt8` where the device has the configuration and the shape fits, refused by
  name where the device has not, `Compiled.int8Layers` says how many layers took it.
  **Step 6**, `compare:ts` on the RTX 5080 (Chrome 151 / Vulkan), the ResNet-18 (CIFAR)
  with 13 of its 20 convolutions on int8 (the stem, the three stride-2 layers and the
  three 1 × 1 downsamples stay f32):

  | batch | f32 fused + captured | **int8 + captured** | ORT Web 1.29.0 | max \|int8 − f32\| |
  |---|---|---|---|---|
  | 1 | 0.66 ms · 48 dispatches | 0.66 ms · 87 dispatches | 3.28–3.85 ms | 3.4e-4 (1.8e-3 of the logits' scale) |
  | 16 | 1.76 ms · 40 dispatches | **1.22 ms** · 79 dispatches | 3.65–4.06 ms | 3.4e-4 |

  **At batch 16 the int8 forward is 1.44× the f32 one and 3× ORT; at batch 1 it is a
  wash**, and the dispatch count says why: an int8 layer is a zeroing copy, the absolute
  maximum, the quantise, the convolution and (split) the slab sum — four dispatches where
  the f32 layer was one — and at batch 1 the replay is launch-bound, so the forty extra
  launches eat what the kernel saves. Step 6's prediction ("batch 16 4.3 → ~3.4; if the
  round trip has been cut by then, ~2.0") is beaten at 1.22 — the round trip was cut and
  the scalar path swept before this plan ran, and the kernel came in at 2.9× rather than
  1.75×. What would take batch 1 with it is a **static per-layer scale** (calibrated once,
  folded into the producing layer's epilogue: the copy, the maximum and the quantise gone,
  the int8 layer one dispatch again) — which needs calibration data, which is Step 4's
  trained network, which the bench does not have (its weights are torch's seed-0 draw,
  so its top-1 is meaningless and the accuracy gate cannot be run). **Step 4 is the open
  item**: a trained ResNet-18 (CIFAR-10, ~10 minutes on the 5080 through cq) exported
  with a labelled test slice for the browser, then the gate as written, then the static
  scale. Held on metal-3 (no int8 configuration, the path inert): `parity:ts` 229,
  `capture:ts` 18 / 18, `device:ts`; `compare:ts` unchanged.

- **2026-09-21, Step 4 — the accuracy gate, on a network trained for it.**
  `tests/browser/train_resnet18_cifar.py` trains `export_resnet18.py`'s ResNet-18 on
  CIFAR-10 on the 5080 through cq (twenty epochs, OneCycle, bf16 autocast, **95 s**, test
  top-1 **93.25 %**) and writes the trained weights, the ONNX twin, a probe and a
  labelled slice of 2,000 test images beside the seed-0 files; `compare:ts` runs the slice
  through the f32 fused forward and the int8 one and prints the gate:

  | forward, RTX 5080, 2,000 CIFAR-10 test images | top-1 |
  |---|---|
  | torch (the training script's own number on the slice) | 92.70 % |
  | borch.ts f32 fused | **92.70 %** |
  | borch.ts int8, 13 of 20 convolutions, per-tensor dynamic activation scale | **92.75 %** (+0.05 points) |

  **Gate "top-1 within 0.5 points of f32": passed**, with room — W8A8 with per-channel
  weights and a per-tensor activation scale costs this network nothing measurable, and
  the logits are 1.8e-3 of their scale from f32 (§6 above). The f32 forward's top-1 is
  torch's to the image, which is the f32 path's own gate met on a trained network for
  the first time. The int8 path is therefore **routed to** where a caller asks
  (`compiled(model, { int8: true })`), on the adapters that have the configuration.
  This run's times: batch 16 int8 1.39 ms against f32 1.76 (the earlier run 1.22 —
  run-to-run range 1.22–1.39), batch 1 0.92 against 0.65 — the batch-1 bill of §6 stands.
  What the plan now has that it did not: a trained network and a calibration set, which
  is what the **static scale** needs — the next step, not in this plan's six, and the one
  that would make batch 1 a gain: per-layer activation scales from the slice, folded
  into the producing layer's epilogue so an int8 layer is one dispatch again.

- **2026-09-21, the static scale — built and measured.** `calibrateInt8(model, pixels,
  batch, shape)`: the int8 layers run f32 over the calibration images accumulating
  `max|x|` and `max|out|` per layer by atomic maximum into two kept words, read back
  once. With the scales, an int8 layer's input is **the producer's int8 twin** — an int8
  layer with a static output scale packs its output four channels a word at its own store
  (`convForwardInt8` / `sumSplitsConv` with `quantOut`) and leaves it on the tensor
  (`Tensor.int8Twin`), so the layer that reads it next has no quantise pass at all — or,
  where the producer is f32 (the stem, a stride-2 layer), the layer's own static input
  scale and one quantise pass. `compare:ts` on the RTX 5080, the same ResNet-18:

  | batch | f32 fused + captured | int8 dynamic | **int8 static** | ORT Web |
  |---|---|---|---|---|
  | 1 | 0.65 ms · 48 dispatches | 0.82 · 87 | **0.77 · 52** | 3.32–3.65 |
  | 16 | 1.75 · 40 | 1.28 · 79 | **1.15 · 44** | 3.67–3.72 |

  Accuracy, the slice split — the scales from one half, the score on the other 1,000
  images: f32 92.40 %, int8 dynamic 92.60 % (+0.20), **int8 static 92.50 % (+0.10)**;
  both gates passed. The logits: static 3.0e-4 from f32 (dynamic 3.4e-4). **At batch 16
  the static int8 forward is 1.52× the f32 one and 3.2× ORT**; the forty extra
  dispatches of the dynamic path are gone (44 against the f32 forward's 40 — the four
  that remain are the quantise passes of the layers fed by f32 producers). **At batch 1
  it is still behind f32** (0.77 against 0.65) with four more dispatches than f32 and
  thirteen of its layers on a split reduction (a slab and a sum each); at that size the
  replay is launch-bound and the int8 kernel's GPU time is not what is paid. Two mistakes
  on the way: a packed word's shifts unparenthesised (WGSL refuses `|` and `<<` mixed),
  and thirty forwards of a hundred images without a scope, which filled the card and
  surfaced as "invalid buffer" in a bind group — the allocation's out-of-memory is
  reported late. Held on metal-3 (the path inert): `capture:ts` 18 / 18, `parity:ts`.

## 5. Risks, and the sentence that retires each

| risk | what would show it | retirement |
|---|---|---|
| a reading of the 8-bit load that is fast and wrong | an int8 kernel faster than the probe | every int8 kernel is held to a CPU int32 reference before it is timed — the probe's rule |
| per-tensor activation scales lose the small channels | Step 4's top-1 gate | per-channel activation scales are the next thing to try, and the gate says whether they are needed |
| the quantise passes eat the GEMM's gain | Step 2's 10 % gate | fuse the quantise into the producing layer's epilogue (the staged store already applies bias and relu) |
| Chrome ships the f32 configuration on Vulkan | `features_probe` | the f32 kernels take over by the existing routing and this path stays for adapters that have only int8 |
