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

### Step 1 — The int8 GEMM in the tree · size S

`kernels.ts` `matmulInt8`: the probe's kernel generalised — M, N, K in multiples of the
configuration, a 32 × 32 output tile a workgroup (four 16 × 16 results), K walked in
32s, `array<i32>` operands packed four to a word with offsets in components, the i32 result
dequantised at the store by a per-row weight scale and a per-tensor activation scale, plus
bias. `kernel_bench --bench=mm` gains an `int8` row where the adapter has the
configuration.

- **Gate**: exact against a CPU int32 reference on three shapes; ≥ 1.5× the scalar tile
  on 1024³ on the 5080 (the probe's 1.75× less the epilogue).

### Step 2 — Activation quantisation · size S

Per tensor, dynamic: one reduction for the absolute maximum (the reduce kernels exist),
one elementwise pass to `round(x / scale)` packed four to a word. Two dispatches a layer;
under `compiled` the second fuses with whatever elementwise tree precedes it, and both
replay.

- **Gate**: the packed tensor round-trips within half a step of the scale; the two
  dispatches cost less than 10 % of the int8 layer they feed at batch 16.

### Step 3 — The convolution · size M

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

### Step 4 — Accuracy · size S · the gate that decides whether it ships

The bench's ResNet-18 with the exported weights, W8A8 through `compiled(model,
{ int8: true })`, against the f32 forward on the same held-out images.

- **Gate**: top-1 within **0.5 points** of f32 on the CIFAR test set; max |logit − torch|
  written into the table beside the time. A path that fails this is not routed to, by
  the same rule that keeps `check=True` honest.

### Step 5 — Routing and the one call · size S

`Device.subgroupInt8` from the configurations; `compiled(model, { int8: true })` folds,
quantises the weights once, records the int8 forward. Off by default — the caller asks
for the accuracy trade — and refused by name on an adapter without the configuration.

- **Gate**: the `compare:ts` table gains an int8 row on the 5080 with its accuracy beside
  it; `capture:ts` holds the int8 replay to its eager int8 forward bit for bit.

### Step 6 — The measurement it was for · size S

The 5080, batch 1 and 16, int8 against f32 against ORT, same page.

- **Predict**: batch 16 4.3 → ~3.4 ms (ORT 3.6). If the round trip (§3) has been cut by
  then, ~2.0.

## 3. The lever this plan points at instead

The captured forward on the 5080 is 4.32 ms for 2.9 ms of GPU; on metal-3 the same
recording is 4.1 ms for 3.9 of GPU. The difference is the submit-and-readback round trip
on Linux/Vulkan — 1.3–1.4 ms against 0.3 — and it is paid once per forward regardless of
kernel. It is not in this plan because it is not int8; it is named here because it is
larger than what this plan can win, and cheaper: a measurement first (where the
milliseconds go between `submit` and the mapped readback — the queue, the staging map,
the fence), then whatever that says. `docs/INFER.md` carries it as its open item.

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

## 5. Risks, and the sentence that retires each

| risk | what would show it | retirement |
|---|---|---|
| a reading of the 8-bit load that is fast and wrong | an int8 kernel faster than the probe | every int8 kernel is held to a CPU int32 reference before it is timed — the probe's rule |
| per-tensor activation scales lose the small channels | Step 4's top-1 gate | per-channel activation scales are the next thing to try, and the gate says whether they are needed |
| the quantise passes eat the GEMM's gain | Step 2's 10 % gate | fuse the quantise into the producing layer's epilogue (the staged store already applies bias and relu) |
| Chrome ships the f32 configuration on Vulkan | `features_probe` | the f32 kernels take over by the existing routing and this path stays for adapters that have only int8 |
