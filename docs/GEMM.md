# The scalar GEMM — the tile every adapter without subgroup matrices runs on

> Plan, written 2026-09-21 before the work. Sibling plans: `docs/INFER.md` (inference,
> whose ledger carries the measurements that led here), `docs/INT8.md` (the int8 path,
> which waits behind this one — §4). Every step is a number on `kernel_bench mm`.

## 0. Why this, and why now

After the readback stall was cut and the scalar conv path's constants were swept on the
RTX 5080 (`docs/INFER.md`, 2026-09-20/21), that card's captured ResNet-18 forward at batch
16 is 1.8 ms of which 1.9 is GPU, and every convolution in it runs at 0.08–0.12 ms a
dispatch — **10–13 TFLOP/s on kernels whose own GEMM tile tops out at 15.8 on 1024³**, on a
card whose f32 peak is about 56. The kernels are at the tile's ceiling; the tile is at a
quarter of the card's.

The tile (`matmul` / `tiledGemm` in `kernels.ts`) is the one written first: 64 × 64 a
workgroup, 256 threads, a 4 × 4 micro-tile a thread, K in sixteens, scalar `f32` loads
with a `select` per element for the edge, two barriers a K-tile, one buffer. Its inner
loop does **sixteen FMAs for eight workgroup-memory reads** — the ratio that pins a
scalar GEMM to shared-memory bandwidth, and the ratio an 8 × 8 micro-tile doubles.

**Who runs it.** Subgroup matrices are a Chrome experiment on Metal and Vulkan. Chrome on
Windows (D3D12), Safari and Firefox have none, and run this tile for every matrix product
and every convolution; so does the 5080 through Vulkan, which is the adapter this plan is
measured on and the proxy for the rest. The subgroup adapters (metal-3) run it for the
shapes the subgroup kernels refuse. It is the path most users are on, and it has never
been tuned.

## 1. The ladder

Each step is a variant of one parametrised kernel (`matmulTiled(M, K, N, cfg)`) raced on
`kernel_bench --bench=mm --sweep=gemm` against the tile as it is, on 1024³, 2048³ and the
two GEMM shapes the deep ResNet layers are (512 × 4608 × 256, 256 × 2304 × 1024), on both
adapters. A step stays if it wins on the 5080 and does not lose on metal-3.

### Step 1 — The micro-tile · size S

8 × 8 a thread on a 128 × 128 tile (256 threads), sixty-four accumulators as sixteen
`vec4`. Workgroup memory: (128 + 128) × 16 × 4 = 16 KiB a buffer — the guaranteed floor
exactly, and `Device.workgroupStorage` says what more the device has.

- **Predict**: 1024³ 15.8 → ≥ 22 TFLOP/s on the 5080 (the shared-read ratio doubles; the
  register file is the unknown). metal-3's scalar tile (4.5) moves less — Apple's
  register file is smaller, and a spill would show as a loss.
- **Gate**: exact against the tile as it is (rel < 1e-5 — the accumulation order changes);
  ≥ 1.3× on 1024³ on the 5080; not slower on metal-3.

### Step 2 — Vector loads · size S

Global and workgroup loads as `vec4<f32>` — four floats a load, a quarter of the load
instructions — where the shape allows it (K and N whole fours; the edge shapes stay on
the scalar staging).

- **Predict**: +15–25 % over Step 1 on the 5080.
- **Gate**: exact; faster than Step 1 on both.

### Step 3 — Double buffering · size S

Two workgroup buffers; the next K-tile's global loads are issued into registers before
this tile's FMAs and stored after, so one barrier a K-tile and the loads overlap the
arithmetic. Workgroup memory doubles (32 KiB at 128 × 128 × 16 — metal-3's limit exactly;
K in eights where it does not fit).

- **Predict**: +10–20 % over Step 2 where memory allows.
- **Gate**: exact; faster than Step 2 on the 5080; the K-in-eights fallback not slower
  than Step 2 on metal-3.

### Step 4 — The tree · size M

The winning configuration into `matmul` (the plain product) and `tiledGemm` (the
convolutions' implicit GEMM, whose B side is a gather and stays scalar — the A side and
the inner loop take the new shape), behind the existing routing: the new tile where the
shape divides it and the device's workgroup storage holds it, the old tile everywhere
else. `tileShape` / `tileDepth` / `scalarMatmulSplit` re-swept for the new tile.

- **Predict**: the 5080's captured forward at batch 16 1.8 → ~1.3 ms; the training step
  11.7 → ~9 (its 8.2 ms of GPU is all this tile's variants); metal-3 unchanged where the
  subgroup kernels run, faster on the shapes they refuse.
- **Gate**: the parity suite; `capture:ts` 18 / 18 on both adapters (replay = eager bit
  for bit — the new kernel is deterministic, so this holds); `compare:ts` on both.

## 2. Risks, and the sentence that retires each

| risk | what would show it | retirement |
|---|---|---|
| sixty-four accumulators spill on Apple | Step 1 slower on metal-3 | the 8 × 4 micro-tile (32 accumulators) is in the sweep; the routing takes the winner per adapter, which is what `Device.workgroupStorage` and the subgroup flags already do |
| the 128 × 128 tile empties the grid on the small deep layers | Step 4's conv shapes slower than the swept 64 × 64 | `tileShape` keeps 64 × 64 in its candidates and the split policy is re-swept; the bench's `--sweep=all` is the same tool |
| `vec4` component stores into workgroup memory serialise | Step 2 slower than Step 1 | A staged scalar, B staged `vec4` — the sweep has both |
| D3D12 behaves unlike Vulkan | nothing here can show it — no D3D12 adapter is measured | the risk is named; the first Windows measurement is the thing to get, and it is a person's |

## 3. Ledger

- **2026-09-21, the sweep on metal-3** (`kernel_bench mm --sweep=gemm`, the minimum of
  five rounds, ms; the slab sum counted where split). Every configuration exact against
  the tile as it is.

  | shape | scalar tile | 64 × 64 r4×4 vec4 | 128 × 128 r8×8 vec4 | 128 × 64 r8×4 vec4 | subgroup (for the eye) |
  |---|---|---|---|---|---|
  | 1024³ | 0.492 | **0.368** | 0.354 + 0.054 (split 4) | 0.342 + 0.031 (split 2) | 0.229 |
  | 2048³ | 3.551 | 2.637 | 2.716 | **2.432** | 1.585 |
  | 512 × 4608 × 256 | 0.280 + 0.009 | 0.209 + 0.009 | 0.211 + 0.024 | **0.197 + 0.009** | 0.132 + 0.008 |
  | 256 × 2304 × 1024 | 0.284 + 0.015 | 0.210 + 0.017 | 0.209 + 0.023 | **0.196 + 0.015** | 0.134 + 0.015 |

  On Apple the register file decides: the 8 × 8 micro-tile is no faster than 4 × 4 with
  the same `vec4` staging (Step 1's prediction, "moves less", held — it moves not at
  all), `vec4` staging alone is **1.34×** on the tile as it is, and the 8 × 4 micro-tile
  on a 128 × 64 tile is the best of the sweep at **1.46×** (2048³ 3.55 → 2.43). Double
  buffering loses everywhere on metal-3 (+3–10 %); the deeper K-tile (32) is a wash;
  the scalar-load 8 × 8 is the slowest of the new ones, so the staging loads matter more
  than the micro-tile here. The 5080's numbers decide the step; metal-3's say the
  routing will be per adapter.

- **2026-09-21, Step 4 on metal-3, ahead of the 5080's verdict.** (a) The plain product:
  `Device.gemmConfigs` per adapter (Apple: 128 × 64 r8×4 `vec4`, then 64 × 64 r4×4 `vec4`;
  others empty until measured), `matmul` takes the first the shape divides, untransposed
  only. `device:ts`: three shapes against the old tile with subgroup matrices forced off,
  rel 0 (the same sequential K walk — bit for bit), the new pipeline taken where a
  configuration fits and the old tile where none does. (b) The convolutions' implicit
  GEMM: `tiledGemm` generalised to a `RM × RN` micro-tile (`ConvTile`; every candidate is
  256 threads) and `tileShape` given the adapter's preferred tile at equal padding
  (`setConvTilesPreferred`, the first GEMM configuration). Its loads stay scalar — the B
  side is a gather — and on metal-3 that is the whole story: `kernel_bench fwd
  --sweep=tiles`, ms, batch 16 / batch 1, the slab sum counted:

  | tile | 512 → 512 at 4 × 4 | 256 → 256 at 8 × 8 | 128 → 128 at 16 × 16 | 512 → 512 at 4 × 4, batch 1 |
  |---|---|---|---|---|
  | 64 × 64 r4×4 (as it was) | 0.318 + 0.024 | 0.315 + 0.024 | 0.317 + 0.031 | **0.095 + 0.005** |
  | 128 × 64 r8×4 | **0.299 + 0.026** | **0.295 + 0.026** | **0.298 + 0.031** | 0.100 + 0.005 |
  | 128 × 128 r8×8 | 0.367 | 0.347 | 0.356 | 0.184 |

  A wash on Apple (+6 % at batch 16, −5 % at batch 1): with scalar gathers the micro-tile
  is not the bottleneck there, and the 8 × 8 loses as it did on the plain product. Kept —
  the gate is "not slower", and the 5080, where the plain product's Step 1 prediction
  stands untested, decides whether the conv's A side gets `vec4` loads (a second binding
  of the weights as `array<vec4<f32>>`) next. Held on metal-3: `parity:ts`, `capture:ts`
  18 / 18, `compare:ts` unchanged (its convolutions are on the subgroup kernels).

- **2026-09-21, the sweep on the RTX 5080 — Steps 1–3 decided.** The worker came back
  (the zombie run of 2026-07-28 had held its GPU slot for a day). Same bench, same
  shapes, the minimum of five rounds, ms:

  | shape | scalar tile | 64 × 64 r4×4 vec4 | 128 × 128 r8×8 vec4 | 128 × 64 r8×4 vec4 | r8×4 dbuf |
  |---|---|---|---|---|---|
  | 1024³ | 0.136 | 0.113 | 0.109 + 0.011 (split 4) · 0.129 unsplit | **0.102** (21.1 TFLOP/s) | 0.151 + 0.011 |
  | 2048³ | 0.812 | 0.693 | 0.807 | **0.671** (25.6) | 1.003 |
  | 512 × 4608 × 256 | 0.080 + 0.004 | 0.065 + 0.004 | 0.067 + 0.007 | **0.058 + 0.005** | 0.087 + 0.005 |
  | 256 × 2304 × 1024 | 0.080 + 0.005 | 0.065 + 0.005 | 0.067 + 0.007 | **0.058 + 0.005** | 0.087 + 0.005 |

  **Step 1's prediction was wrong on both adapters**: the 8 × 8 micro-tile does not reach
  22 TFLOP/s on the 5080 (19.7 split, 16.7 unsplit) and loses to 8 × 4, as it did on
  Apple — sixty-four accumulators cost more occupancy than the shared-read ratio buys,
  on NVIDIA too. **Step 2 is the gain**: `vec4` staging alone is 1.20× on the 5080 and
  1.34× on metal-3, and with the 8 × 4 micro-tile 1.33× / 1.46×. **Step 3 is retired on
  both**: double buffering loses 20–50 % on the 5080 and 3–10 % on metal-3 — the
  register cost of holding the next tile is paid by every thread, and the barrier it
  saves was not the wait. The gate "≥ 1.3× on 1024³ on the 5080" is met by the 8 × 4
  tile (1.33×), not by the one predicted; the deeper K-tile (32) helps the 8 × 8 a
  little (0.105) and is not in the list. So `gemmConfigsFor` returns the same two
  configurations for every vendor: **128 × 64 r8×4 `vec4`, then 64 × 64 r4×4 `vec4`**;
  the old tile keeps the shapes neither divides.

- **2026-09-21, Step 4 on the 5080 — the convolution's verdict.** `kernel_bench fwd
  --sweep=tiles`, the deep layers, ms, the slab sum counted:

  | tile | 512 → 512 at 4 × 4, b16 | 256 → 256 at 8 × 8, b16 | 128 → 128 at 16 × 16, b16 | 512 → 512, b1 | 256 → 256, b1 |
  |---|---|---|---|---|---|
  | 64 × 64 r4×4 (as it was) | **0.100 + 0.007** | **0.099 + 0.007** | **0.099 + 0.007** | **0.029 + 0.004** | **0.023 + 0.004** |
  | 128 × 64 r8×4 | 0.114 + 0.008 | 0.111 + 0.008 | 0.111 + 0.008 | 0.034 + 0.004 | 0.031 + 0.004 |
  | 128 × 128 r8×8 | 0.152 | 0.153 | 0.151 | 0.054 | 0.048 |

  **The micro-tile that wins the plain product loses the convolution, 10–40 %, on the
  card the plan was for.** The reason is in the kernel: the implicit GEMM's B side is a
  gather — one bounds-checked scalar load per element with the (batch, position,
  channel, tap) digits carried — and that is what a K-tile waits on; the FMAs behind it
  are not the bottleneck, so widening the micro-tile buys nothing and the 128-row tile
  costs occupancy on a 512 × 256 product. The `vec4` staging that made the plain product
  1.33× has no purchase on a gather. So `setConvTilesPreferred([])`: the convolutions
  keep their tiles, the generalised `tiledGemm` stays for a kernel that can use it, and
  **Step 4's conv prediction (forward 1.8 → ~1.3 on the 5080) is withdrawn** — the plan's
  gain is the plain product's (every `Linear`, every attention `bmm` off the subgroup
  path, on every adapter without subgroup matrices), not ResNet's convolutions on the
  5080. What would move the convolutions on the scalar path is a kernel whose B side is
  not a gather: the input padded once (`padForGradWeight` exists), a band of rows staged
  as the subgroup kernels stage it, `vec4` loads along the output row — the
  `convForwardSubgroupSmall` shape without the subgroup multiply. Size M, its own gate;
  not taken up here, and named in §4 beside int8 for the decision of what comes next.

- **2026-09-21, the third adapter — Direct3D 12, and a third answer.** The Windows worker
  (an RTX 5050 Laptop, Chrome on Windows 11, `nvidia / blackwell` through D3D12; the
  app-control policy that had blocked it on the 18th was gone) ran the same sweep:

  | shape | scalar tile | 64 × 64 r4×4 vec4 | 128 × 64 r8×4 vec4 | **128 × 128 r8×8 vec4** | r8×8 dbuf |
  |---|---|---|---|---|---|
  | 1024³ | 0.720 | 0.636 | 0.605 + 0.027 | **0.430 + 0.027** (split 4) · 0.539 unsplit | 0.501 + 0.027 |
  | 2048³ | 6.353 | 5.558 | 4.875 | **3.596** (1.77×, 4.8 TFLOP/s) | 3.914 |
  | 512 × 4608 × 256 | 0.425 + 0.004 | 0.345 + 0.005 | 0.343 + 0.008 | **0.248 + 0.014** | 0.292 + 0.014 |
  | 256 × 2304 × 1024 | 0.447 + 0.008 | 0.359 + 0.008 | 0.338 + 0.009 | **0.251 + 0.014** | 0.287 + 0.014 |

  **Under D3D12 the 8 × 8 micro-tile wins, 1.6–1.8×** — the configuration that lost on
  Metal and on Vulkan, and Step 1's prediction as written. The compiler under D3D12 keeps
  sixty-four accumulators in registers where the other two spill or lose occupancy; the
  plain `vec4` staging alone is only 1.13× here (1.34× / 1.20× elsewhere), so on this API
  the micro-tile is the lever and the loads are not. Double buffering loses on all three
  (+9 % here). `gemmConfigsFor` therefore returns the 8 × 8 tile first on Windows —
  the API is not in `GPUAdapterInfo`, and Chrome on Windows is D3D12 — then the 8 × 4,
  then the 64 × 64 (d2e4e17). §2's risk ("D3D12 behaves unlike Vulkan") was real and is
  now measured rather than named. **The convolutions, same sweep, same card**: the
  micro-tiles do not help the gather kernel here either (512 → 512 at 4 × 4, batch 16:
  64 × 64 0.505, 128 × 64 r8×4 0.497, 128 × 128 r8×8 0.645) — the conv verdict holds on
  the third API. Two things the sweep says about this card that are not the GEMM's: the
  split policy over-splits it (512 → 512 at 4 × 4, batch 16: the policy's 32 pieces
  0.493 + 0.014 against 8 pieces 0.465 + 0.005; 64 → 64 at 32 × 32 the policy's 4 against
  unsplit 0.505 — a laptop card with a quarter of the 5080's SMs wants a quarter of the
  workgroups, and WebGPU does not say how many SMs there are); and the direct kernel
  prefers a wider output-channel block (64 → 64 at 32 × 32: 4 × 16 with a slice of 14
  0.289 against the default's 0.343). Both are 5–15 % and both would want a per-card
  number the API does not give — a calibration at `create`, like the kicks. Named, not
  taken.

- **2026-09-21, the scalar staged convolution — built, measured, routed** (§4's first
  candidate; c1b3644, f3455a3). `convForwardStaged`: a block of input channels' band of
  padded rows staged from the unpadded input in row segments (the border is zero at the
  staging, no pad pass), every tap of the block's weights staged as `vec4` rows of output
  channels (`tapMajorWeightsCo`, `[tap][ci][co]`), then the tiled GEMM's inner loop — a
  thread's `RM × RN` FMAs from one `vec4` of weights and `RN` consecutive floats of the
  band. The gather is gone, and with it the reason the micro-tiles lost the convolution:
  `kernel_bench fwd --sweep=staged`, ms, the slab sum counted, the repack (hoisted under
  `compiled`) not:

  | shape, batch 16 | tiled (as it was) | staged 64 × 64 r4×4 kb8 | **staged 128 × 64 r8×4 kb4** | direct |
  |---|---|---|---|---|
  | 512 → 512 at 4 × 4 · metal-3 | 0.318 + 0.024 | 0.302 + 0.133* | **0.264** | 0.796 |
  | 512 → 512 at 4 × 4 · RTX 5080 | 0.100 + 0.007 | 0.099 + 0.040* | **0.080 + 0.007** | 0.238 |
  | 256 → 256 at 8 × 8 · 5080 | 0.099 + 0.007 | 0.097 + 0.016 | **0.078 + 0.016** | 0.169 |
  | 128 → 128 at 16 × 16 · 5080 | 0.098 + 0.007 | 0.097 | **0.077 + 0.012** | 0.093 |
  | 64 → 64 at 32 × 32 · 5080 | 0.101 + 0.011 | 0.099 | 0.144 | **0.084** |
  | 512 → 512 at 4 × 4, batch 1 · 5080 | 0.029 + 0.004 | 0.026 | **0.022** | 0.172 |
  | 256 → 256 at 8 × 8, batch 1 · 5080 | 0.022 + 0.004 | **0.011** | 0.014 | 0.141 |

  (* the "+" on those two rows is the repack, counted there because the row's split was
  one and nothing else was extra.) Exact against the direct kernel on every shape. The
  8 × 4 micro-tile on a 128 × 64 tile with a block of four channels wins the deep layers
  on both adapters, **1.2–1.25× the tiled kernel at batch 16 and 2× at batch 1**, and
  the 64-channel layer stays with the direct kernel. Routed (`STAGED_TILES`, the first
  that fits; `convForwardRun` takes it where no subgroup kernel took the shape, before
  the tiled GEMM). Held: `parity:ts` 229, `capture:ts` 18 / 18 on both adapters.
  `compare:ts` on the 5080: the captured forward **0.66 → 0.54 ms at batch 1**; at
  batch 16 **1.74 → 1.74** — the six deep dispatches went 0.73 → 0.51 ms of GPU, and the
  wall did not follow, because the replay's forty dispatches sit at about 1.6 ms of GPU
  now spread over layers that are each at their bench optimum (the 64- and 128-channel
  direct layers 0.19 / 0.10 a dispatch) and a floor of launches under it. The eager
  training step did not move either (10.6 ms): eager repacks the weights each step
  (`tmwc`, 0.15 ms for a 512 × 512 × 9 weight, six of them) and that eats what the kernel
  saves — the same bill the subgroup path pays in eager, and the reason `compiled`
  hoists it. On metal-3 the stride-2 layers (which the subgroup kernels refuse) moved to
  it: the unfused eager forward at batch 16 7.2 → 4.5 ms. **D3D12 is unmeasured** — the
  Windows worker is in a lecture until its owner says otherwise.

## 4. After this

Two candidates for the 5080's convolutions, which this plan did not move:
- **A scalar staged conv** — the padded plane, a band of rows in workgroup memory, `vec4`
  loads along the row, the FMAs of `tiledGemm`'s inner loop: the gather gone. Serves every
  adapter without subgroup matrices (D3D12, Safari, Firefox). Size M. Predict, from the
  plain product's 1.33×: the deep layers 0.10 → ~0.07 ms a dispatch.
- **`docs/INT8.md` Steps 2–6** — the int8 configuration's 3–3.6× on the GEMM core, on the
  adapters that have it (today: Chrome on NVIDIA over Vulkan), at the accuracy cost that
  plan names. Its Step 3 is the same staged-band kernel with a subgroup multiply in the
  middle, so the two share their staging.
