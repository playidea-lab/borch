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

(filled as the steps land)

## 4. After this

`docs/INT8.md` Steps 2–6 follow when this plan is done: the int8 configuration's 3–3.6×
on the GEMM core applies to whatever GPU time this plan leaves on the 5080, on the
adapters that have the configuration, at the accuracy cost that plan names.
