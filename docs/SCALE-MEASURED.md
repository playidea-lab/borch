# SCALE — measured

> The ledger `docs/SCALE.md` §7 names: every number a gate in the plan produced, with the adapter, the day and the commit. Written by `tests/browser/ceiling.py --record` and by hand from the other gates. A claim about scale that is not here is not made.
### apple / metal-3 — 2026-09-18 · 20c4d9d

| what | measured |
|---|---|
| features | bgra8unorm-storage chromium-experimental-subgroup-matrix clip-distances core-features-and-limits depth-clip-control depth32float-stencil8 dual-source-blending float32-blendable float32-filterable indirect-first-instance primitive-index rg11b10ufloat-renderable shader-f16 subgroup-size-control subgroups texture-component-swizzle texture-compression-astc texture-compression-astc-sliced-3d texture-compression-bc texture-compression-bc-sliced-3d texture-compression-etc2 texture-compression-unaligned texture-formats-tier1 texture-formats-tier2 timestamp-query |
| shader-f16 · subgroups | yes · yes |
| maxBufferSize · maxStorageBufferBindingSize | 4096 MiB · 4096 MiB |
| storage buffers/stage · workgroup storage | 10 · 32 KiB |
| largest single buffer holding its marker | 4096 MiB |
| largest total holding every marker | 32.00 GiB (chunks of 256 MiB) |
| storage quota · used | 10.0 GiB · 0.0 MB |
| hub load imagenet-efficientnet-b0 (21 MB) | 3.4 s · gpu +149 MB (pool +0) · host peak +43 MB = 2.00× file (UA-specific (all bytes)) · faults 0 |
| hub load imagenet-vit-base-patch16-224 (346 MB) | 13.7 s · gpu +1193 MB (pool +0) · host peak +693 MB = 2.00× file (UA-specific (all bytes)) · faults 0 |
| validation faults | 0 |

### nvidia / blackwell — 2026-09-18 · 0756b35 (RTX 5080, Chrome on Linux/Vulkan, cq worker)

> **`shader-f16` is absent here — and that is the point.** The survey put shader-f16 on
> ~84 % of NVIDIA, but that is the D3D12 path; Chrome-on-Linux over Vulkan-Dawn does not
> expose it on this RTX 5080. So this adapter is the measured witness that ADR-003's f32
> fallback is not hypothetical: a recent NVIDIA card, through a mainstream browser, offers
> no half precision. Apple metal-3 above has it; this does not. The f16 path must never be
> the default. Also note `maxStorageBufferBindingSize` is 2048 MiB here, half of Apple's
> 4096 — a window sized to Apple's tier would fail to bind on this card.

| what | measured |
|---|---|
| features | bgra8unorm-storage chromium-experimental-multi-draw-indirect chromium-experimental-subgroup-matrix chromium-experimental-timestamp-query-inside-passes clip-distances core-features-and-limits depth-clip-control depth32float-stencil8 dual-source-blending float32-blendable float32-filterable indirect-first-instance primitive-index rg11b10ufloat-renderable subgroup-size-control subgroups texture-component-swizzle texture-compression-bc texture-compression-bc-sliced-3d texture-formats-tier1 texture-formats-tier2 timestamp-query |
| shader-f16 · subgroups | **no** · yes |
| maxBufferSize · maxStorageBufferBindingSize | 4096 MiB · 2048 MiB |
| storage buffers/stage · workgroup storage | 16 · 48 KiB |
| largest single buffer holding its marker | 4096 MiB |
| largest total holding every marker | 8.00 GiB (chunks of 256 MiB, stopped at the 8 GiB cap) |
| storage quota · used | 10.0 GiB · 0.0 MB |
| hub load imagenet-efficientnet-b0 (21 MB) | 8.4 s · gpu +145 MB (pool +0) · host peak +43 MB = 2.00× file · faults 0 |
| hub load imagenet-vit-base-patch16-224 (346 MB) | 21.5 s · gpu +1193 MB (pool +0) · host peak +693 MB = 2.00× file · faults 0 |
| checkpoint (Step 6) | recomputed backward bit-identical to taped (max \|Δ\| 0.00e0), held 148.6→100.6 KB (68 %), faults 0 |
| subgroup matrix configurations (2026-09-20 · ef40cb2, `features_probe`) | **int8 only**: u8/i8 → u32/i32 at 16×16×32 and 16×8×32 — no f32 8×8×8, so `Device.subgroupMatrix` is off and every conv runs on the scalar kernels (`docs/INFER.md` Step 5) |
| captured training step, ResNet-18 CIFAR batch 16 (2026-09-20 · ef40cb2, `capture:ts`) | eager 13.4 → replay 11.6 ms; 398 intermediates 415.3 → 161.3 MB in 4 arenas; 18/18 bit for bit |
| the readback wait (2026-09-20 · 65b9664, `roundtrip:probe`) | **every wait whose fence is not signalled at the GPU process's first look lands at 2.1–2.7 ms**: 40 tiny dispatches 0.08 ms of GPU → 2.6 of wall under `onSubmittedWorkDone` or `mapAsync`; a 1.57 ms kernel → 2.5. Kicked by `pushErrorScope`/`popErrorScope` round trips until the map resolves: 0.20 and 1.66. `Device.readbackKicks` calibrates **true** here (plain 2.15–2.28 for 0.12 of GPU; metal-3 false, 0.3–1.2 over the GPU). With it: captured ResNet-18 forward 2.66 ms at batch 16 (was 4.32), 0.99 at batch 1 (was 3.00); training 12.3 / 17.4 / 27.6 ms at batch 16 / 32 / 64 |
| int8 subgroup GEMM, 1024³ i8 → i32 (2026-09-20 · cb1e8a2, `int8-sg:probe`) | **0.078 ms · 27.7 TOPS**, exact against a CPU reference with operands in `array<i32>`, offsets and strides in components, `left<i8, K, M>`; the f32 scalar tile on the same GEMM 0.136 ms · 15.8 TFLOP/s — **1.75×** (`docs/INT8.md`) |
| the scalar conv path swept on the card (2026-09-21 · 38b53ba, `kernel_bench fwd --sweep=all`) | split policy 128 / 256 → **1024 / 128** (512 → 512 at 4 × 4, batch 16: split 4 0.179 → 32 0.100 ms); direct weight slice at 48 KiB (128 → 128 at 16 × 16: 0.146 → 0.093); direct grids under 64 workgroups to the split GEMM (batch 1 128 → 128: 0.042 → 0.027). Captured forward **0.65 / 1.74 ms** at batch 1 / 16 (ORT 3.51 / 3.62); training 10.6 / 14.7 / 23.2 |
| the int8 convolution and forward (2026-09-21 · 6fd3001, `kernel_bench --bench=cnvi8`, `compare:ts`) | `convForwardInt8` **exact on nine shapes** against a CPU int32 reference on the GPU's own bytes; 128 → 128 at 16 × 16, batch 16, 0.037 ms against the f32 kernel's 0.105 (2.85×), 512 → 512 at 4 × 4 0.034 against 0.107 (3.15×); the quantiser 0.009–0.012 ms a layer. ResNet-18 forward with 13 of 20 convolutions int8: **1.22 ms at batch 16** (f32 1.76, ORT 3.65–4.06), 0.66 at batch 1 (f32 0.66), max \|int8 − f32\| 3.4e-4; **top-1 on 2,000 CIFAR-10 test images, a network trained on the card (93.25 %): f32 92.70 %, int8 92.75 %** — the half-point gate passed. **Static scales** (b62803f): batch 16 **1.15 ms** · 44 dispatches, batch 1 0.77 · 52; top-1 92.50 % on 1,000 held-out images with scales from the other 1,000 |
| int8 subgroup GEMM in the tree (2026-09-20 · 64e1f78, `kernel_bench --bench=mmi8`) | `matmulInt8` **exact on every entry** of 256³ / 512 × 1024 × 256 / 1024³ / 2048³ against a CPU int32 reference; dequantising store within 1.1e-7. 1024³ **0.038 ms · 56.6 TOPS · 3.59×** the f32 scalar tile (0.136); 2048³ 0.273 ms · 62.9 TOPS · 3.0× (0.821). The probe row above (0.078 ms, 1.75×) was a wall clock carrying the polling stall |
| validation faults | 0 |

### nvidia / blackwell — 2026-09-21 · 825138a (RTX 5050 Laptop, Chrome on Windows 11 / **Direct3D 12**, cq worker DESKTOP-KETRGAE)

> The same vendor name as the row above, through the other API. **`shader-f16` is here**
> (it is not under Vulkan on the 5080), subgroups are, subgroup matrices are not (no
> configuration at all — neither f32 nor int8), so every convolution and every product
> runs on the scalar kernels; and the readback stall of the Vulkan entry is here too —
> the kick calibration turned kicks on (plain 4.43 ms for 0.88 of GPU). The
> app-control policy that had blocked the worker on the 18th was gone on the 21st.

| what | measured |
|---|---|
| features | 24 — `shader-f16`, `subgroups` (32 / 32), `timestamp-query`, `chromium-experimental-multi-draw-indirect`, `texture-compression-unaligned`, … · **no** `chromium-experimental-subgroup-matrix` |
| first run (`first_run`) | first visit 3059 ms in the page / 4.2 s · revisit 2316 / 3.0 · warm 2279 / 2.7 — adapter request 543 ms cold, 115 warm |
| re-tiled GEMM (`kernel_bench mm --sweep=gemm`) | **128 × 128 r8×8 vec4 wins here**: 2048³ 6.35 → 3.60 ms (1.77×, 4.8 TFLOP/s), 1024³ 0.720 → 0.457, the deep ResNet shapes 0.43 → 0.26; the 8 × 4 that wins on Metal and Vulkan 1.30× here; double buffering +9 % |
| the convolutions (`fwd --sweep=all`) | the micro-tiles do not help the gather kernel (512 → 512 at 4 × 4, b16: 64 × 64 0.505, r8×4 0.497, r8×8 0.645); the split policy over-splits this card (32 pieces 0.493 + 0.014 against 8 pieces 0.465 + 0.005); the direct kernel prefers 4 × 16 (64 → 64 at 32 × 32: 0.289 against 0.343) |
| captured training step, ResNet-18 CIFAR batch 16 (`capture:ts`) | eager 32.1 → replay 27.5 ms; 400 intermediates 600.3 → 171.8 MB in 4 arenas; **18 / 18 bit for bit** |
| readback kicks | **on** — plain 4.43 ms for 0.88 of GPU, kicked 1.51 |
| training step, ResNet-18 CIFAR (`compare:ts`) | 32.3 / 53.7 / 98.1 ms at batch 16 / 32 / 64 · TF.js 174.8 / 334.9 / 659.2 |
| inference, ResNet-18 (`compare:ts`) | fused + captured **1.69 ms at batch 1, 7.29 at batch 16** · ORT Web 5.67 / 15.69 · eager (scoped) 5.81 / 8.65 — the first, unscoped, eager run read 152 ms at batch 16 for 22 ms of GPU: a buffer is milliseconds under D3D12 |
| validation faults | 0 |

