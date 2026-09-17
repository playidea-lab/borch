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
| validation faults | 0 |

