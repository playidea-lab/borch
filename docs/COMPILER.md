# The compiler — what the recording is today, and the eight steps that make it a compiler

> Plan, written 2026-09-20 from `device.ts` (`Capture`, `Recorded`), `fuse.ts`,
> `borch_webgpu/_ops.py` (`capture`, `compiled`) and the measurements in `docs/BOOK.md`
> ("A step captured, replayed, and fused"). The inference twin is `docs/INFER.md`, the
> memory twin `docs/SCALE.md`; this one is about the thing both of them lean on. Every
> step has a gate that is a number and a prediction written before the work, so the
> ledger can say which step was wrong. Nothing here touches the numpy core.

## 0. What exists, said precisely

borch has **a trace compiler**: it does not read a program, it records one run of it.

| layer | what it is | where | measured |
|---|---|---|---|
| **recording** | every `run`/copy under `beginCapture` as `Recorded` — pipeline, bind group, grid, the buffers in binding order, and for elementwise/reduce kernels a *recipe* (`Elementwise`/`Reduce`: the WGSL expression, inputs with strides, which binding is the output) | `device.ts:293-330`, `:855-900` | — |
| **replay** | the list re-encoded: no Python, no JS graph, no allocation, no bind-group creation; every buffer the step made is pinned for the capture's life; CPU-side variation moved to the device (Adam's counter, the RNG seed, the scheduler's four floats via `syncHyper`) | `Capture.replay`, `replayRecorded` | ResNet-18 CIFAR step 3.4 → 2.1 ms (38 %), GPT-2blk 34 %, U-Net 28 %, ViT-224 5 % (GPU-bound) — twelve U-Net steps bit-identical to eager |
| **fusion** | elementwise trees → one kernel (same IEEE ops, same order); reductions absorb their producers (serial walk ≤ 64); intermediates provably unread (`internal`, or `detached` and not `held`) never written; a tree stops at the per-stage storage-buffer budget (10 on Metal) | `fuse.ts` | GELU net 139 → 76 dispatches, 0.85 → 0.72 ms; inference 38 → 15, 2.9 → 0.43 ms; U-Net 3 of 223 (already fused by hand) |
| **contract** | `compiled(check=True)`: live-ins snapshotted, replay vs eager rerun compared buffer by buffer — bit for bit plain, `tol`/`state_tol` fused; a difference raises and names the buffer | `_ops.py` `compiled._verify` | the three checks in `capture:py` / `fuse:py` |
| **shape keying** | one recording per tuple of input shapes and plain-value arguments; a new shape records again | `compiled._key` | the last short batch = a second recording |

And its limits, each measured or hit:

- **Shapes are frozen.** A sequence that grows (an NLP batch of varying length, a decode
  loop) records once per length; nothing pads or buckets for the caller (ViT's tokens
  were aligned by hand in `bimm`).
- **Only elementwise and reduce kernels carry a recipe.** A matmul, a conv, a softmax, a
  gather, a BatchNorm kernel has none, so `liveIns()` takes it to read and write every
  buffer it binds ("more than needed, never less") and the fusion pass cannot see through
  it. Every analysis downstream is conservative by exactly this much.
- **No memory plan.** Every intermediate of the step is resident for the capture's life,
  each in its own buffer: the ResNet-18 CIFAR step at batch 16 holds 233 MB and the pool
  517 MB beside it; at batch 64, 700 MB and 1,755 MB. The eager step returns buffers to
  the pool as scopes close; a capture cannot, because the recorded bind groups point at
  them. Liveness would let intermediates share — the recording knows the order.
- **In-place is a copy.** Every in-place op records as a kernel into a fresh buffer and a
  copy back (AdamW's weight decay, `x.copy_`); a compiler that knew the temporary dies at
  the copy would write the target directly. How many copies a step carries has not been
  counted.
- **The optimizer is per parameter unless a hand-written arena catches it.** The Adam/SGD
  arena (one dispatch over a gathered slab, 1.2× on the step) is gated to one parameter
  group, plain Adam, no amsgrad/maximize — a hand-made instance of the general thing a
  compiler does: fuse independent dispatches of the same shape side by side.
- **Kernel choice is a rule, not a measurement.** `subgroupMatmulTile`, the split targets
  (64/512), `SG_AVOID_SPLIT_MAX_K` were swept by hand on one adapter; the rule that
  regressed long-K dW 3× before its guard is the evidence a rule misses shapes.
- **The compiler is Python-only as an API.** `torch.compiled` lives in `_ops.py`;
  borch.ts exposes `Device.beginCapture/endCapture` and `Capture.fuse` but no
  `compiled` sugar — a JS user does the bookkeeping by hand or not at all.
- **Capture and the streaming window refuse each other** (`docs/SCALE.md` decision 6):
  a replay would read a weight the window had evicted.
- **The matmul epilogue is closed** — measured (0.72 → 1.69 ms) and removed, not a gap.

## 1. What "a compiler" would add, and which of it is worth building here

The classic list — IR, shape inference, algebraic rewrites, CSE, layout planning, memory
planning, kernel selection, scheduling — filtered by *what the measurements say the
browser step loses to*:

| capability | what it would recover here | evidence it is needed | verdict |
|---|---|---|---|
| exact read/write sets per dispatch | every analysis below; `liveIns` no longer over-approximates | the "no recipe → reads everything" rule | **build first (Step 0)** |
| memory planning | the pool's 2–2.5× headroom under capture; the bridge to a captured step over a streamed backbone | 517 / 1,755 MB pool beside 233 / 700 MB live | **build (Step 1)** |
| copy elimination | one buffer write per in-place op instead of two | uncounted — Step 2 counts first | **measure, then build** |
| horizontal fusion | the arena's 1.2× for every optimizer and group count; fewer tiny dispatches anywhere | the arena's gate list | **build (Step 3)** |
| shape buckets | one recording per bucket, not per length | ViT's hand alignment; any NLP loop | **build (Step 4)** |
| kernel autotune | the shapes the hand rule gets wrong | the dW 3× regression | **build, cached (Step 5)** |
| a JS `compiled` + `explain` | parity for borch.ts users; a readable account of a replay | the API gap | **build (Step 6)** |
| capture ∧ window | fine-tuning a streamed backbone under replay | SCALE decision 6 | **build after 1 (Step 7)** |
| CSE / algebraic rewrites / constant folding | unknown — count duplicates first | none yet | **probe, then decide** |
| layout planning (NHWC etc.) | none shown | `docs/INFER.md` §2 | **no** |
| dispatch reordering / overlap | WebGPU's queue is in order and Dawn barriers every dispatch; nothing to reorder into | — | **no** |
| matmul epilogue fusion | — | refused by measurement | **no** |

## 2. The steps

### Step 0 — Every dispatch says what it reads and writes · size S

`Device.run(pipeline, buffers, grid, access?)` takes, per binding, `r` / `w` / `rw`; the
kernel builders that already know (`kernels.ts` — every conv, matmul, BatchNorm, softmax,
gather, the optimizer steps) declare it at the call; a call without it keeps today's
conservative default. `Recorded` carries the sets; `liveIns()` and `fuse.ts`'s graph read
them instead of guessing.

- **Gate**: on the three profiled models (ResNet-18 CIFAR, GPT-2blk, U-Net 96) the
  conservative default is used by 0 dispatches; `liveIns()` returns exactly the inputs +
  parameters + optimizer state + running statistics, and `capture:py` still passes bit
  for bit. A wrong declaration — `r` on a buffer the kernel writes — has no test of
  its own here; it is caught by Step 1, where such a buffer gets aliased by the planner
  and `check=True` stops matching eager and names it. Until Step 1 lands, a debug mode
  fills every `r`-only binding with a sentinel after the dispatch and reads it back
  under `capture:py` — a kernel that wrote it shows up as the sentinel gone.

### Step 1 — Memory planning under capture · size M · depends on 0

With exact access sets the recording is a list of intervals: each buffer the step
allocates is live from its first write to its last read. A linear-scan planner assigns
those intervals to offsets in one or a few arenas (`BindSlot` already carries
`{buffer, offset, size}` sub-ranges — the optimizer arena uses them), so intermediates
whose lives do not overlap share bytes. What stays in its own buffer: anything `held`
by the caller, every live-in, and anything a kernel without a recipe touches (none,
after Step 0). The plan is computed once at `endCapture` and the bind groups rebuilt
against the arena; replay is unchanged.

- **Predict**: ResNet-18 CIFAR batch 16 pool beside the step 517 → ≤ 150 MB; batch 64
  1,755 → ≤ 450 MB (the sum of the widest live set, ~1.3× the largest activation layer).
- **Gate**: peak `made − pooled` under a captured step ≤ 1.3× the eager step's peak
  live bytes on the three models; `capture:py` bit for bit; `check=True` clean on the
  fused U-Net and the AdamW decoder.
- **Retire if**: the rebuilt bind groups cost more than the memory saves on a small
  step — they are built once, so this should not happen; if it does, the plan is kept
  and applied only above a size threshold.

### Step 2 — Copy elimination · size S–M · depends on 0

First, count: `Capture.describe()` already names copies; a line in `profile:py`'s
output says how many a step carries and how many bytes they move. Then, for a copy
whose source is a temporary written once, read by nothing else, and dead after the
copy, rewrite the writer's output binding to the copy's destination and drop the copy.

- **Gate**: the count first (recorded in the ledger); then copies per step on the AdamW
  decoder ≤ the number of genuinely aliased writes (expected: the parameter updates
  only), the replay bit for bit, `check=True` clean. If the count says copies are
  under 2 % of a step's bytes and time, the step is retired at the count.

### Step 3 — Horizontal fusion · size M · depends on 0

Independent elementwise dispatches with the same recipe and no data dependence between
them — sixty-two per-parameter SGD updates, the zeroing of every gradient, the bias
adds of a transformer's layers — become one dispatch over a concatenated index space
(a table of `(buffer, offset, n)` segments, the kernel finding its segment by a
prefix-sum lookup). This is what the Adam arena does for one optimizer by hand; the
pass does it for any, including multi-group, AdamW, amsgrad, and the gradient zeroing
the arena never covered. The arena stays for the shapes it already wins, and the pass
is measured against it.

- **Predict**: the optimizer's dispatches in a ResNet-18 step 62 → ≤ 3; the arena's
  1.2× on the SGD/Adam step reproduced by the pass for AdamW and for two parameter
  groups, where the arena refuses today.
- **Gate**: dispatches per step on the three models down by the count of per-parameter
  kernels; step time not up on any of the three; bit for bit against eager (the
  segments compute the same expression in the same order).

### Step 4 — Shape buckets · size S

`compiled(fn, bucket=8)` (or `pad_to`): a tensor argument whose leading non-batch axis
varies is padded to the bucket and recorded once per bucket, with the mask the model
already accepts (`bimm`'s `keyMask`) or a length argument the step reads. The last
short batch is the same mechanism on the batch axis.

- **Gate**: a loop over sequence lengths 1…256 records ≤ 32 times, not 256; each
  bucketed replay bit for bit against the padded eager run; memory of the recordings
  bounded by the bucket count.

### Step 5 — Kernel selection by measurement, cached · size M

At the first recording of a shape, the candidates a rule chooses among today — matmul
tile, split count, the subgroup/tiled/direct conv paths — are each timed once with a
timestamp query (the profiler's instrument), the fastest kept, and the choice cached by
`(adapter, kernel key)` in the same store the hub uses (OPFS/Cache API), so the second
session pays nothing. The hand rules remain the prior and the fallback when timestamps
are unavailable.

- **Predict**: the dW shapes the rule got wrong before its guard (long-K, 32×8 tile,
  3×) are found by the sweep without the guard; ViT-tiny's step ≤ the hand-tuned 27.9 ms.
- **Gate**: no shape in the three models' recordings slower than the rule's choice
  (two runs); first-record overhead ≤ 2× the step count of candidates in milliseconds
  (Burn's autotune, measured on the 4×4-plane conv at 262 ms/step, is the warning of
  what an untuned first load looks like — hence the cache, and a bound).

### Step 6 — `compiled` in borch.ts, and `explain()` · size S

The Python `compiled` (shape key, copies-in, returned tensors, `check`) ported to
`torch.compiled` in borch.ts, on `Device.beginCapture`; and `Capture.explain()` — the
recording as a table of pipeline key, grid, buffers, bytes, and, after a profiled
replay, GPU time by dispatch (the `sig` the profiler files under is already recorded).

- **Gate**: `borch-ts/test/capture.ts` runs the U-Net step through the JS `compiled`
  bit for bit against Python's; `explain()` output is what `profile:py` prints today,
  from the recording alone.

### Step 7 — Capture over a streamed backbone · size M · depends on 1 and SCALE Step 3

The mutual refusal (SCALE decision 6) lifted for one case: the window refills a slot
*in place* (same buffer, same offset) under a capture, so the recorded bind group stays
valid and the replay reads the refilled weight. Step 1's planner knows which arena
regions are live-ins, which is what makes refill-in-place provable.

- **Gate**: the workbench's LoRA fine-tune over the 57 MB ViT window runs under
  `compiled`, loss curve bit for bit against the eager windowed run; refused by name in
  every other combination until this one passes.

### Probe P — Redundancy in a recording · size XS · decides CSE

Count, in the three models' recordings, dispatches with the same pipeline and the same
input buffers (and no intervening write to them) — the thing CSE would remove — and
elementwise kernels whose recipe is an identity or a multiply by a constant one (what
folding would remove).

- **Decision**: ≥ 2 % of a step's dispatches → a CSE pass joins the plan after Step 3;
  less → written in the ledger as "not worth a pass", and the IR question is closed.

## 3. What this plan does not do

- **An IR.** The recording is the IR: a list of dispatches with access sets, recipes
  where there are recipes, and a memory plan. It is enough for every step above, and an
  operator-level graph would exist only to be lowered back into this list.
- **Layout planning, dispatch reordering, matmul epilogues** — §1's table, with the
  reason beside each.
- **Dynamic control flow in a recording.** A step that branches in Python on the step's
  values is two recordings and a caller who picks; nothing here changes that.
- **Decode.** `docs/INFER.md` §4 and `docs/SCALE.md` Step 8.

## 4. Order, parallelism, size

```
0 ──► 1 ──► 7
│     │
│     └──► 2
├──► 3
4 · 5 · 6 · P  (independent of 0)
```

| step | size | depends on | verified where |
|---|---|---|---|
| 0 access sets | S | — | `capture:py`, three models |
| 1 memory plan | M | 0 | peak bytes on three models; `check=True` |
| 2 copies | S–M | 0 | the count, then the AdamW decoder |
| 3 horizontal fusion | M | 0 | dispatches and step time, three models |
| 4 buckets | S | — | a length-varying loop |
| 5 autotune | M | — | the recorded shapes vs the rule, two runs, cached |
| 6 JS `compiled` + `explain` | S | — | `borch-ts/test/capture.ts` |
| 7 capture ∧ window | M | 1, SCALE 3 | the workbench fine-tune |
| P redundancy | XS | — | the ledger |

Two people, two tracks: **0 → 1 → 2/3 → 7** (the recording's analyses, `device.ts` and
`fuse.ts`) and **4 · 5 · 6** (`_ops.py`, the kernel bench, borch.ts's API). Predicted end
state, written down to be contradicted: a captured ResNet-18 CIFAR step at batch 16 with
≤ 150 MB of pool beside it, ~40 dispatches fewer, and no shape it runs slower than the
hand rule; the same `compiled` name in JS and Python; the workbench fine-tune under replay.

## 5. Ledger

- **2026-09-20, Step 0 landed — every dispatch says what it touches.** `bindingAccess`
  (`device.ts`) reads each pipeline's WGSL once at build: `var<uniform>` and
  `var<storage, read>` are reads; a `read_write` binding is a write when every use of its
  name in the body is a plain element assignment, and read-and-write when any use is a
  read, a compound assignment, or the name is taken by address (a pointer, an atomic,
  `arrayLength`). `Recorded.access` carries it; `liveIns()` and `fuse.ts`'s graph read it
  where there is no recipe; `Capture.coverage()` (JS and Python) counts the kinds. Five
  scanner cases in `device:ts` (read/uniform → r, assigned-only → w, `+=` → rw, `&Name`
  → rw, compared → rw). **Gate, `capture:py` on metal-3: U-Net step 257 dispatches —
  20 exact (recipe), 237 declared, 0 guessed; GPT-2blk 388 — 85 exact, 273 declared,
  30 copies, 0 guessed.** Bit for bit against eager on both, `check=True` clean. Two
  things the exact graph changed on its own: the GPT step's live-ins 101 → 100 (the
  over-approximation had counted one buffer a kernel only writes), and its fused
  recording 371 → 370 dispatches (one more tree could be moved once a kernel that only
  reads its inputs stopped counting as their writer). What the scan could not call
  write-only: 189 of the U-Net's declared bindings and 142 of the GPT's are `rw` —
  conservative, and Step 1's planner will say how many of those matter. Two harness
  lessons, both cost twenty minutes: the probe's Python lives inside a JS template
  literal and one backtick in a Python comment ended it (the page never made its worker,
  and the runner waited its full timeout twice — `capture_py.py` now ends the wait on a
  page error); and a headless debug run of the same page is SwiftShader (190 s to the
  first mark against 7 s headed — `launch.py`'s rule, met again).

- **2026-09-20, Step 1 landed — intermediates share bytes.** `plan.ts`: every movable
  intermediate is an interval from the dispatch that first writes it to the last that
  touches it; a first-fit over offsets lays non-overlapping intervals at the same place in
  an arena; `Capture.plan(held)` allocates the arenas, rebinds the records to
  `{buffer, offset, size}` slots and releases the buffers they replace. Not moved: uploads,
  held buffers, sub-range-bound buffers, and any buffer whose first touch is a read (its
  contents predate the recording — including a fresh buffer a kernel accumulates into, whose
  zeros are WebGPU's and would not be zeros in a reused slot). `compiled(plan=True)` is the
  default, run after `fuse` and before `check`. **Gate, `capture:py` on metal-3: U-Net 96 px
  batch 16 — 217 intermediates moved, 380.1 → 147.1 MB in 3 arenas (2.6×), 30 kept as
  live-ins; what the device held 595.5 → 362.6 MB. GPT-2blk fused — 256 moved, 6.1 → 1.5 MB
  (4×), one untouched intermediate released.** Both bit for bit against eager, `check=True`
  clean on the planned recordings, faults 0; replay 6.3 ms a step, unchanged. Two things
  the first two runs taught, each a fault with a sentence: (1) **WebGPU validates a
  buffer's usages per dispatch, not per range** — one arena bound read-only at one offset
  and read-write at another in the same dispatch is "includes writable usage and another
  usage in the same synchronization scope", and a copy's two ends may not be one buffer; so
  the planner colours a conflict graph (read-only vs read-write bindings of each record,
  copy source vs destination) into arenas before it lays offsets — three arenas came out,
  not one. (2) A buffer made under a capture sits in the scope that was open as well as in
  the pinned set; `compiled` records inside the caller's scope, so a buffer the plan
  released was returned again when that scope closed — `unpin` now takes a buffer out of
  every open frame as it pools it. The ResNet-18 prediction (517 → ≤ 150 MB of pool at
  batch 16) waits on Step 6's JS `compiled`; the U-Net's 2.6× is the number to hold it to.

- **2026-09-20, Step 3 — measured three ways, and what stands is not what was planned.**
  (1) *The optimizer arena under a capture*: the reason it was held off (its lazy build,
  recorded, re-ran on every replay) was removed by building it outside the recording, and
  the compiled loop took the arena — and was **slower**: the hand-written net's replay
  0.66 → 1.45 ms (plain), 0.57 → 1.30 (fused); the U-Net's 6.3 → 6.2, noise. The arena's
  eager 1.2× is the JavaScript it skips (one dispatch call, not one per parameter), and a
  replay skips all of that anyway; what remains is its gather and scatter, two copies a
  parameter, which cost more than the per-parameter launches they replace. **Retired**;
  the gate stays `!capturing`, with the measurement beside it in `optim.ts`. The
  prediction "62 → ≤ 3 optimizer dispatches" was the wrong quantity to want under replay.
  (2) *The generic pass*: `Capture.horizontal()` counts what such a pass would merge —
  consecutive elementwise dispatches with one recipe and no dependence. **U-Net: 0 runs.
  GPT-2blk: 0 runs.** Nothing to fuse sideways in either; not built, and the count stays
  in `capture:py`'s line so a model that changes the answer is seen.
  (3) *What did land — Adam's decay into its kernel.* Extending the eager arena to weight
  decay (`adamStep`'s `decay`, coupled and decoupled) first made the arena and the
  per-parameter path differ by a rounding (6e-8 / 3e-7 on `adam_arena_probe`: the kernel
  contracts the multiply and add the two tensor ops rounded apart), and an eager AdamW
  step then no longer matched its own compiled replay (`capture:py` rel|Δparam| 3.6e-3
  over eight scheduled steps). So the per-parameter path took the same kernel arithmetic:
  **eager, arena and replay are one expression again — bit for bit on all three variants
  (`adam_arena_probe`: 0.00e+0 each), and AdamW's two tensor ops a parameter are gone:
  GPT-2blk 388 → 298 dispatches plain, 370 → 280 fused, its 30 recorded copies → 0;
  torch's curve still walked at 7.3e-6.** The invariant that decided it: an eager step
  and its compiled replay must be the same numbers, or `check=True` means nothing.
- **2026-09-20, Step 2 retired at the count.** With AdamW's decay in the kernel the
  U-Net's step records 0 copies and the GPT's 0; the copies the plan meant to eliminate
  were those. `Capture.coverage().copies` keeps the count on every `capture:py` run; a
  model that brings copies back reopens the step.

- **2026-09-20, Step 6 landed — `torch.compiled` and `torch.capture` in JavaScript, and
  `explain()`.** `compile.ts`: the binding's bookkeeping ported — shape key over tensor
  shapes and plain values, the tensor arguments read back once and re-made under the
  capture as uploads (live-ins, never moved), later calls copy in and replay, a new shape
  records again; `fuse` and `plan` on by default with what the function *returns* as the
  held set (JavaScript has no registry of live tensors — the contract is written on the
  file); `check: true` ported whole. `Capture.explain(ns?, head?)` prints the recording
  a dispatch a line with its key, grid, buffers and, after a profiled replay, GPU ms per
  dispatch of its kind. `capture:ts` (census 65 → 66, in the nightly): **an MLP over two
  batch shapes — two recordings, fourteen losses bit for bit, `check` clean on both; the
  ResNet-18 of `bench.ts` at batch 16 — three replays bit for bit against eager, eager
  21.3 → replay 19.1 ms a step, and the number Step 1 waited on: 369 intermediates
  341.8 MB → 129.3 MB in 3 arenas.** The prediction was "517 → ≤ 150 MB of pool beside
  the step"; the like-for-like is the intermediates' bytes, and 129 is under it. The
  first line of `explain` on that recording already says where a person would look next:
  the subgroup conv's `pdw` pad and `tmw` weight repack run every step (`docs/INFER.md`
  Step 2 is the eval half of that).

- **2026-09-20, Step 7 landed — a captured step over a streamed backbone.** The window's
  refill is a staging map (`Window.place`: `mapAsync`, a copy into the slot), which a
  recording cannot re-encode — by the time of a replay the staging buffer holds another
  block. So the recording keeps the *host bytes and the slot* (`Recorded.refill`, written
  by `Window.place` under a capture, the copy itself left out of the recording through
  `Device.unrecorded`), and `Capture.replayAsync` walks the records: the dispatches between
  two refills are encoded and submitted as one run, then the block's bytes are written
  into the slot it had (`Window.refill`) — the queue keeps submit order, so the run before
  reads the old bytes and the run after the new — and on to the next. The window's own
  bookkeeping is untouched: the slots a replay writes are the slots the recording chose,
  free then as now. `torch.captureAsync` and an async `fn` in `compiled` carry it; a
  recording with refills refuses `replay()` by name. **Gate, `capture:ts` on metal-3: a
  four-block LoRA chain over a two-slot window — one recording (8 refills a step: four
  forward, four backward), replayed twice, gives the three eager streamed steps bit for
  bit on the losses and the adapters; `check: true` clean on 20 live-ins; the window 4,096
  of 8,192 bytes used under the recording, one block wide as eagerly; the plan laid its 59
  intermediates 38 → 8 KB.** SCALE decision 6's mutual refusal is lifted for this case —
  refill-in-place under a capture is now measured, not assumed. What is not covered:
  `streamSequential`'s no-grad forward (the same mechanism, not yet exercised), and the
  workbench's Python path, which has no streaming API to record. One fix found on the way:
  a fused kernel's record carried no access set and counted as "guessed" — nine a step
  here, zero in the unfused probes that measured Step 0 — `fuse.ts` now carries it.
- **Step 4 (shape buckets) — deferred, with the reason.** Padding an axis to a bucket is
  correct only where the model consumes the true length (a key mask, as `bimm`'s ViT does
  by hand), so it cannot be a property of `compiled` alone; nothing in the tree varies a
  sequence length under `compiled` today. It comes back with its first caller. **Step 5
  (autotune) — waits on a second adapter**: its gate is two adapters, and the 4090 is off
  the bus (2026-09-14) while the 5080 runs another experiment.

- **2026-09-20, a pass the plan did not list — hoisting the replay-invariant dispatches.**
  `Capture.hoist()`, between `fuse` and `plan` in both `compiled`s: a dispatch that reads
  only constants of the recording (buffers no record writes, not uploaded inputs, not
  refill slots) and writes purely is run once and dropped from the replay, its outputs
  frozen (never planned over, never released before `dispose`). Fixed point, so chains
  hoist whole. Motivated by `docs/INFER.md` Step 2 — the eval forward's weight repacks —
  and it is what Step 0's access sets were for: the eval ResNet-18 replay 41 → 38
  dispatches (the three subgroup layers' `tmw` gone), the training step untouched (257,
  bit for bit; the optimizer writes the weights, so nothing is constant).

## 6. Risks, and the sentence that retires each

| risk | what would show it | retirement |
|---|---|---|
| a wrong access declaration lets the planner alias a live buffer | `check=True` mismatch naming the buffer | the declaration is fixed; the planner never trusts a kernel without one (conservative default = own buffer) |
| the planner's arena exceeds `maxStorageBufferBindingSize` on a small tier | bind group creation refuses | several arenas, each under the tier; the tier is read at init (`device.ts`) |
| horizontal fusion's segment lookup costs more than the launches it saves on tiny steps | step time up on GPT-2blk | a minimum segment count before the pass fires; measured, not assumed |
| autotune's first-record sweep makes the first step seconds long | the first-step clock | bounded candidate count, cached choice, the rule as prior |
| a bucketed replay reads padded rows as data | logits differ from the unpadded eager run | the mask is part of the recipe (`keyMask`), and the gate is bit-for-bit against *padded* eager plus 1e-6 against unpadded |
| refill-in-place under capture reads a half-written slot | a replay that starts before the copy lands | the refill is a recorded copy, ordered in the queue before the dispatches that read it |
| single-op timings swing 2× on the M4 Max | one run says a pass won | every gate is two runs, fenced, adapter printed |
