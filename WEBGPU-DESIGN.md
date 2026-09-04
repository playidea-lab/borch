# borch-webgpu — the design

> The sister library. It does not replace the core `borch`; it goes where the core
> does not. Why it is a separate distribution is in
> [ADR-001 in ROADMAP.md](ROADMAP.md).
>
> **There was no code when this was first written.** It was a document for settling
> what could be settled before starting, and **for writing down what could not be
> settled as unsettled.** There is code now and `borch_webgpu.py` is it — so the
> assumptions below carry measured values, and **the ones that were overturned by
> measuring are left in place.** A prediction that was wrong is more use than one
> that was right.

---

## The goal

**CIFAR-10 ResNet-18 scale training runs in a browser in a few minutes per epoch.**

By ADR-001's arithmetic that demands **about 300×** over where things stand. That
factor governs every decision in this document — a design that cannot produce 300×
is eliminated here.

### Non-goals

| | why |
|---|---|
| replacing the core | the core stays the place to practise syntax. This one is the place to run things |
| all of torch's surface | the curriculum plus what a CIFAR-scale model uses, and nothing more |
| bit equivalence | a non-goal for the core's reason |
| inference optimisation | training is the point. For inference alone, ONNX Runtime Web already exists |

---

## 1. Unverified assumptions — **measuring starts here**

This whole design hangs on the facts below, and **they have not been measured.**
The first task before starting (S0) is not writing code but measuring them.

### A1. Synchronous reads on the TF.js WebGPU backend

Does `tf.Tensor.dataSync()` **actually work on the WebGPU backend, and how
expensive is it.**

WebGPU has no synchronous readback API (only `mapAsync`). So how TF.js handles that
— not supporting it, supporting it with a warning, blocking internally — **decides
outright whether the Python API can stay synchronous.**

- If it works → `loss.item()` stays exactly as it is. **The best case.**
- If it does not, or is fatally slow → section 3's synchronisation alternatives.

> This is the most important line in this document. Every other design decision is
> settled after this answer arrives.

### A2. How mature TF.js WebGPU's **training** path is

TF.js's WebGPU backend matured on the inference side first. What has to be checked
is whether the things training needs exist as actual WebGPU kernels, or fall back
quietly to CPU or WebGL. A fallback means **a GPU-to-CPU round trip every step and
the 300× disappears.**

To be measured: `conv2dBackpropInput`, `conv2dBackpropFilter`, `batchNorm`
backward, `maxPoolBackprop`, and reductions over a large axis.

### A3. Effective throughput

Measure **actual GFLOPS** with GEMM and conv benchmarks. Which row of ADR-001's
table this lands on is settled here.

### The stopping condition

If S0's **estimate for one CIFAR ResNet-18 epoch exceeds five minutes, stop and
choose a different strategy.** The reason for writing it down is plain — unwritten,
"just a little more tuning" runs for months.

---

## 1.5 The S0 results — **measured. Verdict: continue, conditionally**

The measuring environment: Apple Silicon (WebGPU `metal-3`), Chrome, TF.js 4.22.0,
backend `webgpu`. The adapter supports `shader-f16` and `subgroups`. **These are the
results from one machine, one layer size and one version** — they have to be
measured again on another GPU.

### A1 — `dataSync()` **works**

| | the same computation (1024³ matmul → sum → scalar) |
|---|---|
| `dataSync()` | **3.35 ms** |
| `await data()` | 1.72 ms |

The synchronous read does not throw, and costs **+1.6 ms** against the asynchronous
one. It is called once per training step (`loss.item()`), so where a step is tens
of milliseconds it is **negligible.**

> **Conclusion: the Python API stays synchronous.** Alternatives 2, 3 and 4 in
> section 3 will not be needed. The most important unknown in this document
> answered in the best possible direction.

### A2 — the training kernels are **all there**

`conv2d` forward, `conv2d`'s input and filter gradients, `maxPool`'s gradient,
`batchNorm`'s gradient, and the BN gradient decomposed through `moments` — all five
ran without an exception and their gradient shapes and finiteness were right. TF.js
throws rather than falling back quietly when a kernel is missing, so not throwing
is the answer.

### A3 — throughput, and **this measurement's central finding**

GEMM is fast.

| | GFLOPS |
|---|---|
| matmul 512² | 1,007 |
| matmul 1024² | 2,399 |
| matmul 2048² | 3,084 |

**And conv is fast on the forward pass alone.** (Layer size: input
`128×32×32×64`, filter `3×3×64→64`, same padding)

| | GFLOPS | |
|---|---|---|
| conv2d forward | **2,306** | normal |
| `tf.grad`'s dx (`Conv2DBackpropInput`) | **88** | 1/26 of the forward |
| `tf.grad`'s dw (`Conv2DBackpropFilter`) | **130** | 1/18 of the forward |
| forward and backward together | **271** | |

**TF.js's WebGPU conv backward kernels are effectively unoptimised.** This is the
kind A2 cannot catch — it does not throw, it is simply slow.

### And it can be fixed — rewrite the backward **as a forward conv**

| | `tf.grad` | rewritten | factor | numerical agreement |
|---|---|---|---|---|
| dx (kernel spatially reversed, input and output channels swapped) | 88 | **2,917** | **33×** | relative difference 1.3e-06 |
| dw (batch and channel swapped, dY as the filter) | 130 | **807** | **6.2×** | relative difference 1.5e-05 |

Both were written down with the values compared. dw's 1.5e-05 is **an ordering
difference** across 130,000 accumulations, the same place as T4 (bit equivalence)
being a non-goal — and inside the wide-surface harness's tolerance (1e-4).

### The verdict

Against 84 TFLOPs per epoch (counting conv and matmul alone):

| | effective | one epoch | |
|---|---|---|---|
| using `tf.grad` as it is | 271 GFLOPS | **5.1 min** | hits the stopping condition |
| rewriting the backward | ~1,500 GFLOPS | **~1 min** | passes |
| even at half of that | ~750 GFLOPS | ~1.9 min | passes |

> **Continue, with a condition — conv's backward is written here rather than left
> to `tf.grad`.** That is not a burden but the plan all along. This library has its
> own autograd (section 4).

**Three uncertainties remain.** (1) The 84 TFLOPs counts conv and matmul alone, so
the bandwidth-bound operations — BN, ReLU, the residual additions — are missing; a
real epoch is longer than this. (2) It was measured at one layer size. Whether the
same trick holds in the later layers, where the feature maps shrink to 8×8, has not
been looked at. (3) Measuring dx and dw separately sums to 184 ms, which is more
than measuring them together (107 ms) — meaning the measurement has jitter, so the
factors above are a direction rather than a precise value.

### A cancelled item — headless WebGPU

"Does WebGPU come up headless" was written down as an unknown for a while. **It is
removed because its premise was wrong.**

That unknown came from section 6 assuming "stage 2 has to run headless in CI", and
that assumption had nothing behind it. This thing runs on **the real GPU in a
user's machine.** Passing under SwiftShader on a Linux runner with no GPU verifies
a software path nobody uses, and with a different driver it does not guarantee
correctness on a real GPU either.

Verification happens on **a machine with a real GPU** (section 6). Whether headless
works is therefore not a gate.

---

## 1.6 The S3 results — the hand-written backward earned its keep. Now **the transposes** are the bottleneck

Golden 141/141 (values, gradients, training), and CNN training matches torch on the
GPU. Measured on the same layer (`128×64×32×32`, `3×3` 64→64):

| | this library | S0's raw measurement |
|---|---|---|
| conv forward | 279 GFLOPS | **2,306** (raw `tf.conv2d`) |
| conv **backward** (hand-written) | **~1,086** | ~105 (`tf.grad`) |
| forward and backward together | **553** | 271 (the `tf.grad` path) |

**The backward came out as intended — about 10× `tf.grad`.** Section 1.5's basis
for continuing was confirmed in real code.

**And the forward is 8× slower than the raw one.** The cause is **layout** rather
than computation. torch is NCHW and TF.js is NHWC, so every conv walks a 33.5MB
transpose twice (30.4ms of 34.6ms). The backward is fast because a closure holds
the already-transposed input and does not walk it again.

The epoch estimate is 84 TFLOPs ÷ 553 GFLOPS ≈ **2.5 minutes** — inside the
stopping condition (5 minutes). So it continues.

> **S4's first item**: carry 4-D tensors **internally as NHWC** and transpose only
> at the API boundary (`tensor()` and `numpy()`). With the transposes gone the
> forward approaches the raw speed, the combined figure passes 1,000 GFLOPS and the
> epoch falls under 1.5 minutes. In exchange, BatchNorm's channel axis and
> `reshape`/`flatten` have to know about the layout, so the places that can be
> quietly wrong increase — touching it after the golden has grown to 141 is the
> right order.

### A note on how it was measured

The first figure was 156 GFLOPS and **the measuring was wrong.** It was reading
33.5MB of gradients back to the CPU every iteration (183.5ms), and that is transfer
rather than computation. As in S0, the work has to be queued many times and
synchronised once at the end for the kernel speed to come out. The table above is
the re-measured version.

---

## 1.7 S4 — **the goal was reached.** ResNet-18 trains in a browser

A CIFAR-flavoured ResNet-18 with 11,173,962 parameters was actually stood up and
its training step measured. Not an estimate but **a real step with BN, ReLU and the
residual additions all running** (`tests/browser/bench.py`).

| batch | ms/step | one epoch | leaked per step |
|---|---|---|---|
| 16 | 122.1 | 6.36 min | 0.0 |
| 32 | 163.2 | 4.25 min | 0.0 |
| 64 | 239.0 | 3.11 min | 0.0 |
| 128 | 393.8 | 2.57 min | 0.0 |
| **256** | **726.4** | **2.37 min** | 0.0 |

**The 300× ADR-001 demanded came out.** As it stands, 20 epochs is about 50 minutes
— a length of time somebody would actually run in a browser tab.

Two things to record.

- **The estimate was optimistic.** Dividing by one layer's FLOPs gave 2.5 minutes
  and the reality is 2.37–3.1. Only a large batch approaches it, and the
  bandwidth-bound operations were missing from the FLOPs arithmetic. The direction
  was right, and **not deciding on an estimate was the correct call.**
- **A larger batch is more efficient** (an epoch goes from 6.36 to 2.37 minutes
  between 16 and 256). At small batches the individual kernels cannot fill the GPU.

### The NHWC refactor — **done.** 2.37 minutes → 1.9

"The goal is reached, so this is deferred" stood written for a while, and it was
reversed. The golden had grown to 143 by then, so there was a net, and that net
caught five things.

**A tensor carries its layout.** A 4-D tensor may be NHWC internally, and `shape`
always answers in torch's order. conv, pooling, BatchNorm, the activations and the
residual additions all chain in NHWC, and
the transpose happens **once on the way in and once on the way out**.

| batch | transposing | carrying the layout |
|---|---|---|
| 64 | 3.11 min | **1.98 min** (−36%) |
| 128 | 2.57 min | **1.95 min** |
| 256 | 2.37 min | **1.89 min** |

Three rules held.

- **An operation that does not know about layout converts back first**
  (`_canonical`). It can be slow and it cannot be wrong. Measured, the cost of
  converting back is invisible (1.89 → 1.97 minutes, because those operations are
  not on the hot path)
- **A binary operation aligns its pair** (`_align`). Two 4-D tensors means changing
  one; a partner that is not 4-D means converting both back — a 1-D bias attaches
  to the last axis, and which axis that is depends on the layout
- **BatchNorm's backward is written by hand.** Assembled, the axes differ per layout
  and the kernel count grows

**Wherever a handle is passed on, the layout has to be passed with it.**
`detach()` left it out and the internal shape leaked outward — the golden caught it
with five cases immediately.

### Data loading — the plumbing is in, and **it does not change the numbers**

`utils.data` (TensorDataset and DataLoader), `fetch_cached` (an OPFS cache),
`decode_cifar10`, and — for images a visitor already has — `decode_images` (files →
NCHW in [0, 1], labels from the names) with `suspects` (the share of each sample's
k nearest neighbours that disagree with its label; the workbench's review order).
The last two are the core's; the binding only fetches Pillow in Pyodide.

- **The data stays on the CPU.** All of CIFAR-10 on the GPU is 614MB and one batch
  is 3.1MB. Uploading per batch is cheaper and leaves the GPU memory to the model
- **Only the loading is asynchronous.** OPFS has no synchronous API (only inside a
  worker). One `await` during preparation, and **the training loop stays
  synchronous**
- The dataset's address comes from the caller. Baked into the library, the library
  has to be edited when that address disappears

**The upload cost measured as zero.** At batch 128, repeating the same tensor gives
400.8ms/step and re-uploading through the DataLoader gives 399.4ms/step — a 3.1MB
transfer is invisible beside a 400ms step. **Data loading does not change section
1.7's conclusion.**

What was verified: the decoder (shape, dtype, values and length checks on synthetic
bytes), the OPFS round trip, a second call after a fetch giving the same bytes from
the cache, and the DataLoader's batching (the final short batch included).

### It runs on real CIFAR-10 — **the loss goes down**

This could not be confirmed with synthetic data. Random pixels with random labels
have nothing to learn, so the loss does not fall, and then "does it train" stays
unconfirmed forever.

`data_batch_1` (10,000 images) fed to ResNet-18 at batch 128 for 60 steps:

```
load 108ms · x (10000, 3, 32, 32) [0.00, 1.00]
label distribution  [1005, 974, 1032, 1016, 999, 937, 1030, 1001, 1025, 981]
           ↑ **exactly matching** what was measured natively — the browser decode is right

loss 2.4894 → 1.7222   (random guessing is ln 10 = 2.303)
curve [2.49, 3.70, 2.29, 2.33, 2.02, 1.95, 1.83, 1.92, 1.92]
```

The early jump to 3.70 is the usual shape at the start with a large learning rate,
and after it the curve falls below the guessing line. **It learns.**

> **CORS — measured. The originals cannot be fetched.**
>
> ```
> control, jsdelivr : OK status=200 type=cors
> CIFAR original    : No 'Access-Control-Allow-Origin' header  ← blocked
> ```
>
> `cs.toronto.edu` sends no CORS header, so it **cannot be fetched directly from a
> browser.** (The control passed, so the check itself is valid.)
>
> Which is why `cache_put(name, bytes)` is exposed — the user picks a file, or puts
> in bytes fetched from a mirror that does send CORS, and everything after that is
> the same as `fetch_cached`. **This is why the address is not baked into the
> library.**

---

## 2. Why TF.js

**There is no compilation.** Calling `js.tf` directly through Pyodide's JS FFI
keeps this side a `py3-none-any` pure-Python wheel too. Unlike a Rust/wasm backend
or hand-written WGSL kernels, **it is not tied to a Pyodide ABI or an emscripten
version** — the sister library keeps the property the core keeps.

What is borrowed: the WebGPU backend, many kernels, automatic differentiation
(`tf.variableGrads`), memory management. What has to be written: torch's semantics,
the Python API, and every layer section 4 names.

**The price is the semantics.** TF.js's operation rules differ from torch's, and
erasing that difference is this project's reason to exist. Section 5 is the
list.

---

## 3. Where async gets confined

**Operations are synchronous by nature.** `queue.submit()` returns no promise. The
one point that needs asynchrony is **looking at a value** — `.item()`, `print()`,
`.tolist()`. In one step of a training loop that is usually a single
`loss.item()`.

In order of preference:

| | condition | result |
|---|---|---|
| **1. `dataSync()`** | A1 holds | the Python API is **fully synchronous.** The same code as the core |
| 2. a Worker plus `Atomics.wait` | when COOP/COEP headers can be set | fully synchronous. And the GPU has to be on **another thread** — on the same worker, the event loop that would resolve the promise is blocked and it deadlocks |
| 3. JSPI | Chromium-family | fully synchronous. Unsupported on Safari and Firefox, so **it cannot be the main path** |
| 4. `await loss.item()` | always | one `await` stays in the loop. The final fallback |

> Pipelining — reporting the loss one step late to avoid blocking — **is not
> used.** It is a value quietly shifted by one, which is the kind of thing this
> project said it would not do.

---

## 4. How much of the core can be reused — **less than expected**

A fact worth knowing before starting. The core is **tightly coupled** to numpy, so
"just swap the backend" does not work. In the actual code:

- `conv2d` does im2col with `_np.lib.stride_tricks.sliding_window_view`
- `BatchNorm2d.forward` calls `x.data.var(axis=(0, 2, 3), ddof=1)` directly
- `max_pool2d` scatters with `_np.ogrid` and `_np.add.at`
- `Tensor._binary` branches on `.data.dtype` and calls `_np.add` and the rest
- the optimisers update the numpy array directly with `p._array = p.data - lr * g`

Pulling that out into a backend abstraction means **refactoring the core, and that
violates ADR-001** (planting an abstraction in the core). So the conclusion is:

> **The tensor operations and the layers attached directly to them are
> reimplemented. The core is not touched.** This is why it is a "sister library"
> rather than a "backend".

### Reused / reimplemented

| reused as is | reimplemented |
|---|---|
| the six schedulers — Python float arithmetic and nothing else | `Tensor` and autograd |
| `Dataset`, `Subset`, `ConcatDataset`, the samplers | elementwise, reductions, shape manipulation |
| `Optimizer`'s `param_groups` and `state_dict` indexing conventions | `conv2d`, pooling, `BatchNorm`, `LayerNorm` |
| `Module`'s tree structure (`_params`/`_buffers`/the naming conventions) | the optimisers' `step()` formulas |
| the error message specification (`_like_torch`) | `DataLoader._collate` (it hangs on tensor operations) |

**Parameter initialisation uses the core's numpy path as it is and uploads to the
GPU.** The core already implements torch's distribution and torch's `manual_seed`
meaning, and switching to TF.js's RNG breaks that. Initialisation happens once, so
there is no cost either.

---

## 5. torch's semantics — the list of places that will diverge

These were learned the hard way by the core. **Every one of them has to be
confirmed again** on TF.js.

- **dtype promotion** — torch splits by category (bool < integer < float) and
  promotes within a category alone. TF.js refuses most implicit promotion outright.
  The 112-case table has to be run again
- **BatchNorm's two variances** — the biased one (ddof=0) for the normalisation and
  the unbiased one (ddof=1) for updating `running_var`. Biased in both places is off
  by 2.6%. The core was wrong about this for a long time
- **`median`** — with an even count torch takes **the smaller** of the middle two.
  Averaging is quietly different
- **conv padding** — torch takes integer padding and TF.js is built around the
  `same`/`valid` strings
- **two kinds of mask** — a boolean one masks and a float one **adds**
- **accumulation order** — a float32 reduction diverges by order. T4 (bit
  equivalence) is a non-goal here too

---

## 6. Verification — **a three-way comparison does not fit in one process**

This is the central constraint on the harness design.

- real **torch is not in the browser**
- **the GPU path is not in native CPython**

So the current arrangement — calling `torch vs borch` side by side in one process —
does not work for the GPU. It splits into **two stages around a golden file**.

```
stage 1 (native)    run the existing harness and freeze torch's expected values by
                    case name → golden.npz  (plus a hash of the case list)

stage 2 (browser)   load the golden into Pyodide, run the GPU path and compare
                    a different case hash fails — it means the golden is stale
```

### What has to change in the existing harness

- `_wide_cases()` already **takes the library as an argument** through
  `lambda L: ...`. Making it three-way works as is — this shape is the path forward
- `build_cases()` holds two separate closures, `run_real` and `run_nano`. They have
  to be unified into the same `lambda L` shape for a third to fit
- `compare_grad()` monkeypatches `real.tensor`/`nano.tensor` to collect leaves. That
  trick has to be generalised into the library-argument shape too

### The browser runner — **it runs on a machine with a real GPU**

Stage 2 needs a browser, and that browser has to sit on **the same kind of GPU users
actually have**. GitHub-hosted runners have no GPU and fall back to SwiftShader,
which is a path no user ever takes — passing there proves nothing.

So a **self-hosted runner** is used. Registering the development machine as a runner
makes it run automatically on a real GPU. Registering a runner needs repository
settings and a token, so a person does it.

> **The window has to be shown (`--headed`).** This is measured — Playwright's
> headless Chromium has no `navigator.gpu`, so TF.js **falls back to WebGL
> silently.** That is why the runner checks the backend and stops when it is not
> `webgpu`. Without that block you look at WebGL numbers believing you ran on the
> GPU.

The existing CI (native pytest) stays as it is. The core borch is numpy alone, so it
never needed a browser — the new axis attaches **to the GPU path only**.

---

## 7. Memory — TF.js frees by hand

A TF.js tensor has to be released explicitly with `dispose()`, and leaving it to the
non-deterministic timing of Python's GC leaks GPU memory. Hanging it on `__del__` is
not enough.

**The core already has the right hook.** When `backward()` finishes it releases the
graph (`_freed`, `retain_graph`). The lifetime of the intermediate buffers is tied
**exactly to that point** — the moment backward ends, every intermediate tensor in
the graph is disposed.

- parameters, buffers, optimiser state → explicitly **kept**
- graph intermediates → released together at `backward()`
- values inside `no_grad()` → released on leaving the block

> **The last line of this paragraph was wrong.** It originally said "with this rule
> the scope does not have to be exposed in the user API", and measuring again showed
> otherwise.
>
> Python's GC only releases the handle a `Tensor` holds. But the **intermediate
> buffers held by the backward closures** (gelu's `1+erf`, gather's one-hot and the
> like) are not `Tensor`s, and walking the graph does not find them either. Measured,
> **92.7 of them** were left per training step.
>
> So the scope is exposed. It costs one line of difference from the core, which
> beats leaking, and with the scope wrapped around it the leak per step is **0**
> (measured).
>
> ```python
> with torch.scope():
>     opt.zero_grad(); crit(model(x), y).backward(); opt.step()
> ```
>
> Parameters and optimiser state are held alive with `tf.keep`, so they survive
> leaving the scope.

---

## 8. Stages

| | content | what comes out |
|---|---|---|
| ~~**S0**~~ | ~~measure A1, A2, A3~~ | **done — section 1.5. Conditional go-ahead** |
| **S1** | ~~golden-file harness~~ done + browser runner | the runner turning with an empty backend |
| ~~**S2**~~ | ~~`Tensor` + autograd + elementwise, reductions, matmul~~ | **done — golden 124/124, MLP training runs, 0 leak per step** |
| ~~**S3**~~ | ~~`conv2d`, pooling, `BatchNorm`~~ | **done — golden 141/141, CNN training runs. Section 1.6 above** |
| ~~**S4**~~ | ~~ResNet-18, performance, data loading~~ | **target met — 2.37 min/epoch. Section 1.7** |

**S1 coming before S2 is deliberate.** That is the lesson the core learned —
"there was merely no check; there was never evidence that it was right."

---

## 9. Open questions

- **f16** — WebGPU's `shader-f16` is an optional feature. Using it raises throughput
  considerably, but mixed precision is on the core's list of deliberate refusals.
  Should the sister library allow it? (Falling short of 300× in S4 would force this
  answer)
- **the package name** — `borch[webgpu]` extras need index resolution, and this
  repository is still private, so it is not on PyPI. **Making the repository public
  is a precondition**
- **the API name** — go with `import borch_webgpu as torch`, or use the same name as
  the core through a separate import
