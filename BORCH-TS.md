# borch.ts — the design

> A TypeScript tensor library that **trains** in a browser. The WGSL kernels are
> written directly.
>
> This document settles what can be settled before starting, and exists **to
> write down what cannot be settled as unsettled.** The same approach worked when
> standing up the two libraries before it.

---

## Why build it — only what was measured

Three things were confirmed before starting. The fourth could not be, and that is
written down too.

### 1. The spot is empty

No TS library was found that **actually trains** in a browser through a
torch-shaped API.

| | stars | last push | trains | where it stops |
|---|---|---|---|---|
| [webgpu-torch](https://github.com/praeclarum/webgpu-torch) | 646 | 2024-07 | **no** | `conv2d` and `mm` backward are "not implemented" |
| [js-pytorch](https://github.com/eduardoleao052/js-pytorch) | 1,222 | 2024-11 | yes | GPU.js, 16 operations and 11 layers |
| TyTorch | — | 2025-10 | yes | **does not run in a browser** (native libtorch) |

`webgpu-torch` came closest and **stopped at exactly the conv backward.** That was
already solved in the sister library — the derivation that rewrites the backward
as a forward conv.

### 2. Our WGSL keeps up with TF.js

Measured with `tests/browser/wgsl_bench.js` and `wgsl_conv.js`. Every value was
compared against TF.js.

| | against TF.js |
|---|---|
| matrix multiplication (1024³, 2048³, 16384×576×64) | **115–217%** |
| conv (the five shapes ResNet-18 uses) | **72–284%** — three of the five above 100% |

These kernels were written in a day and TF.js has been tuned for years. The two
places sitting at 72–83% still have things untried on them (`vec4` loads, unrolling
the kernel loop, per-shape tile sizes).

### 3. The golden crosses languages

`tests/golden.json` — 798 expected values pinned with real torch. Measured that
**the JSON alone** compares 746 cases on the core and 799 on the sister library.
This is used from day one.

### 4. Demand could not be confirmed

`webgpu-torch` collecting 646 stars and then stopping for two years can be read as
"no demand" and equally as "the author moved on". **We do not know which.** It is
the one assumption this project stands on that has not been measured, and if it is
wrong then everything else being right does not matter — nobody uses it.

---

## What is different — compared against the sister library

The sister library (`borch_webgpu`) sits on TF.js, and **73 places are TF.js
workarounds.** Most of them disappear here.

| | the sister (TF.js) | borch.ts (WGSL) |
|---|---|---|
| layout | NHWC forced — carrying `_relayout` and `_nhwc` around | **we decide it** |
| CPU round trips | 17 places (`einsum`, `quantile`, `linalg`, `cummax` indices, …) | write a kernel |
| `pad` | **quietly wrong values** at rank 5 and above | not somebody else's bug |
| `cast` | int32→float32 **does not change the bits** | absent |
| rank | partial from 7 upwards | we decide it |
| dependencies | TF.js + Pyodide (~10MB) | none |

### What is taken on instead

- **Per-shape shader specialisation is mandatory.** A division's divisor kept in a
  uniform blocks strength reduction, and that alone moved conv from 43% to 284%
  (measured). Which means **a shape signature to a pipeline cache** goes into the
  library's structure. Structure rather than optimisation.
- **WebGPU does not throw past a limit; it quietly does nothing.** This benchmark
  stepped on it twice — 240,000 GFLOPS past the 128MB buffer limit, and "144% of
  TF.js" past the 65,535 workgroup limit. Both are numbers that would have been
  believed without checking the values alongside. **Every kernel gets a value
  comparison attached.**
- The kernel engineering itself. All three times on this benchmark the problem was
  my kernel rather than the platform: an accumulator falling out of registers, a
  dispatch limit exceeded, a division at runtime.

---

## What is settled

### The repository

**It starts inside this repository** (`borch-ts/`). The golden, the CI and what it
is compared against are here. It gets split out when independence is needed —
split first and the golden has to be copied, and the two diverge from that moment.

### Devices

**They are not enumerated, and there is no index.** WebGPU gives no way to count
adapters — `navigator.gpu.requestAdapter()` gives one and that is that, and the
only choice is the `powerPreference` hint. So `'webgpu:1'` has nothing to point at
and there is no reason to build a `device` object. Two strings (`'webgpu'` and
`'cpu'`) are the whole of it.

**VRAM cannot be asked about.** Neither the total nor the free amount is in the
standard. Instead `maxBufferSize` and `maxStorageBufferBindingSize` are requested
at the adapter's maximum and exceeding them throws (`device.ts`) — because past a
limit WebGPU quietly does nothing.

**`'cpu'` is a storage location rather than a compute device.** It means the
values are on the host as a `Float32Array`, and there are no kernels over that.
Putting one into an operation makes the `Tensor.buffer` getter stop with torch's
wording — all 176 operation entry points pass through that getter, so the guard
sits in one place. No CPU kernels were built because there is nowhere for them to
live: rewriting 4,700 lines of operations in JS for a browser without WebGPU runs
head-on into this repository's judgement that not running beats quietly getting
slower.

**It does not fall back; it hands back the reason as a value.** `probe()` splits
`'no-api'` from `'no-adapter'` — the first is a browser or context problem and the
second is a driver blocklist, a virtual machine, or headless. What the user can do
about them differs, so one exception must not cover both.

### The checkpoint format is safetensors — a format only we read throws half of it away

torch's `save`/`load` is pickle. It unpacks Python objects by executing them, so it
cannot be carried into a browser and should not be. A format had to be chosen, and
there were three candidates.

| | |
|---|---|
| a JSON array | simple, and a single float costs seven bytes. A million parameters is 7MB of text |
| our own binary | dense and exact. **We are the only readers** |
| **safetensors** | 8 bytes of header length, a JSON header, a contiguous body. The codec is one file |

The third was chosen. The codec is `serialize.ts` alone, and what that buys is
Python `borch`, numpy and the HF tools reading the same file. **Training in a
browser and carrying the result to your own machine is half of what this project
is about**, and a format only we read throws that half away.

**The dtype is always written as `F32`.** borch's `int64` and `bool` are labels
alone and the values live in a float32 buffer. Writing `I64` into the header
disagrees with a four-byte body and breaks somebody else's reader — the labels ride
separately in `__metadata__`. Somebody else reading gets a float32 array and that
is the right answer; reading it here brings the labels back too.

**The runner actually measures that claim.** Our codec round-tripping with our
codec would work for our own format too, so it proves nothing. `serialize.py`
takes the bytes the browser wrote and opens them **with numpy alone** — it confirms
the values and the labels without a line of borch code.

### Resuming is not a round trip

Saving, reading back and getting the same values asks about **the codec** and
nothing else. The real question is *whether training interrupted and resumed
matches training that ran straight through*, and one momentum, one step counter or
one scheduler epoch
going missing leaves the round trip green and the resume alone diverged. All of it
is deterministic, so **it has to be equal bit for bit** — a tolerance reads missing
state as error.

The same runner also walks **the path that restores the weights alone.** That one
has to diverge for the equivalence check above to be measuring anything. If it does
not, that check is asking nothing.

### The dtype of a reduction — there are four rules (**measured; borch.ts implements part of it**)

One thing was known — that `bool.sum()` is `int64` — and from it, only that there
is a third rule which is neither the "preserve" used by the shape and indexing
operations nor the "promote" used by the value operations. torch was asked about 30
places and the table came out.

| rule | operations | int64 | bool |
|---|---|---|---|
| **accumulating — it promotes bool** | `sum`, `prod`, `cumsum`, `cumprod` | int64 | **int64** |
| **selecting — it keeps the dtype** | `amax`, `amin`, `max`, `min` | int64 | **bool** |
| fixed dtype | `any`, `all` | bool | bool |
| fixed dtype | `argmax`, `argmin`, `nonzero`, `count_nonzero` | int64 | int64 |
| fixed dtype | `logsumexp` | float32 | float32 |
| refused | `mean`, `var`, `std`, `norm` | ✗ | ✗ |
| refused (bool only) | `median`, `argmax`, `argmin` | works | ✗ |

A `dtype=` argument beats all of it.

**The dividing line is a familiar one.** Accumulating **makes** a value (a 3 does
not fit in a boolean cell) and selecting **hands over** a value that was already
there. It is the same line drawn when the dtype labels were fixed — the reductions
were under that rule all along, and rather than "reductions are an exception",
**accumulating and selecting are two different things.**

`mean` being a refusal comes from the same place. A mean is a division, so the
answer does not fit in an integer cell, and torch stops rather than
approximating.

The core (numpy) already gets **24 of those 30 places** right. The six that
diverge are not about the dtype rule but about **whether it refuses** —
`mean`, `median(bool)` and `argmax(bool)` stop in torch and produce a value in the
core, and `logsumexp` gives float64 (a place where numpy's default leaks through).

> **This paragraph said borch.ts's reductions all call `Tensor.make` without a
> dtype and are all float32, and half of that has expired.** Measured in
> `borch-ts/src/tensor.ts` rather than recalled: `Tensor.sum` now reads
> `sum(dtype?: DType)` and its first line is
> `if (dtype !== undefined) return this.castFirst(dtype).sum().to(dtype)`, with the
> same shape on `nansum` and `cumsum`. So `dtype=` **is** accepted and honoured, and
> it converts before accumulating — which is the order that decides the value, since
> folding `[1.7, −2.3, 0.9]` to int64 gives −1 converting first and 0 converting
> last, and torch does the former.
>
> What has not changed is the default path: with no `dtype` it still goes through
> `Tensor.make` unlabelled and comes out float32. So the accurate statement is
> **borch.ts's reductions default to float32, and `dtype=` is honoured on `sum`,
> `nansum` and `cumsum`.**
>
> Marked rather than rewritten, because whether that is now enough to move the
> table into the golden depends on the binding, and nobody has measured that half.
> Translating an expired claim would leave it looking newer and no truer.

Moving the table into the golden and implementing it go in one commit — cases
alone leave it red, and the implementation alone leaves the rule written down
nowhere.

### The constant cache is read-only — caught rather than measured

A one-element tensor is cached by value (`scalarCache` in `tensor.ts`). A training
loop rebuilds the same constants every step, and that much is right — **a constant
is only read.**

The trouble is when something that is not a constant passes through that gate.
Attaching the parameter-group runner immediately produced a WebGPU validation error
in `Adam`:

```
Writable storage buffer binding aliasing found between binding index 2 and 3
```

A size-1 parameter's `m` and `v` were **the same buffer.** Both were built with
`Tensor.zeros([1])`, and that is the global zero constant. Walking the same branch
turned up three more:

| place | what it becomes |
|---|---|
| `Adam`'s m and v (size 1) | the whole command buffer invalid — **it looks like an exception** |
| `SGD`'s momentum buffer (size 1) | overwrites the zero constant the whole program uses |
| `nn.PReLU()` — **its default is one parameter** | the weight *is* the global 0.25 constant, and the optimiser edits it during training |
| `BatchNorm(1)`'s running statistics | overwrites the global 0 and 1 constants |

**Only the first row blows up; the other three are quietly wrong.** Why the golden
missed them is plain — every case supplies its weights from outside, so neither
optimiser state nor running statistics ever pass through.

The fix was to open one more door: `Tensor.owned(shape, value)` skips the cache and
gives its own buffer. **Anything that will be edited in place comes through this
one whatever its value** — parameters, optimiser state, running statistics. Make
the rule about size (be careful when it is 1) and it gets forgotten eventually.

### dtypes

**It starts with float32 alone.** The sister library has three — float32, int64 and
bool — and carrying int64 in a float32 buffer was a workaround chosen because
TF.js's `cast` is broken. Our kernels are under no such compulsion, so it goes in
properly when it is needed — imitating it now inherits the sister's workaround for
no reason.

### Layout

**NCHW.** As in torch. The sister carries NHWC because TF.js's conv is fast only
that way, and our kernels are ours to write.

### autograd

**The same tape structure** as the sister and the core. Three things of one shape
make a fix in one easy to carry to the others, and that has paid off several times
this session.

### Kernel generation

**The WGSL is generated from an operation table.** Writing the name, the forward
expression and the backward expression on one line produces the kernel. It is the
one idea taken from `webgpu-torch`, and it is a good one — several new derivatives
were written by hand this session and each one created a place to be wrong.

### Verification

**It is attached to the golden from the first commit.** The case bodies are not
here, so a case of the same name is written in TS and matched against
`golden.json`'s answer. A case goes in with each operation — the one condition this
repository has held to while growing its surface.

---

## What is not settled

- **The package name and how it is published.** `borch` is in a private repository
  and is not on PyPI yet, and npm needs the same judgement. That is a person's
  decision.
- **How far the API stays torch's.** `x.add_(1)` is awkward in TS and `x.add(1)` is
  natural. Whether the person porting torch code or the person writing TS comes
  first is undecided.
- **Whether the sister library is eventually retired.** Not for now — it is the
  Python user's GPU path.
- **Complex numbers — decided. They go in.** Written out below.

### Complex numbers — they go in. Both things blocking it collapsed

**Two reasons stood written down for a while and both were wrong.**

The first was "the storage is float32 and nothing else". No GPU has a complex type
— CUDA's `cuComplex` is a struct of two floats and **torch's `complex64` is two
float32s too** (measured: 8 bytes per element, and `view_as_real` gives a `(…, 2)`
**view**). It is the same machinery as our `int64` and `bool` being labels over a
float32 buffer, and that machinery already exists.

The second was "it cannot be touched before the autograd convention (Wirtinger) is
measured". That one was a real worry, and **measuring showed no Wirtinger machinery
was needed** — because torch refuses a complex loss outright:

```
(z*z).sum().backward()
  → RuntimeError: grad can be implicitly created only for real scalar outputs
```

If the loss is always real, the convention settles into one — **treat `(re, im)`
as two independent reals, run ordinary real autograd, and bundle them.** Measured
at z = 1+2j:

| loss | `z.grad` | `(∂L/∂re, ∂L/∂im)` |
|---|---|---|
| `(z·z̄).real` | `2+4j` | (2, 4) |
| `abs(z)²` | `2+4j` | (2, 4) |
| `z.real` | `1+0j` | (1, 0) |
| `z.imag` | **`+1j`** | (0, 1) — a conjugate convention would give `-1j` |
| **`(z*z).real`** | **`2-4j`** | (2, −4) |

**The last row is what pins it.** The four above it are compatible with "two
reals" and **cannot be told apart from ordinary complex differentiation** — they
give the same answer or the definition is ambiguous. `(z*z).real` is `a²-b²`, where
the two conventions part (`2-4j` against `2+4j`), and what was measured is the
former.

**One limit remains.** WGSL has no `f64`, so there will never be a `complex128`.
`complex64 + float64` is the only route that produces that combination, so it is
refused loudly there — **the same place and the same shape** as `.double()`, so it
is not a new kind of trap.

#### What else was measured

| | |
|---|---|
| promotion | complex is the top category. Only `complex64 + float64 → complex128` is refused |
| refusals | the ones torch itself blocks — `sign`, `relu`, `max`, `sort`, `floor`, ordered comparisons. `eq` works (there is no order, and there is equality) |
| repr | `tensor([ 1.0000+2.j, -0.5000-1.j])` — it has to be pinned again |
| golden | numpy stores `complex64` in an `.npz` as it is. **The harness needs no change** |
| fft | `rfft(8)` → `(5,)` complex64; `stft(n_fft=4)` → `(3, 9)` |

#### The stages

1. **The core (numpy) alone** — dtypes, construction, `view_as_real/complex`,
   arithmetic, `real/imag/conj`, autograd. numpy already handles `complex64`, so
   **the purpose is verifying the convention.** `tensor.ts` is untouched
2. **borch.ts storage (interleaved) plus kernels** — the most invasive part. **The
   assumption that `size` and the buffer length are 1:1** runs through the code and
   this is the work of pulling it out
3. The binding, the repr, and the golden cases
4. `fft` and `stft` on top of that

Until then `stft` is **a name that refuses.** The real `(…, 2)` path is slated for
removal (measured — `UserWarning: ... will raise an error`), so it would teach a
shape that is about to disappear. Stage 4 produces it properly, in complex.


## The stages

| | contents | what ships | |
|---|---|---|---|
| **T0** | device, buffers, the pipeline cache, `Tensor`, the elementwise operation table, matmul | the golden's elementwise cases pass | ✅ |
| **T1** | the autograd tape, reductions, backward | the `grad::` cases pass | ✅ |
| **T2** | conv, pooling, `nn.Module`, optimisers | MLP training runs | ✅ MLP and CNN both |
| **T3** | ResNet-18 | measure epoch time and accuracy — by the sister's own measure | ✅ ahead on both |

All four stages stand. Golden **845/845 with 0 unwritten**, **30% faster** than the
sister (2.02 → 1.55 minutes per epoch), and **test accuracy 65.5% against 60.4%.**
All measured on a real GPU on the same machine (`apple / metal-3`), and that
condition is recorded with the result.

> Measured again later, the speed held (1.55 minutes per epoch) and **only the
> augmented side beats the sister on accuracy** (64.6% against 60.4%). With
> augmentation off it is 59.3%, below. At the time six of the shortcut layers were
> not being trained and that was acting as regularisation. Both figures and the
> reason are in the T3 accuracy section below. The benchmark's ResNet-18 itself was
> also confirmed to match real torch on the forward pass, the loss and the backward
> pass.

It has reached T2. All 845 golden cases, and on top of them `nn.Module`,
`Sequential`, `Linear`, `ConvND`, `BatchNormND`, `Recurrent` (RNN, LSTM, GRU) and
`MultiheadAttention`, four optimisers (SGD, momentum, Adam, RMSprop), six
schedulers, linear algebra, and `transforms`.

### Why the golden went from 798 to 845 — random numbers do not produce the special values

`relu` passed 798 golden cases unchanged. At an input of **exactly 0** torch gives
a gradient of 0 (`x > 0`, not `x >= 0`) and this flowed 1, and it surfaced while
matching ResNet against real torch (input gradient max diff 1.5e-2).

The cause was not too few relu cases but that **every input was a normal random
number.** A good default that cannot do one thing — exactly 0, two exactly equal
numbers, exactly a boundary, exactly .5 never come up once. Every place a function
kinks is in there.

The `edge::` table collects those inputs. The condition is multiplying **a
different weight per position** when folding — folded uniformly, the difference at
one kinked position is buried in the sum. Five things were caught across the three
libraries, and every one of them has an exactly matching forward pass, so a value
comparison cannot see them.

| what | where |
|---|---|
| `maximum`/`minimum` not splitting the gradient on a tie (torch splits in half) | all three |
| `leakyRelu` (`(1+s)/2` at 0) and `clamp` (half at the boundary), which sit on top of it | borch.ts |
| max pooling ties — torch takes the one position that comes first | core and sister |
| `maxPoolWithArgmax` indices counted without the batch, so **the whole batch lands on the first plane** | sister |
| `topk`/`sort` tie order (reversing an ascending sort reverses the ties too) | core and sister |
| 1-D `max`/`min`/`argmax` giving `(1,)` instead of a scalar | sister |

Pooling ties are not a rare place. **Exactly-zero is everywhere after a ReLU**, so
a window that is entirely zero is a tie every time.

Whether each case actually bites was confirmed by reverting the fix — those cases
alone fail and the rest hold. A case that goes in and bites nothing only raises the
passing count.

### T3 — the sister's baseline (this machine, the same day)

    uv run --with playwright python tests/browser/run.py --lib borch_webgpu --headed --bench

| batch | ms/step | epoch | GPU |
|---|---|---|---|
| 16 | 48.7 | 2.54 min | 136MB |
| 32 | 86.0 | 2.24 min | 181MB |
| 64 | 154.9 | **2.02 min** | 226MB |

**It must not be measured headless.** TF.js then fails to get WebGPU and drops
quietly to WebGL (measured — the runner prints a warning). Those numbers are not
WebGPU's.

### T3 — borch.ts, the same benchmark

    npm run build:ts
    uv run --with playwright python borch-ts/test/bench.py --headed

    adapter: apple / metal-3

| batch | sister (TF.js) | borch.ts | | epoch |
|---|---|---|---|---|
| 16 | 48.7 ms | **38.4 / 40.5 ms** | 27% faster | 2.00 min |
| 32 | 86.0 ms | **62.5 / 63.2 ms** | 38% faster | 1.63 min |
| 64 | 154.9 ms | **118.9 / 119.7 ms** | 30% faster | **1.55 min** |

Measured twice and reproduced. 415 dispatches, 1 submission, 0 leaked. The losses
are 0.0032 / 0.0083 / 0.0100, the same level as the sister's (0.0037 / 0.0059 /
0.0093) — memorising the same batch after seven passes is the correct behaviour,
and that column is what decides whether the training path actually runs.

**The sister's 2.02 minutes per epoch became 1.55.**

### Those numbers stopped reproducing for a while — one kernel was 16× slower

Measuring again later on the same machine with the same command gave **154.6 where
38.1 had been, and 1,921.3 where 118.9 had been.** The losses were unchanged, so the
arithmetic was right and only the speed had gone.

**The shape of the growth** said it was not the machine. Doubling the batch grows
the documented numbers by 1.6–1.9×, and it was growing by 3.5× — a slow machine
would grow all three by the same ratio.

And measuring the individual operations, every one was linear (conv forward and
backward, BatchNorm backward, relu, add), and splitting the same total work by
batch had the larger one come out faster (16×4 = 7.5ms against 64×1 = 6.2ms).
Memory was linear too. **Every part was sound and the sum was 6.5×.**

That is where it stopped. There were **counts** per kernel kind and no **times**,
and the counts stayed at 429 as the batch grew, so they pointed at nothing. So
`timestamp-query` went in (`Device.startProfile()` — turned on, it opens a pass per
dispatch and measures GPU time).

One measurement and it was there: **one `gb` at 94%**, four times larger at twice
the batch. Reading its signature, it was `SliceBackward0`, and
`adaptiveAvgPool(1)` slicing a whole 4×4 was producing **a slice whose input and
output are the same size.**

The cause was `gatherBackward` being `O(input × output)` — for each input element
it walks the whole output looking for the ones that point at it. Both sides are
proportional to the batch, so it squares.

The fix is **tracing back.** When every rule is `lin`, no stride is 0 and the
blocks do not overlap (which is the case for a slice, a transpose, `select` and
`permute`), the output position can be computed from the input position in closed
form — `O(input)`. `expand`, `repeat`, `flip` and `roll` do not meet the condition,
so the walking path stays for them.

After the fix: **38.1 / 62.9 / 119.1 ms** — back down onto the table above. Those
figures were honest, and the run that failed to reproduce them was the
regression.

> One thing this hunt left behind: **wall-clock time alone does not get here.**
> "Which of the 429" was not a question that could be asked, so all the time went
> into measuring the parts one at a time, and that route stops at "every part is
> sound and the sum does not add up". Without an instrument, what accumulates is
> circumstance rather than a cause.

### A correction before that — this document's figures were wrong for a while

    adapter: google / swiftshader     ← headless
    adapter: apple / metal-3          ← --headed

**SwiftShader is a CPU software rasteriser.** The sister's benchmark file says it
has to be run with `--headed`, and the borch.ts benchmark was running headless.
Failing to get an adapter was set up to throw, and **a software adapter is
obtained** — no exception, the wall clock runs, and numbers come out.

So "272× slower than the sister" stood written for a while. That was not a
comparison of libraries but **a comparison of CPU against GPU.** It is the exact
shape of the defect this repository has caught over and over, and this time the one
doing the catching was inside the trap.

Now the benchmark and the golden runner **print the adapter first.** The device
does not change the values, so it makes no difference to whether the golden passes;
left unwritten, the next person measuring performance falls into the same place.

### What was done

The conv the library started with was **one thread per output, a plain loop, no
shared memory.** What `tests/browser/wgsl_conv.js` measured at 72–284% of TF.js was
a **tiled, shared-memory, 4×4 register-blocked** kernel, and that had never been
carried across.

So "our WGSL keeps up with TF.js" earlier in this document was **not a false
statement but a statement about a different thing.** Failing to record that
distinction is this document's fault.

What was carried across, and what went with it:

- **One tiled GEMM skeleton** produces the forward, the input gradient and the
  weight gradient. All three are the same structure with different indexing —
  copied out three times, the day comes when one of them is fixed, and that one
  will be on the gradient side where the value checks cannot see it.
- **Submissions are batched.** A fresh command encoder built and sent per operation
  became one that stacks and sends once, when a value is read. One submission per
  step.
- **`BatchNorm` fusion** — over twenty operations per layer became four kernels.
  The weight and bias gradients are two sums the backward already computes, so they
  are free.
- **Optimiser fusion** — four dispatches per parameter became one.
- **A scalar does not call a kernel.** `x * 0.5` was going through the `fill`
  kernel every time.
- **Buffers are reused by size.** The same shapes repeat every step, so building
  them happens once.
- **`gradWeight` splits its reduction.** It is a GEMM with a small output and a
  large reduction, so depending on the layer it falls to a single workgroup.

**The attributions for the intermediate steps cannot be trusted.** Numbers like
"batching submissions cut 56%" were all measured on SwiftShader, and there is no
reason for the same ratio to hold on a GPU. Only the final figures were measured on
a real adapter, and what contributed how much has to be measured again to be
known.

### What the benchmark caught — this matters more than the speed

Two things came out ahead of the numbers, and both are things **798 golden cases
could not catch.**

1. **Average pooling's backward did not run at all.** `layout: "auto"` drops a
   binding the shader does not read, and the `avg` branch does not look at its
   input. That one surplus buffer made WebGPU invalidate the command buffer — it
   throws no exception — and the gradient simply never appeared. The golden pins
   `adaptive_avg_pool2d` **by value alone**, so it does not ask about this.

2. **There was no weight initialisation at all.** Everything was zero. A network
   starting from zero is symmetric, so the neurons in a layer receive the same
   gradient and move together forever. Every golden case supplies its weights from
   outside, so nobody looks at the initial values.

Both surfaced through one column — "the loss does not move from 2.2685". The wall
clock had been producing perfectly good numbers throughout — **something that
looked like a measurement was coming out.** So the benchmark now produces no
numbers at all if even one validation error occurs.

After the fix the loss falls to **0.0032 / 0.0082 / 0.0098.** That is the same
level as the 0.0037 / 0.0059 / 0.0093 the sister gives on the same benchmark —
memorising the same batch after seven passes is the correct behaviour, and that is
what decides whether the training path actually runs. **The speed was unchanged**
(13,042ms). Which is to say the values do not affect the timing, so the speed
figures above remain valid.

### What was fixed, and what was measured each time

**Nothing was fixed by guessing.** Each step was measured, and what was measured
decided what to fix next.

| what was done | ms/step (batch 16) | dispatches |
|---|---|---|
| at the start | 13,279 | 1,636 |
| porting the tiled conv **forward** | 11,000 (−16%) | 1,636 |
| removing `fill` plus `BatchNorm` fusion | 8,570 (−22%) | **710** |

Why the first port bought only 16% was in the breakdown — **conv was sixty of the
1,636** (3.7%) and the rest was elementwise and reduction plumbing. The kernel
benchmark had been measuring conv, so fixing conv looked like the natural move, and
in an actual step that was not where the time was.

The 286 `fill` dispatches were **calling a kernel to write one scalar.** Every
`x * 0.5` arrived there. It is uploaded without a dispatch and cached by value.

### Where the time was, split by measuring too

    batch 16   8,613.9 ms/step  (forward 1,414.1)

**The backward and the optimiser are 84%.** Which decided what came next — conv's
two backward passes were still plain kernels (walking the batch and the output
positions per weight element: 600 million times in a 512-channel layer), and the
optimiser was four dispatches per parameter (sixty-two parameter tensors → two
hundred and forty).

All three were made to come out of **one** tiled GEMM skeleton. The forward and the
two backwards are the same structure with different indexing — copied out three
times, the day comes when one of them is fixed, and that one will be on the
gradient side where the value checks cannot see it.

### What was measured

    npm run build:ts
    npm run golden:ts

    798 of 798 golden cases written in TS — 0 not yet asked.
    passed 798 / failed 0

Every answer pinned with real torch, matched directly by WGSL in a browser. At the
time that was wider than what the core (746) and the sister (799) saw — the only
one of the three to pass the whole table.

> **This paragraph is a record of the time.** While the table grew from 798 to
> 2,953, borch.ts wrote 2,343 and left 610 unasked, so today **the sister passes
> the whole table and borch.ts is the narrowest of the three.** It is not the number
> that went stale but **the sentence written from it** that reversed — this file
> records a time, so the numbers stay and only the conclusion moves from the
> present tense into the past. **A stale conclusion is worse than a stale number**:
> a number surfaces when it is measured again, and a conclusion is carried away by
> the reader as a present fact.

**Zero failures is not a figure to be trusted on its own, so it was confirmed from
the other direction too.** Four things were broken on purpose and each caught
exactly its own cases.

| what was broken | what caught it |
|---|---|
| 0.001 added to `deg2rad` | `math::deg2rad`, one case |
| the accumulation removed from the scatter | `grad::expand`, `repeat` and `unfold(겹침)` — three, and only where two outputs read the same input |
| conv's stride-divisibility test removed | `grad::conv2d(스트라이드2)/x`, one case — stride 1 always divides evenly and never catches it |
| `roll` detached from the graph | `flow::roll`, one case. **`method::roll` passed unchanged** — right values with the graph cut, the shape the sister went through |

### What the runner prints before the passing count

**The number not asked.** One name written wrongly and that case quietly does not
run while the rest pass, and a green screen at that moment is the worst possible
outcome. A name registered and absent from the golden counts as a failure too.

`borch-ts/test/missing.py` produces the unasked ones by name — a count can be
counted and cannot be navigated.

### Fixed along the way

87 cases were blocked for a while and the cause was **the export** rather than the
implementation. The case functions built their own inputs on the spot with
`np.random.default_rng`, so those inputs never went into `golden.json`, and then an
implementation that is not Python has the expected values and not the inputs. The
inputs moved into `golden_inputs()` — moved without changing a value, and the
harness confirmed that (`golden.py check` named only the newly added keys as
diverged inputs).

This was not borch.ts's problem alone. The golden was made language-neutral so that
"the next implementation does not grow without verification", and in that state the
next implementation would have been blocked at the same 87.

### T3 — accuracy

    uv run --with playwright python borch-ts/test/accuracy.py --epochs 10 --headed

    adapter: apple / metal-3
    10000 training images / 10000 test images, batch 128, 10 epochs

| | sister | borch.ts (then) | borch.ts (now) |
|---|---|---|---|
| augmentation off | | 62.4% (epoch 8) | 59.3% (epoch 8) |
| augmentation on | | **65.5%** (epoch 10) | **64.6%** (epoch 10) |
| as reported | 60.4% | | |

An epoch is 18.5–19.6 seconds (on 10,000 images). Scaled to 50,000 that is 1.57
minutes, which agrees with the 1.55 the step benchmark gave — the two measurements
confirm each other. The figure held when measured again later (18.4–19.7
seconds).

**Augmentation's effect shows up as numbers.** With it off, the training accuracy
reaches 81.5% while the test accuracy turns over at 59.3% — a 22-point gap is
memorisation. With it on, the gap falls to 4 points (68.7% / 64.6%) and at epoch 10
it is **still climbing.** Not raising the training accuracy is what augmentation
is for, and measuring on 10,000 images rather than 50,000 keeping the figures low
is expected too.

#### The numbers being lower now than then — this is not noise

**Running the same condition twice gives ten epochs identical to the decimal
place** (measured). `random.ts`'s stream starts from a fixed place, so a freshly
opened page gives the same initial weights and the same augmentation draws. So the
3.1 points and the 0.9 points are **the result of code changing.**

One change reaches this table and it is a large one — **six shortcut layers were
not being trained at the time.** The benchmark's `Block` held its shortcut in a
plain object where `parameters()` could not see it, so that table was the accuracy
of "a ResNet-18 with six layers frozen". They all learn now.

**More learning producing lower numbers looks strange and is not.** The frozen
layers acted as regularisation, and at 10 epochs over 10,000 images this model wins
by memorising (81.5% training against 59.3% test). More parameters memorise faster
and the test accuracy does not follow by as much — the gap narrowing to 0.9 points
on the augmented side agrees with that explanation.

**One conclusion changes.** It used to read "both conditions beat the sister's
60.4%", and now **only the augmented one does** (64.6% against 60.4%). With
augmentation off it is 59.3%, below. So "ahead" now carries a condition.

### Whether this table's model is the same one as the Python side's also had to be measured

    uv run --with playwright --with numpy --with torch python borch-ts/test/samemodel.py --headed

**The benchmark's ResNet-18 is outside the golden.** It was transcribed by eye
from `tests/browser/bench.py`, so a subtly different block arrangement or BN
placement had no value comparison against it. Different, and both the speed and the
accuracy become **a comparison between two different models.**

So the parameters, the input, the output, the loss and the input gradient are
extracted and matched against real torch. The naming rules differ between the two
languages, so **they are matched by position** — the same structure gives the same
list of shapes in the same order, and a difference parts there first.

The first run had **the structure and the forward pass matching and the input
gradient diverged** (max diff 1.5e-2). Taking pieces off one at a time narrowed it
to `relu` — `step(0.0, x)` is `x >= 0`, so it flows 1 where the input is exactly 0
and torch gives 0.

**798 golden cases did not catch this**, because no relu case had a 0 in its
input. And `grad::BatchNorm2d/x` folds with `sum()`, which makes every upstream
gradient 1, and then **BatchNorm's two backward correction terms cancel exactly** —
that is why its expected value is 4.7e-10, and that case does not ask about the
correction terms at all. So the piecewise comparison folds with **a different
weight per position.**

### The rule this document set for itself

"Measure on the same benchmark, and write down a loss as a loss" is written here,
and that rule caught this document twice — once when it really was behind, and once
when it was **measuring on the wrong device.** The second was the more dangerous.
Being behind is visible in the numbers; measuring wrongly produces numbers that
look perfectly fine.
