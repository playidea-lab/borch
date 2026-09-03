# borch — how far this goes

> The name was `minitorch` first, then `nanotorch`, before arriving here. The
> first two collided with existing projects (`edutorch` did too), because all
> three aimed at the same spot — **"small + educational + torch".** What sets this
> apart from the others is not its size or its purpose but that **it runs in a
> browser**, and the name should say so — limits included.

"Reproduce PyTorch" is not a measurable goal. It is replaced with
**conformance.**

---

## First: why reproduction cannot be the goal

### The denominator is 1,800

| | count |
|---|---|
| `torch.*` | 1,025 |
| `torch.nn.*` | 182 |
| `torch.Tensor` methods | 604 |
| ATen kernels (forward and backward each) | 863 |

What is actually used is **95 of them** — 5%. The other 95% is surface nobody uses
and that can still be wrong.

### Bit equivalence is unreachable

```
summing 100,000 float32s
  numpy  -90.82513427734375
  torch  -90.82499694824219
  diff    1.373e-04          ← 1,000× float32's eps
```

The **order** of summation alone opens that gap. Matching it would mean
reproducing torch's reduction kernel order, and that order changes again with the
SIMD width and the thread count. It is not a goal that can be chased.

### Full reproduction inverts this project's principle

borch's principle is **"an absent feature beats a wrong answer".** Chasing
reproduction shrinks the refusal list and **grows the surface that can be quietly
wrong.** Unless the verification grows in proportion, more coverage means more
danger.

---

## So the goal is this

> **Within the range the curriculum uses: real torch's values, real torch's
> errors, real torch's printed form.**

### Fidelity grades

| grade | meaning | target | at the time (2026-08-14, when this was written) |
|---|---|---|---|
| **T1 values** | values, shapes and gradients equal under `allclose(1e-5)` | **100%** of the supported range | **100%** (132/132) |
| **T2 errors** | the same exception type plus a searchable message | the main errors | **12/12 · 9/9** |
| **T3 printed form** | `print(t)` and `repr` identical | the common ones | **15/15** |
| **T4 bits** | identical bits | **a non-goal** | — |

**Now.** The 132 above was the generated conformance table on the day; the measure
since then is the golden — **4735 golden cases** across three implementations, held to
the count by `tests/test_docs.py` so this sentence cannot go stale the way the column
above did (it sat at 132 for three weeks while the golden passed 4,000).

Writing T4 down as a non-goal is the heart of this document. Left unwritten,
somebody eventually chases it.

### How it is measured

Two layers.

- `tests/test_diff.py` — 76 written by hand. They aim at specific traps
- `tests/conformance.py` — 132 **generated from a table.** Shapes, dtypes and
  arguments multiplied out

The second is the route to growing it. One more row in the table adds dozens of
cases.

```bash
uv run --with numpy --with torch python tests/conformance.py     # the score
uv run --with pytest --with numpy --with torch pytest tests/     # everything
```

---

## The order it happened in

### 1. ~~T2 — match the errors to the real thing~~ · **done**

A message specification was settled — **a Korean explanation plus torch's
canonical English phrase.**

```
행렬곱의 모양이 안 맞습니다 (3x4 @ 3x2) — 앞의 열(4)과 뒤의 행(3)이 같아야 합니다.
(torch: mat1 and mat2 shapes cannot be multiplied (3x4 and 3x2))
```

Korean alone leaves nowhere to search for the answer; copying the English alone
removes the reason this material is in Korean. **Both go in** — the explanation is
read and the English is searched for.

> **This half was reversed later.** The messages are English throughout now, and
> `tests/test_messages.py` holds the rule as "the Python library carries no
> Korean". The shape survived — our sentence plus torch's phrase, through
> `_like_torch(said, torch_phrase)` — and the language of the first half changed.
> The block above is what the message actually looked like at the time and is left
> as it stands; the same call today gives:
>
> ```
> The matmul shapes do not line up (3x4 @ 3x2) — the columns on the left (4) must match the rows on the right (3).
> (torch: mat1 and mat2 shapes cannot be multiplied (3x4 and 3x2))
> ```

And this work turned up **two defects that were not about messages.**

- `.item()` on a three-element tensor was **quietly returning the first value.**
  Precisely the thing this project said it would not do
- Calling `backward()` twice simply passed. torch releases the graph and raises

Both were fixed, along with implementing graph release and `retain_graph=True`.

### 2. ~~T3 — match `repr`~~ · **done**

torch's rules were followed — all-integer values give `1.`, otherwise **four
decimal places**, and a wide range gives exponents. Elements are **right-aligned
to one width** (with negatives in the mix, room appears in front of the
positives), and continuation lines are indented **eight columns**, the width of
`tensor(`. A non-default dtype is attached as a suffix.

And a non-leaf node prints **`grad_fn=<MulBackward0>`** rather than
`requires_grad=True`. That is where a learner sees that a tensor is inside the
graph.

```
>>> torch.tensor([-1.5, 2.0, -0.25])
tensor([-1.5000,  2.0000, -0.2500])
>>> torch.tensor([1.0], requires_grad=True) * 2
tensor([2.], grad_fn=<MulBackward0>)
```

### 3. ~~cover the whole dtype promotion table~~ · **done**

4 dtypes × 4 operations × (tensor, scalar) = **112 cases** were walked, and 54 of
them diverged at first.

One cause accounted for most of them — torch splits by **category** first
(bool < integer < float) and promotes **within that category alone.** A lower
category does not pull a higher one up.

```
float32 + int64   torch: float32     numpy: float64
int64 / int64     torch: float32     numpy: float64
bool - bool       torch: RuntimeError (it points at `~`)
```

Division gives the default floating point dtype even between integers, and
subtraction refuses on booleans. 112/112 at the time.

### 4. ~~view semantics~~ · **done (and this item rested on a false premise)**

This used to say "we copy, and implementing it would be expensive". **That was
written from reading the code without measuring, and it was wrong.** numpy's
reshape, swapaxes and slices already give views and we hold them as they are, so
shared storage had been working from the start.

```python
a = torch.zeros(4); b = a.view(2, 2); b[0, 0] = 9
a          # [9., 0., 0., 0.]  — the same as torch
```

Exactly one thing actually diverged. torch's `view` **refuses a tensor whose
memory order is off** and points at `reshape` — that is where the two differ. It
was matched.

Thirteen storage-sharing facts went into the checker (eight kinds of view, `clone`
independent, `detach` shared, fancy indexing a copy, and the `view`/`reshape`
difference). 13/13.

**The lesson**: writing "this cannot be done" into a roadmap also has to be
measured first. Written unmeasured, it costs hours of planning work that does not
exist.

### 5. ~~`nn.RNN`~~ · **done**

Time is a Python loop. **That slowness is itself the content of chapter 30** —
recurrence cannot be parallelised because the earlier step has to finish before
the later one is visible, and that is why the transformer exists.

`num_layers`, `batch_first`, `nonlinearity`, `bias` and `h_0` all match, and the
parameter names are torch's so the `state_dict` keys match (saving and loading
cross over). The backward passes of `stack` and `cat` were properly fixed along
the way — the RNN stands on them.

### 6. `LSTM`, `GRU` and the transformer encoder · **done**

This goes past what the curriculum asks for. **That is known and it goes in
anyway** — the aim is for this thing to be usable in its own right, and the
decision is recorded.

The lines held while adding them:

- **Parameter names and layout as torch's.** LSTM's gate order is `i, f, g, o`,
  and torch carries the Q, K and V weights bundled into one `in_proj_weight`
  (3E, E). A different order or layout leaves the values plausible and the
  checkpoints incompatible, and that is the quiet kind of wrong
- **In GRU's `n` gate, `r` multiplies the hidden term including its bias.**
  Keeping the bias outside makes it slightly off, and that goes unnoticed
### 7. The decoder and `nn.Transformer` · **done**

The textbook's range stops at the encoder and this was nearly refused on that
basis; it went in. The decoder layer differs from the encoder layer in **one
thing, in the middle** — `multihead_attn` looks at the encoder's output rather
than at itself.

The meaning of the masks was properly fixed along the way. torch's masks come in
two kinds.

- **boolean** — the True positions are masked (-inf)
- **float** — **added** to the scores. The 0/-inf `generate_square_subsequent_mask`
  gives is this

A float mask used to be lumped in as "mask where it is not 0". A causal mask
happens to come out right that way and **a mask that adjusts the weights does
not** — that case is pinned in a test.

### 8. What the integration review turned up · **done**

Code written the way a tutorial writes it was run whole on both sides (MLP
training, a CNN, an LSTM, a transformer, saving and loading). Two of the six
diverged, and **both were the kind the unit comparisons could not catch.**

- **BatchNorm's backward was wrong.** Taking the mean and variance out through
  numpy and using them as constants cut the path x → mean → y. The input gradient
  was off by 1.17 and **the weight gradient never arrived at all** (None). It
  survived a long time because only the forward pass was being compared — the
  worst kind, **"training runs, the loss goes down, and the values differ"**
- **`p.data = ndarray` was being accepted.** torch refuses it. Being more
  permissive is still diverging, and code that ran in the browser breaks on the
  user's own machine

That check went onto five layers — **does the gradient actually arrive at the
weights.** A cut graph shows up as `None`. And that check **caught one more**:
BatchNorm's `running_mean` and `running_var` were not in `state_dict`. Saving and
loading sends **evaluation mode back to the initial values** — training looks fine
and only inference is wrong. It was fixed by bringing in the buffer concept
(`register_buffer`).

The integration scenario stayed as `tests/scenario.py`. Where the unit comparisons
look at one operation at a time, this looks at **the pieces wired together** — all
three defects came out of that.

### 9. What a training loop needs · **done**

It started from "running a training example needs a DataLoader, an optimiser and a
scheduler, surely". **The names were all there and the way to use them differed** —
measuring sixteen things found thirteen divergences.

The largest was `param_groups`. torch's standard path for reading and writing the
learning rate is `opt.param_groups[0]["lr"]`, and the schedulers change it there.
Ours was `opt.lr`, and then **other people's code does not run and other people's
schedulers cannot be used.**

And **my own test was covering that difference.** The StepLR comparison read
`param_groups[0]["lr"]` on torch and `.lr` on nano — reading the two sides
differently is not a check, it is a rationalisation. They read through one helper
now, and it looks at **the whole trajectory rather than one value.**

What was added: `AdamW` and `RMSprop`, six schedulers (`ReduceLROnPlateau`
included), the optimiser `state_dict` (chapter 6's "resume training" hangs on it),
`WeightedRandomSampler` (chapter 5 teaches it and it was missing), `Subset`,
`ConcatDataset`, `Generator` (what pins `random_split`), `sin` and `cos` (chapter
10's positional encoding), `F.mse_loss` and `F.one_hot`.

**The textbook decided the range.** Every name the textbook and the labs mention
was pulled out and only the missing ones filled in; what is left is `amp`,
`backends` and `float16` — all of them the GPU chapter, and all of them deliberate
refusals.

### 10. The wide surface · **done**

Walking the 144 commonly used ones found only **56 of them (38%).** Filling in 88
brought it to 100% — elementwise maths, comparisons, shape manipulation such as
`split`, `chunk`, `gather`, `flip` and `roll`, `topk`, `sort`, `unique` and
`cumsum`, linear algebra, 25 of `nn.functional`, and 15 activation and loss
layers.

**Matching the names was not the end of it.** Everything present was compared by
value and three diverged.

- `median` — with an even element count torch gives **the smaller of the middle
  two.** numpy takes their mean
- `cumsum` and `cumprod` — torch makes `dim` **required.** Given a default,
  other people's code runs differently

The ones with no defined derivative (`sign`, `floor`, `ceil`, `round`) get a
gradient of 0 — torch does the same, because a step function's derivative is 0
almost everywhere.

### 11. The review — the blind spot after widening · **done**

The review immediately after adding the 88. **Right values with the gradient
unexamined** turned up again.

- `topk` and `sort` **were cutting the graph.** Handing back the values alone
  meant no gradient reached the positions taken — in top-k sampling or in a loss
  with a sort in it, **training quietly stops**
- The `requires_grad` of a **leaf** made inside `no_grad` was being turned off.
  torch leaves it on. It is the kind of difference where a parameter made inside
  that block drops out of training

Measuring the coverage found `BatchNorm1d`, `AdaptiveAvgPool2d`, four activation
layers, `ConcatDataset` and `WeightedRandomSampler` **never once walked.** They
all passed, and **there was no evidence they were right beyond there being no
check** — BatchNorm2d had been wrong that long for that reason.

And the questions were changed from "does it run" to "is it right".
`WeightedRandomSampler` used to be checked on length alone and is now checked on
**whether the heavier weights really are drawn more often.**

---

## The roadmap is empty

All the items are closed. What to keep in view when deciding what comes next:

- **What the curriculum does not ask for does not go in.** More surface means more
  places to be quietly wrong
- **Measure, then write.** Item 4 said "this cannot be done" and it was already
  working
- The natural candidates left are not things like `torch.compile` but **what a
  learner actually meets** — the rest of `nn.functional`, `Tensor.scatter_` and
  `gather`, a few more optimisers

---

## What will not be done

| | why |
|---|---|
| **CUDA, distributed, mixed precision** | not in a browser. Imitating them loses the lesson |
| **shipping pre-trained weights from here** | the bytes are somebody else's to host — `borch-hub` fetches what a manifest points at and re-hashes it, and this repository carries none of them |
| **JIT and `torch.compile`** | out of range |
| **speed** | see below |

**That row used to read "pre-trained weights", with nothing narrowing it, and it
had stopped being true.** `borch-hub` has `fetchWeights` and `load`: it resolves
the URL a manifest carries, fetches the bytes, **re-hashes them even out of the
cache**, and throws when the hash disagrees. The landing page's state table lists
that as `partial` — built, not on npm — and says weights do load in a browser,
measured, `access-control-allow-origin: *` on a 44.7MB safetensors.

So this file was promising never to do a thing the front page advertises, and the
old reason — *fetching them is itself the thing to learn* — had become an argument
against a fetcher that exists. What survives is the narrow half, and it is the
half that was always the point: **no weights are carried here.** The row says that
instead of the thing it can no longer say.

It was found by reading the two documents against each other. Nothing checks them,
which is why the row could sit there through the commit that took the same name out
of the landing page's `never` list.

### Speed — measured

This first said "hundreds of times slower than native torch", and **measuring
showed that wrong.** Matrix multiplication calls BLAS on both sides, and on small
tensors torch's dispatcher overhead is the larger cost, so borch is the faster
one.

| | torch (CPU) | borch (native) | borch (browser) |
|---|---|---|---|
| matmul 512² | 0.28ms | 0.13ms | 82.4ms |
| 50 small operations | 0.15ms | 0.17ms | 0.74ms |
| one MLP training step | 0.36ms | 0.18ms | 3.34ms |
| conv2d forward | 0.22ms | 0.44ms | 1.88ms |
| conv2d backward | 3.97ms | 1.11ms | 6.49ms |
| **one MNIST CNN training batch** | — | — | **65.7ms** |

**One MNIST CNN epoch is about two minutes in a browser.** Training really
happens.

What makes it slow is wasm rather than the implementation. Pyodide's BLAS is a
wasm build and cannot use SIMD or multiple threads, so **large matrix
multiplication alone is unusually bad** (294×). At the sizes the exercises
actually use it is 5–10×, and that is not felt.

The boundary is **"up to MNIST scale".** CIFAR and ResNet scale belong on your own
machine or on remote hardware, and that is itself what chapter 8 (GPUs) teaches.

### Why a fast runtime is not the goal

Making it fast would mean rewriting in Rust or C++ and compiling to wasm. At that
moment it loses being **a readable educational implementation** — which is the
whole of this project.

And that spot is already occupied by ONNX Runtime Web and WebGPU (inference only).
If training really has to run in a browser, the answer is remote hardware rather
than borch.

**You cannot have both.** Either a readable implementation or a fast runtime.
borch is the former.

---

## ADR-001: the ceiling is raised by a sister library

- **status**: accepted
- **context**

  With all eleven roadmap items closed, the next move had to be decided. The
  requirement is **running CIFAR and ResNet scale training in a browser**, and
  that collides head-on with the section just above, which writes down "the
  boundary is up to MNIST scale".

  Measuring gives the factor required. The effective throughput now, worked back
  from two minutes per MNIST CNN epoch, is **about 3 GFLOPS.** CIFAR-10 ResNet-18
  is about 1.7 GFLOPs per image per training step, and at 50,000 images that is
  **84 TFLOPs per epoch** — **7.7 hours** at this speed. Bringing an epoch down to
  a few minutes needs **300×.**

  | | one epoch |
  |---|---|
  | now (3 GFLOPS) | 7.7 hours |
  | wasm SIMD (×4) | 1.9 hours |
  | a naive WebGPU shader | ~14 minutes |
  | tuned WebGPU (~1 TFLOPS) | ~1.4 minutes |

  wasm SIMD is 3–5×. **It covers 1% of the 300× and stops there.** Which is to say
  this requirement is unreachable by any route without a GPU.

- **decision**

  The core `borch` **keeps "up to MNIST scale" and the pure-Python 17KB wheel as
  they are.** Raising the ceiling is **a separate distribution's** job
  (`borch-webgpu`). The design is in [WEBGPU-DESIGN.md](WEBGPU-DESIGN.md).

- **rationale**

  - **A wheel's properties are contagious.** A backend in the core stops it being
    `py3-none-any`, and the present verification route — comparing against real
    torch natively — becomes architecture-dependent first
  - **The failure modes move upstream.** WebGPU breaks per browser and per driver.
    In the core, "a learner who came to practise syntax fails at the import
    because of a driver" becomes possible
  - **The harness goes from two-way to three-way.** Laying that load on a core
    holding T1 at 100% breaks the 100% first
  - **A promise breaks.** What the core sells is "the same code with one import
    changed". A `device` concept and asynchronous reads are not compatible with
    that promise

- **alternatives**

  | what was considered | why it was not chosen |
  |---|---|
  | put the backend in the core | the four rationale points above |
  | port libtorch to wasm | person-years buys 1,800 names of surface and **buys no speed.** The bottleneck is wasm rather than the library, so moving it where there is no MKL or FBGEMM does not make it faster |
  | leave things as they are | the ceiling does not rise. It does not answer the requirement |

- **consequences**

  - The "what will not be done · speed" section just above is now **a statement
    about the core alone**
  - **A `device` concept, an asynchronous API or GPU code entering the core is a
    violation of this ADR.** Left unwritten, somebody eventually does it — which is
    why this document exists
  - The core's next tasks are unchanged: widening the conformance table, the
    coverage blind spots, and shipping (making the repository public)
