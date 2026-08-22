# borch

**PyTorch that runs in a browser.** A thin layer over numpy, for practising
PyTorch syntax with nothing installed.

> ## First, what this is **not**
>
> It is not PyTorch. **11%** of the surface, and **0.1%** by code.
> `CUDA`, distributed training, mixed precision, `torch.compile` and pre-trained
> weights are **never coming** — they either cannot exist in a browser, or
> learning them means leaving the browser.
>
> There is one claim to make. **Introductory tutorial code produces the same
> values with one import changed.** That was measured (see conformance, below).
>
> Unrelated to the other projects with similar names —
> [minitorch](https://minitorch.github.io) · [nanotorch](https://pypi.org/project/nanotorch/) ·
> [edutorch](https://pypi.org/project/edutorch/).

## Three things carry this name

All three look at the same golden — the expected values pinned with real PyTorch.
**One table is what makes a divergence visible** — defects caught by comparing the
three against each other are a large share of this repository's history.

| | built on | where it runs | ceiling |
|---|---|---|---|
| **`borch`** (PyPI) | numpy | anywhere, and Pyodide | MNIST scale |
| **`borch`** (npm) — borch.ts | **WGSL directly, zero dependencies** | in a browser only | CIFAR ResNet-18, **1.5 min/epoch** |
| **`borch-webgpu`** (Python) | the borch.ts above | in a browser only | the same thing at **1.6 min/epoch** |

The lower two **stand on the same kernels.** `borch-webgpu` is a 7,863 line
binding calling borch.ts from Python, and the difference (1.5 against 1.6 minutes)
is the cost of one trip through Pyodide.

> **A TF.js version stood there for a while.** The same name, a different
> foundation, **5,307 lines.** Hand-written WGSL was 20% faster on the same
> benchmark (154.7ms → 123.4ms at batch 64) and carried no rank limit, so it was
> removed. That decision and its measurements are in
> [BORCH-TS.md](BORCH-TS.md).

What follows is the Python side. The TypeScript side is in the
[borch.ts](#borchts--typescript-and-wgsl) section.

## See it first — the explanatory pages and the playground

There are two static pages under `site/`. The hero's code runs where it stands,
and the playground takes **both JavaScript and Python** — on the same WGSL
kernels, the loss values agree to the last digit.

```bash
npm run build:ts && npm run site      # http://127.0.0.1:8123/site/
```

No request leaves the machine. Pyodide and numpy come out of the repository
itself (`vendor/`). English is the default and Korean is under `/site/ko/`. The
details are in [site/README.md](site/README.md).

## Installing

One pure-Python wheel. numpy is the only dependency, and Pyodide already has
numpy. It contains the `borch` package and the `borchvision` module.

```bash
uv add ./pyborch-1.4.0-py3-none-any.whl        # the file from a release
```

In a browser (Pyodide), the wheel's bytes are written into the virtual filesystem
and installed with `micropip`.

```js
// The filename has to be kept — micropip reads the package name and version from it.
py.FS.writeFile("/pyborch-1.4.0-py3-none-any.whl", new Uint8Array(wheelBytes));
await py.runPythonAsync(`
import micropip
await micropip.install("emfs:/pyborch-1.4.0-py3-none-any.whl")
`);
```

> **The repository is private, so a release URL cannot go straight into
> `micropip.install()`.** An anonymous request gets a 404 — found out by trying it.
> Once it is public, one URL is the whole of it.

```python
import borch as torch

w = torch.tensor(3.0, requires_grad=True)
loss = (w - 5.0) ** 2
loss.backward()
print(w.grad.item())               # -4.0
```

### How to name it — three ways, and the boundary was measured

`import borch as torch` creates **one name inside that file.**
`from X.Y import Z` looks at a **path** registered in `sys.modules`, which an
alias does not reach. Where exactly that difference bites is pinned by
`tests/test_alias.py`.

| | `torch.nn.Linear` | `from borch.nn import Linear` | `from torch.… import` | somebody else's `import torch` |
|---|---|---|---|---|
| `import borch as torch` | ✅ | ❌ | ❌ | untouched |
| `borch.install("borch")` | ✅ | ✅ | ❌ | untouched |
| `sys.modules["torch"] = borch` | ✅ | ✅ | ✅ | **intercepted** |

**The first line is the default.** Most textbook code reaches things as attributes,
as in `torch.nn.Linear`, and an alias alone covers that.

If `from … import` is needed, **plant it under its own name** —
`borch.install("borch")`. The submodule paths open and nobody else's code is
touched.

Planting it as `torch` is the last resort. After that, **`import torch` inside
somebody else's library** gets the subset too. In one learner's practice
environment that is a convenience; where other code is mixed in it becomes an
error whose cause cannot be found.

```python
import sys, borch
sys.modules["torch"] = borch          # only when it is really needed
```

---

## Why this exists

PyTorch is not ported to WebAssembly. Hundreds of MB of native code, hand-tuned
AVX and NEON kernels that do not carry over to wasm SIMD, and OpenMP threads that
want headers Pyodide does not ship.

**And none of that is needed to learn the syntax.** numpy is already inside
Pyodide.

It is closer to a flight simulator — **the controls are real controls and the
physics underneath is the imitation.** The code a learner types is real PyTorch
code, and it runs unchanged on their own machine.

## The design principle — an absent feature beats a wrong answer

An imitation that is *almost* the same as the real thing teaches whoever learns
from it something false. So **anything outside the range throws rather than
approximating.**

```python
>>> torch.tensor([-1.5, 2.0, -0.25])
tensor([-1.5000,  2.0000, -0.2500])          # the real thing's places and alignment

>>> torch.tensor([1.0], requires_grad=True) * 2
tensor([2.], grad_fn=<MulBackward0>)

>>> torch.randn(3, 4) @ torch.randn(3, 2)
RuntimeError: The matmul shapes do not line up (3x4 @ 3x2) — the columns on the left (4) must match the rows on the right (3).
(torch: mat1 and mat2 shapes cannot be multiplied (3x4 and 3x2))

>>> torch.tensor([1.0]).half()
BorchError: `.half()`(float16) is not in the browser subset.
Use real PyTorch on your own machine (`uv add torch`) — this subset is for practising the syntax, and imitating what is missing teaches the wrong thing.
```

It stops loudly rather than quietly producing a different value.

> The refusal above used to be `nn.LSTM`, and **that stopped being true** —
> `LSTM`, `GRU` and the transformer encoder went in under roadmap item 6, so the
> example was demonstrating a refusal that no longer happens. Replaced with one
> measured just now. This block shows what you get today, unlike the record in
> `ROADMAP.md`, which shows what you got then and is left alone for that reason.

**There is exactly one exception.** `svd_lowrank` and `pca_lowrank` accept
`niter` and do not use it. torch refines a random subspace `niter` times, and that
loop exists because the projection is an approximation. This computes the full SVD
and trims the front, so there is nothing to refine — which means **the values part
from torch's at small `niter`.** The divergence is towards the more accurate side:
ours sits at the limit torch reaches by raising `niter`. It does not stop, because
stopping here would mean refusing an exact answer; it is written down, because **a
divergence is a divergence even when it is more accurate.**

## The file layout — it is not one file

`borch` is **a package.** It began as one file, and that was part of a
distribution story about dropping it in and being done; at 3,300 lines that
justification went first.

| | lines | what |
|---|---|---|
| `_base.py` | 323 | dtypes, the error specification, `repr` |
| `_tensor.py` | 1,711 | `Tensor` and autograd |
| `_ops.py` | 7948 | maths, shapes, `nn.functional` |
| `_nn.py` | 2,881 | layers, recurrence, transformers |
| `_optim.py` | 1,000 | optimisers and schedulers |
| `_fft.py` | 364 | `torch.fft`, `stft`/`istft` |
| `_data.py` | 355 | `Dataset` and `DataLoader` |
| `_rnn.py` | 65 | `nn.utils.rnn` |
| `__init__.py` | 340 | where everything is gathered and planted as `torch` |

**The public names did not change** — 197 of them, the same before and after the
split. `import borch` gives the same thing. `borch_webgpu` has the same shape:
7,863 lines across seven files.

It was not moved by hand. Only the cut points were chosen and a script did the
rest — a person cutting and pasting a file that size quietly loses a line, and
nobody knows until the golden catches it.

## How it is guaranteed

`tests/test_diff.py` **puts the same operation into real torch and into borch and
compares the numbers.** 180 pytest cases, **93% code coverage.**

```bash
uv run --with pytest --with numpy --with torch pytest tests/
```

> **Code coverage cannot be measured on the GPU side.** It runs in a browser
> alone, so `pytest --cov` does not reach it. All that can be said about that side
> is that **3147 golden cases pass**, and that is a surface check rather than a
> line check. The two numbers are not written down as though they were the same
> thing.

What this check caught on its first run: PyTorch's `BatchNorm2d` **uses two
different variances inside one forward pass** — the biased one (ddof=0) for the
normalisation and the unbiased one (ddof=1) for updating `running_var`. Biased in
both places, the output is off by 2.6%. Not the kind of thing that comes out of
reading the code.

### Seven places where green can be a lie

The larger the table grows, the less "N cases pass" is worth. **What is not asked
is not right** is the one rule this repository has repeated, and each of the
places below **actually bit.** They are where to look first when writing a new
case or chasing a defect.

**Getting onto this list requires being able to name the case that was wrong.**
All seven can — `norm(p)`, `nn.Softmax()`, `edge::grad::max(동점)`. An item that
cannot name one is a hunch rather than a lesson, and enough of those turn this
section into a bin. Recording the number of times does the same job — it is what
tells a reader how much to trust the entry.

**One computation under two names can be right on one side only** (5 times).
`norm(p)` was wrong while `linalg.vector_norm(p)` was right; `searchsorted(side=)`
was discarded while `bucketize(right=)` was right; `F.upsample_bilinear` worked
while `Upsample(mode='bilinear')` was quietly nearest-neighbour; and `max_pool2d`
and `max_pool2d_with_indices` were the same story. Ask only the right one and the
wrong one may as well not be in the table.

**The two names can differ down to the argument order**, which is the vicious
end of that branch. `torch.polygamma(n, x)` against `x.polygamma(n)`, and
`torch.lu_solve(b, LU, piv)` against `b.lu_solve(LU, piv)`. The first was wired up
mechanically from a table and caught by a `TypeError` — and **that was luck.** The
arguments happened to be an integer and a tensor, so the types differed; had they
been the same type, a number computed in the reversed order would have come out
and a case would have pinned it.

**Symmetric input hides the defects that break symmetry** (several times). With
all values distinct, three different folding rules give the same answer; with the
window equal to the stride, dropping the stride gives the same answer; and a
diagonal matrix whose upper and lower triangles match never asks about `upper`.
**Choosing the input is half of writing the case.**

**An argument with a domain has to be tried across all of it** (twice). When an
`else` wears one value's name and swallows the rest — `if p == 1: … else: L2` —
shaking the argument does change the answer, so a check for "the argument is used"
passes. The changed answer is simply wrong.

**Where NaN is possible, a mask selects rather than multiplies** (3 times).
`0 × NaN = NaN`, so multiplying by a mask contaminates the very positions it
filtered out. Use `where`. It bit in the core's `median`, in borch.ts's `median`,
and once more in `nanmedian`.

**When what a comment says and what the code asks differ, that comment does not
protect the next person** (once). `values_of` said "it asks whether it is a tensor
first" and actually asked with `hasattr(got, "numpy")`. Our pair forwards an
unfound name as a value and so answers true, while torch's namedtuple does not —
so **the same helper was taking a different branch per library.** It stayed green
in that state for a long time, because nobody had reason to pull `.values` off
that tensor, and building the sparse `Tensor.values` blew up
`reduce::grad::cummax`. A comment is not a check; it is **a record of intent.**
Keeping the intent means making the code ask for it.

**The three compare against each other, so one copying another's hole leaves the
table green** (once, and the most dangerous of them). borch.ts's `i0` left the
gradient at 0, and its comment said **"matched, because the core cuts the graph
too".** The two agreed and both diverged from torch. And it got worse in the
copying — a cut graph stops and a `0` does not. **A gradient of zero and no
gradient are different statements.**

**Machinery added to stop one blindfold creates a blindfold in another shape**
(once, and different in kind from the five above — those are mistakes by whoever
writes a case, and this one is **the checking machinery's own**). Folding a
gradient case with a plain `sum()` makes everything upstream 1, so nothing records
which position failed to move. So a different weight is multiplied in per
position — and building those weights with `arange` makes **the first share 0.**
A case whose output is one cell has that one share as the whole of it, so the
gradient becomes 0 throughout and **an implementation that flows no gradient at
all passes.** `edge::grad::max(동점)` was pinning `[0,0,0,0]` instead of
`[0,1,0,0]`.

Alongside, three more places where the check itself took a side: **an ordinary
tensor has `.values` and `.indices` too** (for the sparse layout). Without asking
whether it is a tuple first, it picks up the tensor's first element, and then a
heap of defects that do not exist appear. Those three and the `arange` above point
opposite ways — **that one shows defects that are not there and this one hides
defects that are.**

And one place has **the opposite symptom** from the five. The five are the side
where a defect is invisible; this one is where **the red carries the wrong name** —
a case touching shared state turns the cases behind it red instead. While
`tensor()` was not taking a copy, one case raised its input array by 1, and
because torch does take a copy and did not leak, sixteen places were wrong **in
the core alone.** Run one at a time they all passed, so the cause was not at its
own address. The remedy differs too — not more cases but **isolation.**

---

## How fast it is

Measured in a browser (Pyodide).

| | time |
|---|---|
| one MLP training step (256×64) | 3.3ms |
| conv2d forward (32×1×28²) | 1.9ms |
| **one MNIST CNN training epoch** | **about 2 minutes** |

Natively it is comparable to torch or faster — both call BLAS, and on small
tensors torch's dispatcher overhead is the larger cost. What makes it slow is
wasm, and within that, large matrix multiplication alone is unusually bad
(Pyodide's BLAS cannot use SIMD or multiple threads).

**Up to MNIST scale, training really happens in a browser.** Above that it is your
own machine or remote hardware — or the GPU distribution below.

## If you need more than that — `borch-webgpu`

This one (the core) is **up to MNIST scale, on numpy.** Crossing that boundary is
a separate distribution's job.

| | the core `borch` | `borch-webgpu` |
|---|---|---|
| built on | numpy | borch.ts (hand-written WGSL) |
| wheel | pure Python, 42KB | not in a wheel (browser-only; the page loads borch.ts) |
| where | anywhere | **in a browser only** |
| ceiling | MNIST scale | **CIFAR ResNet-18 at about 1.6 min/epoch** (measured) |
| readable | that is the whole of it | no. performance is the point |
| in-place | `x.add_(1)`, **and propagation through a view** | `x.add_(1)` works and **propagation through a view is refused** |

### So what percentage

The ceiling had only ever been described **by speed.** "Two minutes an epoch" is
true, and "so what percentage" went unasked for a long time. What was measured is
below — CIFAR-10 with 10,000 training images, 10,000 test images **not used in
training**, ResNet-18, 10 epochs, batch 128.

> **This table was measured on the TF.js foundation.** The package of the same
> name has since moved onto borch.ts and it has not been measured again. The golden
> gave the same values on both foundations at the time, so there is no reason to
> expect a large change — and **what has not been measured is not written down as
> though it had been.**

| after 10 epochs | no augmentation | with augmentation |
|---|---|---|
| training accuracy | 80.9% | 64.8% |
| **test accuracy** | 59.9% | **60.4%** |
| **the gap (overfitting)** | **+21.0%** | **+4.3%** |
| best test | 61.4% (8 epochs) | 62.2% (9 epochs) |

**Looking at the training accuracy alone would have concluded that augmentation
hurts** (80.9% → 64.8%). Looking at the test accuracy alone, they are near enough
identical (59.9% against 60.4%) to conclude that nothing happened. What only
appears when both are read together is that **the overfitting fell from 21.0% to
4.3%**, and that is precisely the job augmentation is there to do. Without it, the
test accuracy turns over and falls after 8 epochs.

The two conditions are run **each on a fresh page.** Run one after the other in one
session, the second model starts after the generator has advanced, so its initial
weights differ — and the thing being measured is the effect of augmentation, with a
difference in initialisation mixed into it.

```bash
# cifar-batch1.bin (training) and cifar-batch-test.bin (test) have to be at the repository root.
# The originals cannot be fetched because of CORS, so they are put there by hand — see the transforms section below.
uv run --with playwright python tests/browser/run.py \
    --lib borch_webgpu --headed --accuracy --epochs 10 --augment off
uv run --with playwright python tests/browser/run.py \
    --lib borch_webgpu --headed --accuracy --epochs 10 --augment on
```

> These are numbers from **10 epochs over 10,000 images.** They are not numbers
> for the whole of CIFAR (50,000) or for a longer run, and they are not for
> comparing against published ResNet-18 figures. What they are here to say is not
> the absolute number but that **there is now a place to measure**, and that
> augmentation does actually work.

**It does not replace the core.** Why the two were not merged is written in
[ADR-001 in the ROADMAP](ROADMAP.md) — in short, a wheel's properties are
contagious, browser and driver failures rise up to the `import`, and the promise
of "the same code with one import changed" is not compatible with `device` and
asynchrony.

The design and the measurements are in [WEBGPU-DESIGN.md](WEBGPU-DESIGN.md).

## The supported range

| | |
|---|---|
| **tensors** | shapes, broadcasting, dtype promotion, indexing, reshape/view/permute/squeeze, split, chunk, flip, roll, gather, narrow, index_select, masked_select |
| **autograd** | `requires_grad`, `backward()`, `.grad`, `no_grad()`, `detach()`, accumulation |
| **reductions** | `sum`, `mean`, `max`, `min`, `prod`, `median`, `norm`, `cumsum`, `topk`, `sort`, `unique`, `std` — backward included |
| **nn** | `Module`, `Linear`, `Conv1d/2d/3d`, `MaxPool1d/2d/3d`, `Upsample`, `Embedding`, `LayerNorm`, `BatchNorm1d/2d/3d`, `Dropout`, `Sequential`, `ModuleList` |
| **recurrence** | `RNN`, `LSTM`, `GRU` — multi-layer, `batch_first`, an initial state. **The top-level function forms too** (`torch.lstm`, `lstm_cell` and six others) — they take the weights as a list. Bidirectionality and inter-layer dropout are refused |
| **transformers** | `MultiheadAttention`, encoder and decoder layers, `nn.Transformer` — boolean and float masks, `norm_first`, gelu |
| **losses** | `MSELoss`, `L1Loss`, `SmoothL1Loss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `NLLLoss` |
| **optim** | `SGD` (momentum, weight_decay), `Adam`, `AdamW`, `RMSprop` — `param_groups`, `state_dict` |
| **schedulers** | `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `LambdaLR`, `ReduceLROnPlateau` |
| **data** | `Dataset`, `TensorDataset`, `Subset`, `ConcatDataset`, `DataLoader`, `WeightedRandomSampler`, `random_split(generator=)`, `collate_fn` |
| **saving** | `state_dict`, `load_state_dict`, `save`/`load`, buffers (`running_mean` and the like) included |
| **nn.functional** | 25 of them — activations, losses, `pad`, `normalize`, `cosine_similarity`, `one_hot`, `layer_norm`, `embedding` |
| **complex** | `complex64` only — `complex`, `polar`, `view_as_real`/`view_as_complex`, `real`/`imag`/`conj`/`angle`/`abs`, arithmetic, autograd. **In all three** (below) |
| **Fourier** | `fft.fft`/`ifft`/`rfft`/`irfft`, `fftfreq`/`rfftfreq`/`fftshift`/`ifftshift`, `stft`/`istft` — `n`, `dim`, `norm` and the backward. **In all three** |

### Complex numbers — `complex64` only

There will **never** be a `complex128`. WGSL has no `f64`, so the half that runs in
a browser cannot carry it. The name is kept, and where promotion tries to produce
it — `complex64 + float64` — it **stops rather than quietly settling for less.**
torch gives `complex128` there. It is this repository's usual choice: standing
still here beats a value that is half right.

The gradient convention was pinned by measurement: **torch refuses `backward()` on
a complex loss.** If the loss is always real, the convention is settled —

    z.grad = ∂L/∂re + i·∂L/∂im

On top of that, **a holomorphic function's backward takes a conjugate**
(multiplication and division: `conj(f'(z))·g`), and `abs`, which produces a real,
does not (`z/|z|`). Real input alone never shows the difference — the conjugate is
the identity over the reals. So the golden asks about all three in one table.

**`conj` diverges from torch's.** torch's `conj` is **lazy** — it raises the
conjugate bit and does not flip the values. So `torch.is_conj(torch.conj(z))` is
`True`, and `view_as_real` refuses, calling it an unresolved conjugate. This one
flips immediately, so that state does not exist at all and **`is_conj` is always
`False`.** The values agree — asked through `conj_physical`, both sides give the
same answer.

**borch.ts stores them interleaved** — one buffer of `[re, im, re, im, …]`. That
makes `view_as_real` and `view_as_complex` **real views** (as they are in torch),
and in exchange the old invariant `size = buffer length` becomes
`buffer length = size × 2`. A kernel that does not know this reads a complex
buffer, sees the first half as reals, and produces a wrong answer **with no
exception** — so **the default is refusal**, and only the operations that know
about complex numbers pass through their own gate. Attaching the `complex64` label
alone through `Tensor.from`, relabelling through `to`, and saving a checkpoint are
blocked for the same reason.

### Fourier — `torch.fft` and `stft`

**It stands on complex numbers.** `stft` was a refusal for a long time and the
refusal said "the complex convention has not been settled". **Because that reason
was precise**, the door opened on the day the convention was settled — written as
"there is no storage", nobody would have asked again after the storage arrived.

`stft` is **an assembly** rather than a new kernel — slice, multiply by the window,
`rfft`. All three are already differentiable names, so **the gradient comes out
right on its own.** Writing the kernel by hand, the forward comes out right
quickly and the backward has to travel through the window and the overlap; getting
it wrong leaves plausible values and training that does not train.

The browser side **runs the DFT directly — O(n²).** Cooley-Tukey is fast at
powers of two alone and other lengths need Bluestein separately, and **the values
are the same either way**; at this project's ceiling the difference is invisible.
The day speed is needed it can change, and until then speed that does not exist is
not written down as though it did. The twiddle factors are **built on the host in
double precision and uploaded** — a shader's `cos` and `sin` have
implementation-defined accuracy, and one rectangular-window `stft` really did fall
outside the golden at a relative error of 2.7e-4 (a size f32 rounding does not
explain).

The hard part of the gradient is not a value but **which half gets counted.**
`rfft`'s backward receives gradient on the stored half alone, so it does **not**
add the conjugate partner (adding it doubles); and `irfft`'s revived conjugate
partners came from the same cells, so it counts **the edges once and the middle
twice.** Both are places that can be wrong while the forward values stay sound.

> **`abs`'s knife edge.** There is a reason the golden's `stft` gradient case uses
> uneven numbers. A ramp signal makes the Nyquist bin **exactly 0**, and there
> `abs` is not differentiable and the sign depends on rounding — accumulating in
> float64 chose +1 and torch's float32 FFT chose 0. The rules did not diverge;
> **the case was standing on the knife edge**, and pinning one of those mounts a
> floating-point accident as the specification.

**`print` is part of the specification too.** Printing a complex number, torch
**measures the real and imaginary parts separately** — in `[1+2j, -0.5-1j]` the
real part demands four decimal places and the imaginary part is integral, giving
`1.0000+2.j`. Measured under one format it comes out `1.0000+2.0000j`, with every
value right and the characters diverged. The padding applies to the real part
alone, so a **negative zero** survives, as in `1.-0.j` — and that one sign caught
this binding losing `-0.0` on its read path. `-0.0 == 0.0`, so a value comparison
would never have caught it.

## torchvision — `transforms` only (`borchvision`)

**The first ten lines of an introductory PyTorch tutorial are torchvision.**

```python
datasets.MNIST(root, transform=transforms.ToTensor())
```

The promise of "the same values with one import changed" catches here first, so
`transforms` exists. It is a separate file because it is
`torchvision.transforms` rather than `torch.transforms` — put inside the core it
would create **a place real torch does not have.**

```python
import borchvision as torchvision
from borchvision import transforms
```

| what is here | **38 of the 41 names `torchvision.transforms` carries — everything except the three that carry a reason.** Composition — `Compose`, `Lambda`, `RandomApply`, `RandomChoice`, `RandomOrder`. Tensors — `ToTensor`, `Normalize`, `LinearTransformation`. Geometry — `Resize`, `CenterCrop`, `RandomCrop`, `RandomResizedCrop`, `FiveCrop`, `TenCrop`, `Pad`, `InterpolationMode`, `RandomRotation`, `RandomAffine`, `RandomPerspective`, `ElasticTransform`, `GaussianBlur`. Policies — `AutoAugment`, `AutoAugmentPolicy`, `RandAugment`, `TrivialAugmentWide`, `AugMix`. Augmentation — `RandomHorizontalFlip`, `RandomVerticalFlip`, `Grayscale`, `RandomGrayscale`, `RandomErasing`, `ColorJitter`, `RandomInvert`, `RandomPosterize`, `RandomSolarize`, `RandomAutocontrast`, `RandomEqualize`, `RandomAdjustSharpness`. Plus `augment_batch`, which torchvision does not have. **`transforms.functional` holds 34 of the 37 names `torchvision.transforms.functional` carries** — `crop`, `center_crop`, `resized_crop`, `five_crop`, `ten_crop`, `pad`, `resize`, `hflip`, `vflip`, `rgb_to_grayscale`, `to_grayscale`, `normalize`, `to_tensor`, `erase`, `get_dimensions`, `get_image_size`, `get_image_num_channels`, `InterpolationMode`, and the photometric `adjust_brightness`, `adjust_contrast`, `adjust_saturation`, `adjust_hue`, `adjust_gamma`, and the pixel rewrites `invert`, `posterize`, `solarize`, `autocontrast`, `equalize`, `adjust_sharpness`, and the grid resampling `rotate`, `affine`, `perspective`, `elastic_transform`, `gaussian_blur` — so `import borchvision.transforms.functional as F` is a line that runs. **What is absent carries a reason** in `tests/torch_gap.py`, and what carries none is the to-do list |
|---|---|
| **`datasets`** | absent. The fetching side is blocked — `cs.toronto.edu` sends no CORS header (measured). And torch's `download=True` keeps the download and reuses it, while Pyodide's filesystem is gone on a refresh. **Once the bytes are in hand it already works** (`fetch_cached`, `cache_put`, `TensorDataset`) |
| **`ops`** | absent. `nms` is short in numpy, so "it is large" would be a false reason; the real one is that nobody stands in front of it — detection needs a pre-trained backbone and COCO-scale data to reach the end |
| **pre-trained weights** | absent. A `.pth` is a pickle, so reading it means imitating torch's internal classes, and getting that subtly wrong brings wrong numbers in correctly shaped weights. ResNet-18 alone is 45MB, and above all, once `pretrained=True` runs people compare against the published top-1 — bit equivalence is an explicit non-goal, so that is a promise it cannot keep |

**The random numbers differ from torch's.** The same seed does not produce
torchvision's picture — torch's generator cannot be used. So the golden compares
only where the probability is pinned at 0 or 1, and whether the draws actually
happen is checked by distribution in `tests/test_vision.py`. Doing one of the two
alone means writing down something unmeasured as measured, under cover of "it is
random, so it cannot be measured".

## borch.ts — TypeScript and WGSL

It does not go through Python. **It does not go through TF.js either** — the
kernels are written directly in WGSL. **Zero** runtime dependencies, and it is
an ES module a browser simply reads (271KB gzipped, 931KB before compression).

```bash
npm install borch
```

```ts
import { init, Tensor, nn, optim, scope, keepAlive } from "borch";

await init();                                   // acquire a WebGPU adapter

const model = new nn.Sequential(
  new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
const opt = new optim.SGD(model.parameters(), 0.05, 0.9);
const crit = new nn.CrossEntropyLoss();

const x = keepAlive(Tensor.from(pixels, [32, 784]));
const y = keepAlive(Tensor.from(labels, [32], { dtype: "int64" }));

for (let i = 0; i < steps; i++) {
  await scope(async () => {                     // release one step's intermediate buffers
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();
    console.log(await loss.item());
  });
}
```

**This example really runs** — `npm run example:ts` executes it as written and
watches the loss go down. Code in documentation rots unless it is run, and this
repository has twice caught installation instructions that did not actually
work.

### From Pyodide, in Python — `borch_webgpu`

It runs Python code **on top of borch.ts.** What the binding does is hide the four
differences above (`await init()`, `await item()`, `scope()`, `.call()`).

```python
import borch_webgpu as torch          # an alias covers most of it
```

WebGPU has no synchronous read and **no `await` appears anyway.** Pyodide's
`run_sync` (JSPI) fills that place — measured (`tests/browser/sync_probe.py`), with
one condition: the page has to enter Python asynchronously. The runner already
does.

If a submodule path is needed, as in `from borch_webgpu.nn import Linear`, call
`borch_webgpu.install()`. It defaults to its own name, so somebody else's
`import torch` is untouched — the same choice as the table above.

It passes **all 3147 golden cases** — nothing in the table is skipped on this side
alone. The core covers 3094 cases, and the remaining 53 are ones the core refuses
on purpose (1-D and 3-D convolutions, ranks 7 and 8), so they are not asked of it.

> That number said 2930 until this translation. The phrasing around it was
> `보는데` rather than `본다`, so `test_docs.py`'s pattern never matched it and the
> figure went stale unwatched while the two beside it stayed current. It is 2938,
> measured. The English wording now matches the pattern, so it is watched.

borch.ts itself has written TS bodies for 2738 cases. The remaining 400 are **two different
things, and counting them as one hides the second**. 360 are **deliberately not
carried across** — the binding (`borch-webgpu`) already goes through borch.ts's kernels on
those cases, so **the values are verified**, and what a TS body would add is not a
value but this side's surface: names and argument order. A good many of them ask
about a Python name alias, so carrying them across would ask the same question
twice. The other 40 are **owed**: the `borchvision` work that arrived after
borch.ts's `vision.ts` was written. `vision.ts` carries the transforms, the
`transforms.functional` namespace that did not exist on this side at all, and the
six pixel rewrites with their six wrappers. What is left has one shape — the
transforms that **read the input between its pixels**: `rotate`, `affine`,
`perspective`, `elastic_transform`, `gaussian_blur` and the classes over them.
Those are a backlog rather than a decision, and calling all 400 deliberate would
make the backlog invisible by counting it as a choice. The runner keeps printing
the total rather than letting it shrink quietly.

> **The owed figure has gone 57 → 19 → 50 → 40 rather than down, and that is the
> figure working.** Forty-five have been carried across while the other session
> froze more; one number cannot show a debt being paid and taken on at the same
> time, so the runner's row carries both. A count that only falls looks like
> progress stalling whenever someone else is building, and a count that only
> reports a total hides that anyone paid.

> Those two figures said 2352 and 608 until they were measured. **Confirming them
> does not need a browser** — the case table registers names without running them,
> so loading `borch-ts/dist/test/cases.js` in node and counting the map is enough,
> and `tests/test_site.py` now does exactly that whenever `dist` exists. A text
> search still cannot do it: `grep -c 'out\.set('` over `cases.ts` gives 848
> against the real 2738, because the names are built programmatically.

### Six places where it diverges from torch

All six turn up in the first ten lines, so they are written down in advance.

| | why |
|---|---|
| `await init()` first | acquiring a WebGPU adapter is asynchronous |
| `await loss.item()` | it brings GPU memory back. The forward and backward passes are synchronous |
| wrap in `using s = scope()` | JS's garbage collection does not release GPU memory in time. One step makes thousands of intermediate buffers |
| `model.call(x)` | JS cannot simply call an object |
| `'cpu'` does not compute | it is where values are put down, not a device with kernels on it (see below) |
| `await opt.step(closure)` | **`LBFGS` alone.** It reads a scalar and branches inside one step (just below) |

**`LBFGS` is slow, and that is a property of the algorithm.** Other optimisers
take one step per set of gradients; this one loops `maxIter` times inside, asking
for the loss and the gradients again each time, and **reads a scalar and branches**
on every iteration — the gradient threshold, the curvature `y·s`, the directional
derivative, the change in loss. All of them are conditions on an `if` or a `break`,
so they cannot live on the GPU, and reading a value is asynchronous here. One
`step()` produces on the order of a hundred GPU-to-host round trips.

Dropping the early exit and running a fixed number of iterations would make it
synchronous, and what that produces is not a synchronous LBFGS but **a different
algorithm.** Use `Adam` on a large model; this name is for solving **a small
problem exactly.** There is no line search (`lineSearchFn`), and passing one stops
loudly.

```ts
const opt = new optim.LBFGS(model.parameters(), 0.1);
await opt.step(() => {          // the closure re-measures the loss and fills the gradients
  opt.zeroGrad();
  const loss = crit.call(model.call(x), y);
  loss.backward();
  return loss;
});
```

`scope()` does not exist in torch — it is TF.js's `tidy`'s place, for TF.js's
reason. Anything that has to survive, such as a parameter, is marked with
`keepAlive` — **without the wrapper the device fills up within a few steps.**

**Two forms, one machine.** The one closer to Python's `with` is the recommended
one.

```ts
for (let i = 0; i < steps; i++) {
  using s = scope();              // it closes at the end of the block
  opt.zeroGrad();
  const loss = crit.call(model.call(x), y);
  loss.backward();
  opt.step();
  console.log(await loss.item());
}

const loss = await scope(async () => { … });   // shorter where the value is taken directly
```

Releasing is synchronous, so it is **`using`** rather than `await using` — and
that one is more widely supported. The block is left only after every `await`
inside it has finished, so the `await loss.item()` above is safe (measured).
Anything to be carried out of the scope is marked with `s.keep(t)`.

**Forgetting either of the two stops loudly.** It did not for a while. A buffer
returning to the pool when a scope closes is not destroyed — that is what the pool
is for — so a tensor carried out unmarked **quietly read whatever the next
allocation wrote over it.** Measured, `[1,2,3,4]` read back as `9,9,9,9`. WebGPU
does not block this either, because it is a valid read of a valid buffer. Now a
buffer returning to the pool raises **a generation count**, and an old tensor
reaching for its values stops right there.

> This library's opening sentence is "it stops loudly rather than quietly
> producing a different value", and the opposite of that was sitting in the core
> training loop. It was a place the golden cannot see — each case gets a clean
> page, so the pool never gets stirred.

**The golden cannot see that place.** Leaking one buffer per step, or dispatching
twice as many kernels, leaves the values equally right and the whole table green.
So there is a separate check that asks about **counts** rather than values —
`npm run cost:ts` compares the dispatches per step, the submissions, and the
buffers a scope let out without releasing, against pinned figures.

Being counts, it is **independent of the adapter.** The benchmark (`bench:ts`)
measures wall-clock time and therefore refuses to answer on a software rasteriser
(that number is the rasteriser's rather than the library's), and these numbers are
decided by the code path and come out the same wherever they run — **running where
the benchmark cannot** is what this check is worth.

The binding side has the same measure:

```bash
uv run --with playwright python tests/browser/run.py --lib borch_webgpu --cost
```

**The two paths give the same numbers** — 53 dispatches and 1 submission per step
on the same model and batch. Which is to say the binding dispatches no extra
kernels, and a divergence is itself the answer. The binding has one more place to
watch — **a Python object holds a JS handle** — so `gc.collect()` is called on
either side of the measurement.

### Where devices are handled

This is `torch.cuda.is_available()`'s place. **It is asynchronous** — acquiring an
adapter is asynchronous and there is no way around it.

```ts
import { init, isAvailable, probe, currentDevice, Tensor } from "borch";

if (!(await isAvailable())) { /* not usable in this browser */ }

const p = await probe();          // when the reason is needed too
if (!p.ok) console.log(p.why);    // 'no-api' | 'no-adapter'

await init({ powerPreference: "high-performance" });   // this is the default
currentDevice();                  // 'webgpu' — null if it never attached
```

Splitting `why` is the point. `no-api` means an old browser or a page that is not
https; `no-adapter` means a driver blocklist, a virtual machine, or headless with
no GPU — what the user can do about them differs, and one exception covering both
loses that split.

`t.device` says where a tensor is; `await t.cpu()` brings it down and `t.webgpu()`
puts it back. **`'cpu'` is a container holding values rather than a device that
computes** — borch has no CPU kernels. A tensor brought down can be read
(`toArray`, `item`, `repr`) and putting it into an operation stops with torch's
wording.

```ts
const g = Tensor.from([1, 2, 3, 4], [2, 2]);   // 'webgpu'
const c = await g.cpu();                        // 'cpu'
await c.item();                                 // works — it is a read
c.sum();                                        // RuntimeError:
                                                // Expected all tensors to be on
                                                // the same device, ...
c.webgpu().sum();                               // works again
```

Only bringing a tensor down is asynchronous, because only that direction is a
round trip. Putting one back is a single write to the queue.

`torch.cuda.synchronize()`'s place is `await device().synchronize()`. Until it
existed, forcing completion meant reading one value, and then **the readback round
trip mixes into the measurement.**

### Writing your own layer

**Putting it in a field registers it.** The same place as what torch does in
`__setattr__`.

```ts
class Net extends nn.Module {
  fc1 = new nn.Linear(4, 8);
  fc2 = new nn.Linear(8, 2);
  override forward(x: Tensor): Tensor {
    return this.fc2.call(this.fc1.call(x).relu());
  }
}
new Net().parameters();       // all four come out
new Net().namedParameters();  // fc1.weight, fc1.bias, fc2.weight, fc2.bias
```

Standing a tensor up as a parameter directly is done with `claim()` — torch's
`nn.Parameter`'s place. A tensor field left unclaimed is treated as a constant and
the optimiser does not step it. **Arrays are not walked** (torch does not register
a Python list either) — use `nn.ModuleList`.

### Parameter groups

```ts
const opt = new optim.SGD([
  { params: backbone.parameters(), lr: 1e-3 },
  { params: head.parameters(),     lr: 1e-2, weightDecay: 0 },
], 1e-3);
opt.addParamGroup({ params: extra.parameters(), lr: 5e-4 });
```

A scheduler drives every group and preserves the ratios between them — the same
result as torch carrying a `base_lrs` per group.

### Random numbers

One `manualSeed` resets the tensor factories, the layer initialisation and dropout
together.

```ts
manualSeed(42);
Tensor.randn([2, 3]);           // standard normal
Tensor.rand([4]);               // [0, 1)
Tensor.randint(0, 10, [8]);     // integers in [low, high), int64
Tensor.randperm(64);
t.randnLike();
```

### `nn.functional` — the lines written as `F.`

torch carries the same operation under two names. `x.relu()` works and so does
`F.relu(x)`, and textbook code uses the first when using layers and the second when
calling a loss or a convolution directly. borch had the first alone, so every line
written as `F.` had to be rewritten wholesale.

```ts
import { nn } from "borch";
const F = nn.functional;                 // the same path as torch.nn.functional

F.relu(x);
F.conv2d(x, weight, bias);
F.crossEntropy(logits, target);
```

**The methods are not removed.** torch has both, so this has both — there is no
reason for code written as `x.relu()` to stop because of this change. `Tensor` does
not get smaller either. This opens a door that was not there rather than clearing
away one that was.

**The five that share a name and are a different operation are not exposed.**
`F.layer_norm`, `F.rms_norm`, `F.pad` and `F.upsample` have a different argument
convention from torch's, and `F.batch_norm` goes out as the layer side's free
function rather than as `Tensor.batchNorm` (which is `layerNorm` with the axes
swapped). Wiring them by name attaches a quietly different operation, so what is
absent is left absent.

### The square-bracket place — `x[...]`

**JavaScript cannot overload `[]`.** So one line of torch's square brackets scatters
here across fifteen branches — `select`, `narrow`, `indexSelect` and the rest — and
whoever transcribes the code had to choose which one per line. `at()` narrows those
branches to one door.

```ts
x.at(0)                     // x[0]           the axis disappears
x.at([null, 1])             // x[:, 1]        null is Python's `:`
x.at(slice(1, 3))           // x[1:3]         the axis stays
x.at([0, slice(1, 3)])      // x[0, 1:3]
x.at(slice(null, null, 2))  // x[::2]
x.at([[0, 2]])              // x[[0, 2]]      two brackets — numpy's shape
x.at(idx)                   // x[idx]         an int64 tensor
```

**Why a slice is a function**: an array alone cannot separate `x.at([1, 3])`
meaning "1 on axis 0 and 3 on axis 1" from "cut 1:3". Python's `x[1:3]` also
resolves to `x[slice(1, 3)]`, so the same name is used — not something new to
learn but the name that was there all along.

**The outermost array is always a list of axes.** Give fewer and the remaining
axes come through whole. Selecting by index means one more layer of wrapping — the
same split as numpy's `x[0, 1]` against `x[[0, 1]]`.

`at()` produces no values. It forwards everything to the existing methods, so
**the golden already guards those values** — the only thing this method is
responsible for is which door it sends you through. The existing methods are
untouched; this adds a door rather than removing one.

**It does not take a boolean mask.** `x[mask]` stays as
`await x.maskedSelect(mask)` — the result's length depends on the values and needs
one read back from the GPU, and making `at()` asynchronous for that one case would
put an `await` on every other use for no reason.

### Feeding it data

`torch.utils.data`'s place. **A batch is a GPU tensor, so it has to be received
inside a `scope()`** — the loader cannot wrap it for you, because the point is for
the tensors to leave the scope.

```ts
const set = new data.TensorDataset(images, labels);
const [train, valid] = data.randomSplit(set, [800, 200]);
const loader = new data.DataLoader(train, { batchSize: 32, shuffle: true });

for (let epoch = 0; epoch < 10; epoch++) {
  for (const [x, y] of loader) {           // a synchronous iterator
    await scope(async () => {
      opt.zeroGrad();
      const loss = crit.call(model.call(x), y);
      loss.backward();
      opt.step();
    });
  }
}
loader.length;    // **the batch count**, not the sample count. As in torch
```

The shuffle follows `manualSeed` — torch keeps a separate generator on the loader
and this uses the one host stream. One seed resets the layer initialisation,
dropout, the tensor factories and the batch order with them. Reshuffling each epoch
is as in torch.

**`sampler` and `num_workers` are absent.** The first because there is nothing to
back it yet (putting down the name alone means what you pass is quietly ignored),
the second because a GPU handle does not cross into a worker. What is here is
`shuffle`, `dropLast`, `Subset`, `randomSplit` and `ConcatDataset`.

### Saving and resuming

**The format is safetensors.** torch's `save`/`load` is pickle, which cannot be
carried into a browser and should not be. Carrying this one instead means
**Python `borch`, numpy and the HF tools read the same file** — and that is what
opens the path from training in a browser to taking the result to your own
machine.

**It carries the nesting as it is** — that is the textbook idiom, and the same
shape as torch's and Python `borch`'s. Non-tensors travel with it too (numbers,
strings, booleans, `null`, arrays).

```ts
import { save, load } from "borch";

const bytes = await save({
  model: model.stateDict(),
  opt: opt.stateDict(),
  sched: sched.stateDict(),
  epoch: 5,
});
// bytes is a Uint8Array — putting it in IndexedDB or downloading it is the caller's job
```

Restoring means **standing the model, the optimiser and the scheduler back up
with the same arguments** and then loading onto them.

```ts
const ck = load(bytes);
model.loadStateDict(ck.model);
opt.loadStateDict(ck.opt);
sched.loadStateDict(ck.sched);
```

The structure is written as JSON into the header's `borch.tree` and the tensors
lie flat as before — **the same scheme as the Python side, so the two read each
other's checkpoints.** A file with no tree (somebody else's safetensors) comes back
as a flat table of tensors.

If the codec underneath is what is wanted, `encode`/`decode` is its place. It
handles a flat `Record<string, Tensor>` and string metadata alone, and comes with
`prefixed`/`unprefixed` for flattening names without collisions and
`numbersToMeta`/`metaToNumbers` for moving numbers into the metadata.

```ts
const { tensors, metadata } = decode(bytes);
```

**Restoring the weights alone is not enough.** The momentum, the step counters and
the scheduler's epoch have to travel with them for the step after resuming to
produce the same numbers as a run that was never interrupted.
`npm run serialize:ts` confirms that bit for bit — the trajectory of ten steps run
straight through and the trajectory of five, interrupted and resumed, have to be
exactly equal to pass, and the same runner opens that file again **with numpy
alone.**

Values always go out as float32. borch's `int64` and `bool` are labels and ride in
the header's `__metadata__` — writing `I64` against a four-byte body breaks
somebody else's reader.

### Where it runs

WebGPU is required. **Without it, it refuses rather than falling back.** The TF.js
version that stood here dropped quietly to WebGL when it could not get WebGPU, and
because of that, performance figures **measured on a CPU software path** were read
as the GPU's for a while. Not running beats quietly getting slower.

Measurements are taken under `--headed` only. A headless browser gives a software
rasteriser (SwiftShader) that **throws no exception and simply produces strange
numbers.** So the runner prints the adapter first, and the benchmark and the
accuracy run refuse outright on a software adapter.

### How much it does

**The golden matches on every case** across two vendors, Apple Metal and NVIDIA
(RTX 4090) — which is to say the hand-written WGSL is not Metal-only. (The 4090
figures were measured against the table as it stood then. That machine has not been
available since the table grew, so it has not been measured again, and what has not
been measured is not written down as though it had been.) The benchmark's
ResNet-18 itself was also confirmed to match real torch on the forward pass, the
loss and the backward pass.

**This table is the evidence for deleting the TF.js version.** It is a record
measured side by side on the same machine and the same benchmark, and the left
column no longer exists.

| CIFAR ResNet-18, batch 64 | the TF.js version (now gone) | **borch.ts** | **borch_webgpu** |
|---|---|---|---|
| ms/step | 154.9 | **118.5** | 123.4 |
| epoch | 2.02 min | **1.55 min** | 1.61 min |
| test accuracy (10 epochs, augmentation on) | 60.4% | **64.6%** | not measured |

The right two columns are **the same kernels.** The 4.9ms difference is the cost of
one trip through Python, and it is the only number that measures what this binding
costs.

> The accuracy is **with augmentation on.** With it off the figure is 59.3%, below
> the sister library's. It read 65.5% / 62.4% for a while, and at that time six of
> the benchmark model's shortcut layers were not being trained — that freezing was
> acting as regularisation. The details are in the T3 accuracy section of
> [BORCH-TS.md](BORCH-TS.md).

The design and the evidence behind the measurements are in
[BORCH-TS.md](BORCH-TS.md).

## What is deliberately not supported

`CUDA`, pre-trained weights, mixed precision, distributed training,
`torch.compile`

**A long refusal list is the intent.** GPUs, saved models and pre-training are
learned by leaving the browser, and imitating them here loses the lesson.

## Conformance

The goal is not "reproduce PyTorch" but **equivalence within the range the
curriculum uses.** Why it was set that way, and what comes next, is in
[ROADMAP.md](ROADMAP.md).

| grade | where it stands |
|---|---|
| **T1 values and gradients** (`allclose 1e-5`) | **100%** — 132 generated cases |
| **T2 error equivalence** | **12/12** — exception types, and 9/9 searchable messages |
| **T3 printed form (`repr`) equivalence** | **15/15** |
| **dtype promotion** | **112/112** — 4 dtypes × 4 operations × tensor and scalar |
| **shared storage** (views and slices) | **13/13** |
| **integration scenarios** | **6/6** — the same code run with one import changed |
| **the wide surface** (maths, shapes, functional) | **67/67** |
| **common API names** | **144/144** |
| T4 bit equivalence | **an explicit non-goal** |

**53 of them ask whether the gradient flows rather than what the value is.** A
check comparing values alone cannot see a cut graph — because the values are
right. The GPU side's `roll` and `masked_select` really were cut that way, and the
golden was entirely green at the time.

And **3147 golden cases** compare all three implementations against **the same
expected values.** The core covers 3094 cases, leaving out the 53 that are
browser-only (things the core refuses on purpose, such as 1-D and 3-D
convolutions) — asking about something that is not there is a wrong answer rather
than a check. Real torch cannot be put into a browser, so the expected values are
pinned natively and carried in.

```bash
uv run --with numpy --with torch python tests/golden.py dump   # stage 1: pin them
uv run --with numpy python tests/golden.py check               # stage 2: compare
uv run --with numpy python tests/export_json.py                # stage 3: export
```

Stage 3 produces `tests/golden.json` (722KB). The point is **making the expected
values usable by an implementation that is not Python** — the numbers obtained by
running real torch are this repository's most expensive asset, and kept inside
Python alone, the next implementation grows without verification.

**The case bodies are not in it.** `lambda L: L.amax(...)` does not become another
language mechanically. The receiving side writes a case of the same name in its own
language and matches its answer here — the expensive half (the numbers) crosses and
the cheap half (one call) is rewritten. borch.ts does exactly that and passes 1779
cases.

**When a name does not match, the runner counts it.** It did not for a while, and
during that time, holding seven names not in the golden while leaving seven of the
golden's own unused looked like "859 of 859, 0 remaining" — the counts matched and
cancelled out.

```bash
uv run --with numpy --with torch python tests/conformance.py
```

## Licence

Apache-2.0 · PI Lab

numpy (BSD-3-Clause) is the only dependency. It is a pure-Python wheel and ships
nothing else bundled with it.

> **Whoever puts it in a browser serves Pyodide alongside it.** Pyodide is
> MPL-2.0, and distributing it in executable form means **telling the recipient how
> to obtain the source** (MPL §3.2). It does not spread into our code — weak
> copyleft works per file, so borch stays Apache-2.0. One line somewhere on the
> page is enough:
>
> > This page includes [Pyodide](https://github.com/pyodide/pyodide), which is
> > licensed under the Mozilla Public License 2.0. The source is available from
> > that repository.
>
> What this leans on and what has to be honoured is collected in
> [THIRD-PARTY.md](THIRD-PARTY.md).
