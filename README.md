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
> is that **2991 golden cases pass**, and that is a surface check rather than a
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

## 지원 범위

| | |
|---|---|
| **텐서** | 모양·브로드캐스팅·dtype 승격 · 인덱싱 · reshape/view/permute/squeeze · split·chunk·flip·roll·gather·narrow·index_select·masked_select |
| **autograd** | `requires_grad` · `backward()` · `.grad` · `no_grad()` · `detach()` · 누적 |
| **축약** | `sum`·`mean`·`max`·`min`·`prod`·`median`·`norm`·`cumsum`·`topk`·`sort`·`unique`·`std` — 역전파 포함 |
| **nn** | `Module` · `Linear` · `Conv1d/2d/3d` · `MaxPool1d/2d/3d` · `Upsample` · `Embedding` · `LayerNorm` · `BatchNorm1d/2d/3d` · `Dropout` · `Sequential` · `ModuleList` |
| **순환** | `RNN` · `LSTM` · `GRU` — 다층 · `batch_first` · 초기 상태. **최상위 함수 꼴도** (`torch.lstm`·`lstm_cell` 등 여덟) — 가중치를 목록으로 받는다. 양방향과 층간 드롭아웃은 거절한다 |
| **트랜스포머** | `MultiheadAttention` · 인코더·디코더 층 · `nn.Transformer` — 불리언·실수 마스크 · `norm_first` · gelu |
| **손실** | `MSELoss`·`L1Loss`·`SmoothL1Loss`·`BCELoss`·`BCEWithLogitsLoss`·`CrossEntropyLoss`·`NLLLoss` |
| **optim** | `SGD`(momentum·weight_decay) · `Adam` · `AdamW` · `RMSprop` — `param_groups` · `state_dict` |
| **스케줄러** | `StepLR` · `MultiStepLR` · `ExponentialLR` · `CosineAnnealingLR` · `LambdaLR` · `ReduceLROnPlateau` |
| **데이터** | `Dataset` · `TensorDataset` · `Subset` · `ConcatDataset` · `DataLoader` · `WeightedRandomSampler` · `random_split(generator=)` · `collate_fn` |
| **저장** | `state_dict` · `load_state_dict` · `save`/`load` · 버퍼(`running_mean` 등) 포함 |
| **nn.functional** | 25종 — 활성·손실·`pad`·`normalize`·`cosine_similarity`·`one_hot`·`layer_norm`·`embedding` |
| **복소수** | `complex64` 만 — `complex`·`polar`·`view_as_real`/`view_as_complex`·`real`/`imag`/`conj`/`angle`/`abs` · 산술 · autograd. **셋 다에 있다** (아래) |
| **푸리에** | `fft.fft`/`ifft`/`rfft`/`irfft` · `fftfreq`/`rfftfreq`/`fftshift`/`ifftshift` · `stft`/`istft` — `n`·`dim`·`norm` 과 역전파. **셋 다에 있다** |

### 복소수 — `complex64` 만

`complex128` 은 **영원히 없다.** WGSL 에 `f64` 가 없어서 브라우저에서 도는 절반이
그것을 못 든다. 그래서 이름은 두되, `complex64 + float64` 처럼 승격이 그것을 만들려는
자리에서 **조용히 내려앉지 않고 멈춘다** — torch 는 여기서 `complex128` 을 준다.
값이 반쯤 맞는 것보다 여기서 서는 편이 낫다는 이 저장소의 같은 선택이다.

기울기 규약은 재서 못 박았다: **torch 는 복소 손실에 `backward()` 를 거절한다.**
손실이 늘 실수라면 규약은 하나로 정해진다 —

    z.grad = ∂L/∂re + i·∂L/∂im

그 위에서 **정칙 함수의 역방향에는 켤레가 붙고**(곱셈·나눗셈: `conj(f'(z))·g`),
실수를 내는 `abs` 에는 안 붙는다(`z/|z|`). 실수 입력만으로는 이 차이가 안 보인다 —
켤레가 실수에서 항등이라서다. 그래서 골든이 셋을 한 표에서 묻는다.

**`conj` 가 torch 와 갈린다.** torch 의 `conj` 는 **게으르다** — 켤레 비트만 세우고
값을 안 뒤집는다. 그래서 `torch.is_conj(torch.conj(z))` 가 `True` 이고,
`view_as_real` 은 "풀지 않은 켤레" 라며 거절한다. 우리 것은 즉시 뒤집으므로
그 상태가 아예 없고, **`is_conj` 는 언제나 `False`** 다. 값은 같다 —
`conj_physical` 로 물으면 양쪽이 같은 답을 낸다.

**borch.ts 의 저장은 인터리브다** — `[re, im, re, im, …]` 한 버퍼. 그래서
`view_as_real`·`view_as_complex` 가 **진짜 뷰**이고(torch 도 그렇다), 대신 오래된
불변식 `size = 버퍼 길이` 가 `버퍼 길이 = size × 2` 로 갈린다. 그것을 모르는 커널이
복소수 버퍼를 읽으면 앞쪽 절반만 실수로 보고 **예외 없이** 틀린 답을 내므로,
**기본값이 거절**이다 — 복소수를 아는 연산만 따로 문을 지난다. `Tensor.from` 으로
형만 `complex64` 라고 붙이는 것, 이름표 갈이(`to`), 체크포인트 저장도 같은 이유로
막혀 있다.

### 푸리에 — `torch.fft` 와 `stft`

**복소수 위에 선다.** `stft` 는 오래 거절이었고 거절문에 "복소수 규약을 안 정했다"
고 적혀 있었다. 그 이유가 **정확했기 때문에** 규약이 정해진 날 문이 열렸다 — "저장이
없다" 로 적어 두었으면 저장이 생긴 뒤에도 아무도 다시 안 물었을 것이다.

`stft` 는 새 커널이 아니라 **조립이다** — 자르고 · 창을 곱하고 · `rfft`. 셋 다 이미
미분되는 이름이라 **기울기가 저절로 맞는다.** 손으로 커널을 쓰면 순방향은 금방 맞고
역방향이 창과 겹침을 다 지나야 해서, 틀리면 값은 그럴듯하고 학습만 안 된다.

브라우저 쪽은 **DFT 를 그대로 돈다 — O(n²) 다.** 쿨리-튜키는 2 의 거듭제곱에서만
빠르고 아닌 길이는 블루스타인이 따로 필요한데, **값은 어느 쪽이든 같고** 이 프로젝트의
천장에서는 차이가 안 보인다. 빨라야 하는 날이 오면 그때 바꾸되, 지금 없는 속도를
있는 것처럼 적지 않는다. 회전인자는 **호스트에서 배정도로 만들어 올린다** —
셰이더의 `cos`/`sin` 은 정확도가 구현에 맡겨져 있고, 실제로 사각창 `stft` 한 자리가
상대오차 2.7e-4 로 골든을 벗어났다(f32 반올림으로는 설명이 안 되는 크기다).

기울기에서 어려운 자리는 값이 아니라 **어느 쪽 반쪽을 세는가** 다. `rfft` 의 역방향은
저장된 반쪽에만 기울기가 오므로 켤레 짝을 **안** 더하고(더하면 두 배), `irfft` 는
되살린 켤레 짝이 같은 칸에서 왔으므로 **가장자리만 한 번, 가운데는 두 번** 센다.
둘 다 순방향 값은 멀쩡한 채로 틀릴 수 있는 자리다.

> **`abs` 의 칼날.** 골든의 `stft` 기울기 케이스가 고르지 않은 수를 쓰는 데는 이유가
> 있다. 경사 신호는 나이퀴스트 칸이 **정확히 0** 이 되는데 거기서 `abs` 는 미분
> 불가능하고 부호가 반올림에 달린다 — 우리는 float64 로 누산해 +1 을, torch 는
> float32 FFT 라 0 을 골랐다. 규칙이 갈린 것이 아니라 **케이스가 칼날 위에 선 것**
> 이고, 그런 것을 굳히면 골든이 부동소수 우연을 명세로 박제한다.

**`print` 도 명세다.** torch 는 복소수를 찍을 때 **실수부와 허수부를 따로 잰다** —
`[1+2j, -0.5-1j]` 에서 실수부는 소수 네 자리를 요구하고 허수부는 정수라
`1.0000+2.j` 가 된다. 한 형식으로 재면 `1.0000+2.0000j` 가 나오는데, 값이 전부
맞는 채로 글자만 갈린다. 자리맞춤도 실수부에만 걸려서 `1.-0.j` 처럼 **음의 0** 이
산다 — 그 부호 하나가 이 결속의 읽기 경로에서 `-0.0` 을 잃고 있던 것을 잡았다.
`-0.0 == 0.0` 이라 값 대조로는 영영 안 걸렸을 자리다.

## torchvision — `transforms` 만 (`borchvision`)

파이토치 입문 튜토리얼의 **첫 열 줄이 torchvision** 이다.

```python
datasets.MNIST(root, transform=transforms.ToTensor())
```

"임포트만 바꿔 같은 값을 낸다"는 약속이 여기서 먼저 걸리므로, `transforms` 는 있다.
별도 파일인 이유는 `torchvision.transforms` 이지 `torch.transforms` 가 아니어서다 —
코어 안에 넣으면 진짜 torch 에 **없는 자리**를 만들게 된다.

```python
import borchvision as torchvision
from borchvision import transforms
```

| 있는 것 | `Compose` · `ToTensor` · `Normalize` · `RandomHorizontalFlip` · `RandomCrop` |
|---|---|
| **`datasets`** | 없다. 받아오는 쪽이 막혀 있다 — `cs.toronto.edu` 가 CORS 헤더를 안 준다(실측). 게다가 torch 의 `download=True` 는 받아두고 재사용하는데 Pyodide 파일시스템은 새로고침에 날아간다. **바이트를 손에 넣은 뒤는 이미 된다** (`fetch_cached`·`cache_put`·`TensorDataset`) |
| **`ops`** | 없다. `nms` 는 numpy 로 짧아서 "크다"는 이유는 거짓이고, 진짜 이유는 아무도 그 앞에 안 선다는 것이다 — 검출은 사전학습 백본과 COCO 급 데이터가 있어야 끝까지 간다 |
| **사전학습 가중치** | 없다. `.pth` 는 pickle 이라 torch 내부 클래스를 흉내 내야 읽히고, 미묘하게 틀리면 모양 맞는 가중치에 틀린 수가 들어온다. ResNet-18 만 45MB 이며, 무엇보다 `pretrained=True` 가 돌면 사람들은 발표된 top-1 과 비교한다 — 비트 동등은 명시적 비목표라 지킬 수 없는 약속이다 |

**난수는 torch 와 다르다.** 같은 씨앗을 줘도 torchvision 과 같은 장면이 나오지 않는다 —
torch 의 난수기를 쓸 수 없어서다. 그래서 골든은 확률을 0·1 로 못 박은 자리만 대조하고,
뽑기가 실제로 도는지는 `tests/test_vision.py` 가 분포로 본다. 둘 중 하나만 하면
"무작위니까 못 잰다"로 안 잰 것을 잰 것처럼 적게 된다.

## borch.ts — TypeScript 와 WGSL

파이썬을 안 거친다. **TF.js 도 안 거친다** — 커널을 WGSL 로 직접 썼다.
런타임 의존성이 **0개**이고, 브라우저가 그냥 읽는 ES 모듈이다(gzip 242KB, 압축 전 834KB).

```bash
npm install borch
```

```ts
import { init, Tensor, nn, optim, scope, keepAlive } from "borch";

await init();                                   // WebGPU 어댑터를 잡는다

const model = new nn.Sequential(
  new nn.Linear(784, 128), new nn.ReLU(), new nn.Linear(128, 10));
const opt = new optim.SGD(model.parameters(), 0.05, 0.9);
const crit = new nn.CrossEntropyLoss();

const x = keepAlive(Tensor.from(pixels, [32, 784]));
const y = keepAlive(Tensor.from(labels, [32], { dtype: "int64" }));

for (let i = 0; i < steps; i++) {
  await scope(async () => {                     // 한 스텝의 중간 버퍼를 놓는다
    opt.zeroGrad();
    const loss = crit.call(model.call(x), y);
    loss.backward();
    opt.step();
    console.log(await loss.item());
  });
}
```

**이 예시는 실제로 돈다** — `npm run example:ts` 가 그대로 실행하고 손실이 내려가는
것까지 본다. 문서의 코드는 안 돌리면 썩고, 이 저장소는 설치 안내가 실제로는 안 듣던
것을 이미 두 번 잡았다.

### Pyodide 에서 파이썬으로 — `borch_webgpu`

파이썬 코드를 **borch.ts 위에서** 돌린다. 결속이 하는 일은 위의 네 가지
차이(`await init()`·`await item()`·`scope()`·`.call()`)를 감추는 것이다.

```python
import borch_webgpu as torch          # 별칭이면 대부분 된다
```

WebGPU 에 동기 읽기가 없는데도 **`await` 이 안 나온다.** Pyodide 의 `run_sync`(JSPI)가
그 자리를 메운다 — 재봤고(`tests/browser/sync_probe.py`), 조건이 하나 있다: 페이지가
비동기로 파이썬에 들어와야 한다. 러너는 이미 그렇게 들어간다.

`from borch_webgpu.nn import Linear` 처럼 하위 경로가 필요하면 `borch_webgpu.install()` 을
부른다. 기본값이 자기 이름이라 남의 `import torch` 는 안 건드린다 — 위의 표와 같은
선택이다.

**골든 2991 건 전부**를 지난다 — 표에서 이쪽만 건너뛰는 것이 하나도 없다. 코어는
그중 2930 건을 보는데, 나머지 53 건은 코어가 일부러 거절하는 것들(1·3 차원 합성곱,
랭크 7·8)이라 안 묻는다.

borch.ts 자신은 2352 건에 TS 본문을 써 두었다. 나머지 608 건은 **일부러 안 옮겼다** —
결속(`borch-webgpu`)이 이미 그 케이스들에서 borch.ts 커널을 지나므로 **값은 검증되고
있고**, TS 본문이 추가로 증명하는 것은 값이 아니라 이쪽 표면(이름과 인자 순서)이다.
그중 상당수는 파이썬 이름 별칭을 묻는 것이라 옮기면 같은 질문이 두 번이 된다.
줄지 않는 그 수를 러너가 계속 찍는다 — 은근히 사라지는 것보다 낫다.

### torch 와 갈리는 여섯 자리

첫 열 줄에서 전부 만나므로 미리 적는다.

| | 왜 |
|---|---|
| `await init()` 을 먼저 | WebGPU 어댑터를 잡는 것이 비동기다 |
| `await loss.item()` | GPU 메모리를 도로 가져온다. 순방향·역방향은 동기다 |
| `using s = scope()` 로 감싼다 | JS 의 쓰레기 수집이 GPU 메모리를 제때 안 놓는다. 한 스텝이 중간 버퍼를 수천 개 만든다 |
| `model.call(x)` | JS 는 객체를 그냥 못 부른다 |
| `'cpu'` 로는 연산이 안 된다 | 값을 내려두는 자리이지 커널이 있는 장치가 아니다 (아래 절) |
| `await opt.step(closure)` | **`LBFGS` 만** 그렇다. 한 걸음 안에서 스칼라를 읽어 분기한다 (바로 아래) |

**`LBFGS` 는 느리고, 그것은 알고리즘의 성질이다.** 다른 옵티마이저는 기울기 한 벌로
한 걸음을 가지만 이쪽은 안에서 `maxIter` 번 돌면서 매번 손실과 기울기를 다시 묻고,
되돌이마다 **스칼라를 읽어 분기한다** — 기울기 문턱, 곡률 `y·s`, 방향 미분, 손실 변화.
전부 `if` 와 `break` 의 조건이라 GPU 위에 둘 수 없고, 값을 읽는 것은 여기서 비동기다.
한 번의 `step()` 이 스텝당 백 번 안팎의 GPU→호스트 왕복을 낸다.

조기 종료를 버리고 고정 횟수로 돌면 동기로 만들 수 있지만, 그렇게 만든 것은 동기
LBFGS 가 아니라 **다른 알고리즘**이다. 큰 모델에는 `Adam` 을 쓰고, 이 이름은 **작은
문제를 정확히** 풀 때 쓴다. 직선 탐색(`lineSearchFn`)은 없고, 주면 시끄럽게 멈춘다.

```ts
const opt = new optim.LBFGS([p], 0.1);
await opt.step(() => {                  // 닫힘이 손실을 다시 재고 기울기를 채운다
  p.grad = null;
  const loss = crit.call(model.call(x), y);
  loss.backward();
  return loss;
});
```

`scope()` 는 torch 에 없다 — TF.js 의 `tidy` 와 같은 자리이고 이유도 같다. 파라미터처럼
살아남아야 하는 것은 `keepAlive` 로 표시한다 — **안 감싸면 몇 스텝 만에 장치가 찬다.**

**꼴이 둘이고 같은 기계다.** 파이썬의 `with` 에 가까운 쪽을 권한다.

```ts
for (let i = 0; i < steps; i++) {
  using s = scope();              // 블록 끝에서 닫힌다
  opt.zeroGrad();
  const loss = crit.call(model.call(x), y);
  loss.backward();
  opt.step();
  console.log(await loss.item());
}

const loss = await scope(async () => { … });   // 값을 그대로 받는 자리는 이쪽이 짧다
```

놓는 일이 동기라 `await using` 이 아니라 **`using`** 이다 — 지원도 그쪽이 넓다.
블록을 벗어나는 시점은 안의 `await` 이 전부 끝난 뒤이므로 위의 `await loss.item()` 은
안전하다(실측). 구역 밖으로 들고 나갈 것은 `s.keep(t)` 로 표시한다.

**둘 중 하나를 까먹으면 시끄럽게 멈춘다.** 한동안 안 그랬다. 구역이 닫힐 때 버퍼는
파괴되지 않고 통에 돌아가므로(그것이 통이 있는 이유다), 표시를 안 하고 밖으로 들고
나간 텐서는 **다음 할당이 덮어쓴 값을 조용히 읽었다** — 재봤더니 `[1,2,3,4]` 가
`9,9,9,9` 로 읽혔다. WebGPU 도 이것은 안 막아 준다. 유효한 버퍼를 유효하게 읽는
것이니까. 이제 버퍼가 통에 돌아갈 때 **삶의 횟수**가 하나 오르고, 옛 텐서가 값에
닿으려 하면 그 자리에서 멈춘다.

> 이 라이브러리의 첫 문장이 「조용히 다른 값을 내느니 시끄럽게 멈춘다」인데, 그
> 반대가 핵심 학습 루프에 있었다. 골든이 못 보는 자리였다 — 케이스마다 페이지가
> 깨끗해서 통이 휘저어질 일이 없다.

**그 자리를 골든이 못 본다.** 스텝마다 버퍼를 하나씩 흘려도, 커널을 두 배로 걸어도
값은 똑같이 맞으므로 표는 전부 초록이다. 그래서 값이 아니라 **세는 것**을 묻는 검사가
따로 있다 — `npm run cost:ts` 가 스텝당 dispatch 수·제출 수·구역이 안 놓고 내보낸
버퍼 수를 굳힌 값과 대조한다.

세는 것이라 **어댑터와 무관하다.** 벤치(`bench:ts`)는 벽시계를 재므로 소프트웨어
래스터라이저에서 답을 거부하는데(그 수는 라이브러리의 수가 아니라 그 래스터라이저의
수다), 이쪽은 코드 경로가 정하는 수라 어디서 돌려도 같다 — **벤치가 못 도는 자리에서
도는 것**이 이 검사의 값어치다.

결속 쪽에도 같은 잣대가 있다:

```bash
uv run --with playwright python tests/browser/run.py --lib borch_webgpu --cost
```

**두 길이 같은 수를 낸다** — 같은 모델·같은 배치에서 스텝당 dispatch 53, 제출 1.
결속이 커널을 더 걸지 않는다는 뜻이고, 갈리면 그 자체가 답이다. 결속 쪽에는 자리가
하나 더 있다 — **파이썬 객체가 JS 손잡이를 쥔다.** 그래서 `gc.collect()` 를 앞뒤로
부르고 잰다.

### 장치를 다루는 자리

`torch.cuda.is_available()` 자리는 이렇다. **비동기다** — 어댑터를 얻는 것이 비동기라
피할 길이 없다.

```ts
import { init, isAvailable, probe, currentDevice, Tensor } from "borch";

if (!(await isAvailable())) { /* 이 브라우저에서는 못 쓴다 */ }

const p = await probe();          // 왜 안 되는지까지 필요하면
if (!p.ok) console.log(p.why);    // 'no-api' | 'no-adapter'

await init({ powerPreference: "high-performance" });   // 기본값이 이것이다
currentDevice();                  // 'webgpu' — 안 붙었으면 null
```

`why` 를 가르는 것이 요점이다. `no-api` 는 브라우저가 낡았거나 https 가 아닌 것이고
`no-adapter` 는 드라이버 차단 목록·가상 머신·GPU 없는 헤드리스다 — 쓰는 사람이 할 수
있는 일이 서로 다른데 예외 하나로 뭉치면 그 갈림이 사라진다.

텐서가 어디 있는지는 `t.device` 가 답하고, `await t.cpu()` 로 내리고 `t.webgpu()` 로
올린다. **`'cpu'` 는 값이 담긴 그릇이지 연산되는 장치가 아니다** — borch 에 CPU 커널은
없다. 내려온 텐서는 읽을 수는 있어도(`toArray`·`item`·`repr`) 연산에 넣으면 torch 와
같은 문구로 멈춘다.

```ts
const g = Tensor.from([1, 2, 3, 4], [2, 2]);   // 'webgpu'
const c = await g.cpu();                        // 'cpu'
await c.item();                                 // 된다 — 읽기다
c.sum();                                        // RuntimeError:
                                                // Expected all tensors to be on
                                                // the same device, ...
c.webgpu().sum();                               // 다시 된다
```

내리는 것만 비동기인 것은 그쪽만 왕복이기 때문이다. 올리는 것은 큐에 쓰기 하나다.

`torch.cuda.synchronize()` 자리는 `await device().synchronize()` 다. 지금까지 완료를
강제하는 방법은 값을 하나 읽는 것이었는데 그러면 **readback 왕복이 측정에 섞인다.**

### 층을 직접 만들 때

**필드에 두면 등록된다.** torch 가 `__setattr__` 로 하는 일과 같은 자리다.

```ts
class Net extends nn.Module {
  fc1 = new nn.Linear(4, 8);
  fc2 = new nn.Linear(8, 2);
  override forward(x: Tensor): Tensor {
    return this.fc2.call(this.fc1.call(x).relu());
  }
}
new Net().parameters();       // 넷 다 나온다
new Net().namedParameters();  // fc1.weight, fc1.bias, fc2.weight, fc2.bias
```

텐서를 직접 파라미터로 둘 때는 `claim()` 으로 세운다 — torch 의 `nn.Parameter` 자리다.
안 세운 텐서 필드는 상수로 보고 옵티마이저가 안 밟는다. **배열은 안 훑는다**(torch 도
파이썬 list 를 등록하지 않는다) — `nn.ModuleList` 를 쓴다.

### 파라미터 그룹

```ts
const opt = new optim.SGD([
  { params: backbone.parameters(), lr: 1e-3 },
  { params: head.parameters(),     lr: 1e-2, weightDecay: 0 },
], 1e-3);
opt.addParamGroup({ params: extra.parameters(), lr: 5e-4 });
```

스케줄러는 그룹 전부를 몰고 그룹 사이 비율을 지킨다 — torch 가 `base_lrs` 를 그룹마다
드는 것과 같은 결과다.

### 난수

`manualSeed` 하나가 텐서 팩토리·층 초기화·dropout 을 같이 되돌린다.

```ts
manualSeed(42);
Tensor.randn([2, 3]);           // 표준정규
Tensor.rand([4]);               // [0, 1)
Tensor.randint(0, 10, [8]);     // [low, high) 정수, int64
Tensor.randperm(64);
t.randnLike();
```

### `nn.functional` — `F.` 로 적힌 줄

torch 는 같은 연산을 두 이름으로 갖는다. `x.relu()` 도 되고 `F.relu(x)` 도 되며,
교재 코드는 층을 쓸 때 앞쪽을, 손실·합성곱을 직접 부를 때 뒤쪽을 쓴다. borch 에는
앞쪽만 있어서 `F.` 로 적힌 줄을 통째로 다시 써야 했다.

```ts
import { nn } from "borch";
const F = nn.functional;                 // torch.nn.functional 과 같은 경로다

F.relu(x);
F.conv2d(x, weight, bias);
F.crossEntropy(logits, target);
```

**메서드를 안 없앤다.** torch 가 둘 다 갖고 있으므로 우리도 둘 다 갖는다 —
`x.relu()` 로 적힌 코드가 이 변경으로 멈출 이유가 없다. `Tensor` 가 작아지지도
않는다. 없던 문을 내는 것이지 있던 것을 치우는 것이 아니다.

**이름이 같은데 연산이 다른 다섯은 안 낸다.** `F.layer_norm`·`F.rms_norm`·`F.pad`·
`F.upsample` 은 torch 와 인자 규약이 다르고, `F.batch_norm` 은 `Tensor.batchNorm`
(축만 바꾼 `layerNorm`)이 아니라 층 쪽 자유 함수로 나간다. 이름으로 이으면 조용히
다른 연산이 걸리므로, 없는 것은 없다고 둔다.

### 대괄호 자리 — `x[...]`

**자바스크립트는 `[]` 를 오버로드할 수 없다.** 그래서 torch 의 대괄호 한 줄이 여기서는
`select`·`narrow`·`indexSelect`… 열다섯 갈래로 흩어지고, 옮겨 적는 사람이 줄마다
어느 것인지 골라야 했다. `at()` 이 그 갈래를 문 하나로 좁힌다.

```ts
x.at(0)                     // x[0]           축이 사라진다
x.at([null, 1])             // x[:, 1]        null 이 파이썬의 `:` 다
x.at(slice(1, 3))           // x[1:3]         축이 남는다
x.at([0, slice(1, 3)])      // x[0, 1:3]
x.at(slice(null, null, 2))  // x[::2]
x.at([[0, 2]])              // x[[0, 2]]      대괄호 둘 — numpy 와 같은 모양
x.at(idx)                   // x[idx]         int64 텐서
```

**슬라이스가 함수인 이유**: `x.at([1, 3])` 이 "축 0 은 1, 축 1 은 3" 인지 "1:3 을
자른다" 인지 배열만으로는 안 갈린다. 파이썬의 `x[1:3]` 도 실은 `x[slice(1, 3)]` 로
풀리므로 같은 이름을 쓴다 — 새로 배울 것이 아니라 원래 그 자리에 있던 이름이다.

**맨 바깥 배열은 언제나 축 목록이다.** 적게 주면 남은 축은 통째로 온다. 번호표로
고르려면 한 겹 더 싼다 — numpy 의 `x[0, 1]` 과 `x[[0, 1]]` 이 갈리는 것과 같다.

`at()` 은 값을 안 만든다. 전부 기존 메서드로 넘기므로 **골든이 이미 그 값들을
지킨다** — 이 메서드가 지는 책임은 어느 문으로 보내는가뿐이다. 기존 메서드는 그대로
있고, 없애는 것이 아니라 문을 하나 더 낸 것이다.

**참·거짓 마스크는 안 받는다.** `x[mask]` 는 `await x.maskedSelect(mask)` 로 남는다 —
결과의 길이가 값에 달려 있어 GPU 에서 한 번 읽어야 알 수 있고, 그것 하나 때문에
`at()` 을 비동기로 만들면 나머지 모든 쓰임이 이유 없이 `await` 를 달게 된다.

### 데이터 먹이기

`torch.utils.data` 자리다. **배치는 GPU 텐서라 `scope()` 안에서 받아야 한다** —
적재기가 대신 감쌀 수 없다. 텐서가 구역 밖으로 나가는 것이 목적이기 때문이다.

```ts
const set = new data.TensorDataset(images, labels);
const [train, valid] = data.randomSplit(set, [800, 200]);
const loader = new data.DataLoader(train, { batchSize: 32, shuffle: true });

for (let epoch = 0; epoch < 10; epoch++) {
  for (const [x, y] of loader) {           // 동기 반복자다
    await scope(async () => {
      opt.zeroGrad();
      const loss = crit.call(model.call(x), y);
      loss.backward();
      opt.step();
    });
  }
}
loader.length;    // 표본 수가 아니라 **배치 수**. torch 와 같다
```

섞기는 `manualSeed` 를 따른다 — torch 는 적재기에 별도 generator 를 두는데 여기서는
호스트 줄기 하나를 쓴다. 씨앗 하나가 층 초기화·dropout·텐서 팩토리에 이어 배치
순서까지 되돌린다. 에폭마다 다시 섞는 것은 torch 와 같다.

**`sampler` 와 `num_workers` 는 없다.** 앞은 지금 받쳐 줄 것이 없어서이고(이름만
놓으면 넣은 것이 조용히 무시된다), 뒤는 워커로 GPU 손잡이가 안 건너가서다. 있는
것은 `shuffle`·`dropLast`·`Subset`·`randomSplit`·`ConcatDataset` 이다.

### 저장하고 이어서 하기

**형식은 safetensors 다.** torch 의 `save`/`load` 는 pickle 이라 브라우저로 옮길 수도
옮겨서도 안 된다. 대신 이쪽을 들면 **파이썬 `borch`·numpy·HF 도구가 같은 파일을
읽는다** — 브라우저에서 학습해 자기 컴퓨터로 가져가는 길이 그것으로 열린다.

**중첩을 그대로 담는다** — 교재의 관용구가 그것이고, torch·파이썬 `borch` 와 같은
모양이다. 텐서가 아닌 것(숫자·글자·참거짓·`null`·배열)도 같이 간다.

```ts
import { save, load } from "borch";

const bytes = await save({
  model: model.stateDict(),
  opt: opt.stateDict(),
  sched: sched.stateDict(),
  epoch: 5,
});
// bytes 는 Uint8Array — IndexedDB 에 넣든 파일로 내리든 쓰는 쪽 몫이다
```

되돌릴 때는 **모델·옵티마이저·스케줄러를 같은 인자로 다시 세운 뒤** 얹는다.

```ts
const ck = load(bytes);
model.loadStateDict(ck.model);
opt.loadStateDict(ck.opt);
sched.loadStateDict(ck.sched);
```

구조는 머리의 `borch.tree` 에 JSON 으로 적히고 텐서는 지금까지처럼 평평하게 눕는다 —
**파이썬 쪽과 같은 스킴이라 서로의 체크포인트를 읽는다.** 나무가 없는 파일(남이 만든
safetensors)을 주면 평평한 텐서 표로 온다.

밑의 코덱이 필요하면 `encode`/`decode` 가 그 자리다. 평평한 `Record<string, Tensor>`
와 문자열 메타데이터만 다루고, 이름을 겹치지 않게 눕히는 `prefixed`·`unprefixed` 와
숫자를 메타데이터로 옮기는 `numbersToMeta`·`metaToNumbers` 가 같이 있다.

```ts
const { tensors, metadata } = decode(bytes);
```

**가중치만 되돌리면 안 된다.** 모멘텀·스텝 계수기·스케줄러의 에폭이 같이 가야 재개한
다음 스텝이 안 끊고 돌린 것과 같은 수를 낸다. `npm run serialize:ts` 가 그것을 비트
단위로 확인한다 — 열 스텝을 통으로 돌린 궤적과 다섯에서 끊었다 이은 궤적이 정확히
같아야 통과하고, 같은 러너가 그 파일을 **numpy 로만** 다시 뜯어 본다.

값은 언제나 float32 로 나간다. borch 의 `int64`·`bool` 은 이름표라 머리의
`__metadata__` 에 실린다 — 4 바이트짜리 몸에 `I64` 라고 적으면 남의 리더가 깨진다.

### 어디서 도는가

WebGPU 가 필요하다. **없으면 폴백하지 않고 거절한다.** 여기 있던 TF.js 판은 WebGPU 를
못 얻으면 WebGL 로 조용히 내려갔고, 그 때문에 한동안 **CPU 소프트웨어 경로에서 잰
성능 수치**를 GPU 의 것으로 읽었다. 조용히 느려지느니 안 도는 편이 낫다.

`--headed` 로만 잰다. 헤드리스 브라우저는 소프트웨어 래스터라이저(SwiftShader)를
주는데 **예외를 안 던지고 수만 이상해진다.** 그래서 러너가 어댑터를 먼저 찍고,
벤치와 정확도는 소프트웨어 어댑터에서 아예 거부한다.

### 얼마나 되나

Apple Metal 과 NVIDIA(RTX 4090) 두 벤더에서 **골든이 전건 같다** — 손으로 쓴
WGSL 이 Metal 전용이 아니라는 뜻이다. (4090 쪽은 그때의 표로 쟀다. 표가 자란 뒤로는
그 기계를 못 써서 다시 안 쟀고, 안 잰 것을 잰 것처럼 적지 않는다.)
벤치의 ResNet-18 자체도 진짜 torch 와 순방향·손실·역방향이 맞는 것을 확인했다.

**TF.js 판을 지운 근거가 이 표다.** 같은 기계·같은 벤치에서 나란히 잰 기록이고,
지금은 왼쪽 열이 없다.

| CIFAR ResNet-18, 배치 64 | TF.js 판 (지금은 없다) | **borch.ts** | **borch_webgpu** |
|---|---|---|---|
| ms/step | 154.9 | **118.5** | 123.4 |
| 에폭 | 2.02분 | **1.55분** | 1.61분 |
| 시험 정확도 (10 에폭, 늘리기 켬) | 60.4% | **64.6%** | 안 쟀다 |

오른쪽 두 열은 **같은 커널**이다. 차이 4.9ms 가 파이썬을 한 번 지나는 값이고,
그것이 이 결속의 값을 재는 유일한 수다.

> 정확도는 **늘리기를 켠 조건**이다. 끈 조건은 59.3% 로 자매보다 낮다. 한동안
> 65.5% / 62.4% 로 적혀 있었는데, 그때는 벤치 모델의 지름길 층 여섯이 학습되지
> 않은 상태였다 — 그 얼어붙음이 규제처럼 굴고 있었다. 자세한 것은
> [BORCH-TS.md](BORCH-TS.md) 의 T3 정확도 절에 있다.

설계와 실측 근거는 [BORCH-TS.md](BORCH-TS.md) 에 있다.

## 일부러 지원하지 않는 것

`CUDA` · 사전학습 가중치 · 혼합정밀도 · 분산 · `torch.compile`

**거절 목록이 긴 것이 의도다.** GPU·저장된 모델·사전학습은 브라우저를 벗어나야 배우는 것들이고,
여기서 흉내 내면 그 교훈이 사라진다.

## 적합성

목표는 "PyTorch 재현"이 아니라 **커리큘럼이 쓰는 범위 안에서의 동등성**이다.
왜 그렇게 잡았는지와 앞으로의 순서는 [ROADMAP.md](ROADMAP.md) 에 있다.

| 등급 | 지금 |
|---|---|
| **T1 값·기울기** (`allclose 1e-5`) | **100%** — 생성 케이스 132개 |
| **T2 오류 동등** | **12/12** — 예외 종류 · 검색 가능한 메시지 9/9 |
| **T3 표현(`repr`) 동등** | **15/15** |
| **dtype 승격** | **112/112** — 4 dtype × 4 연산 × 텐서·스칼라 |
| **저장소 공유** (view·slice) | **13/13** |
| **통합 시나리오** | **6/6** — 같은 코드를 임포트만 바꿔 돌린 결과 |
| **넓은 표면** (수학·모양·functional) | **67/67** |
| **흔한 API 이름** | **144/144** |
| T4 비트 동등 | **명시적 비목표** |

그중 **53건은 값이 아니라 "기울기가 흐르는가"만 묻는다.** 값만 대조하는 검사는
그래프가 끊긴 것을 못 본다 — 값은 맞기 때문이다. 실제로 GPU 쪽의 `roll` 과
`masked_select` 가 그렇게 조용히 끊겨 있었고 그때 골든은 전부 초록이었다.

그리고 **골든 2991건**이 세 구현을 **같은 기대값**에 대조한다. 코어는 브라우저
전용(1·3 차원 합성곱처럼 코어가 일부러 거절하는 것) 53 건을 빼고 2938 건을 본다 —
없는 것을 물으면 그건 검사가 아니라 오답이다. 진짜 torch 를 브라우저에 넣을 수
없어서, 네이티브에서 기대값을 굳혀 브라우저로 들고 간다.

```bash
uv run --with numpy --with torch python tests/golden.py dump   # 1단계: 굳힌다
uv run --with numpy python tests/golden.py check               # 2단계: 대조한다
uv run --with numpy python tests/export_json.py                # 3단계: 밖으로 뽑는다
```

3단계가 `tests/golden.json`(722KB)을 만든다. **파이썬이 아닌 구현도 이 기대값을 쓸 수
있게** 하려는 것이다 — 진짜 torch 를 돌려 얻은 숫자가 이 저장소에서 가장 비싼 자산인데,
파이썬 안에만 두면 다음 구현은 검증 없이 자란다.

**케이스 본문은 안 들어 있다.** `lambda L: L.amax(...)` 는 기계적으로 다른 언어가 되지
않는다. 받는 쪽은 같은 이름의 케이스를 자기 언어로 쓰고, 그 답을 여기서 맞춘다 —
비싼 절반(숫자)은 건너가고 싼 절반(호출 한 줄)은 다시 쓴다. borch.ts 가 실제로 그렇게
쓰고 1779 건을 지난다.

**이름이 안 맞으면 러너가 그것을 센다.** 한동안 안 셌는데, 그때 골든에 없는 이름
일곱을 들고 있으면서 골든의 다른 일곱을 안 쓴 상태가 "859 중 859, 0 건 남음" 으로
보였다 — 개수가 같아서 맞물렸다.

```bash
uv run --with numpy --with torch python tests/conformance.py
```

## 라이선스

Apache-2.0 · PI Lab

의존은 numpy(BSD-3-Clause) 하나다. 순수 파이썬 휠이라 다른 것을 묶어 팔지 않는다.

> **브라우저에 띄우는 쪽은 Pyodide 를 함께 서빙하게 된다.** Pyodide 는 MPL-2.0 이고,
> 실행 형태로 배포하면 **소스를 구할 길을 알려야 한다**(MPL §3.2). 우리 코드로 번지지는
> 않는다 — 파일 단위 약한 카피레프트라 borch 는 Apache-2.0 그대로다.
> 페이지 어딘가에 이 한 줄을 두면 된다:
>
> > 이 페이지는 [Pyodide](https://github.com/pyodide/pyodide) 를 포함하며 Mozilla Public
> > License 2.0 을 따릅니다. 소스는 해당 저장소에서 받을 수 있습니다.
>
> 무엇에 기대고 무엇을 지켜야 하는지는 [THIRD-PARTY.md](THIRD-PARTY.md) 에 정리했다.
