# borch — for agents

> Read this when a task says *in the browser*, *Pyodide*, *JupyterLite*, *WebGPU*, *nothing
> installed*, or *torch cannot be installed here* — and the code wants PyTorch's API. This is
> the page a coding agent needs; `CLAUDE.md` is for sessions working **on** borch, not with it.
> The machine-readable index of these documents is [`llms.txt`](https://playidea-lab.github.io/borch/llms.txt).

## What it is

Three implementations of one API, held to real PyTorch's **values, error messages and
printed form** on thousands of golden cases frozen from torch: a numpy core
(`import borch as torch`), a TypeScript runtime on WebGPU (`borch-ts`, hand-written WGSL, no
dependencies), and a Python binding over that runtime for Pyodide (`borch_webgpu`). Where a
feature is absent it is absent loudly — an error names it — rather than answered wrongly.
The site trains on the visitor's GPU: https://playidea-lab.github.io/borch/site/

## Which one — decide from where the code runs

| The code runs … | Use | Install | First line |
|---|---|---|---|
| in Python on a machine or CI, no GPU needed, torch unavailable | `borch` — the numpy core | `uv pip install pyborch` (or `pip install pyborch`) | `import borch as torch` |
| in Pyodide / JupyterLite / marimo, with WebGPU in the tab | `borch_webgpu` — the same wheel | `%pip install pyborch` | `import borch_webgpu as torch` |
| in Pyodide without WebGPU | `borch` — the numpy core, same wheel | `%pip install pyborch` | `import borch as torch` |
| a page with **no build step** (a doc page, a demo, a paper) | `borch-ts` from a CDN | nothing to install | `import { init, Tensor } from "https://cdn.jsdelivr.net/npm/borch-ts@0.3/+esm"` |
| in a web page, TypeScript or JavaScript | `borch-ts` | `npm install borch-ts` | `import { init, Tensor, nn, optim, scope } from "borch-ts"; await init();` |
| in a page with **no** GPU adapter and only needs a pretrained backbone from the hub | the `cpu` device (`import borch_cpu` / `cpu` namespace) | in the wheel / in `borch-ts` | not for training a model of your own |

One wheel, `pyborch`, carries `borch`, `borch_webgpu`, `borchvision` and `borch_cpu`.

## Ten rules that save a rewrite

1. **Alias, do not shadow.** `import borch as torch` covers attribute access
   (`torch.nn.Linear`). For `from borch.nn import Linear` plant the submodule paths under
   borch's own name first:

   ```python
   import borch
   borch.install("borch")   # then: from borch.nn import Linear
   ```

   The argument matters. Both of the following intercept every other library's
   `import torch` and are for a practice environment only, when the user asks for exactly that:

   ```text
   borch.install()                  # no argument = plants it as `torch`
   sys.modules["torch"] = borch     # the same, by hand
   ```

2. **Check a name before writing it.** [`site/assets/api-index.json`](https://playidea-lab.github.io/borch/site/assets/api-index.json)
   maps every public name to its module (`"AdamW": "optim.AdamW"`);
   [`api.json`](https://playidea-lab.github.io/borch/site/assets/api.json) carries signatures and docs. A name that is not
   there is not implemented. Say so; do not polyfill it quietly.
3. **Never coming, by design:** CUDA, mixed precision, distributed training,
   `torch.compile`. Do not write `.to("cuda")`, `autocast`, `DistributedDataParallel`.
4. **borch-ts differs from torch in six places, all in the first ten lines:**
   `await init()` first · `await loss.item()` (the only asynchronous read) ·
   `using s = scope()` around each step (GPU memory is not garbage-collected in time;
   `s.keep(t)` / `keepAlive` for what leaves the block) · `model.call(x)` instead of
   `model(x)` · `'cpu'` is where values are put down, not a compute device ·
   `await opt.step(closure)` for `LBFGS` alone.
5. **`borch_webgpu` hides four of those** (`init`, `item`, `scope`, `call`): the Python
   reads like torch, with no `await`. It needs the page to enter Python asynchronously;
   JupyterLite, marimo and the site's pages already do.
6. **`init()` refuses a software adapter** (SwiftShader, llvmpipe) rather than pretending
   a CPU rasteriser is a GPU. The switch to ask for one on purpose is in `InitOptions`;
   `probe().software` tells you what you got. A failed `init()` says which flag turns it on.
7. **Errors are torch's own** — same exception type, same message. A refusal is loud on
   purpose; do not catch it and continue with a substitute.
8. **borch-ts computes in float32** (WGSL has no `f64`); complex is `complex64` only, in
   all three implementations. The numpy core follows torch's dtype promotion.
9. **In Pyodide, install the wheel, never torch.** `%pip install pyborch` goes through
   micropip; real PyTorch has no Pyodide build, so `%pip install torch` fails and no
   `try: import torch` fallback will ever succeed there.
10. **Ship what leaves as ONNX:** `torch.onnx.export(model, x, path)` in `borch_webgpu`,
    `onnx.exportOnnx(model, sample)` in TypeScript — the numpy core has no ONNX export.
    Traced from one forward, checked
    against ONNX Runtime Web.

## Reading the source, if you go there

**The comments narrate defects that were fixed, in the past tense, and the sentence after
the story says what happens now.** This repository writes down what went wrong and why,
beside the code that stops it — so a comment can describe a silent wrong answer at length
and be followed by one line saying it now raises.

Measured 2026-09-10: an agent reading `borch-ts/src/optim.ts` carried the story of an
options object reaching the kernel as `[object Object]` into its answer as a live warning
that training would silently do nothing. It has raised `Invalid learning rate: [object
Object]` since the day that comment was written, in the same paragraph.

If a comment tells you something misbehaves, run it before you warn anyone. Nothing here is
a changelog; the guard is usually three lines below the story of why it exists.

## Supported range (short form; the long form is the book)

tensors (broadcasting, dtype promotion, indexing, views, `gather`, `masked_select` …) ·
autograd (`backward`, `.grad`, `no_grad`, `detach`) · reductions with backward · `nn`:
`Module`, `Linear`, `Conv1d/2d/3d`, pooling, `Embedding`, `LayerNorm`, `BatchNorm`,
`Dropout`, `Sequential`, `RNN/LSTM/GRU`, `MultiheadAttention`, `nn.Transformer` · losses:
MSE, L1, SmoothL1, BCE, BCEWithLogits, CrossEntropy, NLL · optim: SGD, Adam, AdamW,
RMSprop, LBFGS · schedulers: StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
LambdaLR, ReduceLROnPlateau · data: `Dataset`, `TensorDataset`, `DataLoader`,
`random_split` · `state_dict` / `save` / `load` · 25 `nn.functional` names · `complex64` ·
`torch.fft` and `stft` · torchvision `transforms` (`borchvision`) · ONNX export.

Absent and refused, with a reason each: `tests/torch_gap.py` prints the ledger.

## Checking support before writing code

Two files are deployed with the site and regenerated on every deployment from the
TypeScript declarations (`site/build_api.py`), so they are never older than the runtime:

- [`api-index.json`](https://playidea-lab.github.io/borch/site/assets/api-index.json) — a flat object, public name →
  module path in borch-ts (`"AdamW": "optim.AdamW"`, `"LayerNorm": "nn.LayerNorm"`,
  `"stft": "fft.stft"`). Under 100 KB; the quick answer to "does this exist".
- [`api.json`](https://playidea-lab.github.io/borch/site/assets/api.json) — `{source, note, modules[], total}`; each
  module is `{name, title, blurb{en,ko}, doc, symbols[], count}` and each symbol
  `{kind, name, signature, doc, members}`. Signatures and TSDoc for the whole surface.

The Python core and the binding follow torch's names, so a borch-ts name under `nn.`,
`optim.`, `fft.`, `data.` or `linalg.` is `torch.nn.…`, `torch.optim.…` and so on in Python.

```python
import json
import urllib.request

INDEX = "https://playidea-lab.github.io/borch/site/assets/api-index.json"
index = json.load(urllib.request.urlopen(INDEX))      # {"AdamW": "optim.AdamW", ...}
for name in ["AdamW", "LayerNorm", "MultiheadAttention", "stft", "autocast", "compile"]:
    print(f"{name:22s} {index.get(name, '— not in borch: say so, do not polyfill')}")
```

## Smoke tests — copy, run, judge

**Python, anywhere** (no browser, no GPU):

```python
import borch as torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
(x * x).sum().backward()
assert x.grad.tolist() == [2.0, 4.0, 6.0]
print(x.grad)  # tensor([2., 4., 6.])
```

**TypeScript, in a page with WebGPU:**

```ts
import { init, Tensor } from "borch-ts";

await init();                                 // a WebGPU adapter, never a software one
const x = Tensor.from([1, 2, 3], [3]);
console.log(await x.mul(x).sum().toArray());   // Float32Array [14]
```

**A page with no build step** (save as `x.html`, double-click it):

```html
<script type="module">
  import { init, Tensor } from "https://cdn.jsdelivr.net/npm/borch-ts@0.3/+esm";
  await init();                                  // refuses a software adapter
  console.log(await Tensor.from([1, 2, 3], [3]).mul(Tensor.from([1, 2, 3], [3])).sum().toArray());
</script>
```

**Pyodide notebook cell:**

```python
%pip install pyborch
import borch_webgpu as torch   # or `import borch as torch` when the tab has no WebGPU

w = torch.tensor(3.0, requires_grad=True)
((w - 5.0) ** 2).backward()
print(w.grad.item())            # -4.0
```

**WebGPU, from a checkout** (needs Playwright; opens a real browser and refuses a software
adapter): `uv run --with playwright python tests/browser/wheel_probe.py --build` builds the
wheel, installs it in Pyodide inside a worker, trains, exports ONNX, and prints the adapter's
name with the losses — the line to trust is the one that names the adapter.

A training loop in torch's shape — `nn.Sequential`, `CrossEntropyLoss`, `Adam`, a
`DataLoader` — runs unchanged on the numpy core and the binding; on borch-ts add the six
lines from rule 4. Ten tutorials that run every block in the page:
https://playidea-lab.github.io/borch/site/tutorials/

## Where the truth is

| Question | Read |
|---|---|
| the whole design, the guarantee, the supported range, what diverges and why | [`docs/BOOK.md`](https://playidea-lab.github.io/borch/docs/BOOK.md) |
| the TypeScript runtime specifically | [`BORCH-TS.md`](https://playidea-lab.github.io/borch/BORCH-TS.md) |
| what conformance means here and what will not be done | [`ROADMAP.md`](https://playidea-lab.github.io/borch/ROADMAP.md) |
| every public name, with module | [`api-index.json`](https://playidea-lab.github.io/borch/site/assets/api-index.json) |
| the browsable API reference | [`site/api/`](https://playidea-lab.github.io/borch/site/api/) |
| all of the documents in one file | [`llms-full.txt`](https://playidea-lab.github.io/borch/llms-full.txt) |
| the source, issues | https://github.com/playidea-lab/borch |

Sister libraries: [`bimm`](https://github.com/playidea-lab/bimm) (a model catalogue,
`bimm-ts` on npm) and [`borch-hub`](https://github.com/playidea-lab/borch-hub) (weights by
manifest and hash).
