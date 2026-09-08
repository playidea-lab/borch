---
name: borch
description: Use when a task wants PyTorch's API (tensors, autograd, nn.Module, optim) in a browser tab, in Pyodide or JupyterLite, on WebGPU, or on a machine where real torch cannot be installed. borch is three implementations of one torch-shaped API held to real PyTorch's values and error messages — this skill picks the entry point, states the rules that save a rewrite, and gives recipes the site's checks run every night. Also use when someone asks for "PyTorch in the browser", "torch without CUDA in a notebook", "train a model on WebGPU", or a torch-compatible library for a place torch does not run.
---

# borch — PyTorch's shape, in a browser tab

One API, three implementations, held to real torch's **values, error messages and printed
form** on thousands of golden cases. The complete agent page is
https://playidea-lab.github.io/borch/AGENTS.md; the machine-readable index of every public name is
https://playidea-lab.github.io/borch/site/assets/api-index.json. This skill is the short form.

## 1. Pick the entry point from where the code runs

| The code runs … | Use | Install | First line |
|---|---|---|---|
| Python on a machine or CI, no GPU | `borch` (numpy core) | `pip install pyborch` | `import borch as torch` |
| Pyodide / JupyterLite / marimo, WebGPU in the tab | `borch_webgpu` (same wheel) | `%pip install pyborch` | `import borch_webgpu as torch` |
| Pyodide without WebGPU | `borch` (same wheel) | `%pip install pyborch` | `import borch as torch` |
| a web page, TypeScript/JavaScript | `borch-ts` | `npm install borch-ts` | `await init()` first |

Never `%pip install torch` in Pyodide — real PyTorch has no Pyodide build, and a
`try: import torch` fallback never succeeds there.

## 2. Rules that save a rewrite

1. Alias, do not shadow: `import borch as torch`. For `from borch.nn import Linear` call
   `borch.install("borch")` first — `install()` with no argument plants it as `torch` and
   intercepts every other library's `import torch`.
2. Check a name before writing it: fetch `api-index.json` (name → module path,
   `"AdamW": "optim.AdamW"`). A name that is not there is not implemented — say so, do not
   polyfill it quietly.
3. Absent by design, never coming: CUDA, mixed precision, distributed training,
   `torch.compile`. Do not write `.to("cuda")`, `autocast`, `DistributedDataParallel`.
4. borch-ts differs from torch in six places: `await init()` first · `await loss.item()` is
   the only asynchronous read · `using s = scope()` around each step, `s.keep(t)` or
   `keepAlive` for what leaves it · `model.call(x)` not `model(x)` · `'cpu'` is not a
   compute device · `await opt.step(closure)` for LBFGS alone.
5. `borch_webgpu` hides four of those — Python reads like torch, no `await`.
6. `init()` refuses a software adapter (SwiftShader, llvmpipe) and says which flag turns a
   real one on; `probe().software` tells you what you got.
7. Errors are torch's own, and a refusal is loud on purpose — do not catch and continue.
8. borch-ts computes in float32; complex is `complex64` only, in all three.
9. Export with `torch.onnx.export(model, x, path)` in `borch_webgpu` or `onnx.exportOnnx(model, sample)`
   in TypeScript; the numpy core has no ONNX export. The file is checked against ONNX Runtime Web.

## 3. Recipes

### A training loop in Python — the numpy core and `borch_webgpu` run this unchanged

```python
import borch as torch          # in Pyodide with WebGPU: import borch_webgpu as torch

torch.manual_seed(0)
x = torch.randn(256, 2)
y = ((x[:, 0] * x[:, 1]) > 0).long()          # XOR-shaped labels

model = torch.nn.Sequential(
    torch.nn.Linear(2, 32), torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(200):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()

acc = (model(x).argmax(1) == y).float().mean().item()
print(f"loss {loss.item():.3f}  acc {acc:.2f}")   # acc well above 0.5
assert acc > 0.8
```

### The same loop in a page — TypeScript on WebGPU

```ts
import { init, Tensor, nn, optim, scope, keepAlive, manualSeed } from "borch-ts";

await init();                                   // a WebGPU adapter, never a software one
manualSeed(0);
const x = keepAlive(Tensor.randn([256, 2]));
const y = keepAlive(x.narrow(1, 0, 1).mul(x.narrow(1, 1, 1)).gt(0).squeeze(1).to("int64"));

const model = new nn.Sequential(new nn.Linear(2, 32), new nn.ReLU(), new nn.Linear(32, 2));
const crit = new nn.CrossEntropyLoss();
const opt = new optim.Adam(model.parameters(), 1e-2);

for (let step = 0; step < 200; step++) {
  using s = scope();                            // frees the step's GPU buffers on exit
  opt.zeroGrad();
  const loss = crit.call(model.call(x), y);
  loss.backward();
  opt.step();
  if (step % 50 === 0) console.log(step, await loss.item());   // the only await
}
```

### Export what you trained as ONNX — `borch_webgpu` and borch-ts; the numpy core does not export

```python
import borch_webgpu as torch   # in the browser; `borch` (numpy) has no torch.onnx

model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
torch.onnx.export(model, torch.randn(1, 4), "/work/model.onnx")   # traced from one forward
```

In TypeScript: `const bytes = onnx.exportOnnx(model, sample);` — then hand the bytes to
ONNX Runtime Web or save them.

### A notebook cell (JupyterLite, marimo, any Pyodide)

```python
%pip install pyborch
import borch_webgpu as torch   # or `import borch as torch` when the tab has no WebGPU

w = torch.tensor(3.0, requires_grad=True)
((w - 5.0) ** 2).backward()
print(w.grad.item())            # -4.0
```

## 4. Where the truth is

- the whole design, the guarantee, the supported range, what diverges and why:
  https://playidea-lab.github.io/borch/docs/BOOK.md
- ten tutorials that run every block in the page: https://playidea-lab.github.io/borch/site/tutorials/
- every public name with its module: https://playidea-lab.github.io/borch/site/assets/api-index.json
- everything in one file: https://playidea-lab.github.io/borch/llms-full.txt
