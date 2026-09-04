# borch

**PyTorch's shape, in a browser tab.** Three implementations of one arithmetic —
a numpy core (`import borch as torch`), a TypeScript runtime on WebGPU (`borch-ts`),
and a Python binding over that runtime for Pyodide (`borch_webgpu`) — held to real
PyTorch's values, errors and printed form within the range a curriculum uses.

- **See it run:** https://playidea-lab.github.io/borch/site/ — the playground trains
  on your GPU; ten lessons and ten tutorials run every code block in the page.
- **The long document** — how the values are guaranteed, the supported range, what is
  deliberately absent and why, borch.ts's design, conformance — is
  [`docs/BOOK.md`](docs/BOOK.md). This page is the door; that one is the house.

## What it is not

Not PyTorch. CUDA, distributed training, mixed precision and `torch.compile` are
never coming — they cannot exist in a browser, or learning them means leaving it.
An absent feature beats a wrong answer, so what is missing is written down as missing:
`tests/torch_gap.py` prints the current count per namespace, and every gap carries a
reason.

## Thirty seconds

**In a browser, nothing installed.** Open the
[playground](https://playidea-lab.github.io/borch/site/playground.html) and press Run.
The landing page times its own first run; measured nightly on a real adapter
(`tests/browser/first_run.py`).

**Python, on your machine.**

```
uv pip install pyborch
```

```python
import borch as torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
(x * x).sum().backward()
print(x.grad)           # tensor([2., 4., 6.])
```

**TypeScript, in a page.**

```
npm install borch-ts
```

```ts
import { init, Tensor } from "borch-ts";

await init();                          // asks for a WebGPU adapter, refuses a software one
const x = Tensor.from([1, 2, 3], [3]);
console.log(await x.mul(x).sum().toArray());   // Float32Array [14]
```

Both examples are run as written by the checks (`tests/test_document_examples.py`,
`borch-ts/test/readme.ts`), so if they stop working the build says so.

## How it is guaranteed

**4744 golden cases** compare all three implementations against the same answers frozen
from real torch — values, shapes, gradients, exception types and messages, and `repr`.
The core runs them natively and in Pyodide; the binding and borch.ts run them in a
browser on a real GPU, and every number they print carries the adapter's name, because a
software adapter answers every WebGPU call correctly and proves nothing about a GPU.

Where this library deliberately parts from torch — a handful of places, each with a
measurement beside it — is listed on the landing page and pinned by
`borch-ts/test/parity.ts`. Everything else that differs is a defect, and the ledgers
(`tests/torch_gap.py`, `borch-ts/test/run.py`) hold zero gaps without a reason.

**Or a notebook.** `%pip install pyborch` then `import borch_webgpu as torch` in any Pyodide — the [notebook page](https://playidea-lab.github.io/borch/site/notebook.html) is JupyterLite with the wheel already on its shelf, training on the tab's GPU 6 s after opening.

**The file leaves as ONNX.** `onnx.exportOnnx(model, sample)` in TypeScript, `torch.onnx.export(model, x, path)` in Python — traced from one forward, written without a dependency, and checked by ONNX Runtime Web reproducing the forward (3.5e-8; see the book).

## Where things are

| | |
|---|---|
| `borch/` | the numpy core — the reference implementation |
| `borch-ts/src/` | the TypeScript runtime: hand-written WGSL, zero dependencies |
| `borch_webgpu/` | the Python binding over borch.ts, for Pyodide |
| `borchvision.py` · `borch-ts/src/vision.ts` | torchvision's transforms, ops, datasets |
| `tests/` | the golden, the ledgers, and the checks that police these documents |
| `site/` | the playground, lessons, tutorials, and the API reference |
| `docs/BOOK.md` | the long document |
| `ROADMAP.md` | what conformance means here, and what will not be done |

Sister libraries: [`bimm`](https://github.com/playidea-lab/bimm) (a model catalogue, on
npm as `bimm-ts`) and [`borch-hub`](https://github.com/playidea-lab/borch-hub) (weights
by manifest and hash).

## Working on it

Every browser check is listed in `.github/workflows/gpu.yml` and run nightly by
`tests/browser/nightly.py`; `CLAUDE.md` holds the three rules a session has to know.
Native tests: `uv run --with pytest --with numpy --with torch --with torchvision --with scipy pytest tests/ -q`.

## Licence

Apache-2.0. Third-party notices are in `THIRD-PARTY.md`.
