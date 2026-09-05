# Other people's work — what this leans on, and what has to be honoured

borch is Apache-2.0. What follows is **the code of other people's that is needed
at run time**, and its terms. The licences are not written from memory —
**each was read out of the file that was downloaded.**

> **This repository does not redistribute any of the below.** `vendor/` is in
> `.gitignore`, and what gets committed is a few lines of sha256
> (`tests/browser/assets.lock`).
> **Whoever puts it in a browser does redistribute them**, though — and the terms
> below apply then.

---

## The core `borch`

| | licence | how it was confirmed |
|---|---|---|
| **numpy** | BSD-3-Clause | `numpy-2.0.2.dist-info/LICENSE.txt` inside the wheel |
| **Pillow** 10.2.0 (`vendor/pyodide/pillow-10.2.0-…whl`, the Pyodide build) | HPND (MIT-CMU) | `PIL/LICENSE` inside the wheel; it decodes the images `ImageFiles` is handed in the browser checks |

That is all of it. One pure-Python wheel; numpy is the only dependency, and Pillow is vendored for the browser checks alone.

Using it in a browser needs Pyodide, and that is **something the host page
loads** rather than something borch ships.

## The browser side: `borch-webgpu` and `borch.ts`

| | licence | how it was confirmed |
|---|---|---|
| **Pyodide** (the Python side only) | **MPL-2.0** | `LICENSE` in pyodide/pyodide 0.27.2 |
| **The CPython standard library** (the Python side only) | PSF License | enclosed in `python_stdlib.zip` |
| **numpy** (the Python side only) | BSD-3-Clause | as above |

**This repository does hold those binaries** — six files under
`vendor/pyodide/`, with their sha256 in `tests/browser/assets.lock`. Holding them
means redistributing them, so where to obtain the source is written here:

| | downloaded from | source |
|---|---|---|
| Pyodide 0.27.2 (MPL-2.0) | `https://cdn.jsdelivr.net/pyodide/v0.27.2/full/` | `https://github.com/pyodide/pyodide` tag `0.27.2` |

## The tutorial data

| | origin | what is in the repository |
|---|---|---|
| **CIFAR-10** | `https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz` (Krizhevsky, 2009) | `site/assets/data/` — the first 2,500 images as a JPEG sprite (1.1MB) |

It is **a subset re-encoded as JPEG** rather than the original, so the pixels are
not the original's. The code that builds it is `site/fetch_data.py`, and with no
randomness in it, it reproduces. Tutorials 4 and 5 read it, and that an accuracy
obtained here must not be compared against the numbers in a paper is written down
in both of those.

MPL-2.0 works **per file**, so it does not spread into our code. Modifying those
files and shipping them would mean releasing the modifications under the same
licence, and they are not modified — `assets.lock` holds that as a check
(`tests/test_site.py::test_vendored_pyodide_matches_its_lock`).

**`borch.ts`'s row in this table is empty.** TypeScript and WGSL and nothing
else, with no run-time dependency. It calls the browser's WebGPU directly.

**TensorFlow.js used to be here** (Apache-2.0, Copyright 2024 Google LLC). The
Python side's GPU implementation stood on it, and the page loaded `tf.min.js` and
`tf-backend-webgpu.min.js` from a CDN. Replacing it with hand-written WGSL
**removed the dependency entirely** — and with it, that much of what a
redistributor has to honour.

---

## What has to be honoured

### MPL-2.0 (Pyodide) — **the source has to be findable**

This one is different in kind from the rest. MPL is **weak copyleft, per file.**

- It **does not spread** into our code. Sitting on one page with borch, which is
  Apache-2.0, does not make borch MPL (the "Larger Work" clause)
- But **distributing Pyodide in executable form** (that is, serving
  `pyodide.asm.wasm` and the rest from a page) means **telling the recipient how
  to obtain those files in source form** (§3.2)

In practice, putting this line next to what is distributed is enough:

> This page includes Pyodide (https://github.com/pyodide/pyodide), which is
> licensed under the Mozilla Public License 2.0. The source is available at the
> address above.

### PSF and BSD-3-Clause — keep the notice

Keep the copyright notice together with the full licence text. Nothing further is
required.

---

## The data

**CIFAR-10** is not in this repository (`.gitignore`). Whoever downloads and uses
it cites it as convention has it.

> Krizhevsky, A. *Learning Multiple Layers of Features from Tiny Images.* 2009.

No explicit licence is attached to it, so anyone planning to redistribute beyond
research and study is safer checking with the source first.

## The relationship to PyTorch

borch **took no code** from PyTorch (BSD-3-Clause). It matched the shape of the
API, and the value comparisons call real torch **in the tests alone** (a `dev`
extra).

The README already carries a warning about the name and about
`sys.modules["torch"] = borch`. It is a place that can turn into a trademark
question, so it is worth going over once before any public distribution.

---

## What this document is not

What is written here is **what was confirmed in the files** and what each
licence's text says. It is not legal advice — the MPL-2.0 item in particular
depends on the actual form of distribution, so having somebody who reads licences
check it before release is the right thing to do.
