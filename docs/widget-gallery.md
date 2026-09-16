# Widget gallery — catalogue & backlog

The gallery (`site/widgets.html`) is borch's embed front door: live ML anyone drops into a
page with one line. This file is the list we build from — what's shipped, and what's worth
authoring next.

**What makes a good gallery widget**
- **Live earns its place.** If a screenshot says it as well, it does not belong here. The
  best widgets *move* — a boundary bending, a curve falling, a ball rolling downhill.
- **Small-scale feasible.** It has to train in seconds on a laptop GPU (MNIST/CIFAR-class,
  small transformers). The browser is the ceiling; teaching-scale is the sweet spot.
- **Embeddable as-is.** JavaScript cell (WebGPU) so it runs on any non-isolated host; the
  Python twin shows only where the host is cross-origin-isolated.
- **One idea per widget.** A card teaches one thing.

Every widget is a lesson cell reached by `embed/embed.html?lesson=<id>&cell=<n>`. A "cheap"
candidate already exists as a cell; an "author" candidate needs a new cell written first
(and it lands in the curriculum too — the gallery and the lessons share one source).

---

## Shipped (15)

| Widget | lesson · cell | Shows |
|---|---|---|
| Attention, live | mini-transformer · 4 | The causal-attention matrix as a heatmap |
| A conv layer's kernels | cnn · 3 | Six 3×3 filters at initialisation |
| A loss curve falling | adam · 3 | Adam training a tiny problem |
| Why depth needs a skip | resnet · 3 | The gradient shrinking with depth |
| Batch norm, drawn | batchnorm · 3 | Activations before and after |
| Dropout's mask | dropout · 3 | The random mask that regularises |
| The singular-value spectrum | eig-svd · 2 | The SVD's singular values, plotted |
| Softmax as probabilities | softmax-ce · 3 | Logits turned into a distribution |
| Vectors & dot products | f-vectors · 3 | Two vectors and the angle between them |
| Broadcasting, drawn | f-broadcast · 2 | How shapes stretch to meet |
| A training loop | training · 3 | Loss coming down over steps |
| Gradients, flowing | autograd · 4 | A gradient read straight off the graph |
| Layer norm, drawn | layernorm · 3 | Each row centred on its own |
| Attention weights | attention · 2 | Which positions attend to which |
| The FFT | signals-fft · 1 | A signal and its frequencies |

---

## Backlog — author these

### ★ Flagship

- **Live decision boundary** — a small MLP on 2-D points (moons / spiral / circles); the
  decision boundary bends to fit as it trains. The most-embedded live-ML widget ever is
  this (TensorFlow Playground). Feasible: evaluate the net on a grid, draw the grid with
  `show()`. **Effort L · highest pull.** Author as a lesson (foundations or basics).

### Live-contrast — the whole point is watching two runs differ

- **Learning rate: too high / too low / right** — the same problem three times; one
  diverges, one crawls, one converges. The commonest beginner mistake, made visible.
  **Effort M.**
- **Overfitting, live** — train on a handful of points; train loss → 0 while a held-out
  curve turns back up. The memorise-vs-generalise story as a curve. (The
  "honest-measurement" capstone is its prose sibling.) **Effort M.**
- **Gradient descent on a loss surface** — a ball tracing its path down a 2-D landscape,
  step by step. **Effort M.**
- **Batch size and noise** — bigger batches, a smoother loss curve. **Effort S–M.**

### Interactive — the reader's input matters (borch-distinctive: it runs in *their* tab)

- **Activation explorer** — a slider over relu / sigmoid / tanh / gelu showing each shape
  *and its gradient*, so vanishing gradients are visible, not asserted. **Effort M.**
- **Initialization matters** — 0 / too-large / Xavier init; watch training start or stall.
  **Effort M.**

### borch-distinctive

- **Same code, JS and Python** — one widget, two language tabs, losses matching digit for
  digit. Demonstrates the "change one import" story live (Python tab shows on isolated
  hosts, e.g. the gallery itself). **Effort S** (an existing dual-language cell may serve).

---

## Backlog — cheap adds (existing cells, not yet in the gallery)

Ready to drop in; no authoring, just add to the gallery list.

| Candidate | lesson · cell | Shows |
|---|---|---|
| Curve fitting | curve-fitting · 2 | A curve learned to fit points |
| Autoencoder | autoencoder · 0 | An image reconstructed through a bottleneck |
| Adversarial example | adversarial · 2 | A tiny pixel change that flips the label |
| Character RNN | char-rnn · 0 | Text generated one character at a time |
| A batch, drawn | data · 4 | Samples stacked into a batch |
| Quickstart | quickstart · 0 | The smallest end-to-end run |

*(Image-classifier cells are deliberately left out — they download CIFAR and train for
several epochs, too heavy for a gallery card.)*

---

## How to add one

1. **Cheap add:** append a row to the `WIDGETS` grid in `site/widgets.html` (+`ko/`) with
   the lesson id and cell index; confirm it renders (all cards mount, no error box).
2. **Author:** write the cell in its lesson first (it joins the curriculum), then add the
   gallery row. Prefer a deterministic visual (`show`/`plot`) over a convergence a reader
   has to tune.
3. Keep the two languages in step — the gallery grid is generated from one list.
