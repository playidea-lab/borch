"""A small ResNet, built and trained whole, run with only the import changed.

`scenario.py` next door runs tutorial-shaped code and compares **what came out at the
end** — an accuracy, a final loss, a sum of logits. That catches a piece that is wrong
from the first step. It cannot say **when** a run started to part, and a divergence that
begins small and compounds arrives at the end looking like any other wrong number.

So this one keeps the loss at **every** step, and the gradient of **every** parameter at
the first one, and compares those. Then a difference has a place and a time: *the third
block's second convolution, before a single weight had moved.*

## Why a residual network in particular

The golden compares one operation against torch at a time and nothing here is a new
kernel. What is new is the **shape of the graph**:

- A residual join is where **two gradients meet and are added.** Every other path in
  this repository is a chain, and a chain never asks whether accumulation at a fork is
  right — an implementation that overwrites instead of adding still trains, and still
  brings the loss down.
- The downsample shortcut makes the two sides of that fork **different lengths**, so an
  ordering mistake shows as a value rather than a shape.
- BatchNorm in `train()` writes its running statistics as a **side effect** while
  gradients flow through the batch statistics. Those buffers are compared here after
  training, because nothing else in this repository reads them back out — an update with
  the wrong momentum, or one taken from the biased variance, leaves the training loss
  untouched and only appears when the model is put in `eval()`.
- The head averages over the spatial dimensions rather than flattening, so the last
  linear layer sees a reduction's gradient.

## Kept determinstic on purpose

Every weight and every input is built in numpy and assigned in, so the two runs start
from **the same bytes**. Nothing is drawn inside either library — one shared stream is a
borch decision, and a scenario that depended on it would be comparing two random walks.

    uv run --with numpy --with torch python tests/resnet.py real
    uv run --with numpy python tests/resnet.py borch
"""

import sys

import numpy as np

LIB = sys.argv[1] if len(sys.argv) > 1 else "borch"
if LIB == "borch":
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import borch as torch
    sys.modules["torch"] = torch
else:
    import torch

nn = torch.nn
F = nn.functional

STEPS = 30
BATCH, CHANNELS, SIDE, CLASSES = 8, 3, 16, 4
WIDTH = 8


class Residual(nn.Module):
    """Two convolutions and a shortcut. `stride` 2 makes the shortcut a 1×1 convolution.

    The join is the point of the whole file: `out + short` is where two gradients meet.
    """

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        # **Identity when the shape survives.** Giving every block a 1×1 anyway would be
        # simpler and would stop the identity path from ever being asked about.
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
            self.shortbn = nn.BatchNorm2d(cout)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short = x if self.short is None else self.shortbn(self.short(x))
        return F.relu(out + short)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(CHANNELS, WIDTH, 3, padding=1, bias=False)
        self.stembn = nn.BatchNorm2d(WIDTH)
        self.b1 = Residual(WIDTH, WIDTH)                    # identity shortcut
        self.b2 = Residual(WIDTH, WIDTH * 2, stride=2)      # 1×1 shortcut, halved
        self.head = nn.Linear(WIDTH * 2, CLASSES)

    def forward(self, x):
        x = F.relu(self.stembn(self.stem(x)))
        x = self.b2(self.b1(x))
        # Average over height and width. `mean` twice rather than a pooling layer, so the
        # last linear sees a reduction's gradient.
        return self.head(x.mean(dim=3).mean(dim=2))


rng = np.random.default_rng(7)
images = rng.standard_normal((BATCH, CHANNELS, SIDE, SIDE)).astype(np.float32)
labels = rng.integers(0, CLASSES, BATCH).astype(np.int64)

model = Net()
# **The same bytes on both sides.** Kaiming init draws, and the two libraries draw
# differently on purpose, so nothing here is left to either one's stream.
for name, p in model.named_parameters():
    shape = tuple(int(v) for v in p.shape)
    fan = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
    scale = float(np.sqrt(2.0 / max(fan, 1)))
    p.data = torch.tensor((rng.standard_normal(shape) * scale).astype(np.float32))

crit = nn.CrossEntropyLoss()
# Weight decay and momentum together: decay applied to the gradient before the momentum
# buffer is a different answer from decay applied after, and both train.
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

x, y = torch.tensor(images), torch.tensor(labels)
out = {}

model.train()
for step in range(STEPS):
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    if step == 0:
        # Before a single weight has moved, so a difference here is the backward pass and
        # nothing else — not the optimizer, not an accumulated drift.
        for name, p in model.named_parameters():
            g = np.asarray(p.grad.numpy(), dtype=np.float64)
            out[f"grad·{name}"] = float(np.sqrt((g * g).sum()))
    out[f"loss·{step:02d}"] = float(loss.item())
    opt.step()

# The buffers BatchNorm wrote on the way. Nothing else in this repository reads these
# back, and a wrong momentum leaves the training loss looking perfect.
#
# **The two libraries hand these out as different types**, which is why this reads
# defensively rather than calling `.numpy()`. torch yields a `Tensor` from
# `named_buffers()`; borch yields the raw ndarray it registered, and
# `num_batches_tracked` as a plain `int`. `state_dict()` is a tensor on both sides —
# that path was built deliberately and says so — so it is only this one that parts.
# Not worked around silently: the divergence is real and is reported separately.
for name, buf in model.named_buffers():
    if "num_batches" in name:
        continue
    raw = buf.numpy() if hasattr(buf, "numpy") else buf
    out[f"buffer·{name}"] = float(np.asarray(raw, dtype=np.float64).sum())

# `eval()` swaps the batch statistics for the running ones — a different path through the
# same layer, and the only one that reads what the loop above accumulated.
model.eval()
with torch.no_grad():
    logits = model(x)
    out["eval·logit sum"] = float(logits.sum().item())
    out["eval·argmax agreement"] = float(
        (logits.argmax(dim=1) == y).sum().item()) / BATCH

for key, value in out.items():
    print(f"{key}\t{value!r}")
