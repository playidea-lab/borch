"""A small ResNet, built and trained whole, on whichever library it is handed.

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

## Kept deterministic on purpose

Every weight and every input is built in numpy and assigned in, so the runs start from
**the same bytes**. Nothing is drawn inside any library — one shared stream is a borch
decision, and a scenario that depended on it would be comparing random walks.

## One body, three surfaces

`report` takes the library rather than importing one, because the same run has to be
askable of the core, of the binding **inside a browser**, and of anything else that grows
a torch-shaped surface. A second copy for the browser would be a second thing to keep in
step, and the one that is not run is the one that drifts.

    uv run --with numpy --with torch python tests/resnet.py real
    uv run --with numpy python tests/resnet.py borch
    uv run --with playwright python tests/browser/run.py --lib borch_webgpu --resnet
"""

import numpy as np

STEPS = 30
BATCH, CHANNELS, SIDE, CLASSES = 8, 3, 16, 4
WIDTH = 8


def minstd(shape, scale, seed):
    """A spread **computed rather than drawn**, so TypeScript can compute the same one.

    Borrowed from `tests/cases.py`, where it was written for the deformable convolution's
    fixture and for the same reason: a fixture from `numpy.random` can be asked of the
    Python surfaces and not of the JavaScript one, and then the comparison stops one
    surface short.

    MINSTD — `s ← s·48271 mod 2³¹−1`. The modulus is small enough that the product stays
    under 2⁵³, so JavaScript's float64 multiply is exact and the two sides agree bit for
    bit. **Restated here rather than imported**: `cases.py` is 800KB of case table that
    the browser page copies in whole, and this file is meant to be loadable on its own.
    The two are pinned to each other by `test_resnet.py`, which fails if they part.
    """
    total = int(np.prod(shape)) or 1
    out = np.empty(total, dtype=np.float32)
    state = seed % 2147483647 or 1
    for i in range(total):
        state = (state * 48271) % 2147483647
        out[i] = (state / 2147483647.0 - 0.5) * 2.0 * scale
    return np.ascontiguousarray(out.reshape(shape))


def _build(torch):
    """The network and its weights — **one construction path**, so `shapes` below reports
    on the same object `report` trains rather than on a second one built alongside it."""
    nn = torch.nn
    F = nn.functional

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
            # **Identity when the shape survives.** Giving every block a 1×1 anyway would
            # be simpler and would stop the identity path from ever being asked about.
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
            # Average over height and width. `mean` twice rather than a pooling layer, so
            # the last linear sees a reduction's gradient.
            return self.head(x.mean(dim=3).mean(dim=2))

    images = minstd((BATCH, CHANNELS, SIDE, SIDE), 1.7, 11)
    # Labels from the same stream, folded into the class count. Drawing them would put
    # this fixture back out of TypeScript's reach for the sake of eight integers.
    labels = np.array([int(abs(v) * 1000) % CLASSES
                       for v in minstd((BATCH,), 1.0, 29)], dtype=np.int64)

    model = Net()
    # **The same bytes on every surface.** Kaiming init draws, and the libraries draw
    # differently on purpose, so nothing here is left to any of their streams — and the
    # generator is one JavaScript can compute, so borch.ts can build this network too
    # rather than being the one surface the comparison cannot reach.
    #
    # The seed is the parameter's position, not a running stream: iteration order is a
    # property of each library's `named_parameters`, and a shared stream would make every
    # later weight depend on it. It matches today (measured, all four), and this way a
    # future reordering shows up as one weight moving rather than all of them.
    for index, (name, p) in enumerate(model.named_parameters()):
        shape = tuple(int(v) for v in p.shape)
        fan = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
        scale = float(np.sqrt(6.0 / max(fan, 1)))      # uniform, matched to Kaiming's variance
        p.data = torch.tensor(minstd(shape, scale, 101 + index * 7))
    return model, images, labels


def report(torch):
    """Trains the network on `torch` and returns every number the comparison reads."""
    nn = torch.nn
    model, images, labels = _build(torch)

    crit = nn.CrossEntropyLoss()
    # Weight decay and momentum together: decay applied to the gradient before the
    # momentum buffer is a different answer from decay applied after, and both train.
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    x, y = torch.tensor(images), torch.tensor(labels)
    out = {}

    model.train()
    for step in range(STEPS):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        if step == 0:
            # Before a single weight has moved, so a difference here is the backward pass
            # and nothing else — not the optimizer, not an accumulated drift.
            for name, p in model.named_parameters():
                g = np.asarray(_numpy(p.grad), dtype=np.float64)
                out[f"grad·{name}"] = float(np.sqrt((g * g).sum()))
        out[f"loss·{step:02d}"] = float(loss.item())
        opt.step()

    # The buffers BatchNorm wrote on the way. Nothing else in this repository reads these
    # back, and a wrong momentum leaves the training loss looking perfect.
    for name, buf in model.named_buffers():
        if "num_batches" in name:
            continue
        out[f"buffer·{name}"] = float(np.asarray(_numpy(buf), dtype=np.float64).sum())

    # `eval()` swaps the batch statistics for the running ones — a different path through
    # the same layer, and the only one that reads what the loop above accumulated.
    model.eval()
    with torch.no_grad():
        logits = model(x)
        out["eval·logit sum"] = float(logits.sum().item())
        out["eval·argmax agreement"] = float(
            (logits.argmax(dim=1) == y).sum().item()) / BATCH
    return out


def shapes(torch):
    """The parameter list and each one's sum right after it is filled.

    **Not part of the comparison** — a way to tell three explanations apart when a surface
    answers differently: a different iteration order (so the init loop hands each weight
    somebody else's draw), an assignment that did not take, or an honest numerical
    divergence. Only the last is a defect, and they look identical from the loss alone.
    """
    got, _images, _labels = _build(torch)
    return "\n".join(
        f"{name} {tuple(int(v) for v in p.shape)}\t{float(np.asarray(_numpy(p.data)).sum()):.8f}"
        for name, p in got.named_parameters())


def stages(torch):
    """The sum after each stage of one forward pass, in `train()` and then in `eval()`.

    **Not part of the comparison** either. When two surfaces start from the same weights
    and the same input and the loss already differs at step zero, the loss says only
    *that* the forward parted. This says where, and the train/eval pair separates a
    difference in the batch statistics from one in the layer itself.
    """
    model, images, _labels = _build(torch)
    F = torch.nn.functional
    x = torch.tensor(images)
    lines = []
    for mode in ("train", "eval"):
        getattr(model, mode)()
        a = model.stem(x)
        b = F.relu(model.stembn(a))
        c = model.b1(b)
        d = model.b2(c)
        e = d.mean(dim=3).mean(dim=2)
        for label, t in (("stem", a), ("stembn+relu", b), ("b1", c), ("b2", d),
                         ("mean", e), ("head", model.head(e))):
            lines.append(f"{mode}·{label}\t{float(t.sum().item()):.8f}")
    return "\n".join(lines)


def _numpy(value):
    """**The libraries hand these out as different types.**

    torch yields a `Tensor` from `named_buffers()`; borch yields the raw ndarray it
    registered, and `num_batches_tracked` as a plain `int`. `state_dict()` is a tensor on
    both — that path was built deliberately and says so — so it is only this one that
    parts. Read defensively here rather than papered over: the divergence is real and is
    reported on its own.
    """
    return value.numpy() if hasattr(value, "numpy") else value


FROZEN = "tests/resnet.json"


def compare(got, root):
    """`got` against the frozen torch answers. Returns the lines that parted.

    **The browser surfaces cannot ask torch.** The native comparison runs both sides live
    and needs no file; a page has no torch to run, so what it compares against has to have
    been written down by something that did. Same two-stage shape as the golden.
    """
    import json
    import pathlib

    path = pathlib.Path(root) / FROZEN
    if not path.exists():
        # **Raised rather than returned.** Coming back as one more line in the list of
        # differences made the caller print "the run parted from torch's" over "there is
        # nothing to compare against" — not measured and measured-wrong wearing the same
        # sentence, which is the failure this whole file was written to stop.
        raise SystemExit(
            f"no frozen answers: {FROZEN} — there is nothing to compare against, which is\n"
            "not the same as a divergence.\n"
            "  first: uv run --with numpy --with torch python tests/resnet.py freeze")
    want = json.loads(path.read_text(encoding="utf-8"))
    # **A reader that found nothing would report agreement.** The count is checked before
    # the values, because "every key I looked at matched" is also true of no keys at all.
    if len(got) < len(want):
        missing = sorted(set(want) - set(got))
        return [f"{len(missing)} of the {len(want)} answers never came back — "
                f"{missing[:4]}{' …' if len(missing) > 4 else ''}"]
    bad = []
    for key, expected in sorted(want.items()):
        mine = got.get(key)
        if mine is None:
            bad.append(f"{key} — absent")
            continue
        rel = abs(expected - mine) / max(1.0, abs(expected))
        if rel > 1e-4:
            bad.append(f"{key} — torch {expected!r} · here {mine!r}  (relative {rel:.2e})")
    return bad


def main(argv):
    import sys

    which = argv[1] if len(argv) > 1 else "borch"
    if which == "freeze":
        import json
        import pathlib

        import torch
        out = pathlib.Path(__file__).resolve().parent.parent / FROZEN
        out.write_text(json.dumps(report(torch), indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"froze the training answers — {len(report(torch))} values → {out}")
        return
    if which == "borch":
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import borch as torch
        sys.modules["torch"] = torch
    else:
        import torch
    for key, value in report(torch).items():
        print(f"{key}\t{value!r}")


if __name__ == "__main__":
    import sys

    main(sys.argv)
