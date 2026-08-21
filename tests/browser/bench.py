"""Measures **a real training step** with ResNet-18 (the CIFAR edition).

A number measured on one layer does not count the bandwidth-bound operations — BN, ReLU, the
residual addition. Whether the goal ("an epoch in N minutes") is met comes only from an actual
step that includes those, which is why this file exists instead of an estimate divided out of
FLOPs.

It runs inside a browser:

    uv run --with playwright python tests/browser/run.py --lib borch_webgpu --headed --bench
"""

import asyncio
import gc as _gc

import js
import numpy as np

import borchvision as vision

CIFAR_TRAIN_IMAGES = 50000
# CIFAR-10's usual values. Without the normalisation the first epoch catches on noticeably slower.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
_shuffle = np.random.default_rng(0)


def _block(L, cin, cout, stride):
    """ResNet's basic block. A 1×1 goes in only when the shortcut has to change shape."""

    class Block(L.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = L.nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.bn1 = L.nn.BatchNorm2d(cout)
            self.conv2 = L.nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
            self.bn2 = L.nn.BatchNorm2d(cout)
            self.shrinks = stride != 1 or cin != cout
            if self.shrinks:
                self.dconv = L.nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
                self.dbn = L.nn.BatchNorm2d(cout)

        def forward(self, x):
            out = L.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            side = self.dbn(self.dconv(x)) if self.shrinks else x
            return L.relu(out + side)

    return Block()


def resnet18(L, num_classes=10):
    """The CIFAR edition — a 3×3 stem with no max-pool. Taking 32×32 through a 7×7 stem shrinks it too far."""

    class Net(L.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = L.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            self.bn = L.nn.BatchNorm2d(64)
            self.body = L.nn.Sequential(
                _block(L, 64, 64, 1), _block(L, 64, 64, 1),
                _block(L, 64, 128, 2), _block(L, 128, 128, 1),
                _block(L, 128, 256, 2), _block(L, 256, 256, 1),
                _block(L, 256, 512, 2), _block(L, 512, 512, 1))
            self.pool = L.nn.AdaptiveAvgPool2d(1)
            self.fc = L.nn.Linear(512, num_classes)

        def forward(self, x):
            x = L.relu(self.bn(self.stem(x)))
            x = self.pool(self.body(x))
            return self.fc(x.flatten(1))

    return Net()


def run(L, batch=32, steps=5, warmup=2):
    """Measures one step's wall-clock time. (a result dict)"""
    rng = np.random.default_rng(0)
    x = L.tensor(rng.standard_normal((batch, 3, 32, 32)).astype(np.float32))
    y = L.tensor(rng.integers(0, 10, batch).astype(np.int64))

    model = resnet18(L)
    n_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    opt = L.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = L.nn.CrossEntropyLoss()

    def one():
        with L.scope():
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            return loss.item()          # has to be read inside the scope

    for _ in range(warmup):
        one()

    # **Python garbage is collected before the baseline is taken.**
    #
    # Otherwise what the warm-up left behind is released late, inside the measurement window,
    # and the division below **turns that one cleanup into a per-step ratio.** A "leak of -24.8"
    # really did come out at batch 32, and changing the step count to 5, 10 and 20 showed the
    # total was always either 0 or exactly -124 — meaning a one-off, not per-step. Being
    # negative it was not leaking, but instrumentation that reports a number other than what its
    # name says cannot be believed the next time it meets a real leak either.
    _gc.collect()
    # **The instrumentation is asked of the library.** `js.tf.memory()` used to be called
    # directly here, and that tied this benchmark to TF.js so it could not run on another
    # implementation — measuring two backends by the same yardstick got stuck on these three
    # lines. For "the same yardstick" to hold, the yardstick must not know one backend.
    before = L.memory()["tensors"]
    t0 = js.performance.now()
    last = None
    for _ in range(steps):
        last = one()
    per_step = (js.performance.now() - t0) / steps
    leak = (L.memory()["tensors"] - before) / steps

    steps_per_epoch = -(-CIFAR_TRAIN_IMAGES // batch)
    return {
        "batch": batch,
        "params": n_params,
        "ms_per_step": round(per_step, 1),
        "epoch_min": round(per_step * steps_per_epoch / 60000, 2),
        "leak_per_step": round(leak, 1),
        "loss": round(float(last), 4),
        "gpu_mb": round(L.memory()["bytes"] / 1e6, 1),
    }


def run_with_loader(L, batch=128, images=5120, steps=5, warmup=2):
    """Measures through the DataLoader — **a CPU→GPU upload is attached to every batch.**

    `run` above keeps using the same tensor. Real training uploads new data every batch, so how
    much that adds to the step time has to be measured separately.

    The reduced image count is deliberate — holding fifty thousand as float32 is 614MB, and
    that would ask about the Python heap rather than what this measurement asks about (the
    upload cost).
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((images, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, 10, images).astype(np.int64)
    loader = L.utils.data.DataLoader(
        L.utils.data.TensorDataset(x, y), batch_size=batch, shuffle=True, drop_last=True)

    model = resnet18(L)
    opt = L.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = L.nn.CrossEntropyLoss()

    def go(limit):
        done, last = 0, None
        for bx, by in loader:
            with L.scope():
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
                last = loss.item()
            done += 1
            if done >= limit:
                break
        return last

    go(warmup)
    t0 = js.performance.now()
    last = go(steps)
    per_step = (js.performance.now() - t0) / steps
    steps_per_epoch = -(-CIFAR_TRAIN_IMAGES // batch)
    return {"batch": batch, "ms_per_step": round(per_step, 1),
            "epoch_min": round(per_step * steps_per_epoch / 60000, 2),
            "loss": round(float(last), 4)}


async def cifar_from(L, url, name="cifar-batch1.bin"):
    """Fetches one chunk of CIFAR-10, caches it and unpacks it.

    Why the URL is an argument: **the original serves no CORS header** (measured). It cannot be
    fetched from a browser directly, so either host it yourself, use a mirror that gives CORS,
    or have the user pick the file and put it in with `cache_put`. Here it is the first — a
    local server serves it.
    """
    raw = await L.fetch_cached(url, name)
    return L.decode_cifar10(raw)


def run_real(L, x, y, batch=128, steps=20, lr=0.05):
    """Trains on real images. Looks at **whether the loss goes down** — with synthetic data the
    labels are random so it cannot go down, and that part went unconfirmed."""
    loader = L.utils.data.DataLoader(
        L.utils.data.TensorDataset(x, y), batch_size=batch, shuffle=True, drop_last=True)
    model = resnet18(L)
    opt = L.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = L.nn.CrossEntropyLoss()

    losses, done = [], 0
    t0 = js.performance.now()
    while done < steps:
        for bx, by in loader:
            with L.scope():
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
                losses.append(round(float(loss.item()), 4))
            done += 1
            if done >= steps:
                break
    per_step = (js.performance.now() - t0) / steps
    return {"steps": steps, "batch": batch,
            "first": losses[0], "last": losses[-1],
            "curve": losses[::max(1, steps // 8)],
            "ms_per_step": round(per_step, 1),
            "epoch_min": round(per_step * -(-CIFAR_TRAIN_IMAGES // batch) / 60000, 2)}


async def _breathe():
    """Gives the event loop room to breathe.

    **Without it the measurement does not reach the end.** Run as a synchronous loop, training
    kills the whole page at around 50-70 seconds (measured: all three times around there). That
    it is not a leak was confirmed separately — between epochs the tensor count sat flat at
    5,148 and the GPU bytes at 136MB. For a measurement whose point is running long, this one
    line is a condition of the measurement itself.
    """
    await asyncio.sleep(0)


async def accuracy(L, model, x, y, batch=250):
    """The fraction correct. Measuring it **on data not used in training** is the whole of this
    function.

    Calling `model.eval()` is the caller's job — if BatchNorm uses the training statistics the
    value moves with the batch composition, and what is measured is batch luck rather than
    accuracy.
    """
    right = 0
    with L.no_grad():
        for i in range(0, len(y), batch):
            with L.scope():
                out = model(L.tensor(_prep(x[i:i + batch])))
                pred = out.numpy().argmax(axis=1)
            right += int((pred == y[i:i + batch]).sum())
            await _breathe()
    return right / len(y)


def _prep(x, augment=False):
    """Takes [0,1] NCHW into the shape the model wants. Augmentation comes **before**
    normalisation — padding the edge with 0 and then normalising is what measures that 0 by the
    same ruler as the other pixels."""
    if augment:
        x = vision.augment_batch(x, crop=32, padding=4, hflip_p=0.5)
    return vision.Normalize(CIFAR_MEAN, CIFAR_STD)(x).astype(np.float32)


async def train_eval(L, train, test, epochs=6, batch=128, lr=0.05, augment=False):
    """Measures **both training accuracy and test accuracy** every epoch.

    Why both are measured is this function's point. Training accuracy alone always rises, so it
    looks like it is going well. **The difference** between the two is overfitting, and that
    difference is what augmentation exists to reduce. Test accuracy alone will not do either —
    when it stops rising, it does not separate not-yet-learned from already-memorised.
    """
    xtr, ytr = train
    xte, yte = test
    model = resnet18(L)
    opt = L.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = L.nn.CrossEntropyLoss()
    n = len(ytr) - len(ytr) % batch
    rows = []
    for _ in range(epochs):
        order = _shuffle.permutation(len(ytr))[:n]
        model.train()
        t0 = js.performance.now()
        last = 0.0
        for i in range(0, n, batch):
            sel = order[i:i + batch]
            bx = _prep(xtr[sel], augment=augment)
            with L.scope():
                opt.zero_grad()
                loss = crit(model(L.tensor(bx)), L.tensor(ytr[sel]))
                loss.backward()
                opt.step()
                last = float(loss.item())
            await _breathe()
        secs = (js.performance.now() - t0) / 1000
        model.eval()
        row = {"train": await accuracy(L, model, xtr, ytr),
               "test": await accuracy(L, model, xte, yte),
               "loss": round(last, 3), "sec": round(secs, 1)}
        rows.append(row)
        # Emitted per epoch. If a long measurement dies partway the return value is lost whole,
        # while what went to the console remains — it really did die once and nothing was saved.
        # The tensor count and bytes go out too. When a long measurement dies, those two numbers
        # are what separate **a leak from the browser killing it** — without them both are
        # plausible and neither can be chosen.
        mem = L.memory()
        js.console.log(f"[bench] epoch {len(rows)}/{epochs} augment={augment} "
                       f"train {row['train']:.3f} test {row['test']:.3f} {row['sec']}s "
                       f"tensors {mem['tensors']} GPU {mem['bytes'] / 1e6:.0f}MB")
    return rows


async def report_accuracy(L, train, test, epochs=6, batch=128, only=None):
    """Runs the augmented side and the unaugmented side **under the same conditions.**

    Measuring one side alone leaves a single number, "accuracy X%", and that number alone cannot
    say whether augmentation was in. It is the same kind of hole as this file measuring nothing
    but ms/step until now.

    `only` runs one side, and **that is the better experiment.** Running both back to back in
    one session leaves the second model with **different initial weights**, because the random
    generator has already advanced — what is being measured is augmentation's effect, and a
    difference in initial values gets mixed in. A freshly loaded page restarts `_shuffle` from
    seed 0, so both sides start from the same place.
    """
    picked = (("without augmentation", False), ("with augmentation", True))
    if only is not None:
        picked = tuple(p for p in picked if p[1] == only)
    out = []
    for tag, aug in picked:
        rows = await train_eval(L, train, test, epochs=epochs, batch=batch, augment=aug)
        out.append(f"{tag}")
        for i, r in enumerate(rows, 1):
            out.append(f"  epoch {i}  train {r['train']:.1%}  test {r['test']:.1%}  "
                       f"gap {r['train'] - r['test']:+.1%}  loss {r['loss']}  {r['sec']}s")
    return ("ResNet-18 (CIFAR) · accuracy\n"
            f"{len(train[1])} training / {len(test[1])} test images, batch {batch}\n"
            + "\n".join(out))


def report(L, batches=(16, 32, 64)):
    lines = []
    for b in batches:
        try:
            r = run(L, batch=b)
            lines.append(
                f"batch {r['batch']:>3}  {r['ms_per_step']:>8.1f} ms/step  "
                f"epoch {r['epoch_min']:>5.2f} min  leak {r['leak_per_step']:>4.1f}  "
                f"GPU {r['gpu_mb']:>6.1f}MB  loss {r['loss']}")
        except Exception as exc:                                    # noqa: BLE001
            lines.append(f"batch {b:>3}  failed: {type(exc).__name__}: {str(exc)[:120]}")
    return "ResNet-18 (CIFAR) · a real training step, per batch size\n" + "\n".join(lines)
