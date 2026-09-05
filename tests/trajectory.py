"""The trajectory golden — **whole training runs** frozen from torch, not single ops.

    stage 1 (native)   uv run --project ~/git/borch python tests/trajectory.py dump
    stage 2 (native)   uv run --project ~/git/borch python -m pytest tests/test_trajectory.py
    stage 2 (browser)  uv run --with playwright python tests/browser/trajectory_py.py

The 4,744 golden cases ask one operation at a time. Both silent bugs of 4 September —
the scalar cache that an in-place write poisoned, the in-place arithmetic that filled a
buffer with NaN — passed every one of them and only showed when a loop ran: the second
step read what the first had corrupted. So this asks the loop. Three recipes, the field
trainer's paths: a linear head on cached features, a small CNN from scratch, and a
two-level U-Net with a skip connection — one logit per pixel.
Everything a run depends on is pinned — the data (numpy, seeded), the initial weights
(written by torch, loaded by both), the order (no shuffle), the optimiser — so the only
thing free to differ is the arithmetic, and that is what is compared: the loss at every
step and the predictions at the end.

The recipes are written against a torch-shaped library `L`, so `build(torch)` and
`build(borch)` are the same code. `dump` runs torch and writes `tests/trajectory.json`
beside the initial weights in `tests/trajectory/`; `check(L)` runs `L` and returns the
largest loss deviation and the prediction agreement.
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WEIGHTS = HERE / "trajectory"
FROZEN = HERE / "trajectory.json"

# ── the data: seeded numpy, the same bytes on every side ──────────────────────────────

def head_data():
    """200 feature rows of 64-d in 5 clusters — what a frozen backbone hands a head."""
    rng = np.random.default_rng(11)
    centres = rng.standard_normal((5, 64)).astype(np.float32) * 2
    y = np.repeat(np.arange(5), 40)
    x = centres[y] + rng.standard_normal((200, 64)).astype(np.float32)
    return x.astype(np.float32), y.astype(np.int64)


def cnn_data():
    """32 images of 3×16×16 in 3 classes — a low-frequency template each, plus noise."""
    rng = np.random.default_rng(12)
    templates = [rng.standard_normal((4, 4, 3)).astype(np.float32) for _ in range(3)]
    idx = np.arange(16) * 4 // 16
    y = np.arange(32) % 3
    x = np.stack([0.5 + 0.3 * templates[k][idx][:, idx] + 0.15 * rng.standard_normal((16, 16, 3)) for k in y])
    return x.transpose(0, 3, 1, 2).astype(np.float32), y.astype(np.int64)


def unet_data():
    """16 images of 3×16×16, each a bright disc on a dark ground, and the disc's mask —
    the smallest segmentation problem that still has an encoder, a decoder and a skip."""
    rng = np.random.default_rng(13)
    yy, xx = np.mgrid[0:16, 0:16]
    x, m = [], []
    for _ in range(16):
        cy, cx, r = rng.uniform(4, 12), rng.uniform(4, 12), rng.uniform(2.5, 4.5)
        disc = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.float32)
        colour = rng.uniform(0.5, 1.0, 3).astype(np.float32)
        img = 0.2 + disc[None] * colour[:, None, None] + 0.1 * rng.standard_normal((3, 16, 16)).astype(np.float32)
        x.append(img); m.append(disc[None])
    return np.stack(x).astype(np.float32), np.stack(m).astype(np.float32)


def build_unet(L):
    """A two-level U-Net: conv → pool → conv → transposed conv → concat with the skip →
    conv → 1×1 to one logit per pixel."""
    class UNet(L.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = L.nn.Sequential(L.nn.Conv2d(3, 8, 3, padding=1), L.nn.ReLU())
            self.pool = L.nn.MaxPool2d(2)
            self.enc2 = L.nn.Sequential(L.nn.Conv2d(8, 16, 3, padding=1), L.nn.ReLU())
            self.up = L.nn.ConvTranspose2d(16, 8, 2, stride=2)
            self.dec = L.nn.Sequential(L.nn.Conv2d(16, 8, 3, padding=1), L.nn.ReLU(), L.nn.Conv2d(8, 1, 1))

        def forward(self, x):
            a = self.enc1(x)
            b = self.enc2(self.pool(a))
            return self.dec(L.cat([self.up(b), a], 1))
    return UNet()


RECIPES = {
    # name: (data, build, optimiser, steps)
    "head": (head_data,
             lambda L: L.nn.Linear(64, 5),
             # Gentle enough that the curve carries information to the last step — at
             # lr 0.1 torch's loss underflowed to 0.0 by step 40 and there was nothing
             # left to compare.
             lambda L, params: L.optim.SGD(params, lr=0.02, momentum=0.9),
             60),
    "cnn": (cnn_data,
            lambda L: L.nn.Sequential(
                L.nn.Conv2d(3, 8, 3, padding=1), L.nn.BatchNorm2d(8), L.nn.ReLU(), L.nn.MaxPool2d(2),
                L.nn.Conv2d(8, 16, 3, padding=1), L.nn.BatchNorm2d(16), L.nn.ReLU(),
                L.nn.AdaptiveAvgPool2d(1), L.nn.Flatten(), L.nn.Linear(16, 3)),
            lambda L, params: L.optim.Adam(params, lr=1e-2),
            40),
    "unet": (unet_data, build_unet,
             lambda L, params: L.optim.Adam(params, lr=1e-2),
             30),
}


def run(L, name, init):
    """One recipe on library `L` from the weights in `init` (a state dict of `L` tensors).
    Returns (losses per step, predictions at the end)."""
    data, build, optimiser, steps = RECIPES[name]
    x_np, y_np = data()
    model = build(L)
    model.load_state_dict(init)
    model.train()
    x, y = L.tensor(x_np), L.tensor(y_np)
    opt = optimiser(L, model.parameters())
    crit = L.nn.BCEWithLogitsLoss() if name == "unet" else L.nn.CrossEntropyLoss()
    losses = []
    scope = getattr(L, "scope", None)               # the browser side frees a step's buffers here
    for _ in range(steps):
        if scope is not None:
            with scope():
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        else:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
    model.eval()
    if scope is not None:
        with scope():
            pred = predict(name, model(x))
    else:
        with L.no_grad():
            pred = predict(name, model(x))
    return losses, pred


def predict(name, out):
    """The run's answer as a flat list of ints: a class per row, or a bit per pixel."""
    if name == "unet":
        return [int(v > 0) for v in out.flatten().tolist()]
    return [int(p) for p in out.argmax(1).tolist()]


def init_path(name):
    return WEIGHTS / f"{name}.safetensors"


def load_init(L, name, path=None):
    """The initial weights as `L` tensors — through borch's own safetensors codec, which
    both sides carry (and which torch reads through here, since the file is plain)."""
    from borch._serialize import parse
    return parse(str(path or init_path(name)), make=lambda arr: L.tensor(np.ascontiguousarray(arr)))


def dump():
    """Stage 1: torch writes the initial weights and the trajectories."""
    import torch
    from borch._serialize import dump as write
    WEIGHTS.mkdir(exist_ok=True)
    frozen = {}
    for name, (_, build, _, _) in RECIPES.items():
        torch.manual_seed(0)
        model = build(torch)
        # The codec asks the converter about every node; a non-tensor answers None.
        write({k: v for k, v in model.state_dict().items()}, str(init_path(name)),
              lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else None)
        losses, pred = run(torch, name, load_init(torch, name))
        frozen[name] = {"losses": losses, "pred": pred}
        truth = RECIPES[name][0]()[1].flatten()
        print(f"{name}: {len(losses)} steps · loss {losses[0]:.4f} → {losses[-1]:.4f} · {sum(p == int(y) for p, y in zip(pred, truth))}/{len(pred)} right")
    FROZEN.write_text(json.dumps(frozen), encoding="utf-8")
    print(f"froze → {FROZEN}")


def check(L, name, init=None, frozen=None):
    """Stage 2: run `L` and measure against torch. Returns a dict with the largest
    relative loss deviation, the step it happened at, and the prediction agreement."""
    frozen = frozen or json.loads(FROZEN.read_text(encoding="utf-8"))
    losses, pred = run(L, name, init if init is not None else load_init(L, name))
    want = frozen[name]
    # Relative to torch's loss, with a floor: below 1e-3 a run has converged, and the
    # float32 noise of two implementations there (2e-6 against 2e-6, measured on the
    # GPU) says nothing about the arithmetic.
    devs = [abs(a - b) / max(abs(b), 1e-3) for a, b in zip(losses, want["losses"])]
    worst = max(range(len(devs)), key=lambda i: devs[i])
    agree = sum(int(a == b) for a, b in zip(pred, want["pred"])) / len(pred)
    return {"steps": len(losses), "worst_rel": devs[worst], "worst_step": worst,
            "loss_torch": want["losses"][worst], "loss_here": losses[worst],
            "final_torch": want["losses"][-1], "final_here": losses[-1], "pred_agree": agree}


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["dump"]:
        dump()
    else:
        import borch
        for name in RECIPES:
            print(name, check(borch, name))
